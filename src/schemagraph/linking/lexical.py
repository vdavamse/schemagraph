"""LinearRAG-style entity activation for schemas, with no LLM and no embeddings.

LinearRAG's insight: *aligned entities, not extracted relations, are the anchors*;
relations stay in the source and are read at inference. For a schema the entities
are tables, columns, glossary terms and sample values; the "sentences" that mention
them are their names, descriptions and tags. We build a bipartite token graph
(``w:<token>`` nodes with ``mention`` edges) at index time and activate it from the
question at query time. The activated seeds feed Personalized PageRank.

Scoring signals (SignalPilot-inspired, recall-first):

* exact token match on a table/column/term name           1.0
* n-gram (bigram/trigram) equal to a table/column name     1.6
* abbreviation / lemma / glossary-synonym expansion        0.8
* fuzzy token match (difflib, >= 0.86)                      0.6
* token appears only in a description or tag               0.35
* a sample value appears verbatim in the question           1.5 (column) - the "value-based" leg
* glossary term or synonym phrase appears in the question   1.5 (term node)
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

from schemagraph.graph.build import SchemaGraph, knode

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get", "give", "has", "have", "how",
    "i", "in", "is", "it", "its", "list", "me", "of", "on", "or", "our", "per", "show", "that", "the",
    "their", "there", "these", "this", "to", "top", "was", "we", "were", "what", "which", "who", "with",
    "all", "any", "each", "find", "return", "using", "where", "when", "than", "then", "into", "over",
    "many", "much", "most", "least", "highest", "lowest", "between", "before", "after", "during",
    "last", "first", "recent", "please", "table", "tables", "column", "columns", "query", "database",
    "select", "rows", "row",
    # question scaffolding that also appears in column descriptions
    "among", "given", "if", "more", "less", "within", "can", "you", "your", "based", "use", "used",
    "calculate", "calculated", "determine", "provide", "identify", "want", "analyze", "understand",
    "also", "same", "those", "only", "both", "other", "such", "like", "should", "would", "could", "will",
    "one", "two", "three", "across", "along", "well", "include", "including", "included", "consider",
    "considering", "excluding", "exclude", "according", "corresponding", "respectively", "specifically",
    "information", "details", "detail", "overall", "whose", "them", "they", "not", "no", "yes", "does", "do",
    "did", "had", "just", "make", "made", "compute", "computed", "report", "listing", "displaying",
    "display", "output", "result", "results", "answer", "question", "need", "needs", "looking", "look",
    "focus", "focusing", "taking", "take", "account", "thus", "so", "but", "else", "while",
}

DESC_WEIGHT = 0.35  # a token that appears only in a table/column description

ABBREVIATIONS: dict[str, str] = {
    "acct": "account", "addr": "address", "amt": "amount", "avg": "average", "cat": "category", "cnt": "count",
    "cust": "customer", "dept": "department", "desc": "description", "dim": "dimension", "dt": "date",
    "emp": "employee", "fct": "fact", "id": "identifier", "inv": "invoice", "loc": "location", "mgr": "manager",
    "no": "number", "num": "number", "org": "organization", "pct": "percent", "prod": "product", "qty": "quantity",
    "stg": "staging", "tbl": "table", "ts": "timestamp", "txn": "transaction", "usr": "user", "yr": "year",
    "mth": "month", "grp": "group", "cd": "code", "nm": "name", "ref": "reference", "src": "source", "agg": "aggregate",
}
ABBREV_REVERSE: dict[str, list[str]] = {}
for _k, _v in ABBREVIATIONS.items():
    ABBREV_REVERSE.setdefault(_v, []).append(_k)

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
MAX_VALUE_TOKENS = 6  # sample values longer than this fall back to a regex scan


def lemma(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("ses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, *, keep_stop: bool = False) -> list[str]:
    text = _CAMEL_RE.sub("_", text)
    toks = [t for t in _SPLIT_RE.split(text.lower()) if t]
    if not keep_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [t for t in toks if len(t) >= 2 or t.isdigit()]


def ngrams(tokens: list[str], n_max: int = 3) -> list[str]:
    out: list[str] = []
    for n in range(2, n_max + 1):
        for i in range(len(tokens) - n + 1):
            out.append("_".join(tokens[i : i + n]))
    return out


# ------------------------------------------------------------------------- index


@dataclass
class LexicalIndex:
    """Inverted index token -> [(node, weight)] plus name index for n-gram matches."""

    postings: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)  # 0..1, ~1 for tokens unique to one object
    names: dict[str, list[str]] = field(default_factory=dict)  # full normalized name -> nodes
    names_nostop: dict[str, list[str]] = field(default_factory=dict)  # same with stopwords dropped, when that differs (date_of_birth -> date_birth)
    values: dict[str, list[str]] = field(default_factory=dict)  # sample value (lower) -> column nodes
    value_grams: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # "_"-joined value words -> [(value, column node)]
    long_values: dict[str, list[str]] = field(default_factory=dict)  # values with more than MAX_VALUE_TOKENS words (regex scan)
    phrases: dict[str, str] = field(default_factory=dict)  # glossary phrase (lower) -> term node
    vocab: list[str] = field(default_factory=list)

    def add(self, token: str, node: str, weight: float) -> None:
        self.postings.setdefault(token, []).append((node, weight))

    def add_name(self, name: str, node: str) -> None:
        norm = "_".join(tokenize(name, keep_stop=True))
        self.names.setdefault(norm, []).append(node)
        nostop = tokenize(name)
        if len(nostop) >= 2 and (alt := "_".join(nostop)) != norm:
            self.names_nostop.setdefault(alt, []).append(node)

    def add_value(self, value: str, node: str) -> None:
        lv = str(value).strip().lower()
        if len(lv) < 3 or lv.replace(".", "").replace("-", "").isdigit():
            return
        self.values.setdefault(lv, []).append(node)
        words = _words(lv)
        if not words:
            return
        if len(words) > MAX_VALUE_TOKENS:
            self.long_values.setdefault(lv, []).append(node)
        else:
            self.value_grams.setdefault("_".join(words), []).append((lv, node))


def _words(text: str) -> list[str]:
    """Whole-word units for value matching: lowercase, split on anything non-alphanumeric."""
    return [w for w in _SPLIT_RE.split(text.lower()) if w]


def _name_tokens(name: str) -> list[str]:
    toks = tokenize(name, keep_stop=True)
    out = list(toks)
    for t in toks:
        if t in ABBREVIATIONS:
            out.append(ABBREVIATIONS[t])
        lt = lemma(t)
        if lt != t:
            out.append(lt)
    return out


def build_index(sg: SchemaGraph, *, add_token_nodes: bool = True, desc_weight: float = DESC_WEIGHT) -> LexicalIndex:
    idx = LexicalIndex()
    g = sg.g
    for n, d in list(g.nodes(data=True)):
        nt = d.get("ntype")
        if nt == "table":
            t = sg.tables[d["fqn"].lower()]
            idx.add_name(t.name, n)
            for tok in _name_tokens(t.name):
                idx.add(tok, n, 1.0)
            for tok in tokenize(t.description or ""):
                idx.add(tok, n, desc_weight)
            for tag in t.tags:
                for tok in tokenize(tag):
                    idx.add(tok, n, 0.5)
            for k, v in t.properties.items():
                if k in {"dbt_name", "business_name"}:
                    for tok in _name_tokens(v):
                        idx.add(tok, n, 0.9)
        elif nt == "column":
            t = sg.tables[d["fqn"].lower()]
            c = t.column(d["name"])
            if c is None:
                continue
            idx.add_name(c.name, n)
            for tok in _name_tokens(c.name):
                idx.add(tok, n, 1.0)
            for tok in tokenize(c.description or ""):
                idx.add(tok, n, desc_weight)
            for tag in c.tags:
                for tok in tokenize(tag):
                    idx.add(tok, n, 0.5)
            for bn in (c.properties.get("business_name"),):
                if bn:
                    for tok in _name_tokens(bn):
                        idx.add(tok, n, 0.9)
            for v in c.sample_values:
                idx.add_value(v, n)
        elif nt == "term":
            term = sg.terms[d["name"]]
            phrases = [term.name, *term.synonyms]
            for p in phrases:
                lp = p.strip().lower()
                if lp:
                    idx.phrases[lp] = n
                for tok in _name_tokens(p):
                    idx.add(tok, n, 1.0)
            for tok in tokenize(term.description or ""):
                idx.add(tok, n, 0.3)
    # dedupe postings (keep max weight per node)
    for tok, posts in idx.postings.items():
        best: dict[str, float] = {}
        for node, w in posts:
            best[node] = max(best.get(node, 0.0), w)
        idx.postings[tok] = sorted(best.items(), key=lambda x: (-x[1], x[0]))
    idx.vocab = sorted(idx.postings)
    n_nodes = max(1, sum(1 for _, d in g.nodes(data=True) if d.get("ntype") in {"table", "column", "term"}))
    norm = math.log(n_nodes + 1)
    for tok, posts in idx.postings.items():
        df = len(posts)
        idx.idf[tok] = max(0.05, math.log((n_nodes - df + 0.5) / (df + 0.5) + 1) / norm)
    if add_token_nodes:
        # LinearRAG mention matrix: token nodes so PPR can flow between objects sharing vocabulary
        for tok, posts in idx.postings.items():
            wn = f"w:{tok}"
            if len(posts) > 200:  # ubiquitous token: skip to avoid hub explosion
                continue
            g.add_node(wn, ntype="token", name=tok)
            for node, w in posts:
                g.add_edge(wn, node, etype="mention", weight=0.3 * w, affinity=0.3 * w)
    return idx


# ------------------------------------------------------------------------- activation


@dataclass
class Activation:
    seeds: dict[str, float]
    matched_terms: list[str]
    matched_values: list[str]
    tokens: list[str]
    reasons: dict[str, list[str]] = field(default_factory=dict)

    def bump(self, node: str, w: float, why: str) -> None:
        self.seeds[node] = self.seeds.get(node, 0.0) + w
        self.reasons.setdefault(node, []).append(why)


def activate(sg: SchemaGraph, idx: LexicalIndex, question: str, *, idf: bool = True, min_numeric_len: int = 4, ngram_stop: bool = True) -> Activation:
    """Activate schema objects from a question.

    ``idf`` scales single-token evidence by how rare the token is across the schema
    (a token shared by half the columns is nearly worthless as a seed); n-gram, value
    and glossary matches are not scaled. Digit-only tokens shorter than
    ``min_numeric_len`` are ignored as single tokens (``5`` from "5-year").
    ``ngram_stop`` also matches question n-grams that keep their stopwords against object
    names ("date of birth" -> ``date_of_birth``, "first name" -> ``first_name``) and
    stopword-free n-grams against the stopword-free form of names.
    """
    q = question.lower()
    raw = tokenize(question, keep_stop=True)
    toks = [t for t in raw if t not in STOPWORDS]
    act = Activation(seeds={}, matched_terms=[], matched_values=[], tokens=toks)

    # 1. glossary phrases (longest first so "gross revenue" beats "revenue")
    for phrase in sorted(idx.phrases, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", q):
            node = idx.phrases[phrase]
            act.bump(node, 1.5, f"glossary phrase '{phrase}'")
            act.matched_terms.append(phrase)
            # propagate straight to glossary targets so they compete as anchors
            for nb in sg.g.neighbors(node):
                if sg.g[node][nb].get("etype") == "glossary":
                    act.bump(nb, 1.2, f"glossary target of '{phrase}'")

    # 2. sample values appearing in the question as whole words (value-based linking); the
    #    question's word n-grams are looked up in the value index instead of scanning every
    #    value with a regex (that scan was 95 % of activation time on 2,500-value schemas)
    words = _words(q)
    seen_pairs: set[tuple[str, str]] = set()
    for size in range(1, MAX_VALUE_TOKENS + 1):
        for i in range(len(words) - size + 1):
            for val, n in idx.value_grams.get("_".join(words[i : i + size]), ()):
                if (val, n) in seen_pairs:
                    continue
                seen_pairs.add((val, n))
                act.bump(n, 1.5, f"value '{val}' in question")
                if val not in act.matched_values:
                    act.matched_values.append(val)
    for val, nodes in idx.long_values.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(val)}(?![a-z0-9])", q):
            for n in nodes:
                act.bump(n, 1.5, f"value '{val}' in question")
            act.matched_values.append(val)

    # 3. n-grams against full object names
    if ngram_stop:
        hit: dict[str, str] = {}  # node -> first gram that matched (a name is one piece of evidence, however many forms match it)
        for ng in dict.fromkeys(ngrams(toks, 3) + ngrams(raw, 3)):
            for n in idx.names.get(ng, []) + idx.names_nostop.get(ng, []):
                hit.setdefault(n, ng)
        for n, ng in hit.items():
            act.bump(n, 1.6, f"n-gram '{ng}' == name")
    else:
        for ng in ngrams(toks, 3):
            for n in idx.names.get(ng, []):
                act.bump(n, 1.6, f"n-gram '{ng}' == name")

    # 4. single tokens, with expansions
    expanded: list[tuple[str, float, str]] = []
    for t in toks:
        if t.isdigit() and len(t) < min_numeric_len:
            continue
        expanded.append((t, 1.0, f"token '{t}'"))
        lt = lemma(t)
        if lt != t:
            expanded.append((lt, 0.8, f"lemma '{lt}'"))
        if t in ABBREVIATIONS:
            expanded.append((ABBREVIATIONS[t], 0.8, f"abbrev '{t}'->'{ABBREVIATIONS[t]}'"))
        for ab in ABBREV_REVERSE.get(t, []):
            expanded.append((ab, 0.8, f"abbrev '{t}'->'{ab}'"))
    seen_tok: set[str] = set()
    for tok, w, why in expanded:
        if tok in seen_tok:
            continue
        seen_tok.add(tok)
        posts = idx.postings.get(tok)
        if posts:
            scale = idx.idf.get(tok, 1.0) if idf else 1.0
            for node, pw in posts:
                act.bump(node, w * pw * scale, why)
        elif len(tok) >= 5:
            for close in difflib.get_close_matches(tok, idx.vocab, n=3, cutoff=0.86):
                scale = idx.idf.get(close, 1.0) if idf else 1.0
                for node, pw in idx.postings.get(close, []):
                    act.bump(node, 0.6 * pw * scale, f"fuzzy '{tok}'~'{close}'")
    return act


def term_targets(sg: SchemaGraph, term_name: str) -> list[str]:
    n = knode(term_name)
    if n not in sg.g:
        return []
    return [sg.g.nodes[m].get("fqn", m) for m in sg.g.neighbors(n) if sg.g[n][m].get("etype") == "glossary"]
