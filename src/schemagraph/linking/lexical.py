"""LinearRAG-style entity activation for schemas, with no LLM and no embeddings.

LinearRAG's insight: *aligned entities, not extracted relations, are the anchors*;
relations stay in the source and are read at inference. For a schema the entities
are tables, columns, glossary terms and sample values; the "sentences" that mention
them are their names, descriptions and tags. We build a bipartite token graph
(``w:<token>`` nodes with ``mention`` edges) at index time and activate it from the
question at query time. The activated seeds feed Personalized PageRank (docs/DESIGN.md,
stage 1; docs/ENTITIES.md traces each entity kind).

Scoring signals (SignalPilot-inspired, recall-first):

* exact token match on a table/column/term name           1.0
* n-gram (bigram/trigram) equal to a table/column name     1.6
* abbreviation / lemma / glossary-synonym expansion        0.8
* fuzzy token match (difflib, >= 0.86)                      0.6
* token appears only in a description                       0.35 (a tag: 0.5)
* a sample value appears verbatim in the question           1.5 (column) - the "value-based" leg
* glossary term or synonym phrase appears in the question   1.5 (term node)
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

import networkx as nx

from schemagraph.graph.build import SchemaGraph, knode
from schemagraph.model import BusinessTerm, Column, Table

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get", "give", "has", "have",
    "how", "i", "in", "is", "it", "its", "list", "me", "of", "on", "or", "our", "per", "show",
    "that", "the", "their", "there", "these", "this", "to", "top", "was", "we", "were", "what",
    "which", "who", "with", "all", "any", "each", "find", "return", "using", "where", "when",
    "than", "then", "into", "over", "many", "much", "most", "least", "highest", "lowest",
    "between", "before", "after", "during", "last", "first", "recent", "please", "table", "tables",
    "column", "columns", "query", "database", "select", "rows", "row",
    # question scaffolding that also appears in column descriptions
    "among", "given", "if", "more", "less", "within", "can", "you", "your", "based", "use", "used",
    "calculate", "calculated", "determine", "provide", "identify", "want", "analyze", "understand",
    "also", "same", "those", "only", "both", "other", "such", "like", "should", "would", "could",
    "will", "one", "two", "three", "across", "along", "well", "include", "including", "included",
    "consider", "considering", "excluding", "exclude", "according", "corresponding",
    "respectively", "specifically", "information", "details", "detail", "overall", "whose", "them",
    "they", "not", "no", "yes", "does", "do", "did", "had", "just", "make", "made", "compute",
    "computed", "report", "listing", "displaying", "display", "output", "result", "results",
    "answer", "question", "need", "needs", "looking", "look", "focus", "focusing", "taking",
    "take", "account", "thus", "so", "but", "else", "while",
}

DESC_WEIGHT = 0.35  # a token that appears only in a table/column description

ABBREVIATIONS: dict[str, str] = {
    "acct": "account", "addr": "address", "amt": "amount", "avg": "average", "cat": "category",
    "cnt": "count", "cust": "customer", "dept": "department", "desc": "description",
    "dim": "dimension", "dt": "date", "emp": "employee", "fct": "fact", "id": "identifier",
    "inv": "invoice", "loc": "location", "mgr": "manager", "no": "number", "num": "number",
    "org": "organization", "pct": "percent", "prod": "product", "qty": "quantity",
    "stg": "staging", "tbl": "table", "ts": "timestamp", "txn": "transaction", "usr": "user",
    "yr": "year", "mth": "month", "grp": "group", "cd": "code", "nm": "name", "ref": "reference",
    "src": "source", "agg": "aggregate",
}


def _reverse_abbreviations(abbreviations: dict[str, str]) -> dict[str, list[str]]:
    """Map each expansion to its abbreviations, in declaration order."""
    reverse: dict[str, list[str]] = {}
    for short, expansion in abbreviations.items():
        reverse.setdefault(expansion, []).append(short)
    return reverse


# Expansion -> abbreviations ("number" -> ["no", "num"]), so a spelled-out question word also
# reaches abbreviated names.
ABBREV_REVERSE: dict[str, list[str]] = _reverse_abbreviations(ABBREVIATIONS)

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
MAX_VALUE_TOKENS = 6  # sample values longer than this fall back to a regex scan
# Digit-only question tokens shorter than this are ignored as single tokens (years and ids
# survive, "3" or "10" do not). The default of ``LinkOptions.min_numeric_len``; BM25 and the
# embedding activator always use this default.
DEFAULT_MIN_NUMERIC_LEN = 4

# Index-time posting weights: how strongly a token found in each place points at its object.
NAME_WEIGHT = 1.0  # a token of the object's own name (or of a glossary term / synonym)
TAG_WEIGHT = 0.5  # a token of one of the object's tags
BUSINESS_NAME_WEIGHT = 0.9  # a token of a curated business or dbt name
TERM_DESC_WEIGHT = 0.3  # a token of a glossary term's description
# Table properties whose value is an alternative name for the table.
BUSINESS_NAME_PROPERTIES = frozenset({"dbt_name", "business_name"})

# Query-time seed weights (multiplied by the posting weight and, with ``idf``, the token's idf).
TOKEN_WEIGHT = 1.0  # the question token itself
EXPANSION_WEIGHT = 0.8  # its lemma or an abbreviation / expansion of it
FUZZY_WEIGHT = 0.6  # a vocabulary token difflib finds close to a question token
NGRAM_NAME_WEIGHT = 1.6  # a question n-gram equal to an object's whole name (not idf-scaled)
VALUE_WEIGHT = 1.5  # a column sample value found verbatim in the question (not idf-scaled)
GLOSSARY_PHRASE_WEIGHT = 1.5  # a glossary term or synonym phrase found in the question
GLOSSARY_TARGET_WEIGHT = 1.2  # each table/column that phrase's term points at

# Fuzzy matching applies only to tokens with no posting of their own.
FUZZY_CUTOFF = 0.86  # difflib similarity floor
FUZZY_MIN_LEN = 5  # shorter tokens are never fuzzy-matched
FUZZY_MAX_MATCHES = 3  # close vocabulary tokens used per question token

# Longest question n-gram matched against object names (bigrams and trigrams).
NGRAM_MAX = 3
# A sample value (and a one-word value's only word) shorter than this is not evidence.
MIN_VALUE_CHARS = 3
# Floor of the normalised idf, so a token every object shares still counts a little.
IDF_FLOOR = 0.05
# A token in more postings than this gets no ``w:`` node: it would be a hub joining everything.
HUB_TOKEN_MAX_POSTINGS = 200
# ``mention`` edge weight (and affinity) = this * posting weight.
MENTION_EDGE_SCALE = 0.3


def lemma(tok: str) -> str:
    """Strip a plural ending: ``categories`` -> ``category``, ``classes`` -> ``class``."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("ses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, *, keep_stop: bool = False) -> list[str]:
    """Split text into lowercase alphanumeric tokens, breaking camelCase and snake_case.

    Tokens shorter than two characters are dropped unless they are digits.

    Args:
        text: Any name, description or question.
        keep_stop: Keep :data:`STOPWORDS` (dropped by default).

    Returns:
        The tokens in text order.
    """
    text = _CAMEL_RE.sub("_", text)
    tokens = [t for t in _SPLIT_RE.split(text.lower()) if t]
    if not keep_stop:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [t for t in tokens if len(t) >= 2 or t.isdigit()]


def ngrams(tokens: list[str], n_max: int = NGRAM_MAX) -> list[str]:
    """All ``_``-joined n-grams of 2 to ``n_max`` consecutive tokens, bigrams first."""
    out: list[str] = []
    for n in range(2, n_max + 1):
        for i in range(len(tokens) - n + 1):
            out.append("_".join(tokens[i : i + n]))
    return out


# ------------------------------------------------------------------------- index


@dataclass
class LexicalIndex:
    """Inverted index from tokens to schema objects, plus the name, value and phrase lookups.

    Attributes:
        postings: Token -> ``[(node, weight)]``, one entry per node (its best weight), sorted by
            descending weight then node id.
        idf: Token -> normalised idf in ``[IDF_FLOOR, ~1]``; ~1 for a token unique to one object.
        names: Full normalised name (``_``-joined tokens, stopwords kept) -> nodes.
        names_nostop: The same with stopwords dropped, only when that differs
            (``date_of_birth`` -> ``date_birth``).
        values: Sample value (stripped, lowercased) -> column nodes.
        value_grams: ``_``-joined words of a value of at most :data:`MAX_VALUE_TOKENS` words ->
            ``[(value, column node)]``, looked up with the question's word n-grams.
        long_values: Values with more than :data:`MAX_VALUE_TOKENS` words -> column nodes
            (matched by a regex scan).
        phrases: Glossary term or synonym phrase (lowercased) -> term node.
        vocab: Every posting token, sorted (the fuzzy-match candidates).
    """

    postings: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    names: dict[str, list[str]] = field(default_factory=dict)
    names_nostop: dict[str, list[str]] = field(default_factory=dict)
    values: dict[str, list[str]] = field(default_factory=dict)
    value_grams: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    long_values: dict[str, list[str]] = field(default_factory=dict)
    phrases: dict[str, str] = field(default_factory=dict)
    vocab: list[str] = field(default_factory=list)

    def add(self, token: str, node: str, weight: float) -> None:
        """Append a ``(node, weight)`` posting for ``token`` (deduplicated later)."""
        self.postings.setdefault(token, []).append((node, weight))

    def add_name(self, name: str, node: str) -> None:
        """Register an object's full name, with and without stopwords, for n-gram matches."""
        normalised = "_".join(tokenize(name, keep_stop=True))
        self.names.setdefault(normalised, []).append(node)
        nostop = tokenize(name)
        if len(nostop) >= 2 and (alt := "_".join(nostop)) != normalised:
            self.names_nostop.setdefault(alt, []).append(node)

    def add_value(self, value: str, node: str) -> None:
        """Register a column sample value, unless it is too short or purely numeric.

        "A++", "10%", "$50", "1,000", "2023/01/15": letter or number residues are not evidence
        ("U.S." -> ``u_s`` is).
        """
        lower_value = str(value).strip().lower()
        if (
            len(lower_value) < MIN_VALUE_CHARS
            or lower_value.replace(".", "").replace("-", "").isdigit()
        ):
            return
        words = _words(lower_value)
        if (
            not words
            or all(word.isdigit() for word in words)
            or (len(words) == 1 and len(words[0]) < MIN_VALUE_CHARS)
        ):
            return
        self.values.setdefault(lower_value, []).append(node)
        if len(words) > MAX_VALUE_TOKENS:
            self.long_values.setdefault(lower_value, []).append(node)
        else:
            self.value_grams.setdefault("_".join(words), []).append((lower_value, node))


def _words(text: str) -> list[str]:
    """Whole-word units for value matching: lowercase, split on anything non-alphanumeric."""
    return [w for w in _SPLIT_RE.split(text.lower()) if w]


def _name_tokens(name: str) -> list[str]:
    """Tokens of a name (stopwords kept), then each token's abbreviation expansion and lemma."""
    tokens = tokenize(name, keep_stop=True)
    out = list(tokens)
    for token in tokens:
        if token in ABBREVIATIONS:
            out.append(ABBREVIATIONS[token])
        lemmatised = lemma(token)
        if lemmatised != token:
            out.append(lemmatised)
    return out


def _index_name_tokens(index: LexicalIndex, name: str, node: str, weight: float) -> None:
    """Post every name token of ``name`` (expansions included) for ``node``."""
    for token in _name_tokens(name):
        index.add(token, node, weight)


def _index_text(index: LexicalIndex, text: str | None, node: str, weight: float) -> None:
    """Post every non-stopword token of a free-text field for ``node``."""
    for token in tokenize(text or ""):
        index.add(token, node, weight)


def _index_tags(index: LexicalIndex, tags: list[str], node: str) -> None:
    """Post every token of every tag for ``node``."""
    for tag in tags:
        _index_text(index, tag, node, TAG_WEIGHT)


def _index_table(index: LexicalIndex, node: str, table: Table, desc_weight: float) -> None:
    """Index a table's name, description, tags and business names. Mutates ``index``."""
    index.add_name(table.name, node)
    _index_name_tokens(index, table.name, node, NAME_WEIGHT)
    _index_text(index, table.description, node, desc_weight)
    _index_tags(index, table.tags, node)
    for key, value in table.properties.items():
        if key in BUSINESS_NAME_PROPERTIES:
            _index_name_tokens(index, value, node, BUSINESS_NAME_WEIGHT)


def _index_column(index: LexicalIndex, node: str, column: Column, desc_weight: float) -> None:
    """Index a column's name, description, tags, business name and sample values.

    Mutates ``index``.
    """
    index.add_name(column.name, node)
    _index_name_tokens(index, column.name, node, NAME_WEIGHT)
    _index_text(index, column.description, node, desc_weight)
    _index_tags(index, column.tags, node)
    business_name = column.properties.get("business_name")
    if business_name:
        _index_name_tokens(index, business_name, node, BUSINESS_NAME_WEIGHT)
    for value in column.sample_values:
        index.add_value(value, node)


def _index_term(index: LexicalIndex, node: str, term: BusinessTerm) -> None:
    """Index a glossary term's name and synonyms (as phrases and tokens) and its description.

    A phrase shared by two terms points at the one indexed last. Mutates ``index``.
    """
    for phrase in [term.name, *term.synonyms]:
        lower_phrase = phrase.strip().lower()
        if lower_phrase:
            index.phrases[lower_phrase] = node
        _index_name_tokens(index, phrase, node, NAME_WEIGHT)
    _index_text(index, term.description, node, TERM_DESC_WEIGHT)


def _dedupe_postings(index: LexicalIndex) -> None:
    """Keep one posting per node (its max weight), sorted by weight desc then node id.

    Mutates ``index.postings`` in place.
    """
    for token, postings in index.postings.items():
        best: dict[str, float] = {}
        for node, weight in postings:
            best[node] = max(best.get(node, 0.0), weight)
        index.postings[token] = sorted(best.items(), key=lambda x: (-x[1], x[0]))


def _count_schema_objects(graph: nx.Graph) -> int:
    """Number of table, column and term nodes (at least 1): the idf document count."""
    return max(
        1,
        sum(1 for _, d in graph.nodes(data=True) if d.get("ntype") in {"table", "column", "term"}),
    )


def _compute_idf(index: LexicalIndex, n_objects: int) -> None:
    """Fill ``index.idf`` with BM25-style idf normalised by ``log(n_objects + 1)``, floored.

    Mutates ``index.idf``.
    """
    norm = math.log(n_objects + 1)
    for token, postings in index.postings.items():
        df = len(postings)
        index.idf[token] = max(
            IDF_FLOOR,
            math.log((n_objects - df + 0.5) / (df + 0.5) + 1) / norm,
        )


def _add_token_nodes(graph: nx.Graph, index: LexicalIndex) -> None:
    """Add a ``w:`` node per non-hub token and a ``mention`` edge to each object it posts.

    This is LinearRAG's mention matrix: PPR can flow between objects sharing vocabulary.
    Mutates ``graph`` in place.
    """
    for token, postings in index.postings.items():
        token_node = f"w:{token}"
        if len(postings) > HUB_TOKEN_MAX_POSTINGS:
            continue
        graph.add_node(token_node, ntype="token", name=token)
        for node, weight in postings:
            graph.add_edge(
                token_node,
                node,
                etype="mention",
                weight=MENTION_EDGE_SCALE * weight,
                affinity=MENTION_EDGE_SCALE * weight,
            )


def build_index(
    schema_graph: SchemaGraph,
    *,
    add_token_nodes: bool = True,
    desc_weight: float = DESC_WEIGHT,
) -> LexicalIndex:
    """Build the lexical index over every table, column and glossary term of the graph.

    Objects are indexed in graph node order; postings keep their best weight per node; idf is
    computed before any token node exists (docs/DESIGN.md, stage 1).

    Args:
        schema_graph: The merged graph. With ``add_token_nodes`` it gains ``w:`` token nodes
            and ``mention`` edges in place, so build it completely first.
        add_token_nodes: Add the token nodes PPR flows through.
        desc_weight: Posting weight of a token found only in a table/column description.

    Returns:
        The index the activator reads.
    """
    index = LexicalIndex()
    graph = schema_graph.graph
    for node, attrs in list(graph.nodes(data=True)):
        node_type = attrs.get("ntype")
        if node_type == "table":
            _index_table(index, node, schema_graph.tables[attrs["fqn"].lower()], desc_weight)
        elif node_type == "column":
            table = schema_graph.tables[attrs["fqn"].lower()]
            column = table.column(attrs["name"])
            if column is not None:
                _index_column(index, node, column, desc_weight)
        elif node_type == "term":
            _index_term(index, node, schema_graph.terms[attrs["name"]])
    _dedupe_postings(index)
    index.vocab = sorted(index.postings)
    _compute_idf(index, _count_schema_objects(graph))
    if add_token_nodes:
        _add_token_nodes(graph, index)
    return index


# ------------------------------------------------------------------------- activation


@dataclass
class Activation:
    """The seeds one question activates, with the evidence behind each.

    Attributes:
        seeds: Node -> summed seed weight (the PPR personalization).
        matched_terms: Glossary phrases found in the question, longest first.
        matched_values: Sample values found in the question, in match order.
        tokens: The question's non-stopword tokens.
        reasons: Node -> one reason string per bump, in bump order.
    """

    seeds: dict[str, float]
    matched_terms: list[str]
    matched_values: list[str]
    tokens: list[str]
    reasons: dict[str, list[str]] = field(default_factory=dict)

    def bump(self, node: str, w: float, why: str) -> None:
        """Add ``w`` to ``node``'s seed weight and record ``why``."""
        self.seeds[node] = self.seeds.get(node, 0.0) + w
        self.reasons.setdefault(node, []).append(why)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Whether ``phrase`` occurs in ``text`` bounded by non-alphanumerics (or the ends)."""
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _activate_glossary(
    schema_graph: SchemaGraph,
    index: LexicalIndex,
    lower_question: str,
    activation: Activation,
) -> None:
    """Seed glossary terms whose phrase is in the question, and their targets.

    Phrases are tried longest first so "gross revenue" is recorded before "revenue"; targets
    are seeded straight away so they compete as anchors. Mutates ``activation``.
    """
    graph = schema_graph.graph
    for phrase in sorted(index.phrases, key=len, reverse=True):
        if _contains_phrase(lower_question, phrase):
            term_node = index.phrases[phrase]
            activation.bump(term_node, GLOSSARY_PHRASE_WEIGHT, f"glossary phrase '{phrase}'")
            activation.matched_terms.append(phrase)
            for neighbor in graph.neighbors(term_node):
                if graph[term_node][neighbor].get("etype") == "glossary":
                    activation.bump(
                        neighbor,
                        GLOSSARY_TARGET_WEIGHT,
                        f"glossary target of '{phrase}'",
                    )


def _activate_values(index: LexicalIndex, lower_question: str, activation: Activation) -> None:
    """Seed columns whose sample value appears in the question as whole words.

    The question's word n-grams are looked up in the value index instead of scanning every
    value with a regex (that scan was 95 % of activation time on 2,500-value schemas); only
    values longer than :data:`MAX_VALUE_TOKENS` words are scanned. Mutates ``activation``.
    """
    words = _words(lower_question)
    seen_pairs: set[tuple[str, str]] = set()
    for size in range(1, MAX_VALUE_TOKENS + 1):
        for i in range(len(words) - size + 1):
            for value, node in index.value_grams.get("_".join(words[i : i + size]), ()):
                if (value, node) in seen_pairs:
                    continue
                seen_pairs.add((value, node))
                activation.bump(node, VALUE_WEIGHT, f"value '{value}' in question")
                if value not in activation.matched_values:
                    activation.matched_values.append(value)
    for value, nodes in index.long_values.items():
        if _contains_phrase(lower_question, value):
            for node in nodes:
                activation.bump(node, VALUE_WEIGHT, f"value '{value}' in question")
            activation.matched_values.append(value)


def _activate_names(
    index: LexicalIndex,
    tokens: list[str],
    raw_tokens: list[str],
    activation: Activation,
    *,
    ngram_stop: bool,
) -> None:
    """Seed objects whose whole name equals a question n-gram.

    Args:
        index: The lexical index.
        tokens: Question tokens without stopwords.
        raw_tokens: Question tokens with stopwords.
        activation: Receives the seeds; mutated in place.
        ngram_stop: Also match n-grams of ``raw_tokens`` and the stopword-free name forms; a
            name is then one piece of evidence however many of its forms match (the first
            matching gram is the reason).
    """
    if ngram_stop:
        first_gram: dict[str, str] = {}
        for gram in dict.fromkeys(ngrams(tokens, NGRAM_MAX) + ngrams(raw_tokens, NGRAM_MAX)):
            for node in index.names.get(gram, []) + index.names_nostop.get(gram, []):
                first_gram.setdefault(node, gram)
        for node, gram in first_gram.items():
            activation.bump(node, NGRAM_NAME_WEIGHT, f"n-gram '{gram}' == name")
    else:
        for gram in ngrams(tokens, NGRAM_MAX):
            for node in index.names.get(gram, []):
                activation.bump(node, NGRAM_NAME_WEIGHT, f"n-gram '{gram}' == name")


def _expand_token(token: str) -> list[tuple[str, float, str]]:
    """The token, its lemma and its abbreviation forms, each as ``(form, weight, reason)``."""
    forms = [(token, TOKEN_WEIGHT, f"token '{token}'")]
    lemmatised = lemma(token)
    if lemmatised != token:
        forms.append((lemmatised, EXPANSION_WEIGHT, f"lemma '{lemmatised}'"))
    if token in ABBREVIATIONS:
        expansion = ABBREVIATIONS[token]
        forms.append((expansion, EXPANSION_WEIGHT, f"abbrev '{token}'->'{expansion}'"))
    for short in ABBREV_REVERSE.get(token, []):
        forms.append((short, EXPANSION_WEIGHT, f"abbrev '{token}'->'{short}'"))
    return forms


def _bump_postings(
    activation: Activation,
    index: LexicalIndex,
    token: str,
    weight: float,
    why: str,
    *,
    idf: bool,
) -> None:
    """Seed every object ``token`` posts, at ``weight * posting weight * idf scale``.

    Args:
        activation: Receives the seeds; mutated in place.
        index: The lexical index.
        token: An indexed token.
        weight: The form's weight (token, expansion or fuzzy).
        why: Reason recorded for every bump.
        idf: Scale by the token's idf (1.0 otherwise).
    """
    scale = index.idf.get(token, 1.0) if idf else 1.0
    for node, posting_weight in index.postings.get(token, []):
        activation.bump(node, weight * posting_weight * scale, why)


def _activate_tokens(
    index: LexicalIndex,
    tokens: list[str],
    activation: Activation,
    *,
    idf: bool,
    min_numeric_len: int,
) -> None:
    """Seed objects from single question tokens and their expansions.

    Each form counts once, at its first occurrence. A form with no posting falls back to up to
    :data:`FUZZY_MAX_MATCHES` close vocabulary tokens when it is long enough.

    Args:
        index: The lexical index.
        tokens: Question tokens without stopwords.
        activation: Receives the seeds; mutated in place.
        idf: Scale single-token evidence by the token's idf.
        min_numeric_len: Digit-only tokens shorter than this are skipped.
    """
    expanded: list[tuple[str, float, str]] = []
    for token in tokens:
        if token.isdigit() and len(token) < min_numeric_len:
            continue
        expanded.extend(_expand_token(token))
    seen_forms: set[str] = set()
    for form, weight, why in expanded:
        if form in seen_forms:
            continue
        seen_forms.add(form)
        if index.postings.get(form):
            _bump_postings(activation, index, form, weight, why, idf=idf)
        elif len(form) >= FUZZY_MIN_LEN:
            for close in difflib.get_close_matches(
                form,
                index.vocab,
                n=FUZZY_MAX_MATCHES,
                cutoff=FUZZY_CUTOFF,
            ):
                _bump_postings(
                    activation,
                    index,
                    close,
                    FUZZY_WEIGHT,
                    f"fuzzy '{form}'~'{close}'",
                    idf=idf,
                )


def activate(
    schema_graph: SchemaGraph,
    idx: LexicalIndex,
    question: str,
    *,
    idf: bool = True,
    min_numeric_len: int = DEFAULT_MIN_NUMERIC_LEN,
    ngram_stop: bool = True,
) -> Activation:
    """Activate schema objects from a question.

    Four stages, in order, each adding to the same seeds: glossary phrases (and their
    targets), sample values, whole-name n-grams, single tokens with expansions
    (docs/DESIGN.md, stage 1; LinearRAG entity activation).

    Args:
        schema_graph: The graph the index was built on.
        idx: The lexical index.
        question: The natural-language question.
        idf: Scale single-token evidence by how rare the token is across the schema (a token
            shared by half the columns is nearly worthless as a seed); n-gram, value and
            glossary matches are not scaled.
        min_numeric_len: Digit-only tokens shorter than this are ignored as single tokens
            (``5`` from "5-year").
        ngram_stop: Also match question n-grams that keep their stopwords against object names
            ("date of birth" -> ``date_of_birth``, "first name" -> ``first_name``) and
            stopword-free n-grams against the stopword-free form of names.

    Returns:
        The seeds, the matched glossary phrases and values, and a reason per bump.
    """
    lower_question = question.lower()
    raw_tokens = tokenize(question, keep_stop=True)
    tokens = [t for t in raw_tokens if t not in STOPWORDS]
    activation = Activation(seeds={}, matched_terms=[], matched_values=[], tokens=tokens)
    _activate_glossary(schema_graph, idx, lower_question, activation)
    _activate_values(idx, lower_question, activation)
    _activate_names(idx, tokens, raw_tokens, activation, ngram_stop=ngram_stop)
    _activate_tokens(idx, tokens, activation, idf=idf, min_numeric_len=min_numeric_len)
    return activation


def term_targets(schema_graph: SchemaGraph, term_name: str) -> list[str]:
    """Fqns (or node ids, for nodes without one) of the objects a glossary term points at."""
    term_node = knode(term_name)
    graph = schema_graph.graph
    if term_node not in graph:
        return []
    return [
        graph.nodes[neighbor].get("fqn", neighbor)
        for neighbor in graph.neighbors(term_node)
        if graph[term_node][neighbor].get("etype") == "glossary"
    ]
