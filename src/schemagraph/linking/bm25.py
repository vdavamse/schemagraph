"""BM25F over one document per table: the sparse-retrieval baseline the graph pipeline is measured against.

Each table is a document with fields (name, column names, descriptions, tags, business names),
each field with its own weight and length normalisation (BM25F, Robertson & Zaragoza 2004).
This is what a text-to-SQL retriever without a graph does; ``LinkOptions(ranker="bm25")`` ranks
tables with it alone and ``ranker="rrf"`` fuses it with the PPR ranking by reciprocal rank.
Query tokens get the same expansions as the graph activator (lemma, abbreviation at 0.8).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from schemagraph.graph.build import SchemaGraph
from schemagraph.linking.lexical import (
    ABBREV_REVERSE,
    ABBREVIATIONS,
    STOPWORDS,
    _name_tokens,
    lemma,
    tokenize,
)

FIELD_WEIGHT = {"name": 1.0, "columns": 1.0, "business": 0.9, "tags": 0.5, "desc": 0.35}


@dataclass
class BM25Index:
    tf: dict[str, dict[str, Counter]] = field(default_factory=dict)  # table fqn -> field -> Counter(token)
    length: dict[str, dict[str, int]] = field(default_factory=dict)  # table fqn -> field -> tokens in field
    avglen: dict[str, float] = field(default_factory=dict)
    df: Counter = field(default_factory=Counter)
    n_docs: int = 0

    def idf(self, tok: str) -> float:
        d = self.df.get(tok, 0)
        return math.log((self.n_docs - d + 0.5) / (d + 0.5) + 1.0)


def build_bm25(sg: SchemaGraph) -> BM25Index:
    idx = BM25Index()
    for t in sg.tables.values():
        fields: dict[str, list[str]] = defaultdict(list)
        fields["name"] += _name_tokens(t.name)
        fields["desc"] += tokenize(t.description or "")
        for tag in t.tags:
            fields["tags"] += tokenize(tag)
        for k, v in t.properties.items():
            if k in {"dbt_name", "business_name"}:
                fields["business"] += _name_tokens(v)
        for c in t.columns:
            fields["columns"] += _name_tokens(c.name)
            fields["desc"] += tokenize(c.description or "")
            for tag in c.tags:
                fields["tags"] += tokenize(tag)
            if c.properties.get("business_name"):
                fields["business"] += _name_tokens(c.properties["business_name"])
        idx.tf[t.fqn] = {f: Counter(toks) for f, toks in fields.items()}
        idx.length[t.fqn] = {f: len(toks) for f, toks in fields.items()}
        for tok in {tok for toks in fields.values() for tok in toks}:
            idx.df[tok] += 1
    idx.n_docs = len(idx.tf)
    for f in FIELD_WEIGHT:
        lens = [ln.get(f, 0) for ln in idx.length.values()]
        idx.avglen[f] = (sum(lens) / len(lens)) if lens else 0.0
    return idx


def query_terms(question: str) -> dict[str, float]:
    """Question tokens with the activator's expansions; weight is the max over forms."""
    out: dict[str, float] = {}
    for t in tokenize(question):
        if t in STOPWORDS or (t.isdigit() and len(t) < 4):
            continue
        forms = [(t, 1.0), (lemma(t), 0.8)]
        if t in ABBREVIATIONS:
            forms.append((ABBREVIATIONS[t], 0.8))
        forms += [(ab, 0.8) for ab in ABBREV_REVERSE.get(t, [])]
        for tok, w in forms:
            out[tok] = max(out.get(tok, 0.0), w)
    return out


def bm25_scores(idx: BM25Index, question: str, *, k1: float = 1.2, b: float = 0.75) -> dict[str, float]:
    terms = query_terms(question)
    scores: dict[str, float] = {}
    for fqn, tfs in idx.tf.items():
        s = 0.0
        for tok, qw in terms.items():
            tf_norm = 0.0
            for f, wf in FIELD_WEIGHT.items():
                c = tfs.get(f, {}).get(tok, 0)
                if c:
                    avg = idx.avglen.get(f) or 1.0
                    tf_norm += wf * c / (1.0 - b + b * idx.length[fqn].get(f, 0) / avg)
            if tf_norm:
                s += qw * idx.idf(tok) * tf_norm * (k1 + 1.0) / (k1 + tf_norm)
        if s > 0:
            scores[fqn] = s
    return scores


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for i, fqn in enumerate(ranking):
            out[fqn] += 1.0 / (k + i + 1)
    return dict(out)
