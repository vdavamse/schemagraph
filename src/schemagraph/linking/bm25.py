"""BM25F over one document per table: the sparse-retrieval baseline for the graph pipeline.

Each table is a document with fields (name, column names, descriptions, tags, business names),
each field with its own weight and length normalisation (BM25F, Robertson & Zaragoza 2004).
This is what a text-to-SQL retriever without a graph does; ``LinkOptions(ranker="bm25")`` ranks
tables with it alone and ``ranker="rrf"`` fuses it with the PPR ranking by reciprocal rank
(docs/DESIGN.md, stage 3). Query tokens get the same expansions as the graph activator (lemma,
abbreviation at 0.8).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from schemagraph.graph.build import SchemaGraph
from schemagraph.linking.lexical import (
    ABBREV_REVERSE,
    ABBREVIATIONS,
    BUSINESS_NAME_PROPERTIES,
    DEFAULT_MIN_NUMERIC_LEN,
    EXPANSION_WEIGHT,
    STOPWORDS,
    TOKEN_WEIGHT,
    _name_tokens,
    lemma,
    tokenize,
)

# Per-field weight, mirroring the lexical index's posting weights.
FIELD_WEIGHT = {"name": 1.0, "columns": 1.0, "business": 0.9, "tags": 0.5, "desc": 0.35}


@dataclass
class BM25Index:
    """Per-table field term frequencies and the corpus statistics BM25F needs.

    Attributes:
        tf: Table fqn -> field -> token counts.
        length: Table fqn -> field -> number of tokens in the field.
        avglen: Field -> average length over all tables (0 where absent).
        df: Token -> number of tables containing it in any field.
        postings: Token -> tables containing it (any field), in table order.
        n_docs: Number of tables.
    """

    tf: dict[str, dict[str, Counter]] = field(default_factory=dict)
    length: dict[str, dict[str, int]] = field(default_factory=dict)
    avglen: dict[str, float] = field(default_factory=dict)
    df: Counter = field(default_factory=Counter)
    postings: dict[str, list[str]] = field(default_factory=dict)
    n_docs: int = 0

    def idf(self, tok: str) -> float:
        """BM25 idf of a token (always positive)."""
        doc_freq = self.df.get(tok, 0)
        return math.log((self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)


def build_bm25(schema_graph: SchemaGraph) -> BM25Index:
    """Build the BM25F index with one document per table of the graph."""
    index = BM25Index()
    for table in schema_graph.tables.values():
        fields: dict[str, list[str]] = defaultdict(list)
        fields["name"] += _name_tokens(table.name)
        fields["desc"] += tokenize(table.description or "")
        for tag in table.tags:
            fields["tags"] += tokenize(tag)
        for key, value in table.properties.items():
            if key in BUSINESS_NAME_PROPERTIES:
                fields["business"] += _name_tokens(value)
        for column in table.columns:
            fields["columns"] += _name_tokens(column.name)
            fields["desc"] += tokenize(column.description or "")
            for tag in column.tags:
                fields["tags"] += tokenize(tag)
            if column.properties.get("business_name"):
                fields["business"] += _name_tokens(column.properties["business_name"])
        index.tf[table.fqn] = {name: Counter(tokens) for name, tokens in fields.items()}
        index.length[table.fqn] = {name: len(tokens) for name, tokens in fields.items()}
        for token in dict.fromkeys(token for tokens in fields.values() for token in tokens):
            index.df[token] += 1
            index.postings.setdefault(token, []).append(table.fqn)
    index.n_docs = len(index.tf)
    for field_name in FIELD_WEIGHT:
        lengths = [by_field.get(field_name, 0) for by_field in index.length.values()]
        index.avglen[field_name] = (sum(lengths) / len(lengths)) if lengths else 0.0
    return index


def query_terms(question: str) -> dict[str, float]:
    """Question tokens with the activator's expansions; weight is the max over forms."""
    out: dict[str, float] = {}
    for token in tokenize(question):
        if token in STOPWORDS or (token.isdigit() and len(token) < DEFAULT_MIN_NUMERIC_LEN):
            continue
        forms = [(token, TOKEN_WEIGHT), (lemma(token), EXPANSION_WEIGHT)]
        if token in ABBREVIATIONS:
            forms.append((ABBREVIATIONS[token], EXPANSION_WEIGHT))
        forms += [(short, EXPANSION_WEIGHT) for short in ABBREV_REVERSE.get(token, [])]
        for form, weight in forms:
            out[form] = max(out.get(form, 0.0), weight)
    return out


def bm25_scores(
    idx: BM25Index,
    question: str,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    """Score tables against a question with BM25F.

    Only the tables in some query token's postings are scored; the rest would score 0.

    Args:
        idx: The BM25F index.
        question: The natural-language question.
        k1: Term-frequency saturation.
        b: Length-normalisation strength, in [0, 1].

    Returns:
        Table fqn -> positive score.
    """
    scores: dict[str, float] = {}
    for token, query_weight in query_terms(question).items():
        tables = idx.postings.get(token)
        if not tables:
            continue
        idf = idx.idf(token)
        for fqn in tables:
            field_tfs = idx.tf[fqn]
            tf_norm = 0.0
            for field_name, field_weight in FIELD_WEIGHT.items():
                count = field_tfs.get(field_name, {}).get(token, 0)
                if count:
                    avg = idx.avglen.get(field_name) or 1.0
                    field_length = idx.length[fqn].get(field_name, 0)
                    tf_norm += field_weight * count / (1.0 - b + b * field_length / avg)
            if tf_norm:
                scores[fqn] = scores.get(fqn, 0.0) + query_weight * idf * tf_norm * (k1 + 1.0) / (
                    k1 + tf_norm
                )
    return {fqn: score for fqn, score in scores.items() if score > 0}


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> dict[str, float]:
    """Fuse rankings: each item scores ``sum 1 / (k + rank)`` over the rankings it appears in.

    Items keep first-appearance order across the rankings, in argument order.
    """
    out: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for i, fqn in enumerate(ranking):
            out[fqn] += 1.0 / (k + i + 1)
    return dict(out)
