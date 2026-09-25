"""The optional ``bm25s`` library as the sparse ranker: an ablation backend for linking/bm25.py.

``LinkOptions(bm25_backend="bm25s")`` scores tables with the bm25s package (xhluca/bm25s, extra
``bm25s``) instead of the hand-rolled BM25F, so the ranking stage can lean on a maintained
library. bm25s has no multi-field model, no field weights and no weighted query terms, so the
adapter approximates BM25F the simple way: one document per table, built from the same fields
as ``table_fields``, with each field's tokens repeated ``FIELD_REPEAT`` times (roughly its BM25F
weight over the smallest one), and one ``get_scores`` call per query-term weight, summed.

Unlike BM25F this shares one length normalisation across fields, so a name hit on a wide table
is diluted by its columns and descriptions, and repetition saturates through a single k1. BM25F
stays the default until the ``bm25s`` backend matches it on Spider 2.0-Lite strict recall and
strict@7 (bench_results/README.md; docs/DESIGN.md, stage 3).
"""

from __future__ import annotations

from collections import defaultdict

from schemagraph.graph.build import SchemaGraph
from schemagraph.linking.bm25 import query_terms, table_fields

# Times each field's tokens are repeated in the table's document: FIELD_WEIGHT / 0.35, rounded.
FIELD_REPEAT = {"name": 3, "columns": 3, "business": 3, "tags": 1, "desc": 1}
# Same saturation and length normalisation as the BM25F default.
K1 = 1.2
B = 0.75
# The bm25s methods benchmarked on Spider 2.0-Lite. Both keep IDF positive, so every table that
# holds a query term scores > 0, as in BM25F. robertson and atire are left out on purpose: bm25s
# clamps robertson's IDF to 0 once a token is in more than half the tables (atire's once it is in
# all of them), so those tables would score 0 and look unmatched.
METHODS = ("lucene", "bm25+")


def _import_bm25s():
    """The ``bm25s`` module, or an ImportError that says how to install it."""
    try:
        import bm25s
    except ImportError as exc:
        raise ImportError(
            "bm25_backend='bm25s' needs the bm25s package: uv sync --extra bm25s"
        ) from exc
    return bm25s


class BM25SIndex:
    """One bm25s index over the graph's tables, for one scoring method.

    Args:
        schema_graph: The graph whose tables become documents, in table order.
        method: A bm25s method in ``METHODS`` (``"lucene"`` or ``"bm25+"``).

    Raises:
        ValueError: ``method`` is not one of ``METHODS``.
    """

    def __init__(self, schema_graph: SchemaGraph, method: str = "lucene"):
        if method not in METHODS:
            raise ValueError(f"unknown bm25_method {method!r}; expected one of {METHODS}")
        bm25s = _import_bm25s()
        self.method = method
        self.fqns: list[str] = []
        # Token -> indexes of the documents containing it; only these can score.
        self.postings: dict[str, list[int]] = defaultdict(list)
        corpus: list[list[str]] = []
        for table in schema_graph.tables.values():
            doc: list[str] = []
            for field_name, tokens in table_fields(table).items():
                doc += tokens * FIELD_REPEAT[field_name]
            for token in dict.fromkeys(doc):
                self.postings[token].append(len(corpus))
            self.fqns.append(table.fqn)
            corpus.append(doc)
        self.postings = dict(self.postings)
        self.retriever = bm25s.BM25(k1=K1, b=B, method=method, dtype="float64", backend="numpy")
        if self.postings:  # bm25s cannot index an empty vocabulary (e.g. only one-letter names)
            self.retriever.index(corpus, show_progress=False)

    def scores(self, question: str) -> dict[str, float]:
        """Score tables against a question; same contract as ``bm25_scores``.

        Query terms are grouped by weight (surface forms, expansions), each group scored with
        one ``get_scores`` call and the vectors summed with their weights. Only tables holding
        some query term are kept: bm25+ gives every document a positive floor per query term.

        Returns:
            Table fqn -> positive score.
        """
        groups: dict[float, list[str]] = defaultdict(list)
        for token, weight in query_terms(question).items():
            if token in self.postings:
                groups[weight].append(token)
        if not groups:
            return {}  # bm25s raises IndexError on an empty token list
        total = None
        for weight, tokens in groups.items():
            vector = weight * self.retriever.get_scores(tokens)
            total = vector if total is None else total + vector
        candidates = {i for tokens in groups.values() for t in tokens for i in self.postings[t]}
        return {self.fqns[i]: float(total[i]) for i in sorted(candidates)}
