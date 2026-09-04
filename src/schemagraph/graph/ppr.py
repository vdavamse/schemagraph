"""HippoRAG-style Personalized PageRank over the heterogeneous schema graph.

Seeds are the nodes activated by the question (tokens, glossary terms, columns,
tables) with their lexical scores; PPR spreads that activation over ``contains``,
``relation``, ``glossary`` and ``mention`` edges so that a table connected to many
activated columns, or one hop from a strongly activated table, rises to the top.

HippoRAG's *node specificity* is reproduced by down-weighting seeds whose name is
shared by many nodes (``id``, ``created_at``, ...) - the analogue of inverse
document frequency for schema objects.
"""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx

from schemagraph.graph.build import SchemaGraph


def specificity_weights(sg: SchemaGraph) -> dict[str, float]:
    """1 / log(1 + number of nodes sharing this name), per node."""
    names = Counter(d.get("name") for _, d in sg.g.nodes(data=True) if d.get("ntype") in {"column", "table"})
    out: dict[str, float] = {}
    for n, d in sg.g.nodes(data=True):
        cnt = names.get(d.get("name"), 1) if d.get("ntype") in {"column", "table"} else 1
        out[n] = 1.0 / math.log(2 + cnt - 1) if cnt > 1 else 1.0
    return out


def personalized_pagerank(
    sg: SchemaGraph,
    seeds: dict[str, float],
    *,
    alpha: float = 0.85,
    specificity: dict[str, float] | None = None,
    max_iter: int = 200,
) -> dict[str, float]:
    """Return PPR scores for every node given seed weights (node id -> weight)."""
    if not seeds:
        return {}
    spec = specificity or specificity_weights(sg)
    personalization = {n: w * spec.get(n, 1.0) for n, w in seeds.items() if n in sg.g and w > 0}
    if not personalization:
        return {}
    total = sum(personalization.values())
    personalization = {n: w / total for n, w in personalization.items()}
    # restrict to the connected components touched by seeds (speed on big graphs)
    touched: set[str] = set()
    for comp in nx.connected_components(sg.g):
        if any(n in comp for n in personalization):
            touched |= comp
    sub = sg.g.subgraph(touched)
    try:
        scores = nx.pagerank(sub, alpha=alpha, personalization=personalization, weight="weight", max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:  # pragma: no cover
        scores = nx.pagerank(sub, alpha=alpha, personalization=personalization, weight="weight", max_iter=max_iter * 5, tol=1e-4)
    return scores


def table_scores(sg: SchemaGraph, node_scores: dict[str, float], *, agg: str = "top3") -> dict[str, float]:
    """Aggregate node scores to tables.

    ``agg="top3"``: own + best + 0.5*second + 0.25*third + 0.02*rest (wide tables with many
    weakly matching columns no longer swamp a table with one strong match).
    ``agg="sum"``:  own + best + 0.25*sum(rest) (the original rule).
    """
    per_table: dict[str, list[float]] = {}
    own: dict[str, float] = {}
    for n, s in node_scores.items():
        d = sg.g.nodes[n]
        if d.get("ntype") == "table":
            own[d["fqn"]] = s
        elif d.get("ntype") == "column":
            per_table.setdefault(d["fqn"], []).append(s)
    out: dict[str, float] = {}
    for fqn in set(own) | set(per_table):
        cols = sorted(per_table.get(fqn, []), reverse=True)
        best = cols[0] if cols else 0.0
        if agg == "sum":
            out[fqn] = own.get(fqn, 0.0) + best + 0.25 * sum(cols[1:])
        else:
            second = cols[1] if len(cols) > 1 else 0.0
            third = cols[2] if len(cols) > 2 else 0.0
            out[fqn] = own.get(fqn, 0.0) + best + 0.5 * second + 0.25 * third + 0.02 * sum(cols[3:])
    return out
