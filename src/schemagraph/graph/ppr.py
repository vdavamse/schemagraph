"""HippoRAG-style Personalized PageRank over the heterogeneous schema graph.

Seeds are the nodes activated by the question (tokens, glossary terms, columns,
tables) with their lexical scores; PPR spreads that activation over ``contains``,
``relation``, ``glossary`` and ``mention`` edges so that a table connected to many
activated columns, or one hop from a strongly activated table, rises to the top.

HippoRAG's *node specificity* is reproduced by down-weighting seeds whose name is
shared by many nodes (``id``, ``created_at``, ...) - the analogue of inverse
document frequency for schema objects.

The walk itself is the same power iteration ``networkx.pagerank`` runs, but on a
row-normalised sparse matrix built once per graph (:class:`PPRMatrix`) instead of
being re-derived from a NetworkX subgraph view on every question: on a 4,000-column
schema that conversion cost 300-400 ms per link, the iteration itself a few ms.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np
import scipy.sparse as sp

from schemagraph.graph.build import SchemaGraph

log = logging.getLogger("schemagraph")


def specificity_weights(sg: SchemaGraph) -> dict[str, float]:
    """1 / log(1 + number of nodes sharing this name), per node."""
    names = Counter(d.get("name") for _, d in sg.g.nodes(data=True) if d.get("ntype") in {"column", "table"})
    out: dict[str, float] = {}
    for n, d in sg.g.nodes(data=True):
        cnt = names.get(d.get("name"), 1) if d.get("ntype") in {"column", "table"} else 1
        out[n] = 1.0 / math.log(2 + cnt - 1) if cnt > 1 else 1.0
    return out


class PPRMatrix:
    """Row-stochastic transition matrix of the graph for one edge attribute.

    ``edge_attr`` names the attribute read as *affinity* (transition mass): ``weight``
    reproduces the historical behaviour, where relation edges carry their join cost
    (FK 1.0, inferred 2.5); ``affinity`` uses the per-kind values in
    :data:`schemagraph.graph.build.PPR_AFFINITY`. Build it after the lexical index
    has added its token nodes; the graph must not change afterwards.
    """

    def __init__(self, sg: SchemaGraph, edge_attr: str = "weight") -> None:
        self.edge_attr = edge_attr
        self.nodes: list[str] = list(sg.g.nodes())
        self.index: dict[str, int] = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for u, v, d in sg.g.edges(data=True):
            w = float(d.get(edge_attr, d.get("weight", 1.0)))
            if w <= 0 or u == v:
                continue
            i, j = self.index[u], self.index[v]
            rows += [i, j]
            cols += [j, i]
            vals += [w, w]
        adj = sp.csr_array((vals, (rows, cols)), shape=(n, n), dtype=float)
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.zeros_like(deg)
        nz = deg != 0
        inv[nz] = 1.0 / deg[nz]
        self.transition = (sp.diags(inv) @ adj).tocsr()
        self.dangling = np.flatnonzero(~nz)

    def run(self, personalization: dict[str, float], *, alpha: float = 0.85, max_iter: int = 500, tol: float = 1e-12) -> dict[str, float]:
        n = len(self.nodes)
        if n == 0:
            return {}
        p = np.zeros(n)
        for node, w in personalization.items():
            i = self.index.get(node)
            if i is not None and w > 0:
                p[i] += w
        total = p.sum()
        if total <= 0:
            return {}
        p /= total
        # start from the teleport vector: mass never enters components the seeds do not touch,
        # so those nodes stay exactly zero (what restricting to the touched components achieved)
        # tolerance is per node (L1 change < n * tol), as in networkx, but 1e-12 instead of 1e-6: at 1e-6 a
        # 7,000-node schema stopped with an L1 error near 1e-2, enough to reorder near-tied tables at rank 1
        # the L1 error contracts by alpha per step from at most 2, so derive the budget from alpha:
        # a public ppr_alpha of 0.95 needs ~540 steps where 0.85 needs ~170
        if 0.0 < alpha < 1.0:
            max_iter = max(max_iter, math.ceil(math.log(n * tol / 2.0) / math.log(alpha)) + 1)
        x = p.copy()
        err = float("inf")
        for _ in range(max_iter):
            xlast = x
            x = alpha * (x @ self.transition + x[self.dangling].sum() * p) + (1.0 - alpha) * p
            err = float(np.abs(x - xlast).sum())
            if err < n * tol:
                break
        else:
            log.warning("PPR did not converge in %d iterations (L1 change %.2e, alpha %.2f)", max_iter, err, alpha)
        return {self.nodes[i]: float(x[i]) for i in np.flatnonzero(x > 0)}


def personalized_pagerank(
    sg: SchemaGraph,
    seeds: dict[str, float],
    *,
    alpha: float = 0.85,
    specificity: dict[str, float] | None = None,
    max_iter: int = 200,
    matrix: PPRMatrix | None = None,
) -> dict[str, float]:
    """Return PPR scores for every node reachable from the seeds (node id -> score)."""
    if not seeds:
        return {}
    spec = specificity or specificity_weights(sg)
    personalization = {n: w * spec.get(n, 1.0) for n, w in seeds.items() if n in sg.g and w > 0}
    if not personalization:
        return {}
    return (matrix or PPRMatrix(sg)).run(personalization, alpha=alpha, max_iter=max_iter)


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
    for fqn in sorted(set(own) | set(per_table)):  # stable order: ties must not depend on the hash seed
        cols = sorted(per_table.get(fqn, []), reverse=True)
        best = cols[0] if cols else 0.0
        if agg == "sum":
            out[fqn] = own.get(fqn, 0.0) + best + 0.25 * sum(cols[1:])
        else:
            second = cols[1] if len(cols) > 1 else 0.0
            third = cols[2] if len(cols) > 2 else 0.0
            out[fqn] = own.get(fqn, 0.0) + best + 0.5 * second + 0.25 * third + 0.02 * sum(cols[3:])
    return out
