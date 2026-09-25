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

# Node types whose shared names make a seed less specific (HippoRAG node specificity).
SPECIFIC_NTYPES = {"column", "table"}


def specificity_weights(schema_graph: SchemaGraph) -> dict[str, float]:
    """Return ``1 / log(1 + number of nodes sharing this name)`` per node.

    Only tables and columns are counted; every other node, and any node whose name is unique,
    gets 1.0.
    """
    name_counts = Counter(
        attrs.get("name")
        for _, attrs in schema_graph.graph.nodes(data=True)
        if attrs.get("ntype") in SPECIFIC_NTYPES
    )
    weights: dict[str, float] = {}
    for node, attrs in schema_graph.graph.nodes(data=True):
        if attrs.get("ntype") in SPECIFIC_NTYPES:
            count = name_counts.get(attrs.get("name"), 1)
        else:
            count = 1
        weights[node] = 1.0 / math.log(2 + count - 1) if count > 1 else 1.0
    return weights


class PPRMatrix:
    """Row-stochastic transition matrix of the graph for one edge attribute.

    ``edge_attr`` names the attribute read as *affinity* (transition mass): ``weight``
    reproduces the historical behaviour, where relation edges carry their join cost
    (FK 1.0, inferred 2.5); ``affinity`` uses the per-kind values in
    :data:`schemagraph.graph.build.PPR_AFFINITY`. Build it after the lexical index
    has added its token nodes; the graph must not change afterwards.

    Attributes:
        edge_attr: The edge attribute read as transition mass.
        nodes: Every graph node, in graph insertion order (the matrix row order).
        index: Row index of each node.
        transition: ``D^-1 A``: the symmetric affinity matrix with each row divided by its sum.
        dangling: Row indices of nodes with no positive-affinity edge.
    """

    def __init__(self, schema_graph: SchemaGraph, edge_attr: str = "weight") -> None:
        self.edge_attr = edge_attr
        self.nodes: list[str] = list(schema_graph.graph.nodes())
        self.index: dict[str, int] = {node: i for i, node in enumerate(self.nodes)}
        node_count = len(self.nodes)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for u, v, attrs in schema_graph.graph.edges(data=True):
            affinity = float(attrs.get(edge_attr, attrs.get("weight", 1.0)))
            if affinity <= 0 or u == v:
                continue
            i, j = self.index[u], self.index[v]
            rows += [i, j]
            cols += [j, i]
            vals += [affinity, affinity]
        adjacency = sp.csr_array((vals, (rows, cols)), shape=(node_count, node_count), dtype=float)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse_degree = np.zeros_like(degree)
        has_edges = degree != 0
        inverse_degree[has_edges] = 1.0 / degree[has_edges]
        self.transition = (sp.diags(inverse_degree) @ adjacency).tocsr()
        self.dangling = np.flatnonzero(~has_edges)

    def run(
        self,
        personalization: dict[str, float],
        *,
        alpha: float = 0.85,
        max_iter: int = 500,
        tol: float = 1e-12,
    ) -> dict[str, float]:
        """Run the personalized power iteration from a teleport distribution.

        Args:
            personalization: Teleport mass per node id; unknown nodes and non-positive values
                are ignored, the rest is normalised to sum to 1.
            alpha: Damping factor, the probability of following an edge rather than teleporting.
            max_iter: Minimum iteration budget; raised to what ``alpha`` needs to reach ``tol``.
            tol: Per-node convergence tolerance (stops when the L1 change is below
                ``len(nodes) * tol``).

        Returns:
            Score per node id for every node with a positive score; empty when there is no
            positive teleport mass.
        """
        node_count = len(self.nodes)
        if node_count == 0:
            return {}
        p = np.zeros(node_count)
        for node, mass in personalization.items():
            i = self.index.get(node)
            if i is not None and mass > 0:
                p[i] += mass
        total = p.sum()
        if total <= 0:
            return {}
        p /= total
        # start from the teleport vector: mass never enters components the seeds do not touch,
        # so those nodes stay exactly zero (what restricting to the touched components achieved)
        # tolerance is per node (L1 change < n * tol), as in networkx, but 1e-12 instead of 1e-6:
        # at 1e-6 a 7,000-node schema stopped with an L1 error near 1e-2, enough to reorder
        # near-tied tables at rank 1
        # the L1 error contracts by alpha per step from at most 2, so derive the budget from
        # alpha: a public ppr_alpha of 0.95 needs ~540 steps where 0.85 needs ~170
        if 0.0 < alpha < 1.0:
            max_iter = max(
                max_iter,
                math.ceil(math.log(node_count * tol / 2.0) / math.log(alpha)) + 1,
            )
        x = p.copy()
        err = float("inf")
        for _ in range(max_iter):
            xlast = x
            x = alpha * (x @ self.transition + x[self.dangling].sum() * p) + (1.0 - alpha) * p
            err = float(np.abs(x - xlast).sum())
            if err < node_count * tol:
                break
        else:
            log.warning(
                "PPR did not converge in %d iterations (L1 change %.2e, alpha %.2f)",
                max_iter,
                err,
                alpha,
            )
        return {self.nodes[i]: float(x[i]) for i in np.flatnonzero(x > 0)}


def personalized_pagerank(
    schema_graph: SchemaGraph,
    seeds: dict[str, float],
    *,
    alpha: float = 0.85,
    specificity: dict[str, float] | None = None,
    max_iter: int = 200,
    matrix: PPRMatrix | None = None,
) -> dict[str, float]:
    """Spread the question's seed activation over the graph with Personalized PageRank.

    Stage 2 of the linker (HippoRAG's PPR over the knowledge graph; see ``docs/DESIGN.md``).
    Each seed's weight is multiplied by its node specificity before it becomes teleport mass.

    Args:
        schema_graph: The graph to walk.
        seeds: Lexical activation per node id; seeds absent from the graph or non-positive
            are dropped.
        alpha: Damping factor, the probability of following an edge rather than teleporting.
        specificity: Specificity multiplier per node; computed with
            :func:`specificity_weights` when None or empty.
        max_iter: Minimum iteration budget (see :meth:`PPRMatrix.run`).
        matrix: A prebuilt transition matrix for ``schema_graph``; built on the fly (reading
            ``weight``) when None.

    Returns:
        PPR score per node id for every node reachable from the seeds.
    """
    if not seeds:
        return {}
    spec = specificity or specificity_weights(schema_graph)
    personalization = {
        node: weight * spec.get(node, 1.0)
        for node, weight in seeds.items()
        if node in schema_graph.graph and weight > 0
    }
    if not personalization:
        return {}
    return (matrix or PPRMatrix(schema_graph)).run(personalization, alpha=alpha, max_iter=max_iter)


def table_scores(
    schema_graph: SchemaGraph,
    node_scores: dict[str, float],
    *,
    agg: str = "top3",
) -> dict[str, float]:
    """Fold node scores into one score per table fqn.

    A table scores its own node plus a weighted aggregate of its columns' scores:

    * ``agg="top3"``: own + best + 0.5*second + 0.25*third + 0.02*rest (wide tables with many
      weakly matching columns no longer swamp a table with one strong match).
    * ``agg="sum"``: own + best + 0.25*sum(rest) (the original rule).

    Args:
        schema_graph: The graph the node ids belong to.
        node_scores: PPR score per node id; nodes other than tables and columns are ignored.
        agg: Column aggregation rule, ``"top3"`` or ``"sum"``.

    Returns:
        Score per table fqn, keyed in sorted fqn order.
    """
    column_scores_by_table: dict[str, list[float]] = {}
    own: dict[str, float] = {}
    for node, score in node_scores.items():
        attrs = schema_graph.graph.nodes[node]
        if attrs.get("ntype") == "table":
            own[attrs["fqn"]] = score
        elif attrs.get("ntype") == "column":
            column_scores_by_table.setdefault(attrs["fqn"], []).append(score)
    scores: dict[str, float] = {}
    # stable order: ties must not depend on the hash seed
    for fqn in sorted(set(own) | set(column_scores_by_table)):
        cols = sorted(column_scores_by_table.get(fqn, []), reverse=True)
        best = cols[0] if cols else 0.0
        if agg == "sum":
            scores[fqn] = own.get(fqn, 0.0) + best + 0.25 * sum(cols[1:])
        else:
            second = cols[1] if len(cols) > 1 else 0.0
            third = cols[2] if len(cols) > 2 else 0.0
            # top3 weights: 1, 0.5, 0.25 for the three best columns, 0.02 for the long tail
            scores[fqn] = (
                own.get(fqn, 0.0) + best + 0.5 * second + 0.25 * third + 0.02 * sum(cols[3:])
            )
    return scores
