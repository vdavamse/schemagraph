"""PathRAG-style flow-based pruning of candidate join paths.

PathRAG's observation: graph retrieval fails from *redundancy*, not scarcity. Given
candidate paths between anchor tables, assign each anchor a unit of resource and
propagate it along the candidate subgraph with decay ``alpha``; stop when the
resource at a node falls under ``theta``. A path's *reliability* is the mean
resource of its nodes. Keep the top-``k`` paths by reliability.

Applied to schemas this prefers short paths through well-connected, activated
tables and drops long detours through hubs that happen to also connect the anchors.
"""

from __future__ import annotations

import networkx as nx

from schemagraph.graph.build import SchemaGraph
from schemagraph.model import Edge, JoinPath, JoinStep

# Which relation to describe a join step with when two tables have several: lower wins.
# Declared join keys first, then catalog relations, inferred keys, lineage, anything else.
STEP_KIND_PREFERENCE: dict[str, int] = {
    "foreign_key": 0,
    "relationship_test": 0,
    "join_hint": 0,
    "catalog_relation": 1,
    "inferred": 2,
    "lineage": 3,
}
UNKNOWN_STEP_KIND_PREFERENCE = 4


def flow_resources(
    sub: nx.Graph,
    anchors: list[str],
    *,
    alpha: float = 0.8,
    theta: float = 0.05,
    node_prior: dict[str, float] | None = None,
) -> dict[str, float]:
    """Propagate a unit of resource from each anchor with decay and sum it per node.

    Breadth-first from each anchor: a node keeps its incoming resource (boosted by its prior)
    and splits ``alpha`` of it evenly among its unvisited neighbours, each share divided by the
    edge's join cost; a share below ``theta`` is dropped.

    Args:
        sub: Candidate subgraph of the table graph (edge ``weight`` is a join cost).
        anchors: Anchor node ids; those not in ``sub`` are ignored.
        alpha: Fraction of a node's resource passed on to its neighbours.
        theta: Smallest share that keeps propagating.
        node_prior: Optional extra weight per node; a node's received resource is multiplied
            by ``1 + prior``.

    Returns:
        Resource per node of ``sub`` (zero for nodes no anchor reached), summed over anchors.
    """
    resource: dict[str, float] = dict.fromkeys(sub.nodes, 0.0)
    prior = node_prior or {}
    for anchor in anchors:
        if anchor not in sub:
            continue
        frontier = {anchor: 1.0}
        visited = {anchor}
        while frontier:
            next_frontier: dict[str, float] = {}
            for node, amount in frontier.items():
                resource[node] += amount * (1.0 + prior.get(node, 0.0))
                unvisited = [
                    neighbor for neighbor in sub.neighbors(node) if neighbor not in visited
                ]
                if not unvisited:
                    continue
                share = alpha * amount / len(unvisited)
                for neighbor in unvisited:
                    inverse_cost = 1.0 / sub[node][neighbor].get("weight", 1.0)
                    value = share * inverse_cost
                    if value >= theta:
                        next_frontier[neighbor] = next_frontier.get(neighbor, 0.0) + value
            visited |= set(next_frontier)
            frontier = next_frontier
    return resource


def prune_paths(
    schema_graph: SchemaGraph,
    paths: list[list[str]],
    anchors: list[str],
    *,
    alpha: float = 0.8,
    theta: float = 0.05,
    top_k: int = 8,
    node_prior: dict[str, float] | None = None,
) -> list[JoinPath]:
    """Score candidate paths by flow reliability and keep the best ``top_k``.

    Stage 6 of the linker (PathRAG's flow-based pruning; see ``docs/DESIGN.md``). A path's
    reliability is the mean :func:`flow_resources` of its nodes with a mild length penalty;
    a path that is a contiguous sub-path of another candidate is dropped as redundant.

    Args:
        schema_graph: The graph the paths were found in.
        paths: Candidate paths as table-node-id lists.
        anchors: Anchor table node ids the resource flows from.
        alpha: Fraction of a node's resource passed on to its neighbours.
        theta: Smallest share that keeps propagating.
        top_k: Maximum number of paths returned.
        node_prior: Optional extra weight per node (see :func:`flow_resources`).

    Returns:
        The kept paths, most reliable first, each with its join steps and rounded reliability.
    """
    if not paths:
        return []
    table_graph = schema_graph.table_graph()
    nodes = {node for path in paths for node in path}
    sub = table_graph.subgraph(nodes).copy()
    resources = flow_resources(sub, anchors, alpha=alpha, theta=theta, node_prior=node_prior)
    scored: list[tuple[float, list[str]]] = []
    for path in paths:
        reliability = sum(resources.get(node, 0.0) for node in path) / max(len(path), 1)
        # mild length penalty so equal-flow shorter paths win
        reliability = reliability / (1.0 + 0.1 * (len(path) - 2))
        scored.append((reliability, path))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    kept = _drop_subpaths(scored)
    kept.sort(key=lambda item: (-item[0], len(item[1])))
    join_paths: list[JoinPath] = []
    for reliability, path in kept[:top_k]:
        join_paths.append(
            JoinPath(
                tables=[schema_graph.graph.nodes[node]["fqn"] for node in path],
                steps=_join_steps(schema_graph, path),
                reliability=round(reliability, 4),
            )
        )
    return join_paths


def _drop_subpaths(scored: list[tuple[float, list[str]]]) -> list[tuple[float, list[str]]]:
    """Drop paths that are contiguous sub-paths of a longer kept path (redundant for the LLM).

    Walks ``scored`` in order; a path already covered by a kept one is skipped, and kept paths
    covered by the new one are evicted.
    """
    kept: list[tuple[float, list[str]]] = []
    for reliability, path in scored:
        if any(_is_subpath(path, other) for _, other in kept):
            continue
        kept = [(score, other) for score, other in kept if not _is_subpath(other, path)]
        kept.append((reliability, path))
    return kept


def _is_subpath(short: list[str], long: list[str]) -> bool:
    """Whether ``short`` appears contiguously in the strictly longer ``long``, either direction."""
    if len(short) >= len(long):
        return False
    size = len(short)
    for i in range(len(long) - size + 1):
        window = long[i : i + size]
        if window == short or window == list(reversed(short)):
            return True
    return False


def _join_condition(relation: Edge) -> str:
    """Human-readable ON clause (or explanation when keys are unknown) for one relation."""
    if relation.from_columns and relation.to_columns:
        return " AND ".join(
            f"{relation.from_table}.{from_column} = {relation.to_table}.{to_column}"
            for from_column, to_column in zip(
                relation.from_columns,
                relation.to_columns,
                strict=False,
            )
        )
    if relation.kind == "lineage":
        return (
            f"{relation.from_table} feeds {relation.to_table} "
            "(dbt lineage; join keys not declared)"
        )
    return relation.description or f"{relation.from_table} -> {relation.to_table} ({relation.kind})"


def _join_steps(schema_graph: SchemaGraph, path: list[str]) -> list[JoinStep]:
    """One join step per consecutive table pair with a relation, from its preferred relation."""
    steps: list[JoinStep] = []
    for u, v in zip(path, path[1:], strict=False):
        from_fqn = schema_graph.graph.nodes[u]["fqn"]
        to_fqn = schema_graph.graph.nodes[v]["fqn"]
        relations = schema_graph.relations(from_fqn, to_fqn)
        if not relations:
            continue
        best = sorted(
            relations,
            key=lambda relation: STEP_KIND_PREFERENCE.get(
                relation.kind,
                UNKNOWN_STEP_KIND_PREFERENCE,
            ),
        )[0]
        steps.append(
            JoinStep(
                from_table=from_fqn,
                to_table=to_fqn,
                kind=best.kind,
                on=_join_condition(best),
                source=best.source,
            )
        )
    return steps
