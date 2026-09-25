"""SchemaGraphSQL-style path selection over the table graph.

Given anchor tables (sources = tables whose columns appear in filters, destinations =
tables whose columns appear in the output), enumerate *all shortest simple paths*
between every (source, destination) pair and return the union. Bridge tables that are
semantically irrelevant but structurally mandatory are included by construction.

Deviation from the paper: paths are shortest by *weighted* length, where declared
foreign keys cost 1.0 and weaker evidence (catalog relations, inferred keys) costs more,
so a path through real join keys beats an equally-short path through weaker evidence.
Lineage edges are not join paths at all (``SchemaGraph.table_graph`` drops them by
default); they are rendered as context on the tables they connect.

Paths come from Yen's k-shortest-paths (``nx.shortest_simple_paths``), which yields
simple paths in non-decreasing weighted length, so enumeration stops at the first path
longer than ``best + max_extra``. The previous implementation enumerated every simple
path up to a hop cutoff and filtered by length, which is exponential on dense
inferred-edge clusters (1.8 s on a 12-clique; benchmark p99 over 600 ms).
"""

from __future__ import annotations

from itertools import combinations, product

import networkx as nx

from schemagraph.graph.build import JOIN_KINDS, SchemaGraph, tnode

# Safety cap on paths enumerated (kept or not) between one anchor pair.
MAX_PATHS_PER_PAIR = 64
# Slack on the length comparison so float sums of equal edge costs count as ties.
LENGTH_TOLERANCE = 1e-9


def _path_length(tg: nx.Graph, path: list[str]) -> float:
    """Weighted length of a path: the sum of its edges' join costs."""
    return sum(tg[u][v]["weight"] for u, v in zip(path, path[1:], strict=False))


def shortest_paths_between(
    tg: nx.Graph,
    a: str,
    b: str,
    max_extra: float = 0.0,
    cutoff: int = 6,
) -> list[list[str]]:
    """Find the simple paths from a to b within ``max_extra`` of the shortest weighted length.

    The shortest path is always returned; the tied or near-tied alternatives must also
    have at most ``cutoff`` hops. At most :data:`MAX_PATHS_PER_PAIR` paths are enumerated.

    Args:
        tg: Table graph (see ``SchemaGraph.table_graph``) whose edge ``weight`` is a join cost.
        a: Start node id.
        b: End node id.
        max_extra: How much longer than the shortest path an alternative may be, in join cost.
        cutoff: Maximum hops of an alternative path.

    Returns:
        The paths as node-id lists, in non-decreasing weighted length; ``[[a]]`` when
        ``a == b``, empty when either node is missing or no path exists.
    """
    if a == b:
        return [[a]]
    if a not in tg or b not in tg:
        return []
    paths: list[list[str]] = []
    best: float | None = None
    try:
        for i, path in enumerate(nx.shortest_simple_paths(tg, a, b, weight="weight")):
            # counts enumerated paths: ties over the hop cutoff are not free
            if i >= MAX_PATHS_PER_PAIR:
                break
            length = _path_length(tg, path)
            if best is None:
                best = length
                paths.append(path)
                continue
            if length > best + max_extra + LENGTH_TOLERANCE:
                break
            if len(path) - 1 <= cutoff:
                paths.append(path)
    except nx.NetworkXNoPath:
        return []
    return paths


def union_of_shortest_paths(
    schema_graph: SchemaGraph,
    sources: list[str],
    destinations: list[str] | None = None,
    *,
    max_extra: float = 0.0,
    cutoff: int = 6,
    kinds: frozenset[str] | set[str] | None = JOIN_KINDS,
) -> tuple[list[list[str]], set[str]]:
    """Connect anchor tables through the union of their shortest join paths.

    Stage 5 of the linker (SchemaGraphSQL's union of shortest paths; see ``docs/DESIGN.md``):
    this is what pulls bridge tables into the linked schema. If ``destinations`` is None,
    pairs are formed among the sources themselves (the common case: "connect all anchor
    tables"); otherwise every (source, destination) pair is connected. A path already found
    in either direction is not repeated.

    Args:
        schema_graph: The graph whose table projection is searched.
        sources: Anchor table fqns; those not in the projection are ignored.
        destinations: Optional second set of anchor table fqns.
        max_extra: How much longer than the shortest path an alternative may be, in join cost.
        cutoff: Maximum hops of an alternative path.
        kinds: Relation kinds walked (default: join-capable ones; ``None`` adds lineage).

    Returns:
        ``(paths, tables)``: the candidate paths as table-node-id lists, and the fqns of every
        table on those paths plus every anchor found in the projection.
    """
    tg = schema_graph.table_graph(kinds=kinds)
    source_nodes = [tnode(s) for s in sources if tnode(s) in tg]
    destination_nodes = (
        [tnode(d) for d in destinations if tnode(d) in tg] if destinations else None
    )
    if destination_nodes:
        pairs = list(product(source_nodes, destination_nodes))
    else:
        pairs = list(combinations(source_nodes, 2))
    paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for start, end in pairs:
        if start == end:
            continue
        for path in shortest_paths_between(tg, start, end, max_extra=max_extra, cutoff=cutoff):
            key = tuple(path)
            if key in seen or tuple(reversed(path)) in seen:
                continue
            seen.add(key)
            paths.append(path)
    union = {schema_graph.graph.nodes[node]["fqn"] for path in paths for node in path}
    union.update(
        schema_graph.graph.nodes[node]["fqn"] for node in source_nodes + (destination_nodes or [])
    )
    return paths, union


def connected_components_of(schema_graph: SchemaGraph, fqns: list[str]) -> list[set[str]]:
    """Group the given tables by the connected component of the join graph they fall in.

    Tables missing from the join graph are dropped; components holding none of them are skipped.
    """
    tg = schema_graph.table_graph()
    nodes = [tnode(fqn) for fqn in fqns if tnode(fqn) in tg]
    components: list[set[str]] = []
    for component in nx.connected_components(tg):
        hit = {tg.nodes[node]["fqn"] for node in nodes if node in component}
        if hit:
            components.append(hit)
    return components
