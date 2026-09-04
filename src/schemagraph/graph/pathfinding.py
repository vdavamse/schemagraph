"""SchemaGraphSQL-style path selection over the table graph.

Given anchor tables (sources = tables whose columns appear in filters, destinations =
tables whose columns appear in the output), enumerate *all shortest simple paths*
between every (source, destination) pair and return the union. Bridge tables that are
semantically irrelevant but structurally mandatory are included by construction.

Deviation from the paper: paths are shortest by *weighted* length, where declared
foreign keys cost 1.0 and weaker evidence (lineage, inferred) costs more, so a path
through real join keys beats an equally-short path through lineage edges.
"""

from __future__ import annotations

from itertools import combinations, product

import networkx as nx

from schemagraph.graph.build import SchemaGraph, tnode


def shortest_paths_between(tg: nx.Graph, a: str, b: str, max_extra: float = 0.0, cutoff: int = 6) -> list[list[str]]:
    """All simple paths from a to b whose weighted length is within ``max_extra`` of the shortest."""
    if a == b:
        return [[a]]
    if a not in tg or b not in tg:
        return []
    try:
        best = nx.shortest_path_length(tg, a, b, weight="weight")
    except nx.NetworkXNoPath:
        return []
    hop_cutoff = min(cutoff, int(best / min(d["weight"] for _, _, d in tg.edges(data=True)) + 1)) if tg.number_of_edges() else cutoff
    out: list[list[str]] = []
    for path in nx.all_simple_paths(tg, a, b, cutoff=hop_cutoff):
        length = sum(tg[u][v]["weight"] for u, v in zip(path, path[1:], strict=False))
        if length <= best + max_extra + 1e-9:
            out.append(path)
    if not out:  # numerical corner case: fall back to one shortest path
        out.append(nx.shortest_path(tg, a, b, weight="weight"))
    return out


def union_of_shortest_paths(
    sg: SchemaGraph,
    sources: list[str],
    destinations: list[str] | None = None,
    *,
    max_extra: float = 0.0,
    cutoff: int = 6,
) -> tuple[list[list[str]], set[str]]:
    """Return (candidate paths, union of table fqns on those paths).

    If ``destinations`` is None, pairs are formed among the sources themselves
    (the common case: "connect all anchor tables").
    """
    tg = sg.table_graph()
    src = [tnode(s) for s in sources if tnode(s) in tg]
    dst = [tnode(d) for d in destinations if tnode(d) in tg] if destinations else None
    pairs = list(product(src, dst)) if dst else list(combinations(src, 2))
    paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for a, b in pairs:
        if a == b:
            continue
        for p in shortest_paths_between(tg, a, b, max_extra=max_extra, cutoff=cutoff):
            key = tuple(p)
            if key in seen or tuple(reversed(p)) in seen:
                continue
            seen.add(key)
            paths.append(p)
    union = {sg.g.nodes[n]["fqn"] for p in paths for n in p}
    union.update(sg.g.nodes[n]["fqn"] for n in src + (dst or []))
    return paths, union


def connected_components_of(sg: SchemaGraph, fqns: list[str]) -> list[set[str]]:
    tg = sg.table_graph()
    nodes = [tnode(f) for f in fqns if tnode(f) in tg]
    comps: list[set[str]] = []
    for comp in nx.connected_components(tg):
        hit = {tg.nodes[n]["fqn"] for n in nodes if n in comp}
        if hit:
            comps.append(hit)
    return comps
