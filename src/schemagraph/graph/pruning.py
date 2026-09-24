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
from schemagraph.model import JoinPath, JoinStep


def flow_resources(
    sub: nx.Graph,
    anchors: list[str],
    *,
    alpha: float = 0.8,
    theta: float = 0.05,
    node_prior: dict[str, float] | None = None,
) -> dict[str, float]:
    """Resource per node after decayed propagation from each anchor (summed over anchors)."""
    resource: dict[str, float] = dict.fromkeys(sub.nodes, 0.0)
    prior = node_prior or {}
    for a in anchors:
        if a not in sub:
            continue
        frontier = {a: 1.0}
        visited = {a}
        while frontier:
            nxt: dict[str, float] = {}
            for n, r in frontier.items():
                resource[n] += r * (1.0 + prior.get(n, 0.0))
                nbrs = [m for m in sub.neighbors(n) if m not in visited]
                if not nbrs:
                    continue
                share = alpha * r / len(nbrs)
                for m in nbrs:
                    w = 1.0 / sub[n][m].get("weight", 1.0)
                    val = share * w
                    if val >= theta:
                        nxt[m] = nxt.get(m, 0.0) + val
            visited |= set(nxt)
            frontier = nxt
    return resource


def prune_paths(schema_graph: SchemaGraph,
    paths: list[list[str]],
    anchors: list[str],
    *,
    alpha: float = 0.8,
    theta: float = 0.05,
    top_k: int = 8,
    node_prior: dict[str, float] | None = None,
) -> list[JoinPath]:
    """Score candidate paths by flow reliability and keep the best ``top_k``."""
    if not paths:
        return []
    tg = schema_graph.table_graph()
    nodes = {n for p in paths for n in p}
    sub = tg.subgraph(nodes).copy()
    res = flow_resources(sub, anchors, alpha=alpha, theta=theta, node_prior=node_prior)
    scored: list[tuple[float, list[str]]] = []
    for p in paths:
        rel = sum(res.get(n, 0.0) for n in p) / max(len(p), 1)
        # mild length penalty so equal-flow shorter paths win
        rel = rel / (1.0 + 0.1 * (len(p) - 2))
        scored.append((rel, p))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    # drop paths that are contiguous sub-paths of a longer kept path (redundant for the LLM)
    kept: list[tuple[float, list[str]]] = []
    for rel, p in scored:
        if any(_is_subpath(p, q) for _, q in kept):
            continue
        kept = [(r2, q) for r2, q in kept if not _is_subpath(q, p)]
        kept.append((rel, p))
    kept.sort(key=lambda x: (-x[0], len(x[1])))
    out: list[JoinPath] = []
    for rel, p in kept[:top_k]:
        out.append(JoinPath(tables=[schema_graph.graph.nodes[n]["fqn"] for n in p], steps=_steps(schema_graph, p), reliability=round(rel, 4)))
    # PathRAG serializes ascending by reliability (most reliable last, closest to the question)
    return out


def _is_subpath(short: list[str], long: list[str]) -> bool:
    if len(short) >= len(long):
        return False
    n = len(short)
    for i in range(len(long) - n + 1):
        window = long[i : i + n]
        if window == short or window == list(reversed(short)):
            return True
    return False


def _steps(schema_graph: SchemaGraph, path: list[str]) -> list[JoinStep]:
    steps: list[JoinStep] = []
    for u, v in zip(path, path[1:], strict=False):
        fu, fv = schema_graph.graph.nodes[u]["fqn"], schema_graph.graph.nodes[v]["fqn"]
        rels = schema_graph.relations(fu, fv)
        if not rels:
            continue
        best = sorted(rels, key=lambda r: ({"foreign_key": 0, "relationship_test": 0, "join_hint": 0, "catalog_relation": 1, "inferred": 2, "lineage": 3}.get(r.kind, 4)))[0]
        if best.from_columns and best.to_columns:
            on = " AND ".join(f"{best.from_table}.{a} = {best.to_table}.{b}" for a, b in zip(best.from_columns, best.to_columns, strict=False))
        elif best.kind == "lineage":
            on = f"{best.from_table} feeds {best.to_table} (dbt lineage; join keys not declared)"
        else:
            on = best.description or f"{best.from_table} -> {best.to_table} ({best.kind})"
        steps.append(JoinStep(from_table=fu, to_table=fv, kind=best.kind, on=on, source=best.source))
    return steps
