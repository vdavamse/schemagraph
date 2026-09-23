"""Merge snapshots into one heterogeneous NetworkX graph.

Node ids
--------
* ``t:<table_fqn>``          table / model / source
* ``c:<table_fqn>.<column>`` column
* ``k:<term>``               business-glossary term (from Collibra, dbt semantic layer, users)
* ``w:<token>``              lexical token (LinearRAG-style entity anchor; built by ``linking.lexical``)

Edges (undirected ``nx.Graph`` with attribute lists) carry ``etype``:

* ``contains``    table <-> column
* ``relation``    table <-> table, with ``relations: list[Edge]`` (FK, lineage, tests, catalog, hints)
* ``fk_col``      column <-> column for column-level FKs (used for join-key selection)
* ``glossary``    term <-> table/column
* ``mention``     token <-> table/column/term  (added by the lexical indexer)

The table-level projection used for path-finding is derived with :func:`table_graph`.

Merge order is explicit: :func:`build_graph` sorts snapshots by :func:`snapshot_priority`
(a per-connection ``priority`` if set, else :data:`SOURCE_PRIORITY` by source type, with the
user's glossary and join hints first), and the first source to fill a field wins. The build
runs in three passes: every snapshot's tables, then every relation edge, then every glossary
term, so a foreign key or term that points at a table another source introspects never
depends on which source loaded first. :meth:`SchemaGraph.add_snapshot` is the incremental
path: it resolves one snapshot's edges and terms against the tables loaded so far, and a
stub created for a referenced-only table is replaced when the real table arrives.
"""

from __future__ import annotations

import networkx as nx

from schemagraph.model import BusinessTerm, Edge, SchemaSnapshot, Table

# How trustworthy each edge kind is as a *join path*. Lower weight == preferred path.
RELATION_WEIGHT: dict[str, float] = {
    "foreign_key": 1.0,
    "relationship_test": 1.0,
    "join_hint": 1.0,
    "catalog_relation": 1.3,
    "lineage": 1.6,
    "inferred": 2.5,
}

# How much PPR activation flows across a relation edge of each kind (``affinity`` attribute).
# Kept separate from RELATION_WEIGHT: that one is a *cost* for path-finding, and reading it as
# transition mass made an inferred edge carry 2.5x the flow of a declared foreign key. Uniform
# by default; only ``LinkOptions.ppr_edge_attr="affinity"`` reads it (the benchmark decides).
PPR_AFFINITY: dict[str, float] = {
    "foreign_key": 1.0,
    "relationship_test": 1.0,
    "join_hint": 1.0,
    "catalog_relation": 1.0,
    "lineage": 1.0,
    "inferred": 1.0,
}


# Merge precedence by the ``source_type`` a connector emits (not its registry name: the
# unity_catalog connector emits "unity", aws_glue emits "glue"); lower merges first and wins
# conflicting scalar fields. Curated sources beat introspected ones; the user's own glossary
# and hints beat everything.
SOURCE_PRIORITY: dict[str, int] = {
    "user": 0,
    "collibra": 10,
    "dbt": 20,
    "unity": 30,
    "duckdb": 40,
    "ddl": 50,
    "glue": 60,
}
DEFAULT_PRIORITY = 70

# Relation kinds that describe a *join*. Lineage (dbt ref()/source(), Unity table lineage) is
# provenance: two models fed by the same source are related but cannot be joined through it,
# so it stays out of the path-finding projection and is rendered as context instead.
JOIN_KINDS: frozenset[str] = frozenset({"foreign_key", "relationship_test", "join_hint", "catalog_relation", "inferred"})


def snapshot_priority(snap: SchemaSnapshot) -> int:
    return snap.priority if snap.priority is not None else SOURCE_PRIORITY.get(snap.source_type, DEFAULT_PRIORITY)


def merge_key(snap: SchemaSnapshot) -> tuple[bool, int, str]:
    """Sort key for merging: the user's curation first whatever a connection's priority, then
    :func:`snapshot_priority` (any int, as set), then source name for a stable tie-break."""
    return (snap.source_type != "user", snapshot_priority(snap), snap.source)


def tnode(fqn: str) -> str:
    return f"t:{fqn.lower()}"


def cnode(fqn: str, col: str) -> str:
    return f"c:{fqn.lower()}.{col.lower()}"


def knode(term: str) -> str:
    return f"k:{term.strip().lower()}"


class SchemaGraph:
    """Container for the merged graph plus lookup tables."""

    def __init__(self) -> None:
        self.g: nx.Graph = nx.Graph()
        self.tables: dict[str, Table] = {}  # lower fqn -> Table (merged)
        self.terms: dict[str, BusinessTerm] = {}  # lower name -> term
        self.sources: dict[str, str] = {}  # source name -> source_type
        self._stubs: set[str] = set()  # lower fqns of referenced-only placeholder tables

    # ---------------------------------------------------------------- building
    def add_snapshot(self, snap: SchemaSnapshot) -> None:
        """Merge one snapshot incrementally: its tables, then its edges and terms, resolved
        against every table loaded so far. :func:`build_graph` is the order-independent path."""
        self.add_tables(snap)
        for e in snap.edges:
            self.add_edge(e)
        for b in snap.terms:
            self.add_term(b)

    def add_tables(self, snap: SchemaSnapshot) -> None:
        snap.stamp()
        if snap.tables or snap.edges or snap.terms:
            self.sources[snap.source] = snap.source_type
        for t in snap.tables:
            self._merge_table(t)

    def _merge_table(self, t: Table) -> None:
        key = t.fqn.lower()
        existing = self.tables.get(key)
        replaced = False
        if "stub" in t.properties:  # only the graph marks stubs; a round-tripped snapshot cannot
            t = t.model_copy(update={"properties": {k: v for k, v in t.properties.items() if k != "stub"}})
        if existing is not None and key in self._stubs:
            # a referenced-only placeholder: the real table replaces it; the node keeps the edges
            # already attached to it and remembers who referenced it in ``source``
            real = t.model_copy(deep=True)
            real.source = ",".join(dict.fromkeys(filter(None, real.source.split(",") + existing.source.split(","))))
            self.tables[key] = real
            self._stubs.discard(key)
            self.g.add_node(tnode(t.fqn), ntype="table", fqn=real.fqn, name=real.name.lower())
            existing = None
            replaced = True
        elif existing is None:
            self.tables[key] = t.model_copy(deep=True)
            self.g.add_node(tnode(t.fqn), ntype="table", fqn=t.fqn, name=t.name.lower())
        if existing is not None:
            # merge: fill blanks, union columns, union tags/properties
            existing.description = existing.description or t.description
            existing.owner = existing.owner or t.owner
            existing.row_count = existing.row_count if existing.row_count is not None else t.row_count
            if existing.kind == "table" and t.kind != "table":
                existing.kind = t.kind
            for tag in t.tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)
            existing.properties = {**t.properties, **existing.properties}
            existing.source = ",".join(dict.fromkeys(filter(None, existing.source.split(",") + [t.source])))
            for c in t.columns:
                ec = existing.column(c.name)
                if ec is None:
                    existing.columns.append(c.model_copy(deep=True))
                else:
                    ec.description = ec.description or c.description
                    ec.data_type = ec.data_type or c.data_type
                    ec.nullable = ec.nullable if ec.nullable is not None else c.nullable
                    ec.is_primary_key = ec.is_primary_key or c.is_primary_key
                    for v in c.sample_values:
                        if v not in ec.sample_values and len(ec.sample_values) < 20:
                            ec.sample_values.append(v)
                    for tag in c.tags:
                        if tag not in ec.tags:
                            ec.tags.append(tag)
                    ec.properties = {**c.properties, **ec.properties}
            for pk in t.primary_key:
                if pk not in existing.primary_key:
                    existing.primary_key.append(pk)
        table = self.tables[key]
        tn = tnode(table.fqn)
        for c in table.columns:
            cn = cnode(table.fqn, c.name)
            if cn not in self.g:
                self.g.add_node(cn, ntype="column", fqn=table.fqn, name=c.name.lower(), table=tn)
                self.g.add_edge(tn, cn, etype="contains", weight=0.5, affinity=0.5)
        if replaced:
            # the stub had no columns: attach the column-level FKs and glossary targets that
            # could only resolve to the table while it was a placeholder
            for m in list(self.g.neighbors(tn)):
                if self.g[tn][m].get("etype") == "relation":
                    for r in self.g[tn][m]["relations"]:
                        self._add_fk_cols(r)
            for term in list(self.terms.values()):
                if any(tg.lower().startswith(key + ".") for tg in term.targets):
                    self.add_term(term)

    def add_edge(self, e: Edge) -> None:
        a, b = tnode(e.from_table), tnode(e.to_table)
        for n, fqn in ((a, e.from_table), (b, e.to_table)):
            if n not in self.g:
                # referenced table not introspected: create a stub so the path exists
                stub = Table(name=fqn.split(".")[-1], schema=".".join(fqn.split(".")[1:-1]) or None, catalog=fqn.split(".")[0] if fqn.count(".") >= 2 else None, source=e.source, properties={"stub": "true"})
                if fqn.lower() not in self.tables:
                    self.tables[fqn.lower()] = stub
                    self._stubs.add(fqn.lower())
                self.g.add_node(n, ntype="table", fqn=fqn, name=stub.name.lower())
        if a == b:
            return
        data = self.g.get_edge_data(a, b)
        if data is None or data.get("etype") != "relation":
            self.g.add_edge(a, b, etype="relation", relations=[], weight=RELATION_WEIGHT.get(e.kind, 2.0), affinity=PPR_AFFINITY.get(e.kind, 1.0))
            data = self.g.get_edge_data(a, b)
        rels: list[Edge] = data["relations"]
        if all(r.key != e.key for r in rels):
            rels.append(e)
        data["weight"] = min(RELATION_WEIGHT.get(r.kind, 2.0) for r in rels)  # cost: best evidence wins
        data["affinity"] = max(PPR_AFFINITY.get(r.kind, 1.0) for r in rels)
        self._add_fk_cols(e)

    def _add_fk_cols(self, e: Edge) -> None:
        for fc, tc in zip(e.from_columns, e.to_columns, strict=False):
            ca, cb = cnode(e.from_table, fc), cnode(e.to_table, tc)
            if ca in self.g and cb in self.g:
                self.g.add_edge(ca, cb, etype="fk_col", weight=0.8, affinity=0.8)

    def add_term(self, b: BusinessTerm) -> None:
        key = b.name.strip().lower()
        if not key:
            return
        existing = self.terms.get(key)
        if existing is None:
            self.terms[key] = b.model_copy(deep=True)
        else:
            existing.description = existing.description or b.description
            for s in b.synonyms:
                if s not in existing.synonyms:
                    existing.synonyms.append(s)
            for t in b.targets:
                if t not in existing.targets:
                    existing.targets.append(t)
        term = self.terms[key]
        kn = knode(term.name)
        if kn not in self.g:
            self.g.add_node(kn, ntype="term", name=key)
        # re-resolve from scratch: a target that fell back to its table while the table was a
        # stub (no columns yet) must not keep that edge once the column exists
        self.g.remove_edges_from([(kn, m) for m in list(self.g.neighbors(kn)) if self.g[kn][m].get("etype") == "glossary"])
        for target in term.targets:
            n = self._resolve_target(target)
            if n is not None:
                self.g.add_edge(kn, n, etype="glossary", weight=0.7, affinity=0.7)

    def _resolve_target(self, target: str) -> str | None:
        lt = target.lower()
        if lt in self.tables:
            return tnode(lt)
        # table.column: split from the right
        if "." in lt:
            tbl, col = lt.rsplit(".", 1)
            if tbl in self.tables:
                n = cnode(tbl, col)
                return n if n in self.g else tnode(tbl)
        # bare table name match (unique)
        matches = [k for k in self.tables if k.split(".")[-1] == lt]
        if len(matches) == 1:
            return tnode(matches[0])
        return None

    # ---------------------------------------------------------------- queries
    def table(self, fqn: str) -> Table | None:
        return self.tables.get(fqn.lower())

    def find_table(self, name: str) -> Table | None:
        """Match on FQN, or unique suffix match on ``schema.table`` / ``table``."""
        t = self.table(name)
        if t:
            return t
        ln = name.lower()
        matches = [k for k in self.tables if k == ln or k.endswith("." + ln)]
        if len(matches) == 1:
            return self.tables[matches[0]]
        return None

    def relations(self, fqn_a: str, fqn_b: str) -> list[Edge]:
        data = self.g.get_edge_data(tnode(fqn_a), tnode(fqn_b))
        if not data or data.get("etype") != "relation":
            return []
        return list(data["relations"])

    def all_edges(self) -> list[Edge]:
        out: list[Edge] = []
        for _, _, d in self.g.edges(data=True):
            if d.get("etype") == "relation":
                out.extend(d["relations"])
        return out

    def table_graph(self, kinds: frozenset[str] | set[str] | None = JOIN_KINDS) -> nx.Graph:
        """Projection with only table nodes and ``relation`` edges of the given kinds (for path-finding).

        The default keeps join-capable kinds only (:data:`JOIN_KINDS`); ``kinds=None`` keeps every
        relation, lineage included. Edge ``weight`` is the best join cost among the kept kinds.
        """
        tg = nx.Graph()
        for n, d in self.g.nodes(data=True):
            if d.get("ntype") == "table":
                tg.add_node(n, **d)
        for a, b, d in self.g.edges(data=True):
            if d.get("etype") != "relation":
                continue
            rels = d["relations"] if kinds is None else [r for r in d["relations"] if r.kind in kinds]
            if rels:
                tg.add_edge(a, b, weight=min(RELATION_WEIGHT.get(r.kind, 2.0) for r in rels), relations=rels)
        return tg

    def lineage(self, fqn: str) -> tuple[list[str], list[str]]:
        """(upstream, downstream) table fqns connected to ``fqn`` by ``lineage`` relations."""
        n = tnode(fqn)
        up: set[str] = set()
        down: set[str] = set()
        if n not in self.g:
            return [], []
        for m in self.g.neighbors(n):
            d = self.g[n][m]
            if d.get("etype") != "relation":
                continue
            for r in d["relations"]:
                if r.kind != "lineage":
                    continue
                if r.from_table.lower() == fqn.lower():
                    down.add(r.to_table)
                else:
                    up.add(r.from_table)
        return sorted(up), sorted(down)

    def stats(self) -> dict[str, int]:
        ntypes: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            ntypes[d.get("ntype", "?")] = ntypes.get(d.get("ntype", "?"), 0) + 1
        etypes: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            etypes[d.get("etype", "?")] = etypes.get(d.get("etype", "?"), 0) + 1
        return {
            "tables": ntypes.get("table", 0),
            "columns": ntypes.get("column", 0),
            "terms": ntypes.get("term", 0),
            "tokens": ntypes.get("token", 0),
            "relations": len(self.all_edges()),
            "relation_pairs": etypes.get("relation", 0),
            "sources": len(self.sources),
        }


def build_graph(snapshots: list[SchemaSnapshot]) -> SchemaGraph:
    """Merge snapshots in priority order (see :func:`merge_key`): all tables first, then
    all relation edges, then all glossary terms, so resolution never depends on merge order."""
    sg = SchemaGraph()
    ordered = sorted(snapshots, key=merge_key)
    for s in ordered:
        sg.add_tables(s)
    for s in ordered:
        for e in s.edges:
            sg.add_edge(e)
    for s in ordered:
        for b in s.terms:
            sg.add_term(b)
    return sg
