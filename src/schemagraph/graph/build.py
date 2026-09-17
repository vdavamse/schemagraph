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

    # ---------------------------------------------------------------- building
    def add_snapshot(self, snap: SchemaSnapshot) -> None:
        snap.stamp()
        if snap.tables or snap.edges or snap.terms:
            self.sources[snap.source] = snap.source_type
        for t in snap.tables:
            self._merge_table(t)
        for e in snap.edges:
            self.add_edge(e)
        for b in snap.terms:
            self.add_term(b)

    def _merge_table(self, t: Table) -> None:
        key = t.fqn.lower()
        existing = self.tables.get(key)
        if existing is None:
            self.tables[key] = t.model_copy(deep=True)
            self.g.add_node(tnode(t.fqn), ntype="table", fqn=t.fqn, name=t.name.lower())
        else:
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

    def add_edge(self, e: Edge) -> None:
        a, b = tnode(e.from_table), tnode(e.to_table)
        for n, fqn in ((a, e.from_table), (b, e.to_table)):
            if n not in self.g:
                # referenced table not introspected: create a stub so the path exists
                stub = Table(name=fqn.split(".")[-1], schema=".".join(fqn.split(".")[1:-1]) or None, catalog=fqn.split(".")[0] if fqn.count(".") >= 2 else None, source=e.source, properties={"stub": "true"})
                self.tables.setdefault(fqn.lower(), stub)
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
        # column-level FK edges
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

    def table_graph(self) -> nx.Graph:
        """Projection with only table nodes and ``relation`` edges (for path-finding)."""
        tg = nx.Graph()
        for n, d in self.g.nodes(data=True):
            if d.get("ntype") == "table":
                tg.add_node(n, **d)
        for a, b, d in self.g.edges(data=True):
            if d.get("etype") == "relation":
                tg.add_edge(a, b, weight=d["weight"], relations=d["relations"])
        return tg

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
    sg = SchemaGraph()
    for s in snapshots:
        sg.add_snapshot(s)
    return sg
