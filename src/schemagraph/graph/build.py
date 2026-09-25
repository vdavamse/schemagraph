"""Merge snapshots into one heterogeneous NetworkX graph.

Node ids
--------
* ``t:<table_fqn>``          table / model / source
* ``c:<table_fqn>.<column>`` column
* ``k:<term>``               business-glossary term (from Collibra, dbt semantic layer, users)
* ``w:<token>``              lexical token (LinearRAG-style entity anchor; built by
                             ``linking.lexical``)

Edges (undirected ``nx.Graph`` with attribute lists) carry ``etype``:

* ``contains``    table <-> column
* ``relation``    table <-> table, with ``relations: list[Edge]`` (FK, lineage, tests, catalog,
                  hints)
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

from collections import Counter

import networkx as nx

from schemagraph.model import BusinessTerm, Column, Edge, SchemaSnapshot, Table

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
JOIN_KINDS: frozenset[str] = frozenset(
    {"foreign_key", "relationship_test", "join_hint", "catalog_relation", "inferred"}
)

# Join cost of a relation kind missing from RELATION_WEIGHT (between lineage 1.6 and inferred 2.5).
UNKNOWN_RELATION_WEIGHT = 2.0
# PPR affinity of a relation kind missing from PPR_AFFINITY.
DEFAULT_AFFINITY = 1.0
# Weight (and affinity) of the non-relation edges: table-column, column-column FK, term-object.
CONTAINS_WEIGHT = 0.5
FK_COLUMN_WEIGHT = 0.8
GLOSSARY_WEIGHT = 0.7
# A merged column keeps at most this many sample values across all sources.
MAX_SAMPLE_VALUES = 20
# Property that marks a placeholder table created for an edge endpoint nobody introspected.
_STUB_PROPERTY = "stub"


def snapshot_priority(snap: SchemaSnapshot) -> int:
    """Merge priority of a snapshot: its connection's ``priority`` if set, else by source type."""
    if snap.priority is not None:
        return snap.priority
    return SOURCE_PRIORITY.get(snap.source_type, DEFAULT_PRIORITY)


def merge_key(snap: SchemaSnapshot) -> tuple[bool, int, str]:
    """Sort key for merging snapshots.

    The user's curation sorts first whatever a connection's priority, then
    :func:`snapshot_priority` (any int, as set), then the source name for a stable tie-break.
    """
    return (snap.source_type != "user", snapshot_priority(snap), snap.source)


def tnode(fqn: str) -> str:
    """Node id of a table: ``t:<lowercased fqn>``."""
    return f"t:{fqn.lower()}"


def cnode(fqn: str, col: str) -> str:
    """Node id of a column: ``c:<lowercased table fqn>.<lowercased column>``."""
    return f"c:{fqn.lower()}.{col.lower()}"


def knode(term: str) -> str:
    """Node id of a glossary term: ``k:<stripped, lowercased term name>``."""
    return f"k:{term.strip().lower()}"


def _relation_weight(kind: str) -> float:
    """Join cost of one relation kind (see :data:`RELATION_WEIGHT`)."""
    return RELATION_WEIGHT.get(kind, UNKNOWN_RELATION_WEIGHT)


def _relation_affinity(kind: str) -> float:
    """PPR transition affinity of one relation kind (see :data:`PPR_AFFINITY`)."""
    return PPR_AFFINITY.get(kind, DEFAULT_AFFINITY)


def _merge_sources(first: list[str], second: list[str]) -> str:
    """Join two lists of source names into one comma-separated string, deduplicated in order."""
    return ",".join(dict.fromkeys(filter(None, first + second)))


def _append_missing(target: list[str], items: list[str], limit: int | None = None) -> None:
    """Append each item not already in ``target``, while ``target`` is shorter than ``limit``.

    Mutates ``target`` in place.
    """
    for item in items:
        if item not in target and (limit is None or len(target) < limit):
            target.append(item)


def _without_stub_marker(table: Table) -> Table:
    """Return ``table`` without the stub property.

    Only the graph marks stubs, so a snapshot that round-tripped a stub must not keep the marker.
    """
    if _STUB_PROPERTY not in table.properties:
        return table
    properties = {k: v for k, v in table.properties.items() if k != _STUB_PROPERTY}
    return table.model_copy(update={"properties": properties})


def _stub_table(fqn: str, source: str) -> Table:
    """Placeholder for a table an edge references but no snapshot introspected.

    The last dotted part is the name. The catalog is the first part only when there are three or
    more; the schema is whatever lies between catalog and name. So a two-part ``schema.name``
    keeps only its name: the stub's ``fqn`` is ``name`` while it is stored under ``schema.name``.
    """
    parts = fqn.split(".")
    return Table(
        name=parts[-1],
        schema=".".join(parts[1:-1]) or None,
        catalog=parts[0] if fqn.count(".") >= 2 else None,
        source=source,
        properties={_STUB_PROPERTY: "true"},
    )


def _merge_column_fields(existing: Column, incoming: Column) -> None:
    """Fill ``existing``'s blank fields from ``incoming`` and union samples, tags and properties.

    The existing (higher-priority) column wins every conflict. Mutates ``existing`` in place.
    """
    existing.description = existing.description or incoming.description
    existing.data_type = existing.data_type or incoming.data_type
    if existing.nullable is None:
        existing.nullable = incoming.nullable
    existing.is_primary_key = existing.is_primary_key or incoming.is_primary_key
    _append_missing(existing.sample_values, incoming.sample_values, limit=MAX_SAMPLE_VALUES)
    _append_missing(existing.tags, incoming.tags)
    existing.properties = {**incoming.properties, **existing.properties}


def _merge_table_fields(existing: Table, incoming: Table) -> None:
    """Merge a lower-priority copy of a table into the one already in the graph.

    Blank scalar fields are filled, a plain ``table`` kind is upgraded to a more specific one
    (view, model, ...), and tags, properties, sources, columns and primary-key columns are
    unioned. The existing table wins every conflict. Mutates ``existing`` in place.
    """
    existing.description = existing.description or incoming.description
    existing.owner = existing.owner or incoming.owner
    if existing.row_count is None:
        existing.row_count = incoming.row_count
    if existing.kind == "table" and incoming.kind != "table":
        existing.kind = incoming.kind
    _append_missing(existing.tags, incoming.tags)
    existing.properties = {**incoming.properties, **existing.properties}
    existing.source = _merge_sources(existing.source.split(","), [incoming.source])
    for column in incoming.columns:
        existing_column = existing.column(column.name)
        if existing_column is None:
            existing.columns.append(column.model_copy(deep=True))
        else:
            _merge_column_fields(existing_column, column)
    _append_missing(existing.primary_key, incoming.primary_key)


class SchemaGraph:
    """The merged schema graph plus lookup tables for its tables, terms and sources.

    Attributes:
        graph: The heterogeneous NetworkX graph (node and edge types in the module docstring).
            On ``relation`` edges ``weight`` is a join *cost*; on every other edge it is a
            transition affinity. Must not be mutated once a ``Linker`` has been built on it.
        tables: Merged tables by lowercased fqn, stubs included.
        terms: Merged glossary terms by stripped, lowercased name.
        sources: Source type of every snapshot that contributed anything, by source name.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.tables: dict[str, Table] = {}
        self.terms: dict[str, BusinessTerm] = {}
        self.sources: dict[str, str] = {}
        self._stubs: set[str] = set()  # lowercased fqns of referenced-only placeholder tables

    # ---------------------------------------------------------------- building
    def add_snapshot(self, snap: SchemaSnapshot) -> None:
        """Merge one snapshot incrementally.

        Its tables go in first, then its edges and terms, resolved against every table loaded
        so far. :func:`build_graph` is the order-independent path.
        """
        self.add_tables(snap)
        for edge in snap.edges:
            self.add_edge(edge)
        for term in snap.terms:
            self.add_term(term)

    def add_tables(self, snap: SchemaSnapshot) -> None:
        """Merge a snapshot's tables (not its edges or terms) and register its source."""
        snap.stamp()
        if snap.tables or snap.edges or snap.terms:
            self.sources[snap.source] = snap.source_type
        for table in snap.tables:
            self._merge_table(table)

    def _merge_table(self, table: Table) -> None:
        """Merge one table: insert it, replace a stub with it, or merge it into the existing one.

        Whatever the case, every column gets a node joined to its table, and a replaced stub gets
        the column-level FKs and glossary targets that could not resolve while it had no columns.
        """
        key = table.fqn.lower()
        table = _without_stub_marker(table)
        existing = self.tables.get(key)
        replaced_stub = existing is not None and key in self._stubs
        if replaced_stub:
            self._replace_stub(key, table, existing)
        elif existing is None:
            self._insert_table(key, table)
        else:
            _merge_table_fields(existing, table)
        merged = self.tables[key]
        self._add_column_nodes(merged)
        if replaced_stub:
            self._reattach_after_stub(key, tnode(merged.fqn))

    def _insert_table(self, key: str, table: Table) -> None:
        """Store a copy of a table seen for the first time and add its node."""
        self.tables[key] = table.model_copy(deep=True)
        self.graph.add_node(tnode(table.fqn), ntype="table", fqn=table.fqn, name=table.name.lower())

    def _replace_stub(self, key: str, table: Table, stub: Table) -> None:
        """Replace a referenced-only placeholder with the real table.

        The node keeps the edges already attached to it, and the table remembers in ``source``
        the sources that referenced it while it was a stub.
        """
        real = table.model_copy(deep=True)
        real.source = _merge_sources(real.source.split(","), stub.source.split(","))
        self.tables[key] = real
        self._stubs.discard(key)
        self.graph.add_node(tnode(table.fqn), ntype="table", fqn=real.fqn, name=real.name.lower())

    def _add_column_nodes(self, table: Table) -> None:
        """Add a node and a ``contains`` edge for every column of ``table`` not yet in the graph."""
        table_node = tnode(table.fqn)
        for column in table.columns:
            column_node = cnode(table.fqn, column.name)
            if column_node not in self.graph:
                self.graph.add_node(
                    column_node,
                    ntype="column",
                    fqn=table.fqn,
                    name=column.name.lower(),
                    table=table_node,
                )
                self.graph.add_edge(
                    table_node,
                    column_node,
                    etype="contains",
                    weight=CONTAINS_WEIGHT,
                    affinity=CONTAINS_WEIGHT,
                )

    def _reattach_after_stub(self, key: str, table_node: str) -> None:
        """Resolve what could only point at the table while it was a column-less stub.

        That is the column-level FKs of the relations already attached to its node, and the
        glossary terms with a ``table.column`` target in it.
        """
        for neighbor in list(self.graph.neighbors(table_node)):
            if self.graph[table_node][neighbor].get("etype") == "relation":
                for relation in self.graph[table_node][neighbor]["relations"]:
                    self._add_fk_columns(relation)
        for term in list(self.terms.values()):
            if any(target.lower().startswith(key + ".") for target in term.targets):
                self.add_term(term)

    def add_edge(self, edge: Edge) -> None:
        """Add one table-to-table relation, creating stub tables for unknown endpoints.

        Two tables share a single ``relation`` edge holding every piece of evidence in
        ``relations``; its ``weight`` is the cheapest join cost and its ``affinity`` the largest
        affinity among them. A self-relation adds only the stub, never an edge.
        """
        from_node, to_node = tnode(edge.from_table), tnode(edge.to_table)
        self._ensure_table_node(from_node, edge.from_table, edge.source)
        self._ensure_table_node(to_node, edge.to_table, edge.source)
        if from_node == to_node:
            return
        attrs = self.graph.get_edge_data(from_node, to_node)
        if attrs is None or attrs.get("etype") != "relation":
            self.graph.add_edge(
                from_node,
                to_node,
                etype="relation",
                relations=[],
                weight=_relation_weight(edge.kind),
                affinity=_relation_affinity(edge.kind),
            )
            attrs = self.graph.get_edge_data(from_node, to_node)
        relations: list[Edge] = attrs["relations"]
        if all(relation.key != edge.key for relation in relations):
            relations.append(edge)
        attrs["weight"] = min(_relation_weight(relation.kind) for relation in relations)
        attrs["affinity"] = max(_relation_affinity(relation.kind) for relation in relations)
        self._add_fk_columns(edge)

    def _ensure_table_node(self, node: str, fqn: str, source: str) -> None:
        """Create a stub table and its node when an edge endpoint is not in the graph yet."""
        if node in self.graph:
            return
        stub = _stub_table(fqn, source)
        if fqn.lower() not in self.tables:
            self.tables[fqn.lower()] = stub
            self._stubs.add(fqn.lower())
        self.graph.add_node(node, ntype="table", fqn=fqn, name=stub.name.lower())

    def _add_fk_columns(self, edge: Edge) -> None:
        """Link the column pairs of a relation with ``fk_col`` edges when both columns exist."""
        for from_column, to_column in zip(edge.from_columns, edge.to_columns, strict=False):
            from_node = cnode(edge.from_table, from_column)
            to_node = cnode(edge.to_table, to_column)
            if from_node in self.graph and to_node in self.graph:
                self.graph.add_edge(
                    from_node,
                    to_node,
                    etype="fk_col",
                    weight=FK_COLUMN_WEIGHT,
                    affinity=FK_COLUMN_WEIGHT,
                )

    def add_term(self, term: BusinessTerm) -> None:
        """Merge one glossary term and (re)link it to the tables and columns it targets.

        A term already present keeps its description and gains the new synonyms and targets.
        Its ``glossary`` edges are rebuilt from scratch, so a target that fell back to its table
        while the table was a stub moves to the column once the column exists.
        """
        key = term.name.strip().lower()
        if not key:
            return
        existing = self.terms.get(key)
        if existing is None:
            self.terms[key] = term.model_copy(deep=True)
        else:
            existing.description = existing.description or term.description
            _append_missing(existing.synonyms, term.synonyms)
            _append_missing(existing.targets, term.targets)
        merged = self.terms[key]
        term_node = knode(merged.name)
        if term_node not in self.graph:
            self.graph.add_node(term_node, ntype="term", name=key)
        self._clear_glossary_edges(term_node)
        for target in merged.targets:
            target_node = self._resolve_target(target)
            if target_node is not None:
                self.graph.add_edge(
                    term_node,
                    target_node,
                    etype="glossary",
                    weight=GLOSSARY_WEIGHT,
                    affinity=GLOSSARY_WEIGHT,
                )

    def _clear_glossary_edges(self, term_node: str) -> None:
        """Remove every ``glossary`` edge of a term node."""
        glossary_edges = [
            (term_node, neighbor)
            for neighbor in list(self.graph.neighbors(term_node))
            if self.graph[term_node][neighbor].get("etype") == "glossary"
        ]
        self.graph.remove_edges_from(glossary_edges)

    def _resolve_target(self, target: str) -> str | None:
        """Node a glossary target points at, or None when it matches nothing.

        Tried in order: an exact table fqn; ``table.column`` (split from the right), falling back
        to the table while the column has no node; a bare table name that matches exactly one
        table.
        """
        lower_target = target.lower()
        if lower_target in self.tables:
            return tnode(lower_target)
        if "." in lower_target:
            table_key, column_name = lower_target.rsplit(".", 1)
            if table_key in self.tables:
                column_node = cnode(table_key, column_name)
                return column_node if column_node in self.graph else tnode(table_key)
        matches = [key for key in self.tables if key.split(".")[-1] == lower_target]
        if len(matches) == 1:
            return tnode(matches[0])
        return None

    # ---------------------------------------------------------------- queries
    def table(self, fqn: str) -> Table | None:
        """Table by exact fqn (case-insensitive), or None."""
        return self.tables.get(fqn.lower())

    def find_table(self, name: str) -> Table | None:
        """Match on FQN, or unique suffix match on ``schema.table`` / ``table``."""
        table = self.table(name)
        if table:
            return table
        lower_name = name.lower()
        matches = [
            key for key in self.tables if key == lower_name or key.endswith("." + lower_name)
        ]
        if len(matches) == 1:
            return self.tables[matches[0]]
        return None

    def relations(self, fqn_a: str, fqn_b: str) -> list[Edge]:
        """Every relation (FK, lineage, hint, ...) recorded between two tables."""
        attrs = self.graph.get_edge_data(tnode(fqn_a), tnode(fqn_b))
        if not attrs or attrs.get("etype") != "relation":
            return []
        return list(attrs["relations"])

    def all_edges(self) -> list[Edge]:
        """Every table-to-table relation in the graph."""
        edges: list[Edge] = []
        for _, _, attrs in self.graph.edges(data=True):
            if attrs.get("etype") == "relation":
                edges.extend(attrs["relations"])
        return edges

    def table_graph(self, kinds: frozenset[str] | set[str] | None = JOIN_KINDS) -> nx.Graph:
        """Table-only projection with the ``relation`` edges of the given kinds, for path-finding.

        The default keeps join-capable kinds only (:data:`JOIN_KINDS`); ``kinds=None`` keeps every
        relation, lineage included. Edge ``weight`` is the best join cost among the kept kinds.
        """
        projection = nx.Graph()
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("ntype") == "table":
                projection.add_node(node, **attrs)
        for node_a, node_b, attrs in self.graph.edges(data=True):
            if attrs.get("etype") != "relation":
                continue
            if kinds is None:
                kept = attrs["relations"]
            else:
                kept = [relation for relation in attrs["relations"] if relation.kind in kinds]
            if kept:
                weight = min(_relation_weight(relation.kind) for relation in kept)
                projection.add_edge(node_a, node_b, weight=weight, relations=kept)
        return projection

    def lineage(self, fqn: str) -> tuple[list[str], list[str]]:
        """(upstream, downstream) table fqns connected to ``fqn`` by ``lineage`` relations."""
        table_node = tnode(fqn)
        upstream: set[str] = set()
        downstream: set[str] = set()
        if table_node not in self.graph:
            return [], []
        for neighbor in self.graph.neighbors(table_node):
            attrs = self.graph[table_node][neighbor]
            if attrs.get("etype") != "relation":
                continue
            for relation in attrs["relations"]:
                if relation.kind != "lineage":
                    continue
                if relation.from_table.lower() == fqn.lower():
                    downstream.add(relation.to_table)
                else:
                    upstream.add(relation.from_table)
        return sorted(upstream), sorted(downstream)

    def stats(self) -> dict[str, int]:
        """Node, relation and source counts (served by the API and shown in the UI)."""
        node_types = Counter(attrs.get("ntype", "?") for _, attrs in self.graph.nodes(data=True))
        edge_types = Counter(
            attrs.get("etype", "?") for _, _, attrs in self.graph.edges(data=True)
        )
        return {
            "tables": node_types.get("table", 0),
            "columns": node_types.get("column", 0),
            "terms": node_types.get("term", 0),
            "tokens": node_types.get("token", 0),
            "relations": len(self.all_edges()),
            "relation_pairs": edge_types.get("relation", 0),
            "sources": len(self.sources),
        }


def build_graph(snapshots: list[SchemaSnapshot]) -> SchemaGraph:
    """Merge snapshots into one graph, order-independently.

    Snapshots merge in :func:`merge_key` order in three passes: all tables first, then all
    relation edges, then all glossary terms, so resolution never depends on merge order.
    """
    schema_graph = SchemaGraph()
    ordered = sorted(snapshots, key=merge_key)
    for snap in ordered:
        schema_graph.add_tables(snap)
    for snap in ordered:
        for edge in snap.edges:
            schema_graph.add_edge(edge)
    for snap in ordered:
        for term in snap.terms:
            schema_graph.add_term(term)
    return schema_graph
