"""Core data model shared by every connector, the graph builder and the linker.

A connector introspects one source (a DDL file, a DuckDB database, a dbt project,
Unity Catalog, AWS Glue, Collibra, ...) and returns a :class:`SchemaSnapshot`.
Snapshots are merged into a single graph; every table, edge and term keeps its
``source`` so provenance survives the merge.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EdgeKind = Literal[
    "foreign_key",  # declared FK (DDL, DuckDB, Unity constraints, Collibra "references")
    "lineage",  # dbt ref()/source() DAG, Unity table lineage
    "relationship_test",  # dbt `relationships` test (FK-equivalent)
    "catalog_relation",  # generic curated relation from a catalog (Collibra)
    "join_hint",  # human-authored join hint
    "inferred",  # name-based inference (last resort)
]

TableKind = Literal["table", "view", "model", "source", "seed", "snapshot", "external"]


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Column(BaseModel):
    """A column of a table, with optional type, description, key flag, samples and tags."""

    model_config = ConfigDict(extra="forbid")

    name: str
    data_type: str | None = None
    description: str | None = None
    nullable: bool | None = None
    is_primary_key: bool = False
    sample_values: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)


class Table(BaseModel):
    """A table, view or dbt model/source, with its columns and the source that described it.

    ``schema_name`` is serialised as ``schema``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    schema_name: str | None = Field(default=None, alias="schema")
    catalog: str | None = None
    kind: TableKind = "table"
    description: str | None = None
    columns: list[Column] = Field(default_factory=list)
    primary_key: list[str] = Field(default_factory=list)
    row_count: int | None = None
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)
    source: str = ""

    @property
    def fqn(self) -> str:
        """Dotted ``catalog.schema.name``, skipping the parts that are unset."""
        return ".".join(p for p in (self.catalog, self.schema_name, self.name) if p)

    def column(self, name: str) -> Column | None:
        """Return the column called ``name`` (case-insensitive), or None."""
        lower_name = name.lower()
        for column in self.columns:
            if column.name.lower() == lower_name:
                return column
        return None


class Edge(BaseModel):
    """A relation between two tables. ``from`` is the referencing / upstream side."""

    model_config = ConfigDict(extra="forbid")

    kind: EdgeKind
    from_table: str
    to_table: str
    from_columns: list[str] = Field(default_factory=list)
    to_columns: list[str] = Field(default_factory=list)
    description: str | None = None
    confidence: float = 1.0
    source: str = ""

    @property
    def key(self) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
        """Case-insensitive identity: kind, both tables and both column lists, lowercased."""
        return (
            self.kind,
            self.from_table.lower(),
            self.to_table.lower(),
            tuple(c.lower() for c in self.from_columns),
            tuple(c.lower() for c in self.to_columns),
        )


class BusinessTerm(BaseModel):
    """A glossary entry: a business word and the schema objects it refers to.

    ``targets`` are table FQNs or ``table_fqn.column`` references.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    source: str = ""


class SchemaSnapshot(BaseModel):
    """Everything one connector introspected from one source: tables, edges and glossary terms."""

    model_config = ConfigDict(extra="forbid")

    source: str
    source_type: str
    tables: list[Table] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    terms: list[BusinessTerm] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    warnings: list[str] = Field(default_factory=list)
    # merge order: lower merges first and wins conflicting fields;
    # None = by source_type (graph.build.SOURCE_PRIORITY)
    priority: int | None = None

    def stamp(self) -> SchemaSnapshot:
        """Fill ``source`` on every child object that does not carry one."""
        for table in self.tables:
            table.source = table.source or self.source
        for edge in self.edges:
            edge.source = edge.source or self.source
        for term in self.terms:
            term.source = term.source or self.source
        return self

    def table(self, fqn: str) -> Table | None:
        """Return the table with this FQN (case-insensitive), or None."""
        lower_fqn = fqn.lower()
        for table in self.tables:
            if table.fqn.lower() == lower_fqn:
                return table
        return None


# --------------------------------------------------------------------------- linking results


class LinkedColumn(BaseModel):
    """A column kept in a linked table, with its score and why it was kept."""

    name: str
    data_type: str | None = None
    description: str | None = None
    score: float = 0.0
    reason: str | None = None


class LinkedTable(BaseModel):
    """A table in the linked sub-schema, with its score, anchor flag and selected columns."""

    fqn: str
    score: float
    is_anchor: bool = False
    columns: list[LinkedColumn] = Field(default_factory=list)
    description: str | None = None
    kind: TableKind = "table"


class JoinStep(BaseModel):
    """One hop of a join path: two tables, the relation kind and the join condition."""

    from_table: str
    to_table: str
    kind: EdgeKind
    on: str
    source: str = ""


class JoinPath(BaseModel):
    """A chain of tables connecting anchors, with its steps and a reliability score."""

    tables: list[str]
    steps: list[JoinStep]
    reliability: float


class LinkResult(BaseModel):
    """The sub-schema linked to a question: tables, join paths, anchors, glossary hits and DDL."""

    question: str
    tables: list[LinkedTable]
    join_paths: list[JoinPath]
    anchors: list[str]
    terms_matched: list[str]
    glossary: dict[str, list[str]] = Field(default_factory=dict)
    ddl: str = ""
    stats: dict[str, float | int | str] = Field(default_factory=dict)
    ranking: list[tuple[str, float]] = Field(
        default_factory=list,
        description="Top candidate tables by score (debug only)",
    )
