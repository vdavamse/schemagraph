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
    return datetime.now(UTC)


class Column(BaseModel):
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
        return ".".join(p for p in (self.catalog, self.schema_name, self.name) if p)

    def column(self, name: str) -> Column | None:
        lname = name.lower()
        for c in self.columns:
            if c.name.lower() == lname:
                return c
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
    model_config = ConfigDict(extra="forbid")

    source: str
    source_type: str
    tables: list[Table] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    terms: list[BusinessTerm] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    warnings: list[str] = Field(default_factory=list)
    priority: int | None = None  # merge order: lower merges first and wins conflicting fields; None = by source_type (graph.build.SOURCE_PRIORITY)

    def stamp(self) -> SchemaSnapshot:
        """Fill ``source`` on every child object that does not carry one."""
        for t in self.tables:
            t.source = t.source or self.source
        for e in self.edges:
            e.source = e.source or self.source
        for b in self.terms:
            b.source = b.source or self.source
        return self

    def table(self, fqn: str) -> Table | None:
        lf = fqn.lower()
        for t in self.tables:
            if t.fqn.lower() == lf:
                return t
        return None


# --------------------------------------------------------------------------- linking results


class LinkedColumn(BaseModel):
    name: str
    data_type: str | None = None
    description: str | None = None
    score: float = 0.0
    reason: str | None = None


class LinkedTable(BaseModel):
    fqn: str
    score: float
    is_anchor: bool = False
    columns: list[LinkedColumn] = Field(default_factory=list)
    description: str | None = None
    kind: TableKind = "table"


class JoinStep(BaseModel):
    from_table: str
    to_table: str
    kind: EdgeKind
    on: str
    source: str = ""


class JoinPath(BaseModel):
    tables: list[str]
    steps: list[JoinStep]
    reliability: float


class LinkResult(BaseModel):
    question: str
    tables: list[LinkedTable]
    join_paths: list[JoinPath]
    anchors: list[str]
    terms_matched: list[str]
    glossary: dict[str, list[str]] = Field(default_factory=dict)
    ddl: str = ""
    stats: dict[str, float | int | str] = Field(default_factory=dict)
    ranking: list[tuple[str, float]] = Field(default_factory=list, description="Top candidate tables by score (debug only)")
