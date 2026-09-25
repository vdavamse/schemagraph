"""DuckDB connector: introspect a local DuckDB file (tables, columns, PK/FK, samples)."""

from __future__ import annotations

from typing import ClassVar

import duckdb
from pydantic import BaseModel, Field

from schemagraph.connectors.base import register
from schemagraph.model import Column, Edge, SchemaSnapshot, Table


class DuckDBConfig(BaseModel):
    """A local DuckDB database file, read-only, with optional row counts and sample values."""

    path: str = Field(description="Path to the .duckdb file")
    schemas: list[str] | None = Field(
        default=None,
        description="Restrict to these schemas (default: all non-system)",
    )
    sample_values: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Sample distinct values per text column",
    )
    row_counts: bool = True


# Schemas that describe the database itself, never introspected.
_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "main.information_schema"}
# Data-type prefixes of the columns that get sample values.
TEXT_TYPE_PREFIXES = ("VARCHAR", "TEXT", "STRING", "CHAR")
# A sample value is truncated to this many characters.
MAX_SAMPLE_CHARS = 80

# Tables and views with a literal table type: duckdb_tables() has no table_type column
# (DuckDB 1.5).
_TABLES_SQL = """
    SELECT database_name, schema_name, table_name, 'BASE TABLE', comment
    FROM duckdb_tables() WHERE NOT internal
    UNION ALL
    SELECT database_name, schema_name, view_name, 'VIEW', comment
    FROM duckdb_views() WHERE NOT internal
"""
_COLUMNS_SQL = """
    SELECT schema_name, table_name, column_name, data_type, is_nullable, comment, column_index
    FROM duckdb_columns() WHERE NOT internal ORDER BY schema_name, table_name, column_index
"""
_CONSTRAINTS_SQL = """
    SELECT schema_name, table_name, constraint_type, constraint_column_names,
           referenced_table, referenced_column_names
    FROM duckdb_constraints()
"""

# (schema, table name) -> table
_TableMap = dict[tuple[str, str], Table]


def _read_tables(
    con: duckdb.DuckDBPyConnection,
    cfg: DuckDBConfig,
    source: str,
) -> _TableMap:
    """Every non-system table and view the config selects, by (schema, name)."""
    tables: _TableMap = {}
    for _db, schema, name, table_type, comment in con.execute(_TABLES_SQL).fetchall():
        if schema in _SYSTEM_SCHEMAS:
            continue
        if cfg.schemas and schema not in cfg.schemas:
            continue
        tables[(schema, name)] = Table(
            name=name,
            schema=schema,
            catalog=None,
            kind="view" if str(table_type).upper() == "VIEW" else "table",
            description=comment,
            source=source,
        )
    return tables


def _read_columns(con: duckdb.DuckDBPyConnection, tables: _TableMap) -> None:
    """Append every column to its table in column order. Mutates ``tables`` in place."""
    rows = con.execute(_COLUMNS_SQL).fetchall()
    for schema, table_name, column_name, data_type, nullable, comment, _index in rows:
        table = tables.get((schema, table_name))
        if table is None:
            continue
        table.columns.append(
            Column(
                name=column_name,
                data_type=data_type,
                nullable=bool(nullable),
                description=comment,
            )
        )


def _read_constraints(
    con: duckdb.DuckDBPyConnection,
    tables: _TableMap,
    source: str,
) -> list[Edge]:
    """Apply primary keys to the tables and return the foreign-key edges.

    Mutates ``tables`` in place (primary keys).
    """
    edges: list[Edge] = []
    rows = con.execute(_CONSTRAINTS_SQL).fetchall()
    for schema, table_name, constraint_type, key_columns, ref_table, ref_cols in rows:
        table = tables.get((schema, table_name))
        if table is None:
            continue
        key_columns = list(key_columns or [])
        if constraint_type == "PRIMARY KEY":
            table.primary_key = key_columns
            for column in table.columns:
                if column.name in key_columns:
                    column.is_primary_key = True
        elif constraint_type == "FOREIGN KEY" and ref_table:
            referenced = tables.get((schema, ref_table))
            to_fqn = referenced.fqn if referenced else f"{schema}.{ref_table}"
            edges.append(
                Edge(
                    kind="foreign_key",
                    from_table=table.fqn,
                    to_table=to_fqn,
                    from_columns=key_columns,
                    to_columns=list(ref_cols or []),
                    source=source,
                )
            )
    return edges


def _add_row_counts_and_samples(
    con: duckdb.DuckDBPyConnection,
    tables: _TableMap,
    cfg: DuckDBConfig,
    snap: SchemaSnapshot,
) -> None:
    """Query each table's row count and text columns' distinct sample values.

    Mutates the tables in place; a failed row count becomes a snapshot warning, a failed
    sample query is skipped.

    Args:
        con: Open connection.
        tables: Tables by (schema, name).
        cfg: Connector config (``row_counts``, ``sample_values``).
        snap: Snapshot that receives warnings.
    """
    for table in tables.values():
        quoted = f'"{table.schema_name}"."{table.name}"'
        if cfg.row_counts:
            try:
                table.row_count = con.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
            except duckdb.Error as e:  # pragma: no cover
                snap.warnings.append(f"row count failed for {table.fqn}: {e}")
        if cfg.sample_values:
            for column in table.columns:
                if column.data_type and column.data_type.upper().startswith(TEXT_TYPE_PREFIXES):
                    try:
                        values = con.execute(
                            f'SELECT DISTINCT "{column.name}" FROM {quoted}'
                            f' WHERE "{column.name}" IS NOT NULL LIMIT {cfg.sample_values}'
                        ).fetchall()
                        column.sample_values = [str(v[0])[:MAX_SAMPLE_CHARS] for v in values]
                    except duckdb.Error:  # pragma: no cover
                        pass


def introspect_duckdb(cfg: DuckDBConfig, source: str = "duckdb") -> SchemaSnapshot:
    """Read tables, columns, keys, row counts and samples from a DuckDB file (read-only)."""
    snap = SchemaSnapshot(source=source, source_type="duckdb")
    con = duckdb.connect(cfg.path, read_only=True)
    try:
        tables = _read_tables(con, cfg, source)
        _read_columns(con, tables)
        snap.edges.extend(_read_constraints(con, tables, source))
        _add_row_counts_and_samples(con, tables, cfg, snap)
        snap.tables = list(tables.values())
    finally:
        con.close()
    return snap.stamp()


@register
class DuckDBConnector:
    """Connector over a local DuckDB file."""

    type_name: ClassVar[str] = "duckdb"
    Config = DuckDBConfig

    def __init__(self, name: str, config: DuckDBConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        """Open the file read-only and count its tables."""
        con = duckdb.connect(self.config.path, read_only=True)
        try:
            count_sql = "SELECT count(*) FROM duckdb_tables() WHERE NOT internal"
            count = con.execute(count_sql).fetchone()[0]
        finally:
            con.close()
        return f"ok: {count} tables"

    def introspect(self) -> SchemaSnapshot:
        """Read the database into a snapshot."""
        return introspect_duckdb(self.config, self.name)
