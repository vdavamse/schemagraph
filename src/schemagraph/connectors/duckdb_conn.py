"""DuckDB connector: introspect a local DuckDB file (tables, columns, PK/FK, samples)."""

from __future__ import annotations

from typing import ClassVar

import duckdb
from pydantic import BaseModel, Field

from schemagraph.connectors.base import register
from schemagraph.model import Column, Edge, SchemaSnapshot, Table


class DuckDBConfig(BaseModel):
    path: str = Field(description="Path to the .duckdb file")
    schemas: list[str] | None = Field(default=None, description="Restrict to these schemas (default: all non-system)")
    sample_values: int = Field(default=5, ge=0, le=50, description="Sample distinct values per text column")
    row_counts: bool = True


_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "main.information_schema"}


def introspect_duckdb(cfg: DuckDBConfig, source: str = "duckdb") -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="duckdb")
    con = duckdb.connect(cfg.path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT database_name, schema_name, table_name, 'BASE TABLE', comment
            FROM duckdb_tables() WHERE NOT internal
            UNION ALL
            SELECT database_name, schema_name, view_name, 'VIEW', comment
            FROM duckdb_views() WHERE NOT internal
            """
        ).fetchall()
        tables: dict[str, Table] = {}
        for _db, schema, name, ttype, comment in rows:
            if schema in _SYSTEM_SCHEMAS:
                continue
            if cfg.schemas and schema not in cfg.schemas:
                continue
            t = Table(
                name=name,
                schema=schema,
                catalog=None,
                kind="view" if str(ttype).upper() == "VIEW" else "table",
                description=comment,
                source=source,
            )
            tables[(schema, name)] = t

        cols = con.execute(
            """
            SELECT schema_name, table_name, column_name, data_type, is_nullable, comment, column_index
            FROM duckdb_columns() WHERE NOT internal ORDER BY schema_name, table_name, column_index
            """
        ).fetchall()
        for schema, tname, cname, dtype, nullable, comment, _idx in cols:
            t = tables.get((schema, tname))
            if t is None:
                continue
            t.columns.append(Column(name=cname, data_type=dtype, nullable=bool(nullable), description=comment))

        cons = con.execute(
            """
            SELECT schema_name, table_name, constraint_type, constraint_column_names,
                   referenced_table, referenced_column_names
            FROM duckdb_constraints()
            """
        ).fetchall()
        for schema, tname, ctype, ccols, ref_table, ref_cols in cons:
            t = tables.get((schema, tname))
            if t is None:
                continue
            ccols = list(ccols or [])
            if ctype == "PRIMARY KEY":
                t.primary_key = ccols
                for c in t.columns:
                    if c.name in ccols:
                        c.is_primary_key = True
            elif ctype == "FOREIGN KEY" and ref_table:
                ref_t = tables.get((schema, ref_table))
                to_fqn = ref_t.fqn if ref_t else f"{schema}.{ref_table}"
                snap.edges.append(
                    Edge(
                        kind="foreign_key",
                        from_table=t.fqn,
                        to_table=to_fqn,
                        from_columns=ccols,
                        to_columns=list(ref_cols or []),
                        source=source,
                    )
                )

        for t in tables.values():
            q = f'"{t.schema_name}"."{t.name}"'
            if cfg.row_counts:
                try:
                    t.row_count = con.execute(f"SELECT count(*) FROM {q}").fetchone()[0]
                except duckdb.Error as e:  # pragma: no cover
                    snap.warnings.append(f"row count failed for {t.fqn}: {e}")
            if cfg.sample_values:
                for c in t.columns:
                    if c.data_type and c.data_type.upper().startswith(("VARCHAR", "TEXT", "STRING", "CHAR")):
                        try:
                            vals = con.execute(
                                f'SELECT DISTINCT "{c.name}" FROM {q} WHERE "{c.name}" IS NOT NULL LIMIT {cfg.sample_values}'
                            ).fetchall()
                            c.sample_values = [str(v[0])[:80] for v in vals]
                        except duckdb.Error:  # pragma: no cover
                            pass
        snap.tables = list(tables.values())
    finally:
        con.close()
    return snap.stamp()


@register
class DuckDBConnector:
    type_name: ClassVar[str] = "duckdb"
    Config = DuckDBConfig

    def __init__(self, name: str, config: DuckDBConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        con = duckdb.connect(self.config.path, read_only=True)
        try:
            n = con.execute("SELECT count(*) FROM duckdb_tables() WHERE NOT internal").fetchone()[0]
        finally:
            con.close()
        return f"ok: {n} tables"

    def introspect(self) -> SchemaSnapshot:
        return introspect_duckdb(self.config, self.name)
