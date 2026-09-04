"""DDL connector: parse ``CREATE TABLE`` statements (any sqlglot dialect) into a snapshot.

This is what the frontend's "paste your DDL" box feeds. Handles inline and
table-level PRIMARY KEY / FOREIGN KEY constraints, column comments (`COMMENT 'x'`),
and ``ALTER TABLE ... ADD FOREIGN KEY`` statements.
"""

from __future__ import annotations

from typing import ClassVar

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp

from schemagraph.connectors.base import register
from schemagraph.model import Column, Edge, SchemaSnapshot, Table


class DDLConfig(BaseModel):
    ddl: str = Field(description="One or more CREATE TABLE statements")
    dialect: str | None = Field(default=None, description="sqlglot dialect, e.g. postgres, snowflake, bigquery, duckdb")
    default_schema: str | None = Field(default=None, description="Schema to assume for unqualified tables")
    default_catalog: str | None = None


def _ident(node: exp.Expression | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, exp.Identifier):
        return node.name
    return node.name or None


def _table_parts(t: exp.Table, cfg: DDLConfig) -> tuple[str | None, str | None, str]:
    catalog = t.catalog or cfg.default_catalog or None
    schema = t.db or cfg.default_schema or None
    return catalog, schema, t.name


def _fqn(catalog: str | None, schema: str | None, name: str) -> str:
    return ".".join(p for p in (catalog, schema, name) if p)


def _resolve_ref(t: exp.Table, cfg: DDLConfig, known: dict[str, str]) -> str:
    """Resolve a referenced table name to an FQN, preferring already-parsed tables."""
    catalog, schema, name = _table_parts(t, cfg)
    fqn = _fqn(catalog, schema, name)
    if fqn.lower() in known:
        return known[fqn.lower()]
    # unqualified reference: find a unique table with that bare name
    matches = [v for k, v in known.items() if k.split(".")[-1] == name.lower()]
    if len(matches) == 1:
        return matches[0]
    return fqn


def parse_ddl(cfg: DDLConfig, source: str = "ddl") -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="ddl")
    try:
        statements = sqlglot.parse(cfg.ddl, read=cfg.dialect, error_level=sqlglot.ErrorLevel.IGNORE)
    except sqlglot.errors.ParseError as e:  # pragma: no cover - defensive
        snap.warnings.append(f"parse error: {e}")
        statements = []

    known: dict[str, str] = {}
    pending_fks: list[tuple[str, exp.ForeignKey]] = []

    for stmt in statements:
        if stmt is None:
            continue
        if isinstance(stmt, exp.Create) and isinstance(stmt.this, exp.Schema):
            schema_node: exp.Schema = stmt.this
            tnode = schema_node.this
            if not isinstance(tnode, exp.Table):
                continue
            catalog, schema, name = _table_parts(tnode, cfg)
            kind = "view" if stmt.args.get("kind", "").upper() == "VIEW" else "table"
            table = Table(name=name, schema=schema, catalog=catalog, kind=kind, source=source)
            comment_props = [p for p in (stmt.args.get("properties") or []) if isinstance(p, exp.SchemaCommentProperty)]
            if comment_props:
                table.description = comment_props[0].this.name
            for e in schema_node.expressions:
                if isinstance(e, exp.ColumnDef):
                    col = Column(name=e.name, data_type=e.args["kind"].sql(dialect=cfg.dialect) if e.args.get("kind") else None)
                    for c in e.constraints:
                        ck = c.kind
                        if isinstance(ck, exp.PrimaryKeyColumnConstraint):
                            col.is_primary_key = True
                            table.primary_key.append(col.name)
                        elif isinstance(ck, exp.NotNullColumnConstraint):
                            col.nullable = bool(ck.args.get("allow_null"))
                        elif isinstance(ck, exp.CommentColumnConstraint):
                            col.description = ck.this.name
                        elif isinstance(ck, exp.Reference):
                            ref = ck.this
                            if isinstance(ref, exp.Schema) and isinstance(ref.this, exp.Table):
                                pending_fks.append(
                                    (
                                        table.fqn,
                                        exp.ForeignKey(
                                            expressions=[exp.to_identifier(col.name)],
                                            reference=ck,
                                        ),
                                    )
                                )
                    table.columns.append(col)
                elif isinstance(e, exp.PrimaryKey):
                    for pk in e.expressions:
                        cname = _ident(pk) or pk.name
                        table.primary_key.append(cname)
                        c = table.column(cname)
                        if c:
                            c.is_primary_key = True
                elif isinstance(e, exp.ForeignKey):
                    pending_fks.append((table.fqn, e))
                elif isinstance(e, exp.Constraint):
                    for sub in e.expressions:
                        if isinstance(sub, exp.ForeignKey):
                            pending_fks.append((table.fqn, sub))
                        elif isinstance(sub, exp.PrimaryKey):
                            for pk in sub.expressions:
                                cname = _ident(pk) or pk.name
                                table.primary_key.append(cname)
                                c = table.column(cname)
                                if c:
                                    c.is_primary_key = True
            table.primary_key = list(dict.fromkeys(table.primary_key))
            snap.tables.append(table)
            known[table.fqn.lower()] = table.fqn
        elif isinstance(stmt, exp.Alter) and isinstance(stmt.this, exp.Table):
            catalog, schema, name = _table_parts(stmt.this, cfg)
            owner_fqn = known.get(_fqn(catalog, schema, name).lower(), _fqn(catalog, schema, name))
            for action in stmt.args.get("actions") or []:
                for fk in action.find_all(exp.ForeignKey):
                    pending_fks.append((owner_fqn, fk))
        elif isinstance(stmt, exp.Comment):
            # COMMENT ON TABLE t IS '...' / COMMENT ON COLUMN t.c IS '...'
            kind = (stmt.args.get("kind") or "").upper()
            target = stmt.this
            text = stmt.expression.name if stmt.expression is not None else None
            if kind == "TABLE" and isinstance(target, exp.Table):
                fqn = _resolve_ref(target, cfg, known)
                t = snap.table(fqn)
                if t:
                    t.description = text
            elif kind == "COLUMN" and isinstance(target, exp.Column):
                tbl = target.table
                for t in snap.tables:
                    if t.name.lower() == tbl.lower():
                        c = t.column(target.name)
                        if c:
                            c.description = text

    for owner_fqn, fk in pending_fks:
        ref = fk.args.get("reference")
        if not isinstance(ref, exp.Reference):
            continue
        ref_schema = ref.this
        if not isinstance(ref_schema, exp.Schema) or not isinstance(ref_schema.this, exp.Table):
            continue
        to_fqn = _resolve_ref(ref_schema.this, cfg, known)
        from_cols = [_ident(c) or c.name for c in fk.expressions]
        to_cols = [_ident(c) or c.name for c in ref_schema.expressions]
        snap.edges.append(
            Edge(
                kind="foreign_key",
                from_table=owner_fqn,
                to_table=to_fqn,
                from_columns=from_cols,
                to_columns=to_cols,
                source=source,
            )
        )
    if not snap.tables:
        snap.warnings.append("no CREATE TABLE statements found")
    return snap.stamp()


@register
class DDLConnector:
    type_name: ClassVar[str] = "ddl"
    Config = DDLConfig

    def __init__(self, name: str, config: DDLConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        snap = parse_ddl(self.config, self.name)
        return f"parsed {len(snap.tables)} tables, {len(snap.edges)} foreign keys"

    def introspect(self) -> SchemaSnapshot:
        return parse_ddl(self.config, self.name)
