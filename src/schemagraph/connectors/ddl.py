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
    """Pasted DDL: one or more CREATE TABLE statements in any sqlglot dialect."""

    ddl: str = Field(description="One or more CREATE TABLE statements")
    dialect: str | None = Field(
        default=None,
        description="sqlglot dialect, e.g. postgres, snowflake, bigquery, duckdb",
    )
    default_schema: str | None = Field(
        default=None,
        description="Schema to assume for unqualified tables",
    )
    default_catalog: str | None = None


# A foreign key waiting for every table to be parsed: (owning table fqn, FOREIGN KEY node).
_PendingFK = tuple[str, exp.ForeignKey]


def _ident(node: exp.Expression | None) -> str | None:
    """Name of an identifier-like node, or None when absent or empty."""
    if node is None:
        return None
    if isinstance(node, exp.Identifier):
        return node.name
    return node.name or None


def _table_parts(table_expr: exp.Table, cfg: DDLConfig) -> tuple[str | None, str | None, str]:
    """(catalog, schema, name) of a table expression, filling the configured defaults."""
    catalog = table_expr.catalog or cfg.default_catalog or None
    schema = table_expr.db or cfg.default_schema or None
    return catalog, schema, table_expr.name


def _fqn(catalog: str | None, schema: str | None, name: str) -> str:
    """Dot-join the non-empty parts of a table name."""
    return ".".join(p for p in (catalog, schema, name) if p)


def _resolve_ref(table_expr: exp.Table, cfg: DDLConfig, known: dict[str, str]) -> str:
    """Resolve a referenced table name to an FQN, preferring already-parsed tables."""
    catalog, schema, name = _table_parts(table_expr, cfg)
    fqn = _fqn(catalog, schema, name)
    if fqn.lower() in known:
        return known[fqn.lower()]
    # unqualified reference: find a unique table with that bare name
    matches = [v for k, v in known.items() if k.split(".")[-1] == name.lower()]
    if len(matches) == 1:
        return matches[0]
    return fqn


def _parse_statements(cfg: DDLConfig, snap: SchemaSnapshot) -> list[exp.Expression | None]:
    """Parse the DDL leniently; a hard parse error becomes a snapshot warning."""
    try:
        return sqlglot.parse(cfg.ddl, read=cfg.dialect, error_level=sqlglot.ErrorLevel.IGNORE)
    except sqlglot.errors.ParseError as e:  # pragma: no cover - defensive
        snap.warnings.append(f"parse error: {e}")
        return []


def _is_create_table(stmt: exp.Expression) -> bool:
    """Whether a statement is a CREATE with a column list (table or view)."""
    return isinstance(stmt, exp.Create) and isinstance(stmt.this, exp.Schema)


def _apply_primary_key(table: Table, key: exp.PrimaryKey) -> None:
    """Record a table-level PRIMARY KEY. Mutates ``table`` in place."""
    for pk in key.expressions:
        column_name = _ident(pk) or pk.name
        table.primary_key.append(column_name)
        column = table.column(column_name)
        if column:
            column.is_primary_key = True


def _column_from_def(
    column_def: exp.ColumnDef,
    table: Table,
    cfg: DDLConfig,
    pending_fks: list[_PendingFK],
) -> Column:
    """Build a column from its definition and apply its inline constraints.

    Args:
        column_def: The ``ColumnDef`` node.
        table: Owning table; an inline PRIMARY KEY is appended to its ``primary_key``.
        cfg: Parser config (the dialect renders the data type).
        pending_fks: Inline REFERENCES are appended here as foreign keys.

    Returns:
        The column (not yet added to ``table``).
    """
    kind = column_def.args.get("kind")
    column = Column(
        name=column_def.name,
        data_type=kind.sql(dialect=cfg.dialect) if kind else None,
    )
    for constraint in column_def.constraints:
        constraint_kind = constraint.kind
        if isinstance(constraint_kind, exp.PrimaryKeyColumnConstraint):
            column.is_primary_key = True
            table.primary_key.append(column.name)
        elif isinstance(constraint_kind, exp.NotNullColumnConstraint):
            column.nullable = bool(constraint_kind.args.get("allow_null"))
        elif isinstance(constraint_kind, exp.CommentColumnConstraint):
            column.description = constraint_kind.this.name
        elif isinstance(constraint_kind, exp.Reference):
            ref = constraint_kind.this
            if isinstance(ref, exp.Schema) and isinstance(ref.this, exp.Table):
                foreign_key = exp.ForeignKey(
                    expressions=[exp.to_identifier(column.name)],
                    reference=constraint_kind,
                )
                pending_fks.append((table.fqn, foreign_key))
    return column


def _constraint_members(
    constraint: exp.Constraint,
    table: Table,
    pending_fks: list[_PendingFK],
) -> None:
    """Apply the FOREIGN KEY / PRIMARY KEY members of a named ``CONSTRAINT`` clause."""
    for sub in constraint.expressions:
        if isinstance(sub, exp.ForeignKey):
            pending_fks.append((table.fqn, sub))
        elif isinstance(sub, exp.PrimaryKey):
            _apply_primary_key(table, sub)


def _table_from_create(
    stmt: exp.Create,
    cfg: DDLConfig,
    source: str,
    pending_fks: list[_PendingFK],
) -> Table | None:
    """Build a table from a CREATE statement, queueing its foreign keys.

    Args:
        stmt: A CREATE statement accepted by :func:`_is_create_table`.
        cfg: Parser config.
        source: Snapshot source name stamped on the table.
        pending_fks: Every FOREIGN KEY / REFERENCES of the table is appended here.

    Returns:
        The table, or None when the CREATE target is not a table.
    """
    schema_node: exp.Schema = stmt.this
    table_expr = schema_node.this
    if not isinstance(table_expr, exp.Table):
        return None
    catalog, schema, name = _table_parts(table_expr, cfg)
    kind = "view" if stmt.args.get("kind", "").upper() == "VIEW" else "table"
    table = Table(name=name, schema=schema, catalog=catalog, kind=kind, source=source)
    comment_props = [
        p
        for p in (stmt.args.get("properties") or [])
        if isinstance(p, exp.SchemaCommentProperty)
    ]
    if comment_props:
        table.description = comment_props[0].this.name
    for element in schema_node.expressions:
        if isinstance(element, exp.ColumnDef):
            table.columns.append(_column_from_def(element, table, cfg, pending_fks))
        elif isinstance(element, exp.PrimaryKey):
            _apply_primary_key(table, element)
        elif isinstance(element, exp.ForeignKey):
            pending_fks.append((table.fqn, element))
        elif isinstance(element, exp.Constraint):
            _constraint_members(element, table, pending_fks)
    table.primary_key = list(dict.fromkeys(table.primary_key))
    return table


def _alter_foreign_keys(
    stmt: exp.Alter,
    cfg: DDLConfig,
    known: dict[str, str],
    pending_fks: list[_PendingFK],
) -> None:
    """Queue the foreign keys added by an ``ALTER TABLE`` statement."""
    catalog, schema, name = _table_parts(stmt.this, cfg)
    owner_fqn = known.get(_fqn(catalog, schema, name).lower(), _fqn(catalog, schema, name))
    for action in stmt.args.get("actions") or []:
        for fk in action.find_all(exp.ForeignKey):
            pending_fks.append((owner_fqn, fk))


def _apply_comment(
    stmt: exp.Comment,
    cfg: DDLConfig,
    known: dict[str, str],
    snap: SchemaSnapshot,
) -> None:
    """Apply ``COMMENT ON TABLE`` / ``COMMENT ON COLUMN``. Mutates ``snap``'s tables in place."""
    kind = (stmt.args.get("kind") or "").upper()
    target = stmt.this
    text = stmt.expression.name if stmt.expression is not None else None
    if kind == "TABLE" and isinstance(target, exp.Table):
        fqn = _resolve_ref(target, cfg, known)
        table = snap.table(fqn)
        if table:
            table.description = text
    elif kind == "COLUMN" and isinstance(target, exp.Column):
        table_name = target.table
        for table in snap.tables:
            if table.name.lower() == table_name.lower():
                column = table.column(target.name)
                if column:
                    column.description = text


def _resolve_foreign_keys(
    pending_fks: list[_PendingFK],
    cfg: DDLConfig,
    known: dict[str, str],
    source: str,
) -> list[Edge]:
    """Turn queued foreign keys into edges, resolving targets against the parsed tables."""
    edges: list[Edge] = []
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
        edges.append(
            Edge(
                kind="foreign_key",
                from_table=owner_fqn,
                to_table=to_fqn,
                from_columns=from_cols,
                to_columns=to_cols,
                source=source,
            )
        )
    return edges


def parse_ddl(cfg: DDLConfig, source: str = "ddl") -> SchemaSnapshot:
    """Parse DDL text into a stamped snapshot.

    Foreign keys are resolved after every statement is read, so a reference to a table
    created later in the script still points at its fully qualified name.
    """
    snap = SchemaSnapshot(source=source, source_type="ddl")
    statements = _parse_statements(cfg, snap)
    # lowercased fqn -> fqn as declared, for every table parsed so far
    known: dict[str, str] = {}
    pending_fks: list[_PendingFK] = []

    for stmt in statements:
        if stmt is None:
            continue
        if _is_create_table(stmt):
            table = _table_from_create(stmt, cfg, source, pending_fks)
            if table is None:
                continue
            snap.tables.append(table)
            known[table.fqn.lower()] = table.fqn
        elif isinstance(stmt, exp.Alter) and isinstance(stmt.this, exp.Table):
            _alter_foreign_keys(stmt, cfg, known, pending_fks)
        elif isinstance(stmt, exp.Comment):
            _apply_comment(stmt, cfg, known, snap)

    snap.edges.extend(_resolve_foreign_keys(pending_fks, cfg, known, source))
    if not snap.tables:
        snap.warnings.append("no CREATE TABLE statements found")
    return snap.stamp()


@register
class DDLConnector:
    """Connector over pasted DDL text."""

    type_name: ClassVar[str] = "ddl"
    Config = DDLConfig

    def __init__(self, name: str, config: DDLConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        """Parse the DDL and report what was found."""
        snap = parse_ddl(self.config, self.name)
        return f"parsed {len(snap.tables)} tables, {len(snap.edges)} foreign keys"

    def introspect(self) -> SchemaSnapshot:
        """Parse the DDL into a snapshot."""
        return parse_ddl(self.config, self.name)
