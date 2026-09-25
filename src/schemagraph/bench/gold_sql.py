"""Gold column sets parsed from the public gold SQL of Spider 2.0.

Same recipe as EviLink and the Snowflake SQL-Schema-Retrieval suite: parse the gold query with
sqlglot, resolve every column reference to a base table, and keep only columns that exist in that
table's schema, so CTE aliases and computed names drop out. Unqualified columns that several
referenced tables could own are attributed to all of them (recall-oriented gold, slightly harsher
for the linker than a hand annotation would be).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

# sqlglot dialect name per Spider 2.0 dialect (anything else is passed through as is).
DIALECT = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}


@dataclass
class GoldColumns:
    """Columns, base tables and unresolved names of one gold query.

    Attributes:
        columns: ``(schema key, lowercase column)`` pairs.
        tables: Schema keys of the base tables referenced.
        unresolved: Column names that matched no base table.
        parsed: Whether sqlglot parsed the query into at least one statement.
    """

    columns: set[tuple[str, str]] = field(default_factory=set)
    tables: set[str] = field(default_factory=set)
    unresolved: set[str] = field(default_factory=set)
    parsed: bool = True


def _norm(s: str) -> str:
    """Strip whitespace and identifier quotes, then lowercase."""
    return s.strip().strip('`"').lower()


def _table_fqn(table: exp.Table) -> str:
    """Normalised dotted name of a table reference (catalog and db parts when present)."""
    parts = []
    for key in ("catalog", "db"):
        part = table.args.get(key)
        if part is not None:
            parts.append(_norm(part.name))
    parts.append(_norm(table.name))
    # BigQuery backticked `project.dataset.table` may arrive as one identifier
    return ".".join(x for part in parts for x in part.split(".") if x)


def _parse_trees(sql: str, dialect: str) -> list[exp.Expression] | None:
    """Parsed statements of ``sql``, or None when it does not parse into any statement."""
    try:
        trees = sqlglot.parse(sql, read=DIALECT.get(dialect, dialect))
    except Exception:
        return None
    trees = [tree for tree in trees if tree is not None]
    return trees or None


def _cte_names(trees: list[exp.Expression]) -> set[str]:
    """Normalised names of every CTE defined in the statements."""
    ctes: set[str] = set()
    for tree in trees:
        for cte in tree.find_all(exp.CTE):
            ctes.add(_norm(cte.alias_or_name))
    return ctes


def _resolve_tables(
    trees: list[exp.Expression],
    ctes: set[str],
    resolve: Callable[[str], str | None],
    out: GoldColumns,
) -> dict[str, str]:
    """Resolve every base-table reference to its schema key.

    Args:
        trees: Parsed statements.
        ctes: CTE names, which are not base tables.
        resolve: Raw table name -> schema key, or None if unknown.
        out: Collects the schema keys in ``tables``. Mutated in place.

    Returns:
        Qualifier -> schema key: an explicit alias always maps (last wins); the bare table name
        and the full name map to the first table that uses them.
    """
    alias_map: dict[str, str] = {}
    for tree in trees:
        for table in tree.find_all(exp.Table):
            fqn = _table_fqn(table)
            if fqn in ctes:
                continue
            key = resolve(fqn)
            if key is None:
                continue
            out.tables.add(key)
            if table.alias:
                alias_map[_norm(table.alias)] = key
            alias_map.setdefault(fqn.split(".")[-1], key)
            alias_map.setdefault(fqn, key)
    return alias_map


def _resolve_columns(
    trees: list[exp.Expression],
    ctes: set[str],
    alias_map: dict[str, str],
    schema: dict[str, set[str]],
    out: GoldColumns,
) -> None:
    """Attribute every column reference to the base tables that own it.

    A known qualifier picks its table; anything else (no qualifier, a CTE or an unknown one) is
    tried against every referenced table. Names no candidate owns go to ``out.unresolved``.

    Args:
        trees: Parsed statements.
        ctes: CTE names.
        alias_map: Qualifier -> schema key from :func:`_resolve_tables`.
        schema: Schema key -> lowercase column names.
        out: Holds the referenced ``tables``; collects ``columns`` and ``unresolved``. Mutated
            in place.
    """
    for tree in trees:
        for column in tree.find_all(exp.Column):
            name = _norm(column.name)
            if not name or name == "*":
                continue
            qualifier = _norm(column.table) if column.table else ""
            if qualifier and qualifier in alias_map:
                candidates = [alias_map[qualifier]]
            elif qualifier and qualifier in ctes:
                candidates = sorted(out.tables)
            else:
                candidates = sorted(out.tables)
            hits = [table for table in candidates if name in schema.get(table, set())]
            if hits:
                out.columns.update((table, name) for table in hits)
            else:
                out.unresolved.add(name)


def gold_columns(
    sql: str,
    dialect: str,
    schema: dict[str, set[str]],
    resolve: Callable[[str], str | None],
) -> GoldColumns:
    """Parse a gold query into the base-table columns it references.

    Args:
        sql: The gold SQL (one or more statements).
        dialect: Spider 2.0 dialect (``bigquery``, ``snowflake``, ``sqlite``).
        schema: Maps a schema key (canonical table name) to its lowercase column names.
        resolve: Maps a raw table name from the SQL to that key (or ``None`` if unknown).

    Returns:
        The gold columns; ``parsed`` is False (and everything empty) when the SQL does not parse.
    """
    out = GoldColumns()
    trees = _parse_trees(sql, dialect)
    if trees is None:
        out.parsed = False
        return out
    ctes = _cte_names(trees)
    alias_map = _resolve_tables(trees, ctes, resolve, out)
    _resolve_columns(trees, ctes, alias_map, schema, out)
    return out
