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

DIALECT = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}


@dataclass
class GoldColumns:
    columns: set[tuple[str, str]] = field(default_factory=set)  # (schema key, lowercase column)
    tables: set[str] = field(default_factory=set)  # schema keys of base tables referenced
    unresolved: set[str] = field(default_factory=set)  # column names that matched no base table
    parsed: bool = True


def _norm(s: str) -> str:
    return s.strip().strip('`"').lower()


def _table_fqn(t: exp.Table) -> str:
    parts = []
    for key in ("catalog", "db"):
        p = t.args.get(key)
        if p is not None:
            parts.append(_norm(p.name))
    parts.append(_norm(t.name))
    # BigQuery backticked `project.dataset.table` may arrive as one identifier
    return ".".join(x for part in parts for x in part.split(".") if x)


def gold_columns(sql: str, dialect: str, schema: dict[str, set[str]], resolve: Callable[[str], str | None]) -> GoldColumns:
    """``schema`` maps a schema key (canonical table name) to its lowercase column names;
    ``resolve`` maps a raw table name from the SQL to that key (or ``None`` if unknown)."""
    out = GoldColumns()
    try:
        trees = sqlglot.parse(sql, read=DIALECT.get(dialect, dialect))
    except Exception:
        out.parsed = False
        return out
    trees = [t for t in trees if t is not None]
    if not trees:
        out.parsed = False
        return out
    ctes: set[str] = set()
    for tree in trees:
        for cte in tree.find_all(exp.CTE):
            ctes.add(_norm(cte.alias_or_name))
    alias_map: dict[str, str] = {}
    for tree in trees:
        for t in tree.find_all(exp.Table):
            fqn = _table_fqn(t)
            if fqn in ctes:
                continue
            key = resolve(fqn)
            if key is None:
                continue
            out.tables.add(key)
            if t.alias:
                alias_map[_norm(t.alias)] = key
            alias_map.setdefault(fqn.split(".")[-1], key)
            alias_map.setdefault(fqn, key)
    for tree in trees:
        for c in tree.find_all(exp.Column):
            col = _norm(c.name)
            if not col or col == "*":
                continue
            qual = _norm(c.table) if c.table else ""
            if qual and qual in alias_map:
                cands = [alias_map[qual]]
            elif qual and qual in ctes:
                cands = sorted(out.tables)
            else:
                cands = sorted(out.tables)
            hits = [t for t in cands if col in schema.get(t, set())]
            if hits:
                out.columns.update((t, col) for t in hits)
            else:
                out.unresolved.add(col)
    return out
