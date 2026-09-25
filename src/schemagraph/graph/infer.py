"""Name-based relationship inference for catalogs that declare no foreign keys.

Two conservative rules, both emitting ``inferred`` edges (lowest trust weight):

1. **Reference-by-name**: a column ``<x>_id`` / ``<x>id`` (also ``_key``, ``_code``,
   ``_number``, ``_no``, ``_num``) in table A, with ``x`` at least three characters and not a
   generic word, and a table named ``x`` / ``xs`` / ``x`` with ``ies`` / ``x_*`` in the same
   schema. The target column is the first of ``id``, A's column name, ``<x>_id``, ``<x>id`` or
   any primary-key column  ->  A.<x>_id -> X.<target>.
2. **Shared key column**: a key-looking column name (the suffixes above) or a primary-key
   column, present in two or more tables of the same schema. Generic names (``id``, ``code``,
   ``name``, ...) and names shared by more than ``max_fanout`` tables are skipped (they're hubs,
   not keys). If one holder has it as primary key, the others link to that table; otherwise
   every pair links.

Meant for Glue / BigQuery / Snowflake public datasets. Turn on per source; never
mixed with declared FKs on the same pair (declared evidence already wins by weight).
"""

from __future__ import annotations

import re
from collections import defaultdict

from schemagraph.model import Column, Edge, SchemaSnapshot, Table

# A key-looking column name: ``<stem>_id``, ``<stem>id``, ``<stem>_key``, ``<stem>_code``, ...
_KEY_SUFFIX = re.compile(r"^(?P<stem>.+?)_?(id|key|code|number|no|num)$", re.I)
# Column names (and reference stems) too generic to identify a table or a shared key.
_GENERIC_NAMES = {
    "id",
    "key",
    "code",
    "number",
    "name",
    "value",
    "type",
    "date",
    "time",
    "year",
    "month",
    "day",
    "status",
}
# Shortest reference stem rule 1 accepts (``<x>_id`` with ``x`` shorter is too ambiguous).
MIN_STEM_LEN = 3
# Confidence of each rule's edges: a name reference to a table, a shared key that is a primary
# key in one table, and a shared key with no primary key (every holder paired with every other).
REFERENCE_CONFIDENCE = 0.6
PK_SHARED_CONFIDENCE = 0.7
PAIRWISE_SHARED_CONFIDENCE = 0.4


class _EdgeCollector:
    """Accumulates inferred edges for one snapshot, in emission order.

    Skips self-pairs, pairs that already have a declared edge (either direction) and a column
    pair already emitted in either direction.
    """

    def __init__(self, source: str, declared: set[tuple[str, str]]) -> None:
        self.source = source
        self.declared = declared
        self.edges: list[Edge] = []
        self._seen: set[tuple[str, str, str, str]] = set()

    def add(
        self,
        from_table: Table,
        from_column: str,
        to_table: Table,
        to_column: str,
        why: str,
        confidence: float,
    ) -> None:
        """Emit ``from_table.from_column -> to_table.to_column`` unless it is skipped.

        Args:
            from_table: Referencing table.
            from_column: Column name in ``from_table``.
            to_table: Referenced table.
            to_column: Column name in ``to_table``.
            why: Edge description explaining the rule that fired.
            confidence: Edge confidence in [0, 1].
        """
        from_key, to_key = from_table.fqn.lower(), to_table.fqn.lower()
        if from_key == to_key:
            return
        if (from_key, to_key) in self.declared:
            return
        key = (from_key, from_column.lower(), to_key, to_column.lower())
        reverse_key = (to_key, to_column.lower(), from_key, from_column.lower())
        if key in self._seen or reverse_key in self._seen:
            return
        self._seen.add(key)
        self.edges.append(
            Edge(
                kind="inferred",
                from_table=from_table.fqn,
                to_table=to_table.fqn,
                from_columns=[from_column],
                to_columns=[to_column],
                description=why,
                confidence=confidence,
                source=self.source,
            )
        )


def _schema_of(table: Table) -> str:
    """Lowercased ``catalog.schema`` of a table (the parts that are set)."""
    return ".".join(part for part in (table.catalog, table.schema_name) if part).lower()


def _table_stem(name: str) -> str:
    """Lowercased table name without a trailing ``_*`` wildcard (sharded BigQuery tables)."""
    stem = name.lower()
    if stem.endswith("_*"):
        stem = stem[:-2]
    return stem


def _declared_pairs(snap: SchemaSnapshot) -> set[tuple[str, str]]:
    """Lowercased table pairs of the snapshot's declared edges, in both directions."""
    declared = {(edge.from_table.lower(), edge.to_table.lower()) for edge in snap.edges}
    declared |= {(b, a) for a, b in declared}
    return declared


def _group_by_schema(tables: list[Table], same_schema_only: bool) -> dict[str, list[Table]]:
    """Tables per lowercased schema, or all in one group when ``same_schema_only`` is False."""
    groups: dict[str, list[Table]] = defaultdict(list)
    for table in tables:
        groups[_schema_of(table) if same_schema_only else ""].append(table)
    return groups


def _tables_by_stem(tables: list[Table]) -> dict[str, list[Table]]:
    """Tables by name stem, also registered under the singular of a plural (``s``/``ies``) stem."""
    by_stem: dict[str, list[Table]] = defaultdict(list)
    for table in tables:
        stem = _table_stem(table.name)
        by_stem[stem].append(table)
        if stem.endswith("s"):
            by_stem[stem[:-1]].append(table)
        if stem.endswith("ies"):
            by_stem[stem[:-3] + "y"].append(table)
    return by_stem


def _reference_stem(column_name: str) -> str | None:
    """The table stem a key-looking column refers to, or None when it names no table."""
    match = _KEY_SUFFIX.match(column_name)
    if not match:
        return None
    stem = match.group("stem").lower()
    if stem in _GENERIC_NAMES or len(stem) < MIN_STEM_LEN:
        return None
    return stem


def _referenced_column(table: Table, column_name: str, stem: str) -> Column | None:
    """First column of ``table`` a ``<stem>`` reference can point at, or None.

    That is ``id``, the referencing column's own name, ``<stem>_id``, ``<stem>id``, or any
    primary-key column.
    """
    names = {"id", column_name.lower(), f"{stem}_id", f"{stem}id"}
    candidates = (
        column
        for column in table.columns
        if column.name.lower() in names or column.is_primary_key
    )
    return next(candidates, None)


def _reference_by_name(tables: list[Table], collector: _EdgeCollector) -> None:
    """Rule 1: link a ``<x>_id`` column to a table named ``x`` of the same group."""
    by_stem = _tables_by_stem(tables)
    for table in tables:
        for column in table.columns:
            stem = _reference_stem(column.name)
            if stem is None:
                continue
            for referenced in by_stem.get(stem, []):
                if referenced is table:
                    continue
                target = _referenced_column(referenced, column.name, stem)
                if target is not None:
                    collector.add(
                        table,
                        column.name,
                        referenced,
                        target.name,
                        f"name match {column.name} -> {referenced.name}.{target.name}",
                        REFERENCE_CONFIDENCE,
                    )


def _shared_key_holders(tables: list[Table]) -> dict[str, list[tuple[Table, str]]]:
    """``(table, column name)`` holders of every key-looking, non-generic lowercased column name."""
    holders: dict[str, list[tuple[Table, str]]] = defaultdict(list)
    for table in tables:
        for column in table.columns:
            lower_name = column.name.lower()
            if lower_name in _GENERIC_NAMES or "." in lower_name:
                continue
            if _KEY_SUFFIX.match(lower_name) or column.is_primary_key:
                holders[lower_name].append((table, column.name))
    return holders


def _is_primary_key_column(table: Table, column_name: str) -> bool:
    """Whether ``table`` has a column ``column_name`` marked as primary key."""
    column = table.column(column_name)
    return bool(column and column.is_primary_key)


def _shared_keys(tables: list[Table], collector: _EdgeCollector, max_fanout: int) -> None:
    """Rule 2: link tables of the same group that share a key-looking column name.

    A name held by fewer than two or more than ``max_fanout`` tables is skipped. When a holder
    has the column as primary key, every other holder links to the first such table; otherwise
    every pair of holders is linked.
    """
    for lower_name, holders in _shared_key_holders(tables).items():
        if len(holders) < 2 or len(holders) > max_fanout:
            continue
        pk_holders = [
            (table, column) for table, column in holders if _is_primary_key_column(table, column)
        ]
        if pk_holders:
            pk_table, pk_column = pk_holders[0]
            for table, column in holders:
                if table is not pk_table:
                    collector.add(
                        table,
                        column,
                        pk_table,
                        pk_column,
                        f"shared key {lower_name} (pk in {pk_table.name})",
                        PK_SHARED_CONFIDENCE,
                    )
        else:
            for i in range(len(holders)):
                for j in range(i + 1, len(holders)):
                    collector.add(
                        holders[i][0],
                        holders[i][1],
                        holders[j][0],
                        holders[j][1],
                        f"shared key {lower_name}",
                        PAIRWISE_SHARED_CONFIDENCE,
                    )


def infer_edges(
    snap: SchemaSnapshot,
    *,
    max_fanout: int = 12,
    same_schema_only: bool = True,
) -> list[Edge]:
    """Infer ``inferred`` relation edges from column and table naming conventions.

    Per schema group, rule 1 (reference by name) runs before rule 2 (shared key column); see
    the module docstring for both. Pairs with a declared edge are never inferred.

    Args:
        snap: The snapshot to read tables and declared edges from (not modified).
        max_fanout: Rule 2 skips a key name held by more than this many tables (a hub, not a
            key).
        same_schema_only: Only relate tables of the same ``catalog.schema``; when False the
            whole snapshot is one group.

    Returns:
        The inferred edges, in emission order, with the snapshot's source.
    """
    collector = _EdgeCollector(snap.source, _declared_pairs(snap))
    for tables in _group_by_schema(snap.tables, same_schema_only).values():
        _reference_by_name(tables, collector)
        _shared_keys(tables, collector, max_fanout)
    return collector.edges


def with_inferred_edges(snap: SchemaSnapshot, **options) -> SchemaSnapshot:
    """Append :func:`infer_edges` to a snapshot's edges and return the snapshot.

    Mutates ``snap`` in place. ``options`` are passed to :func:`infer_edges`.
    """
    snap.edges.extend(infer_edges(snap, **options))
    return snap
