"""Name-based relationship inference for catalogs that declare no foreign keys.

Two conservative rules, both emitting ``inferred`` edges (lowest trust weight):

1. **Reference-by-name**: a column ``<x>_id`` / ``<x>id`` / ``<x>_key`` in table A and a
   table named ``x`` / ``xs`` / ``x_*`` in the same schema that has a column
   ``id`` / ``<x>_id`` / ``<x>id``  ->  A.<x>_id -> X.id.
2. **Shared key column**: a key-looking column name (``*_id``, ``*_key``, ``*_code``,
   ``*_number``, ``id``) present in two or more tables of the same schema. Columns
   shared by more than ``max_fanout`` tables are skipped (they're hubs, not keys).

Meant for Glue / BigQuery / Snowflake public datasets. Turn on per source; never
mixed with declared FKs on the same pair (declared evidence already wins by weight).
"""

from __future__ import annotations

import re
from collections import defaultdict

from schemagraph.model import Edge, SchemaSnapshot, Table

_KEY_SUFFIX = re.compile(r"^(?P<stem>.+?)_?(id|key|code|number|no|num)$", re.I)
_GENERIC = {"id", "key", "code", "number", "name", "value", "type", "date", "time", "year", "month", "day", "status"}


def _schema_of(t: Table) -> str:
    return ".".join(p for p in (t.catalog, t.schema_name) if p).lower()


def _table_stem(name: str) -> str:
    n = name.lower()
    if n.endswith("_*"):
        n = n[:-2]
    return n


def infer_edges(snap: SchemaSnapshot, *, max_fanout: int = 12, same_schema_only: bool = True) -> list[Edge]:
    declared = {(e.from_table.lower(), e.to_table.lower()) for e in snap.edges}
    declared |= {(b, a) for a, b in declared}
    out: list[Edge] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(a: Table, ca: str, b: Table, cb: str, why: str, conf: float) -> None:
        if a.fqn.lower() == b.fqn.lower():
            return
        if (a.fqn.lower(), b.fqn.lower()) in declared:
            return
        key = (a.fqn.lower(), ca.lower(), b.fqn.lower(), cb.lower())
        rkey = (b.fqn.lower(), cb.lower(), a.fqn.lower(), ca.lower())
        if key in seen or rkey in seen:
            return
        seen.add(key)
        out.append(Edge(kind="inferred", from_table=a.fqn, to_table=b.fqn, from_columns=[ca], to_columns=[cb], description=why, confidence=conf, source=snap.source))

    groups: dict[str, list[Table]] = defaultdict(list)
    for t in snap.tables:
        groups[_schema_of(t) if same_schema_only else ""].append(t)

    for tables in groups.values():
        by_stem: dict[str, list[Table]] = defaultdict(list)
        for t in tables:
            stem = _table_stem(t.name)
            by_stem[stem].append(t)
            if stem.endswith("s"):
                by_stem[stem[:-1]].append(t)
            if stem.endswith("ies"):
                by_stem[stem[:-3] + "y"].append(t)
        # rule 1: <x>_id -> table x
        for a in tables:
            for c in a.columns:
                m = _KEY_SUFFIX.match(c.name)
                if not m:
                    continue
                stem = m.group("stem").lower()
                if stem in _GENERIC or len(stem) < 3:
                    continue
                for b in by_stem.get(stem, []):
                    if b is a:
                        continue
                    target = next((bc for bc in b.columns if bc.name.lower() in {"id", c.name.lower(), f"{stem}_id", f"{stem}id"} or bc.is_primary_key), None)
                    if target is not None:
                        add(a, c.name, b, target.name, f"name match {c.name} -> {b.name}.{target.name}", 0.6)
        # rule 2: shared key column names
        holders: dict[str, list[tuple[Table, str]]] = defaultdict(list)
        for t in tables:
            for c in t.columns:
                ln = c.name.lower()
                if ln in _GENERIC or "." in ln:
                    continue
                if _KEY_SUFFIX.match(ln) or c.is_primary_key:
                    holders[ln].append((t, c.name))
        for ln, hs in holders.items():
            if len(hs) < 2 or len(hs) > max_fanout:
                continue
            # prefer connecting to a table that has it as primary key, else pairwise
            pks = [(t, c) for t, c in hs if (t.column(c) and t.column(c).is_primary_key)]
            if pks:
                pk_t, pk_c = pks[0]
                for t, c in hs:
                    if t is not pk_t:
                        add(t, c, pk_t, pk_c, f"shared key {ln} (pk in {pk_t.name})", 0.7)
            else:
                for i in range(len(hs)):
                    for j in range(i + 1, len(hs)):
                        add(hs[i][0], hs[i][1], hs[j][0], hs[j][1], f"shared key {ln}", 0.4)
    return out


def with_inferred_edges(snap: SchemaSnapshot, **kw) -> SchemaSnapshot:
    snap.edges.extend(infer_edges(snap, **kw))
    return snap
