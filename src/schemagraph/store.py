"""DuckDB persistence: connections, snapshots, user glossary and join hints.

One file (``.schemagraph/schemagraph.duckdb`` by default). The graph itself is
rebuilt in memory from the stored snapshots at load time - NetworkX is the working
representation, DuckDB is the durable one. Secrets in connection configs may be
written as ``${ENV_VAR}`` and are substituted when a connector is instantiated.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from schemagraph.model import BusinessTerm, Edge, SchemaSnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS connections (
    name TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    config JSON NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    priority INTEGER
);
CREATE TABLE IF NOT EXISTS snapshots (
    source TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    payload JSON NOT NULL,
    n_tables INTEGER NOT NULL,
    n_edges INTEGER NOT NULL,
    n_terms INTEGER NOT NULL,
    warnings JSON NOT NULL,
    built_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS glossary (
    name TEXT PRIMARY KEY,
    description TEXT,
    synonyms JSON NOT NULL,
    targets JSON NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
CREATE TABLE IF NOT EXISTS join_hints (
    id INTEGER PRIMARY KEY,
    from_table TEXT NOT NULL,
    to_table TEXT NOT NULL,
    from_columns JSON NOT NULL,
    to_columns JSON NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL
);
CREATE SEQUENCE IF NOT EXISTS join_hints_seq START 1;
"""

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# A config key containing any of these (case-insensitive) holds a secret and is redacted.
SECRET_KEY_MARKERS = ("token", "secret", "password", "key")

_CONNECTION_COLUMNS_SQL = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'connections'
"""

_UPSERT_CONNECTION_SQL = """
INSERT INTO connections (name, type, config, created_at, updated_at, priority)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (name) DO UPDATE SET
  type = excluded.type,
  config = excluded.config,
  updated_at = excluded.updated_at,
  priority = {keep}
"""

_CONNECTIONS_SQL = """
SELECT name, type, config, created_at, updated_at, priority
FROM connections
ORDER BY name
"""

_SNAPSHOT_SUMMARY_SQL = """
SELECT n_tables, n_edges, n_terms, built_at, warnings
FROM snapshots
WHERE source = ?
"""

_UPSERT_SNAPSHOT_SQL = """
INSERT INTO snapshots (
  source, source_type, payload, n_tables, n_edges, n_terms, warnings, built_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (source) DO UPDATE SET
  source_type = excluded.source_type,
  payload = excluded.payload,
  n_tables = excluded.n_tables,
  n_edges = excluded.n_edges,
  n_terms = excluded.n_terms,
  warnings = excluded.warnings,
  built_at = excluded.built_at
"""

_SNAPSHOTS_WITH_PRIORITY_SQL = """
SELECT s.payload, c.priority
FROM snapshots s
LEFT JOIN connections c ON c.name = s.source
ORDER BY s.source
"""

_UPSERT_TERM_SQL = """
INSERT INTO glossary (name, description, synonyms, targets, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (name) DO UPDATE SET
  description = excluded.description,
  synonyms = excluded.synonyms,
  targets = excluded.targets,
  updated_at = excluded.updated_at
"""

_INSERT_JOIN_HINT_SQL = """
INSERT INTO join_hints (
  id, from_table, to_table, from_columns, to_columns, description, created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_JOIN_HINTS_SQL = """
SELECT id, from_table, to_table, from_columns, to_columns, description
FROM join_hints
ORDER BY id
"""


def substitute_env(obj: Any) -> Any:
    """Replace ``${ENV_VAR}`` references in strings, recursing into dicts and lists.

    Unset variables are left as the literal ``${ENV_VAR}`` text.
    """
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: substitute_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_env(v) for v in obj]
    return obj


def _is_secret(key: str, value: Any) -> bool:
    """Whether a config entry is a literal secret (``${ENV}`` references are shown as is)."""
    return (
        any(marker in key.lower() for marker in SECRET_KEY_MARKERS)
        and isinstance(value, str)
        and not value.startswith("${")
    )


def redact(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with literal secret values replaced by ``***``."""
    out = {}
    for key, value in config.items():
        if _is_secret(key, value):
            out[key] = "***"
        else:
            out[key] = value
    return out


def _connection_summary(
    name: str,
    type_name: str,
    config: str,
    created: datetime | None,
    updated: datetime | None,
    priority: int | None,
    snap: tuple[Any, ...] | None,
) -> dict[str, Any]:
    """Describe one stored connection (redacted config) and its snapshot, if built.

    Args:
        name: Connection name.
        type_name: Connector type.
        config: The stored config as JSON text.
        created: When the connection was first registered.
        updated: When it was last re-registered.
        priority: Explicit merge priority, or None for the source-type order.
        snap: The snapshot's ``(n_tables, n_edges, n_terms, built_at, warnings)`` row, or None
            when the connection has never been built.

    Returns:
        The JSON-ready summary the API and CLI list.
    """
    return {
        "name": name,
        "type": type_name,
        "config": redact(json.loads(config)),
        "priority": priority,
        "created_at": created.isoformat() if created else None,
        "updated_at": updated.isoformat() if updated else None,
        "built": bool(snap),
        "n_tables": snap[0] if snap else 0,
        "n_edges": snap[1] if snap else 0,
        "n_terms": snap[2] if snap else 0,
        "built_at": snap[3].isoformat() if snap and snap[3] else None,
        "warnings": json.loads(snap[4]) if snap else [],
    }


class Store:
    """The DuckDB file holding connections, raw snapshots, the user glossary and join hints.

    Creates the tables on open and migrates stores that predate per-connection priorities.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        for stmt in _SCHEMA.strip().split(";"):
            if stmt.strip():
                self.con.execute(stmt)
        columns = {row[0] for row in self.con.execute(_CONNECTION_COLUMNS_SQL).fetchall()}
        # stores created before 2026-09-17 (explicit merge order per connection)
        if "priority" not in columns:
            self.con.execute("ALTER TABLE connections ADD COLUMN priority INTEGER")

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.con.close()

    # ------------------------------------------------------------ connections
    def upsert_connection(
        self,
        name: str,
        type_name: str,
        config: dict[str, Any],
        priority: int | None = None,
        *,
        clear_priority: bool = False,
    ) -> None:
        """Insert or update a connection.

        Args:
            name: Connection name (the primary key).
            type_name: Registered connector type.
            config: Connector config, stored as given (``${ENV}`` references unsubstituted).
            priority: Merge order (lower merges first and wins conflicting fields). None keeps
                the stored value on re-register (the UI form has no priority field); a new
                connection with None merges by source type.
            clear_priority: Reset a stored priority to that default.
        """
        now = datetime.now(UTC)
        keep = "NULL" if clear_priority else "COALESCE(excluded.priority, connections.priority)"
        self.con.execute(
            _UPSERT_CONNECTION_SQL.format(keep=keep),
            [name, type_name, json.dumps(config), now, now, None if clear_priority else priority],
        )

    def delete_connection(self, name: str) -> None:
        """Delete a connection and its snapshot."""
        self.con.execute("DELETE FROM connections WHERE name = ?", [name])
        self.con.execute("DELETE FROM snapshots WHERE source = ?", [name])

    def connections(self) -> list[dict[str, Any]]:
        """Summaries of every connection, by name, with redacted configs and snapshot counts."""
        rows = self.con.execute(_CONNECTIONS_SQL).fetchall()
        out = []
        for name, type_name, config, created, updated, priority in rows:
            snap = self.con.execute(_SNAPSHOT_SUMMARY_SQL, [name]).fetchone()
            out.append(
                _connection_summary(name, type_name, config, created, updated, priority, snap)
            )
        return out

    def connection(self, name: str) -> tuple[str, dict[str, Any]] | None:
        """Return a connection's ``(type, config)``, or None when it is not registered."""
        row = self.con.execute(
            "SELECT type, config FROM connections WHERE name = ?",
            [name],
        ).fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1])

    # ------------------------------------------------------------ snapshots
    def save_snapshot(self, snap: SchemaSnapshot) -> None:
        """Insert or replace the snapshot stored for ``snap.source``."""
        self.con.execute(
            _UPSERT_SNAPSHOT_SQL,
            [
                snap.source,
                snap.source_type,
                snap.model_dump_json(by_alias=True),
                len(snap.tables),
                len(snap.edges),
                len(snap.terms),
                json.dumps(snap.warnings),
                snap.created_at,
            ],
        )

    def snapshots(self) -> list[SchemaSnapshot]:
        """All stored snapshots, each carrying its connection's merge ``priority``.

        The merge order is decided by ``build_graph``, not here.
        """
        rows = self.con.execute(_SNAPSHOTS_WITH_PRIORITY_SQL).fetchall()
        out = []
        for payload, priority in rows:
            snap = SchemaSnapshot.model_validate_json(payload)
            if priority is not None:
                snap.priority = priority
            out.append(snap)
        return out

    def snapshot(self, source: str) -> SchemaSnapshot | None:
        """Return the snapshot stored for ``source``, or None."""
        row = self.con.execute(
            "SELECT payload FROM snapshots WHERE source = ?",
            [source],
        ).fetchone()
        return SchemaSnapshot.model_validate_json(row[0]) if row else None

    # ------------------------------------------------------------ user glossary / hints
    def upsert_term(self, term: BusinessTerm) -> None:
        """Insert or update a user glossary term, keyed by its stripped, lowercased name."""
        self.con.execute(
            _UPSERT_TERM_SQL,
            [
                term.name.strip().lower(),
                term.description,
                json.dumps(term.synonyms),
                json.dumps(term.targets),
                datetime.now(UTC),
            ],
        )

    def delete_term(self, name: str) -> None:
        """Delete a user glossary term by name (case-insensitive)."""
        self.con.execute("DELETE FROM glossary WHERE name = ?", [name.strip().lower()])

    def terms(self) -> list[BusinessTerm]:
        """The user glossary, by name, as terms sourced ``user``."""
        rows = self.con.execute(
            "SELECT name, description, synonyms, targets FROM glossary ORDER BY name"
        ).fetchall()
        return [
            BusinessTerm(
                name=name,
                description=description,
                synonyms=json.loads(synonyms),
                targets=json.loads(targets),
                source="user",
            )
            for name, description, synonyms, targets in rows
        ]

    def add_join_hint(self, edge: Edge) -> int:
        """Store a user join hint and return its id."""
        hint_id = self.con.execute("SELECT nextval('join_hints_seq')").fetchone()[0]
        self.con.execute(
            _INSERT_JOIN_HINT_SQL,
            [
                hint_id,
                edge.from_table,
                edge.to_table,
                json.dumps(edge.from_columns),
                json.dumps(edge.to_columns),
                edge.description,
                datetime.now(UTC),
            ],
        )
        return hint_id

    def delete_join_hint(self, hid: int) -> None:
        """Delete a join hint by id."""
        self.con.execute("DELETE FROM join_hints WHERE id = ?", [hid])

    def join_hints(self) -> list[tuple[int, Edge]]:
        """Every join hint, by id, as ``(id, join_hint edge sourced user)`` pairs."""
        rows = self.con.execute(_JOIN_HINTS_SQL).fetchall()
        return [
            (
                hint_id,
                Edge(
                    kind="join_hint",
                    from_table=from_table,
                    to_table=to_table,
                    from_columns=json.loads(from_columns),
                    to_columns=json.loads(to_columns),
                    description=description,
                    source="user",
                ),
            )
            for hint_id, from_table, to_table, from_columns, to_columns, description in rows
        ]

    def user_snapshot(self) -> SchemaSnapshot:
        """Glossary + join hints as a synthetic snapshot at priority 0, merged first.

        Human curation wins conflicting fields.
        """
        return SchemaSnapshot(
            source="user",
            source_type="user",
            priority=0,
            terms=self.terms(),
            edges=[edge for _, edge in self.join_hints()],
        ).stamp()
