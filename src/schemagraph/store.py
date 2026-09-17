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
    updated_at TIMESTAMP NOT NULL
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


def substitute_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: substitute_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_env(v) for v in obj]
    return obj


def redact(config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in config.items():
        if any(s in k.lower() for s in ("token", "secret", "password", "key")) and isinstance(v, str) and not v.startswith("${"):
            out[k] = "***"
        else:
            out[k] = v
    return out


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        for stmt in _SCHEMA.strip().split(";"):
            if stmt.strip():
                self.con.execute(stmt)
        cols = {r[0] for r in self.con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'connections'").fetchall()}
        if "priority" not in cols:  # added 2026-09-17: explicit merge order per connection
            self.con.execute("ALTER TABLE connections ADD COLUMN priority INTEGER")

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------ connections
    def upsert_connection(self, name: str, type_name: str, config: dict[str, Any], priority: int | None = None) -> None:
        """``priority``: merge order (lower merges first and wins conflicting fields); None = by source type."""
        now = datetime.now(UTC)
        self.con.execute(
            """
            INSERT INTO connections (name, type, config, created_at, updated_at, priority) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (name) DO UPDATE SET type = excluded.type, config = excluded.config, updated_at = excluded.updated_at,
              priority = excluded.priority
            """,
            [name, type_name, json.dumps(config), now, now, priority],
        )

    def delete_connection(self, name: str) -> None:
        self.con.execute("DELETE FROM connections WHERE name = ?", [name])
        self.con.execute("DELETE FROM snapshots WHERE source = ?", [name])

    def connections(self) -> list[dict[str, Any]]:
        rows = self.con.execute("SELECT name, type, config, created_at, updated_at, priority FROM connections ORDER BY name").fetchall()
        out = []
        for name, type_name, config, created, updated, priority in rows:
            snap = self.con.execute("SELECT n_tables, n_edges, n_terms, built_at, warnings FROM snapshots WHERE source = ?", [name]).fetchone()
            out.append(
                {
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
            )
        return out

    def connection(self, name: str) -> tuple[str, dict[str, Any]] | None:
        row = self.con.execute("SELECT type, config FROM connections WHERE name = ?", [name]).fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1])

    # ------------------------------------------------------------ snapshots
    def save_snapshot(self, snap: SchemaSnapshot) -> None:
        self.con.execute(
            """
            INSERT INTO snapshots (source, source_type, payload, n_tables, n_edges, n_terms, warnings, built_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source) DO UPDATE SET source_type = excluded.source_type, payload = excluded.payload,
              n_tables = excluded.n_tables, n_edges = excluded.n_edges, n_terms = excluded.n_terms,
              warnings = excluded.warnings, built_at = excluded.built_at
            """,
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
        """All stored snapshots, each carrying its connection's merge ``priority`` (order is decided by ``build_graph``)."""
        rows = self.con.execute("SELECT s.payload, c.priority FROM snapshots s LEFT JOIN connections c ON c.name = s.source ORDER BY s.source").fetchall()
        out = []
        for payload, priority in rows:
            snap = SchemaSnapshot.model_validate_json(payload)
            if priority is not None:
                snap.priority = priority
            out.append(snap)
        return out

    def snapshot(self, source: str) -> SchemaSnapshot | None:
        row = self.con.execute("SELECT payload FROM snapshots WHERE source = ?", [source]).fetchone()
        return SchemaSnapshot.model_validate_json(row[0]) if row else None

    # ------------------------------------------------------------ user glossary / hints
    def upsert_term(self, term: BusinessTerm) -> None:
        self.con.execute(
            """
            INSERT INTO glossary (name, description, synonyms, targets, updated_at) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (name) DO UPDATE SET description = excluded.description, synonyms = excluded.synonyms,
              targets = excluded.targets, updated_at = excluded.updated_at
            """,
            [term.name.strip().lower(), term.description, json.dumps(term.synonyms), json.dumps(term.targets), datetime.now(UTC)],
        )

    def delete_term(self, name: str) -> None:
        self.con.execute("DELETE FROM glossary WHERE name = ?", [name.strip().lower()])

    def terms(self) -> list[BusinessTerm]:
        rows = self.con.execute("SELECT name, description, synonyms, targets FROM glossary ORDER BY name").fetchall()
        return [BusinessTerm(name=n, description=d, synonyms=json.loads(s), targets=json.loads(t), source="user") for n, d, s, t in rows]

    def add_join_hint(self, edge: Edge) -> int:
        hid = self.con.execute("SELECT nextval('join_hints_seq')").fetchone()[0]
        self.con.execute(
            "INSERT INTO join_hints (id, from_table, to_table, from_columns, to_columns, description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [hid, edge.from_table, edge.to_table, json.dumps(edge.from_columns), json.dumps(edge.to_columns), edge.description, datetime.now(UTC)],
        )
        return hid

    def delete_join_hint(self, hid: int) -> None:
        self.con.execute("DELETE FROM join_hints WHERE id = ?", [hid])

    def join_hints(self) -> list[tuple[int, Edge]]:
        rows = self.con.execute("SELECT id, from_table, to_table, from_columns, to_columns, description FROM join_hints ORDER BY id").fetchall()
        return [
            (i, Edge(kind="join_hint", from_table=f, to_table=t, from_columns=json.loads(fc), to_columns=json.loads(tc), description=d, source="user"))
            for i, f, t, fc, tc, d in rows
        ]

    def user_snapshot(self) -> SchemaSnapshot:
        """Glossary + join hints as a synthetic snapshot merged last (human curation wins)."""
        return SchemaSnapshot(source="user", source_type="user", terms=self.terms(), edges=[e for _, e in self.join_hints()]).stamp()
