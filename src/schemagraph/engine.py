"""The Engine: one object the CLI, API and MCP server all drive.

Owns the Store, (re)builds the in-memory SchemaGraph from stored snapshots, holds
the Linker, and runs connectors.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from schemagraph import connectors  # noqa: F401  (registers connector types)
from schemagraph.connectors.base import config_schema, connector_types, make_connector
from schemagraph.graph.build import SchemaGraph, build_graph
from schemagraph.linking.linker import Linker, LinkOptions
from schemagraph.model import BusinessTerm, Edge, LinkResult, SchemaSnapshot, Table
from schemagraph.store import Store, substitute_env

log = logging.getLogger("schemagraph")

DEFAULT_HOME = Path(os.environ.get("SCHEMAGRAPH_HOME", ".schemagraph"))


class Engine:
    def __init__(self, home: str | Path | None = None, *, llm: Any | None = "auto"):
        self.home = Path(home) if home else DEFAULT_HOME
        self.store = Store(self.home / "schemagraph.duckdb")
        self._lock = threading.RLock()
        self.graph: SchemaGraph = SchemaGraph()
        self.linker: Linker | None = None
        self._llm = llm
        self.reload()

    # ----------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.store.close()

    def reload(self) -> None:
        with self._lock:
            snaps = self.store.snapshots()
            snaps.append(self.store.user_snapshot())
            self.graph = build_graph(snaps)
            llm = self._resolve_llm()
            self.linker = Linker(self.graph, llm=llm)

    def _resolve_llm(self):
        if self._llm != "auto":
            return self._llm
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return None
        try:
            from schemagraph.llm.anchors import ClaudeAnchorPicker

            return ClaudeAnchorPicker()
        except Exception as e:  # pragma: no cover
            log.warning("LLM anchor picker unavailable: %s", e)
            return None

    @property
    def has_llm(self) -> bool:
        return self.linker is not None and self.linker.llm is not None

    # ----------------------------------------------------------- connections
    def connector_types(self) -> list[str]:
        return connector_types()

    def connector_schema(self, type_name: str) -> dict[str, Any]:
        return config_schema(type_name)

    def add_connection(self, name: str, type_name: str, config: dict[str, Any], *, build: bool = True) -> SchemaSnapshot | None:
        make_connector(type_name, name, substitute_env(config))  # validate config shape
        self.store.upsert_connection(name, type_name, config)
        if build:
            return self.build(name)
        return None

    def remove_connection(self, name: str) -> None:
        with self._lock:
            self.store.delete_connection(name)
        self.reload()

    def check_connection(self, name: str) -> str:
        row = self.store.connection(name)
        if not row:
            raise KeyError(name)
        type_name, config = row
        return make_connector(type_name, name, substitute_env(config)).check()

    def build(self, name: str | None = None) -> SchemaSnapshot | list[SchemaSnapshot]:
        """Introspect one connection (or all) and persist the snapshot(s)."""
        names = [name] if name else [c["name"] for c in self.store.connections()]
        out: list[SchemaSnapshot] = []
        for n in names:
            row = self.store.connection(n)
            if not row:
                raise KeyError(n)
            type_name, config = row
            conn = make_connector(type_name, n, substitute_env(config))
            log.info("building %s (%s)", n, type_name)
            snap = conn.introspect()
            snap.source = n
            with self._lock:
                self.store.save_snapshot(snap)
            out.append(snap)
        self.reload()
        return out[0] if name else out

    def connections(self) -> list[dict[str, Any]]:
        return self.store.connections()

    # ----------------------------------------------------------- glossary / hints
    def upsert_term(self, term: BusinessTerm) -> None:
        self.store.upsert_term(term)
        self.reload()

    def delete_term(self, name: str) -> None:
        self.store.delete_term(name)
        self.reload()

    def add_join_hint(self, edge: Edge) -> int:
        hid = self.store.add_join_hint(edge)
        self.reload()
        return hid

    def delete_join_hint(self, hid: int) -> None:
        self.store.delete_join_hint(hid)
        self.reload()

    # ----------------------------------------------------------- queries
    def link(self, question: str, **kw: Any) -> LinkResult:
        assert self.linker is not None
        opts = LinkOptions(**kw)
        if opts.use_llm and not self.has_llm:
            opts.use_llm = False
        with self._lock:
            return self.linker.link(question, opts)

    def explain(self, question: str) -> dict[str, Any]:
        assert self.linker is not None
        return self.linker.explain(question)

    def tables(self) -> list[Table]:
        return sorted(self.graph.tables.values(), key=lambda t: t.fqn)

    def table(self, fqn: str) -> Table | None:
        return self.graph.find_table(fqn)

    def edges(self) -> list[Edge]:
        return self.graph.all_edges()

    def join_path(self, a: str, b: str) -> list[list[str]]:
        from schemagraph.graph.pathfinding import union_of_shortest_paths

        ta, tb = self.graph.find_table(a), self.graph.find_table(b)
        if not ta or not tb:
            raise KeyError(a if not ta else b)
        paths, _ = union_of_shortest_paths(self.graph, [ta.fqn], [tb.fqn])
        return [[self.graph.g.nodes[n]["fqn"] for n in p] for p in paths]

    def stats(self) -> dict[str, Any]:
        s = self.graph.stats()
        s["llm"] = self.has_llm
        s["home"] = str(self.home)
        return s
