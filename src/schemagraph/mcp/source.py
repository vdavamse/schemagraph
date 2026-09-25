"""What the MCP server reads: a schema graph, a linker over it and a few lookups.

:class:`SchemaSource` is the interface. :class:`~schemagraph.engine.Engine` satisfies it as is;
:class:`LinkerSource` adapts a bare :class:`~schemagraph.linking.linker.Linker` (the benchmark
builds one per database and has no Store); :class:`ScopedSource` restricts either to the tables
of one connection.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Protocol, runtime_checkable

from schemagraph.engine import find_join_paths
from schemagraph.graph.build import SchemaGraph
from schemagraph.linking.linker import Linker, LinkOptions
from schemagraph.linking.render import render_ddl
from schemagraph.model import BusinessTerm, Edge, JoinPath, LinkedTable, LinkResult, Table

# A scoped link asks the linker for this many times the budget, so the tables left after dropping
# other connections' tables still fill it.
SCOPED_OVERFETCH = 2


@runtime_checkable
class SchemaSource(Protocol):
    """A schema graph plus the lookups the MCP tools serve.

    Implementations read their graph on every call: an Engine reload swaps it, never mutates it.
    """

    @property
    def graph(self) -> SchemaGraph:
        """The merged schema graph (for relations and glossary terms)."""
        ...

    def link(self, question: str, **overrides: Any) -> LinkResult:
        """Link a question to a sub-schema; ``overrides`` are :class:`LinkOptions` fields."""
        ...

    def tables(self) -> list[Table]:
        """Every table, sorted by FQN."""
        ...

    def table(self, fqn: str) -> Table | None:
        """Look a table up by FQN or unambiguous suffix."""
        ...

    def edges(self) -> list[Edge]:
        """Every table-to-table relation."""
        ...

    def terms(self) -> list[BusinessTerm]:
        """Every glossary term."""
        ...

    def join_path(self, a: str, b: str) -> list[list[str]]:
        """Shortest join path(s) between two tables, as lists of FQNs; KeyError when unknown."""
        ...

    def stats(self) -> dict[str, Any]:
        """Graph counts and settings."""
        ...


class LinkerSource:
    """A :class:`SchemaSource` over a bare :class:`Linker` and the graph it indexes.

    Unlike the Engine it adds nothing to the options: ``link`` runs ``LinkOptions(**overrides)``,
    so embeddings are on only when the caller asks for them. Links run one at a time, as the
    Engine's do: the linker builds its PPR matrix, BM25 index and embedder lazily and unguarded,
    so concurrent first links would each build them.

    Attributes:
        linker: The linker; its ``schema_graph`` is the source's graph.
        dialect: SQL dialect of the database behind the graph, reported by :meth:`stats`.
    """

    def __init__(self, linker: Linker, *, dialect: str | None = None):
        self.linker = linker
        self.dialect = dialect
        self._lock = threading.Lock()

    @property
    def graph(self) -> SchemaGraph:
        """The graph the linker indexes."""
        return self.linker.schema_graph

    def link(self, question: str, **overrides: Any) -> LinkResult:
        """Link ``question`` with ``LinkOptions(**overrides)``."""
        with self._lock:
            return self.linker.link(question, LinkOptions(**overrides))

    def tables(self) -> list[Table]:
        """Every table in the graph, sorted by FQN."""
        return sorted(self.graph.tables.values(), key=lambda table: table.fqn)

    def table(self, fqn: str) -> Table | None:
        """Look a table up by FQN or unambiguous suffix."""
        return self.graph.find_table(fqn)

    def edges(self) -> list[Edge]:
        """Every relation edge in the graph."""
        return self.graph.all_edges()

    def terms(self) -> list[BusinessTerm]:
        """Every glossary term in the graph."""
        return list(self.graph.terms.values())

    def join_path(self, a: str, b: str) -> list[list[str]]:
        """Shortest join path(s), as :meth:`Engine.join_path` finds them.

        Raises:
            KeyError: Either table is unknown.
        """
        return find_join_paths(self.graph, a, b)

    def stats(self) -> dict[str, Any]:
        """Graph counts plus whether the linker has an LLM and the dialect."""
        stats: dict[str, Any] = dict(self.graph.stats())
        stats["llm"] = self.linker.llm is not None
        stats["dialect"] = self.dialect
        return stats


def _table_sources(table: Table) -> set[str]:
    """The connection names in a table's ``source`` (a merged table lists them comma-joined)."""
    return {name.strip() for name in table.source.split(",")}


def _target_table(schema_graph: SchemaGraph, target: str) -> Table | None:
    """The table a glossary target (``fqn``, ``fqn.column`` or a unique bare name) points at."""
    table = schema_graph.table(target)
    if table is None and "." in target:
        table = schema_graph.table(target.rsplit(".", 1)[0])
    return table or schema_graph.find_table(target)


class ScopedSource:
    """A :class:`SchemaSource` restricted to the tables of one connection.

    A table is in scope when ``connection`` is one of the names in its ``source``. Lookups and
    counts see only those tables, and glossary terms only their targets in those tables. When
    nothing is out of scope (a single-connection store), :meth:`link` runs unchanged; otherwise
    it over-fetches and filters (see :meth:`link`).

    Attributes:
        source: The unscoped source.
        connection: The connection (snapshot source name) to keep.
    """

    def __init__(self, source: SchemaSource, connection: str):
        self.source = source
        self.connection = connection
        # (a weak reference to the graph it was computed for, whether the scope drops any table
        # of it): not id(), which a new graph can reuse after a reload frees the old one, and
        # weak, so the cache does not keep a replaced graph alive
        self._filters_cache: tuple[weakref.ref[SchemaGraph], bool] | None = None

    @property
    def graph(self) -> SchemaGraph:
        """The unscoped graph; the scope applies to the lookups, not to relation data."""
        return self.source.graph

    def in_scope(self, fqn: str) -> bool:
        """Whether the table ``fqn`` (exact, case-insensitive) belongs to the connection."""
        table = self.graph.table(fqn)
        return table is not None and self.connection in _table_sources(table)

    def _filters(self) -> bool:
        """Whether the scope drops at least one table of the current graph (cached per graph)."""
        schema_graph = self.graph
        if self._filters_cache is None or self._filters_cache[0]() is not schema_graph:
            drops = not all(
                self.connection in _table_sources(table) for table in schema_graph.tables.values()
            )
            self._filters_cache = (weakref.ref(schema_graph), drops)
        return self._filters_cache[1]

    def link(self, question: str, **overrides: Any) -> LinkResult:
        """Link ``question`` and keep only this connection's tables.

        When the scope drops tables, the link over-fetches :data:`SCOPED_OVERFETCH` times the
        budget (after adaptive widening, see :meth:`_budget`) with ``bypass_if_fits`` off (a
        doubled budget would otherwise return whole unrelated schemas), drops other
        connections' tables, keeps every table on a surviving join path (bridges are never
        cut), fills the rest of the budget in rank order and re-renders the DDL.
        """
        if not self._filters():
            return self.source.link(question, **overrides)
        options = LinkOptions(**overrides)
        max_tables = self._budget(options)
        result = self.source.link(
            question,
            **{**overrides, "max_tables": max_tables * SCOPED_OVERFETCH, "bypass_if_fits": False},
        )
        result = self._restrict(result, max_tables)
        if options.render:
            result.ddl = render_ddl(self.graph, result)
        return result

    def _budget(self, options: LinkOptions) -> int:
        """The table budget an unscoped link of ``options`` has, adaptive widening included.

        The widening is decided on the whole graph, as the linker decides it.
        """
        if options.adaptive_budget and len(self.graph.tables) > options.large_threshold:
            return max(options.max_tables, options.max_tables_large)
        return options.max_tables

    def _restrict(self, result: LinkResult, max_tables: int) -> LinkResult:
        """Copy of ``result`` with this connection's tables, bridges first, up to ``max_tables``.

        Glossary matches follow the rule of :meth:`terms`: a matched term keeps only its targets
        in scope, and a term with none left is dropped from ``glossary`` and ``terms_matched``.
        """
        glossary = self._scoped_glossary(result.glossary)
        dropped_terms = set(result.glossary) - set(glossary)
        kept = [table for table in result.tables if self.in_scope(table.fqn)]
        paths = [
            path for path in result.join_paths if all(self.in_scope(fqn) for fqn in path.tables)
        ]
        tables = _fill_budget(kept, paths, max_tables)
        names = {table.fqn.lower() for table in tables}
        return result.model_copy(
            update={
                "tables": tables,
                "anchors": [anchor for anchor in result.anchors if anchor.lower() in names],
                "join_paths": [
                    path for path in paths if {fqn.lower() for fqn in path.tables} <= names
                ],
                "glossary": glossary,
                "terms_matched": [
                    term for term in result.terms_matched if term not in dropped_terms
                ],
            }
        )

    def _scoped_glossary(self, glossary: dict[str, list[str]]) -> dict[str, list[str]]:
        """Matched glossary phrases with only their in-scope targets; phrases left empty go."""
        scoped = {}
        for phrase, targets in glossary.items():
            kept = [target for target in targets if self._target_in_scope(target)]
            if kept:
                scoped[phrase] = kept
        return scoped

    def tables(self) -> list[Table]:
        """This connection's tables, sorted by FQN."""
        return [table for table in self.source.tables() if self.connection in _table_sources(table)]

    def table(self, fqn: str) -> Table | None:
        """Look a table up by FQN or by a suffix unambiguous among this connection's tables."""
        table = self.graph.find_table(fqn)
        if table is not None:
            return table if self.connection in _table_sources(table) else None
        lower_name = fqn.lower()
        matches = [
            table for table in self.tables() if table.fqn.lower().endswith("." + lower_name)
        ]
        return matches[0] if len(matches) == 1 else None

    def edges(self) -> list[Edge]:
        """Relations whose two tables both belong to this connection."""
        return [
            edge
            for edge in self.source.edges()
            if self.in_scope(edge.from_table) and self.in_scope(edge.to_table)
        ]

    def terms(self) -> list[BusinessTerm]:
        """Glossary terms with a target in this connection's tables, listing only those targets.

        Targets decide, not the term's ``source``: a user glossary term (source ``user``) is
        kept for the connections it points into, and a term merged from several connections
        shows each scope its own targets. A term with no target in scope is dropped.
        """
        scoped = []
        for term in self.source.terms():
            targets = [target for target in term.targets if self._target_in_scope(target)]
            if targets:
                scoped.append(term.model_copy(update={"targets": targets}))
        return scoped

    def _target_in_scope(self, target: str) -> bool:
        """Whether a glossary target points at a table of this connection."""
        table = _target_table(self.graph, target)
        return table is not None and self.connection in _table_sources(table)

    def join_path(self, a: str, b: str) -> list[list[str]]:
        """Join paths between two of this connection's tables that stay inside it.

        Raises:
            KeyError: Either table is unknown or belongs to another connection.
        """
        table_a, table_b = self.table(a), self.table(b)
        if table_a is None or table_b is None:
            raise KeyError(a if table_a is None else b)
        paths = self.source.join_path(table_a.fqn, table_b.fqn)
        return [path for path in paths if all(self.in_scope(fqn) for fqn in path)]

    def stats(self) -> dict[str, Any]:
        """The source's stats with table, column, relation and term counts for this connection."""
        tables = self.tables()
        edges = self.edges()
        pairs = {frozenset((edge.from_table.lower(), edge.to_table.lower())) for edge in edges}
        return {
            **self.source.stats(),
            "tables": len(tables),
            "columns": sum(len(table.columns) for table in tables),
            "relations": len(edges),
            "relation_pairs": len(pairs),
            "terms": len(self.terms()),
            "sources": 1,
            "connection": self.connection,
        }


def _fill_budget(
    ranked: list[LinkedTable],
    paths: list[JoinPath],
    max_tables: int,
) -> list[LinkedTable]:
    """Tables of ``ranked`` on any of ``paths``, then the best-ranked others up to ``max_tables``.

    The result keeps rank order. Tables on a join path are all kept, even past the budget.
    """
    on_path = {fqn.lower() for path in paths for fqn in path.tables}
    chosen = {table.fqn for table in ranked if table.fqn.lower() in on_path}
    for table in ranked:
        if len(chosen) >= max_tables:
            break
        chosen.add(table.fqn)
    return [table for table in ranked if table.fqn in chosen]
