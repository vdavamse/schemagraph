"""MCP server exposing the schema graph to any agent (Claude Code, Databao-style clients).

Tools are read-only and never execute SQL; the agent that calls these owns SQL generation
and execution against its own connection. (Read-only execution exists only in the optional
agent layer, ``schemagraph ask``.)

The server reads a :class:`~schemagraph.mcp.source.SchemaSource`: the Engine by default, a bare
Linker through :class:`~schemagraph.mcp.source.LinkerSource`, either scoped to one connection
by :class:`~schemagraph.mcp.source.ScopedSource`. It runs over stdio (:meth:`FastMCP.run`) or
streamable HTTP at ``/mcp`` (``schemagraph mcp --transport http``, the ``/mcp`` mount of
``schemagraph serve``, or :func:`schemagraph.mcp.http.serve_http`).
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from schemagraph.mcp.source import SchemaSource, ScopedSource
from schemagraph.model import Table

# Instructions the MCP client shows the agent.
_INSTRUCTIONS = (
    "Schema context for text-to-SQL. Call link_schema first; it returns the tables, columns and "
    "join paths needed for a question as annotated DDL."
)
# ``search_tables`` lists at most this many column names per table.
MAX_SEARCH_COLUMNS = 50
# FastMCP configures logging when none is set; INFO would log every request of an in-process server.
_LOG_LEVEL = "WARNING"
# Host the HTTP transport binds by default; loopback turns on DNS-rebinding protection.
DEFAULT_HOST = "127.0.0.1"
# The hosts FastMCP turns DNS-rebinding protection (Host/Origin checks) on for.
_PROTECTED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

log = logging.getLogger("schemagraph.mcp")

_Result = TypeVar("_Result")


async def _in_thread(function: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
    """Run a blocking source call in a worker thread.

    FastMCP runs tools on the event loop, which ``schemagraph serve`` shares with the API: a link
    waiting on the Engine lock (a reload, a build) or on Claude must not stall other requests.
    """
    return await anyio.to_thread.run_sync(functools.partial(function, *args, **kwargs))


def _matches_query(table: Table, lower_query: str) -> bool:
    """Whether a lowercased query is in the table's FQN, description or any column name."""
    return (
        lower_query in table.fqn.lower()
        or lower_query in (table.description or "").lower()
        or any(lower_query in column.name.lower() for column in table.columns)
    )


def _search_tables(source: SchemaSource, query: str, limit: int) -> list[dict[str, Any]]:
    """Up to ``limit`` tables matching ``query`` (case-insensitive substring), in FQN order."""
    lower_query = query.lower()
    out = []
    for table in source.tables():
        if _matches_query(table, lower_query):
            out.append(
                {
                    "fqn": table.fqn,
                    "kind": table.kind,
                    "description": table.description,
                    "columns": [c.name for c in table.columns][:MAX_SEARCH_COLUMNS],
                    "source": table.source,
                }
            )
        if len(out) >= limit:
            break
    return out


def _table_detail(source: SchemaSource, fqn: str) -> dict[str, Any]:
    """One table as JSON plus every relation touching it, or an ``error`` entry when unknown."""
    table = source.table(fqn)
    if not table:
        return {"error": f"unknown table {fqn}"}
    lower_fqn = table.fqn.lower()
    relations = [
        edge.model_dump()
        for edge in source.edges()
        if lower_fqn in (edge.from_table.lower(), edge.to_table.lower())
    ]
    return {
        **json.loads(table.model_dump_json(by_alias=True)),
        "fqn": table.fqn,
        "relations": relations,
    }


def _join_path_detail(source: SchemaSource, from_table: str, to_table: str) -> dict[str, Any]:
    """Join paths with every relation of each hop, or an ``error`` entry for an unknown table."""
    try:
        paths = source.join_path(from_table, to_table)
    except KeyError as error:
        return {"error": f"unknown table {error}"}
    detailed = []
    for path in paths:
        steps = []
        for a, b in zip(path, path[1:], strict=False):
            relations = source.graph.relations(a, b)
            steps.append([relation.model_dump() for relation in relations])
        detailed.append({"tables": path, "steps": steps})
    return {"paths": detailed}


def _glossary(source: SchemaSource) -> list[dict[str, Any]]:
    """Every business term of ``source`` as JSON."""
    return [term.model_dump() for term in source.terms()]


def create_server(
    source: SchemaSource | None = None,
    *,
    connection: str | None = None,
    host: str = DEFAULT_HOST,
) -> FastMCP:
    """Build the FastMCP server with read-only tools over ``source``.

    Tool names, parameters and docstrings are what agents see; they are part of the public
    surface.

    Args:
        source: What the tools read: an :class:`~schemagraph.engine.Engine` (a new one when
            None) or a :class:`~schemagraph.mcp.source.LinkerSource`.
        connection: Restrict every tool to this connection's tables
            (:class:`~schemagraph.mcp.source.ScopedSource`); None serves the whole graph.
        host: Host the streamable-HTTP transport is served on. On a loopback host FastMCP
            accepts only loopback ``Host``/``Origin`` headers (DNS-rebinding protection); on
            any other host it checks neither, and a warning is logged.

    Returns:
        The server; ``run()`` serves stdio, ``streamable_http_app()`` an ASGI app with the
        endpoint at ``/mcp``.
    """
    if source is None:
        from schemagraph.engine import Engine

        source = Engine()
    if connection is not None:
        source = ScopedSource(source, connection)
    if host not in _PROTECTED_HOSTS:
        log.warning(
            "MCP over HTTP on %s has no DNS-rebinding protection: any page a browser on this "
            "network opens can call its tools. Serve it on 127.0.0.1 unless it is firewalled.",
            host,
        )
    server = FastMCP("schemagraph", instructions=_INSTRUCTIONS, host=host, log_level=_LOG_LEVEL)
    _register_link_tools(server, source)
    _register_lookup_tools(server, source)
    return server


def _register_link_tools(server: FastMCP, source: SchemaSource) -> None:
    """Register ``link_schema`` and ``link_schema_json``."""

    @server.tool()
    async def link_schema(
        question: str,
        max_tables: int = 20,
        columns: str = "relevant",
        use_llm: bool = False,
        adaptive_budget: bool = True,
    ) -> str:
        """Return the sub-schema (annotated DDL with join paths and glossary) needed to answer a natural-language question.

        Args:
            question: The user's question in natural language.
            max_tables: Upper bound on tables returned.
            columns: "relevant" (pruned) or "all".
            use_llm: Let a Claude call choose anchor tables (needs ANTHROPIC_API_KEY).
            adaptive_budget: On schemas of more than 30 tables, raise max_tables to at least 20. Pass false to keep a small max_tables as is.
        """  # noqa: E501
        result = await _in_thread(
            source.link,
            question,
            max_tables=max_tables,
            columns=columns,
            use_llm=use_llm,
            adaptive_budget=adaptive_budget,
        )
        return result.ddl

    @server.tool()
    async def link_schema_json(
        question: str,
        max_tables: int = 20,
        adaptive_budget: bool = True,
        include_ddl: bool = False,
    ) -> dict[str, Any]:
        """Same as link_schema but structured: tables with scores, columns with reasons, join paths, matched glossary terms.

        Args:
            question: The user's question in natural language.
            max_tables: Upper bound on tables returned.
            adaptive_budget: On schemas of more than 30 tables, raise max_tables to at least 20. Pass false to keep a small max_tables as is.
            include_ddl: Also fill "ddl" with the annotated DDL link_schema returns (empty otherwise).
        """  # noqa: E501
        result = await _in_thread(
            source.link,
            question,
            max_tables=max_tables,
            adaptive_budget=adaptive_budget,
            render=include_ddl,
        )
        return result.model_dump()


def _register_lookup_tools(server: FastMCP, source: SchemaSource) -> None:
    """Register the table, join-path, glossary and stats lookups."""

    @server.tool()
    async def search_tables(query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find tables by name or description substring."""
        return await _in_thread(_search_tables, source, query, limit)

    @server.tool()
    async def get_table(fqn: str) -> dict[str, Any]:
        """Full detail for one table: columns (types, descriptions, samples), primary key, and every relation with provenance."""  # noqa: E501
        return await _in_thread(_table_detail, source, fqn)

    @server.tool()
    async def find_join_path(from_table: str, to_table: str) -> dict[str, Any]:
        """Shortest join path(s) between two tables over foreign keys, catalog relations, dbt relationship tests, join hints and inferred keys; when none exists, the dbt lineage route (provenance, no declared join keys)."""  # noqa: E501
        return await _in_thread(_join_path_detail, source, from_table, to_table)

    @server.tool()
    async def list_glossary() -> list[dict[str, Any]]:
        """Business terms with the tables/columns they map to."""
        return await _in_thread(_glossary, source)

    @server.tool()
    async def graph_stats() -> dict[str, Any]:
        """Counts of tables, columns, relations, terms and sources in the loaded graph."""
        return await _in_thread(source.stats)


def main() -> None:  # pragma: no cover
    """Run the server over stdio."""
    create_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
