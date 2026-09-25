"""The orchestrator's view of the schema: a few memoised lookups over one schemagraph MCP server.

The generator reads the schema through the same server's tools (``MCPToolset``); the orchestrator
and the checks read it through :class:`SchemaClient`, so an answer never touches the graph
directly. Linking is deterministic, so every lookup is memoised for the client's lifetime.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Any, TypeVar

from fastmcp import Client

from schemagraph.graph.build import JOIN_KINDS
from schemagraph.model import Edge

# The rendered DDL's header ends with this line; the header repeats the link text.
_HEADER_END = "-- Linked tables:"

_Value = TypeVar("_Value")


def ddl_body(ddl: str) -> str:
    """Return rendered DDL without its header.

    The ``-- Question:`` line repeats the link text, which spans several lines when external
    knowledge is appended, so the cut goes through the ``-- Linked tables:`` line instead.
    """
    start = ddl.find(_HEADER_END)
    if start < 0:
        return ddl.strip()
    end = ddl.find("\n", start)
    return "" if end < 0 else ddl[end + 1 :].strip()


@dataclass(frozen=True)
class LinkedSchema:
    """One linked sub-schema, as the generator and the critic see it.

    Attributes:
        ddl: The annotated DDL without its header.
        tables: FQNs of the linked tables, in rank order.
    """

    ddl: str
    tables: tuple[str, ...]


def _is_join(edge: Edge) -> bool:
    return edge.kind in JOIN_KINDS


def _joins_every_hop(path: dict[str, Any]) -> bool:
    """Whether every hop of a ``find_join_path`` path has a join-capable relation."""
    return all(any(relation["kind"] in JOIN_KINDS for relation in step) for step in path["steps"])


def _forget_failure(
    cache: dict[Any, asyncio.Future[Any]],
    key: Hashable,
    task: asyncio.Future[Any],
) -> None:
    """Drop a failed or cancelled memo task (reading its exception, so asyncio logs nothing)."""
    failed = task.cancelled() or task.exception() is not None
    if failed and cache.get(key) is task:
        del cache[key]


class SchemaClient:
    """Memoised schema lookups over one MCP session to a schemagraph server.

    The memo holds each lookup's task, so concurrent requests for one key share a single tool
    call; a failed call is forgotten and retried by the next request.

    Use it as an async context manager to hold one session open across many lookups; outside
    one, each lookup opens and closes its own session.

    Attributes:
        mcp_url: The server's streamable-HTTP endpoint, e.g. ``http://127.0.0.1:8765/mcp``.
    """

    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
        self._client = Client(mcp_url)
        self._links: dict[tuple[str, int, bool], asyncio.Future[LinkedSchema]] = {}
        self._tables: dict[str, asyncio.Future[dict[str, Any] | None]] = {}
        self._paths: dict[tuple[str, str], asyncio.Future[list[list[str]]]] = {}

    async def __aenter__(self) -> SchemaClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.__aexit__(*exc_info)

    @staticmethod
    async def _memo(
        cache: dict[Any, asyncio.Future[_Value]],
        key: Hashable,
        fetch: Callable[[], Awaitable[_Value]],
    ) -> _Value:
        """Return ``fetch()``'s result for ``key``, sharing one task among concurrent callers.

        Callers wait with :func:`asyncio.wait`, which never cancels what it waits on, so a
        caller cancelled by its node timeout does not cancel the task for the others (unlike
        ``asyncio.shield``, it logs nothing when the task later fails). A task that fails or is
        cancelled forgets itself when it finishes, whether or not anyone still awaits it, so
        the next call retries.
        """
        task = cache.get(key)
        if task is None:
            task = cache[key] = asyncio.ensure_future(fetch())
            task.add_done_callback(functools.partial(_forget_failure, cache, key))
        if not task.done():  # a done task may belong to an earlier event loop (a reused client)
            await asyncio.wait((task,))
        return task.result()

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Call one tool and return its structured result (a list comes wrapped in ``result``)."""
        async with self._client:
            result = await self._client.call_tool(tool, arguments)
        return result.structured_content

    async def link(self, text: str, *, max_tables: int, adaptive_budget: bool) -> LinkedSchema:
        """Link ``text`` to a sub-schema of at most ``max_tables`` tables.

        Args:
            text: The question, with any external knowledge appended.
            max_tables: The table budget.
            adaptive_budget: Let the linker widen a small budget on large schemas.

        Returns:
            The DDL (header stripped) and the linked tables.
        """

        async def fetch() -> LinkedSchema:
            result = await self._call(
                "link_schema_json",
                {
                    "question": text,
                    "max_tables": max_tables,
                    "adaptive_budget": adaptive_budget,
                    "include_ddl": True,
                },
            )
            tables = tuple(table["fqn"] for table in result["tables"])
            return LinkedSchema(ddl=ddl_body(result["ddl"]), tables=tables)

        return await self._memo(self._links, (text, max_tables, adaptive_budget), fetch)

    async def table(self, fqn: str) -> dict[str, Any] | None:
        """Return ``get_table``'s detail (columns, relations) for a table, or None when unknown.

        A found table is also memoised under its returned FQN, so ``orders`` and
        ``public.orders`` share one call once either has been fetched.
        """

        async def fetch() -> dict[str, Any] | None:
            detail = await self._call("get_table", {"fqn": fqn})
            if "error" in detail:
                return None
            self._tables.setdefault(detail["fqn"], self._tables[fqn])
            return detail

        return await self._memo(self._tables, fqn, fetch)

    async def resolve(self, name: str) -> str | None:
        """Return the FQN a raw table name (quoted, bare or qualified) resolves to, or None."""
        detail = await self.table(name.strip().strip('`"'))
        return detail["fqn"] if detail else None

    async def _join_edges(self, fqn: str) -> list[Edge]:
        """Join-capable relations touching a table (none when it is unknown)."""
        detail = await self.table(fqn)
        if detail is None:
            return []
        edges = [Edge.model_validate(relation) for relation in detail["relations"]]
        return [edge for edge in edges if _is_join(edge)]

    async def join_relations(self, a: str, b: str) -> list[Edge]:
        """Return the join-capable relations between two tables, in either direction."""
        fqn_a, fqn_b = await self.resolve(a), await self.resolve(b)
        if not fqn_a or not fqn_b:
            return []
        pair = {fqn_a.lower(), fqn_b.lower()}
        return [
            edge
            for edge in await self._join_edges(fqn_a)
            if {edge.from_table.lower(), edge.to_table.lower()} == pair
        ]

    async def neighbours(self, fqn: str) -> list[str]:
        """Return the tables one join-capable relation away from ``fqn``, sorted."""
        own = await self.resolve(fqn)
        if own is None:
            return []
        others = set()
        for edge in await self._join_edges(own):
            for end in (edge.from_table, edge.to_table):
                if end.lower() != own.lower():
                    others.add(end)
        return sorted(others)

    async def join_path(self, a: str, b: str) -> list[list[str]]:
        """Return the shortest join paths from ``a`` to ``b`` as FQN lists; empty when unknown.

        Only paths whose every hop is a join-capable relation count; the server's lineage
        fallback is not a join.
        """

        async def fetch() -> list[list[str]]:
            result = await self._call("find_join_path", {"from_table": a, "to_table": b})
            paths = result.get("paths", [])
            return [path["tables"] for path in paths if _joins_every_hop(path)]

        return await self._memo(self._paths, (a, b), fetch)

