"""Serve the MCP server over streamable HTTP.

:func:`run_http` blocks (``schemagraph mcp --transport http``). :func:`serve_http` runs the
server in a background thread on an ephemeral port for the duration of a ``with`` block, which
is how the agent layer and the benchmark give their agents an MCP URL without a separate
process; :func:`serve_http_async` is the same for callers on an event loop. :func:`serve_asgi`
does it for any ASGI app, e.g. the FastAPI app with its ``/mcp`` mount.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP

from schemagraph.mcp.server import DEFAULT_HOST

# Path of the streamable-HTTP endpoint, on its own server and on ``schemagraph serve``.
MCP_PATH = "/mcp"
# Port of ``schemagraph mcp --transport http`` (``schemagraph serve`` uses 8765).
DEFAULT_MCP_PORT = 8766
# Seconds to wait for a background server to start, and then to stop.
STARTUP_TIMEOUT = 30.0
SHUTDOWN_TIMEOUT = 10.0
# Poll interval while waiting for a background server to start, in seconds.
_POLL_SECONDS = 0.01


def run_http(server: FastMCP, *, host: str = DEFAULT_HOST, port: int = DEFAULT_MCP_PORT) -> None:
    """Serve ``server`` over streamable HTTP at ``http://host:port/mcp`` until interrupted."""
    uvicorn.run(server.streamable_http_app(), host=host, port=port)


@contextmanager
def serve_http(server: FastMCP, *, host: str = DEFAULT_HOST) -> Iterator[str]:
    """Serve ``server`` over streamable HTTP in a daemon thread for the ``with`` block.

    A server can be served once: its session manager does not restart. Build a new one per
    ``with`` block.

    Args:
        server: The MCP server, e.g. from :func:`~schemagraph.mcp.server.create_server`.
        host: Interface to bind; the port is ephemeral.

    Yields:
        The endpoint URL, ``http://host:port/mcp``.
    """
    with serve_asgi(server.streamable_http_app(), host=host) as base_url:
        yield base_url + MCP_PATH


@asynccontextmanager
async def serve_http_async(server: FastMCP, *, host: str = DEFAULT_HOST) -> AsyncIterator[str]:
    """:func:`serve_http` for an ``async with`` block: start and stop run in a worker thread.

    Starting waits for uvicorn and stopping joins its thread, which would otherwise stall the
    caller's event loop.

    Args:
        server: The MCP server, e.g. from :func:`~schemagraph.mcp.server.create_server`.
        host: Interface to bind; the port is ephemeral.

    Yields:
        The endpoint URL, ``http://host:port/mcp``.
    """
    served = serve_http(server, host=host)
    url = await asyncio.to_thread(served.__enter__)
    try:
        yield url
    finally:
        await asyncio.to_thread(served.__exit__, None, None, None)


@contextmanager
def serve_asgi(app: Any, *, host: str = DEFAULT_HOST) -> Iterator[str]:
    """Run an ASGI app under uvicorn in a daemon thread, on an ephemeral port.

    The app's lifespan runs (startup before the URL is yielded, shutdown on exit).

    Args:
        app: The ASGI app.
        host: Interface to bind.

    Yields:
        The base URL, ``http://host:port`` (no trailing slash).

    Raises:
        RuntimeError: The server did not start within :data:`STARTUP_TIMEOUT` seconds.
    """
    config = uvicorn.Config(app, host=host, port=0, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="schemagraph-mcp-http", daemon=True)
    thread.start()
    try:
        _wait_started(server, thread)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(SHUTDOWN_TIMEOUT)


def _wait_started(server: uvicorn.Server, thread: threading.Thread) -> None:
    """Block until ``server`` accepts connections.

    Raises:
        RuntimeError: The thread died (e.g. a lifespan error) or the timeout passed.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("MCP HTTP server exited during startup")
        if time.monotonic() > deadline:
            raise RuntimeError(f"MCP HTTP server did not start within {STARTUP_TIMEOUT}s")
        time.sleep(_POLL_SECONDS)
