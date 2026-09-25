"""MCP server: read-only schema-context tools for agents, over stdio or streamable HTTP."""

from schemagraph.mcp.http import serve_http, serve_http_async
from schemagraph.mcp.server import create_server
from schemagraph.mcp.source import LinkerSource, SchemaSource, ScopedSource

__all__ = [
    "LinkerSource",
    "SchemaSource",
    "ScopedSource",
    "create_server",
    "serve_http",
    "serve_http_async",
]
