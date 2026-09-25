"""MCP server: schema sources, connection scoping, link parameters and the HTTP transport."""

from __future__ import annotations

import asyncio
import time

import anyio
import httpx
import pytest
from typer.testing import CliRunner

from schemagraph.api.app import create_app
from schemagraph.cli import app as cli_app
from schemagraph.engine import Engine
from schemagraph.graph.build import build_graph
from schemagraph.linking.linker import Linker
from schemagraph.mcp import LinkerSource, SchemaSource, ScopedSource, create_server, serve_http
from schemagraph.mcp.http import serve_asgi
from schemagraph.model import BusinessTerm
from tests.conftest import STORE_DDL

QUESTION = "revenue by product category for customers in california"


def _engine(tmp_path, *connections: tuple[str, str, str]) -> Engine:
    """An Engine with one ``ddl`` connection per ``(name, ddl, schema)``."""
    engine = Engine(tmp_path / "home", llm=None, embed=False)
    for name, ddl, schema in connections:
        engine.add_connection(name, "ddl", {"ddl": ddl, "dialect": "postgres", "default_schema": schema})
    return engine


@pytest.fixture
def store_engine(tmp_path):
    engine = _engine(tmp_path, ("store", STORE_DDL, "public"))
    yield engine
    engine.close()


@pytest.fixture
def two_store_engine(tmp_path):
    """The store plus a second connection whose ``customer`` table is called ``client``."""
    engine = _engine(
        tmp_path,
        ("store", STORE_DDL, "public"),
        ("other", STORE_DDL.replace("customer", "client"), "crm"),
    )
    yield engine
    engine.close()


def _call(server, tool: str, arguments: dict):
    """Call a tool in process and return its structured result (``result`` unwrapped)."""
    _, structured = anyio.run(server.call_tool, tool, arguments)
    return structured["result"] if set(structured) == {"result"} else structured


def _same_link(a, b) -> bool:
    """Two link results agree on everything but the timing in ``stats``."""
    return a.model_dump(exclude={"stats"}) == b.model_dump(exclude={"stats"})


def _wide_ddl(n_tables: int) -> str:
    """A schema of ``n_tables`` small tables, all mentioning orders (above ``large_threshold``)."""
    return "\n".join(
        f"CREATE TABLE orders_{i} (id INT PRIMARY KEY, order_total NUMERIC);" for i in range(n_tables)
    )


# ------------------------------------------------------------------ sources


def test_engine_and_linker_source_satisfy_the_protocol(store_engine, store_snapshot):
    linker_source = LinkerSource(Linker(build_graph([store_snapshot])), dialect="postgres")
    assert isinstance(store_engine, SchemaSource)
    assert isinstance(linker_source, SchemaSource)
    assert isinstance(ScopedSource(store_engine, "store"), SchemaSource)


def test_linker_source_answers_like_the_engine(store_engine):
    snaps = store_engine.store.snapshots() + [store_engine.store.user_snapshot()]
    source = LinkerSource(Linker(build_graph(snaps)), dialect="postgres")
    assert [t.fqn for t in source.tables()] == [t.fqn for t in store_engine.tables()]
    assert source.table("customer").fqn == "public.customer"
    assert len(source.edges()) == len(store_engine.edges())
    assert source.join_path("customer", "shipment") == store_engine.join_path("customer", "shipment")
    with pytest.raises(KeyError):
        source.join_path("customer", "nope")
    direct = store_engine.link(QUESTION, max_tables=4)
    assert _same_link(source.link(QUESTION, max_tables=4), direct)
    stats = source.stats()
    assert stats["tables"] == 7 and stats["dialect"] == "postgres" and stats["llm"] is False


def test_concurrent_first_links_build_the_ppr_matrix_once(store_snapshot, monkeypatch):
    from schemagraph.graph import ppr

    builds: list[int] = []
    build = ppr.PPRMatrix.__init__

    def slow_build(self, *args, **kwargs):
        builds.append(1)
        time.sleep(0.1)  # widen the window in which a concurrent link would build it again
        build(self, *args, **kwargs)

    monkeypatch.setattr(ppr.PPRMatrix, "__init__", slow_build)
    server = create_server(LinkerSource(Linker(build_graph([store_snapshot]))))

    async def links() -> None:
        async with anyio.create_task_group() as group:
            for index in range(4):
                group.start_soon(server.call_tool, "link_schema", {"question": f"revenue {index}"})

    anyio.run(links)
    assert len(builds) == 1


# ------------------------------------------------------------------ scoping


def test_scoped_link_is_unchanged_on_a_single_connection(store_engine):
    scoped = ScopedSource(store_engine, "store")
    for max_tables in (3, 5):
        direct = store_engine.link(QUESTION, max_tables=max_tables, adaptive_budget=False)
        result = scoped.link(QUESTION, max_tables=max_tables, adaptive_budget=False)
        assert _same_link(result, direct)


def test_scoped_link_keeps_bridges_and_drops_other_connections(two_store_engine):
    result = ScopedSource(two_store_engine, "store").link(QUESTION, max_tables=3, adaptive_budget=False)
    names = {t.fqn for t in result.tables}
    assert names and all(name.startswith("public.") for name in names)
    for path in result.join_paths:
        assert set(path.tables) <= names  # no join path loses its bridge tables
    assert all(anchor in names for anchor in result.anchors)
    assert "crm." not in result.ddl and "public." in result.ddl  # re-rendered for the scope
    unrendered = ScopedSource(two_store_engine, "store").link(QUESTION, max_tables=3, render=False)
    assert unrendered.ddl == ""


def test_scoped_link_widens_the_budget_like_an_unscoped_one(tmp_path):
    engine = _engine(tmp_path, ("wide", _wide_ddl(40), "public"), ("store", STORE_DDL, "shop"))
    scoped = ScopedSource(engine, "wide")
    widened = scoped.link("order total", max_tables=5)
    assert len(widened.tables) == 20 == len(engine.link("order total", max_tables=5).tables)
    assert all(table.fqn.startswith("public.") for table in widened.tables)
    assert len(scoped.link("order total", max_tables=5, adaptive_budget=False).tables) == 5
    engine.close()


def test_scope_cache_follows_a_reloaded_graph(tmp_path):
    engine = _engine(tmp_path, ("store", STORE_DDL, "public"))
    scoped = ScopedSource(engine, "store")
    assert scoped._filters() is False
    engine.add_connection(
        "other", "ddl", {"ddl": STORE_DDL, "dialect": "postgres", "default_schema": "crm"}
    )
    assert scoped._filters() is True  # the reload swapped the graph
    assert scoped._filters_cache[0]() is engine.graph
    engine.close()


def test_scoped_lookups_see_one_connection(two_store_engine):
    scoped = ScopedSource(two_store_engine, "other")
    assert {t.fqn.split(".")[0] for t in scoped.tables()} == {"crm"}
    assert scoped.table("orders").fqn == "crm.orders"  # ambiguous in the graph, unique in scope
    assert two_store_engine.table("orders") is None
    assert scoped.table("public.orders") is None
    assert all(e.from_table.startswith("crm.") and e.to_table.startswith("crm.") for e in scoped.edges())
    assert scoped.join_path("client", "shipment")[0][0] == "crm.client"
    with pytest.raises(KeyError):
        scoped.join_path("public.customer", "shipment")
    stats = scoped.stats()
    assert stats["tables"] == 7 and stats["connection"] == "other" and stats["sources"] == 1
    assert stats["relations"] == len(scoped.edges())


def test_scoped_terms_keep_only_targets_in_scope(two_store_engine):
    two_store_engine.upsert_term(
        BusinessTerm(name="revenue", targets=["public.orders.total_amount", "crm.orders.total_amount"])
    )
    two_store_engine.upsert_term(BusinessTerm(name="client", targets=["crm.client"]))
    scoped = ScopedSource(two_store_engine, "store")
    assert [(term.name, term.targets) for term in scoped.terms()] == [
        ("revenue", ["public.orders.total_amount"])
    ]
    assert scoped.stats()["terms"] == 1
    assert {term.name for term in ScopedSource(two_store_engine, "other").terms()} == {
        "revenue",
        "client",
    }


def test_create_server_scopes_every_tool(two_store_engine):
    two_store_engine.upsert_term(BusinessTerm(name="client", targets=["crm.client"]))
    two_store_engine.upsert_term(BusinessTerm(name="buyer", targets=["public.customer"]))
    two_store_engine.upsert_term(
        BusinessTerm(name="big spender", targets=["crm.client", "public.customer"])
    )
    server = create_server(two_store_engine, connection="store")
    tools = {tool.name for tool in anyio.run(server.list_tools)}
    checked = {
        "search_tables",
        "get_table",
        "find_join_path",
        "list_glossary",
        "graph_stats",
        "link_schema",
        "link_schema_json",
    }
    assert tools == checked  # a new tool must be added to this test
    found = _call(server, "search_tables", {"query": "orders"})
    assert found and all(row["fqn"].startswith("public.") for row in found)
    assert "error" in _call(server, "get_table", {"fqn": "crm.client"})
    detail = _call(server, "get_table", {"fqn": "orders"})
    assert detail["fqn"] == "public.orders"
    assert all("crm." not in str(relation) for relation in detail["relations"])
    assert "error" in _call(server, "find_join_path", {"from_table": "crm.client", "to_table": "shipment"})
    paths = _call(server, "find_join_path", {"from_table": "customer", "to_table": "shipment"})["paths"]
    assert paths and all(fqn.startswith("public.") for path in paths for fqn in path["tables"])
    glossary = _call(server, "list_glossary", {})
    assert [(term["name"], term["targets"]) for term in glossary] == [
        ("big spender", ["public.customer"]),
        ("buyer", ["public.customer"]),
    ]
    stats = _call(server, "graph_stats", {})
    assert stats["connection"] == "store" and stats["tables"] == 7 and stats["terms"] == 2
    assert "crm." not in _call(server, "link_schema", {"question": QUESTION, "max_tables": 3})
    linked = _call(server, "link_schema_json", {"question": QUESTION, "max_tables": 3, "include_ddl": True})
    assert linked["tables"] and all(table["fqn"].startswith("public.") for table in linked["tables"])
    assert "crm." not in linked["ddl"]
    spender = _call(
        server,
        "link_schema_json",
        {"question": "big spender or client revenue", "max_tables": 3, "include_ddl": True},
    )
    assert spender["glossary"] == {"big spender": ["public.customer"]}
    assert "client" not in spender["terms_matched"]  # a term only other connections have
    assert "crm." not in spender["ddl"] and "big spender = public.customer" in spender["ddl"]


# ------------------------------------------------------------------ link parameters


def test_link_schema_adaptive_budget_false_keeps_a_small_budget(tmp_path):
    engine = _engine(tmp_path, ("wide", _wide_ddl(40), "public"))
    server = create_server(engine)
    widened = _call(server, "link_schema_json", {"question": "order total", "max_tables": 3})
    tight = _call(
        server,
        "link_schema_json",
        {"question": "order total", "max_tables": 3, "adaptive_budget": False},
    )
    assert len(tight["tables"]) < len(widened["tables"]) == 20
    ddl = _call(server, "link_schema", {"question": "order total", "max_tables": 3, "adaptive_budget": False})
    assert ddl.count("CREATE TABLE") == len(tight["tables"])
    engine.close()


def test_link_schema_json_include_ddl(store_engine):
    server = create_server(store_engine)
    assert _call(server, "link_schema_json", {"question": QUESTION})["ddl"] == ""
    with_ddl = _call(server, "link_schema_json", {"question": QUESTION, "include_ddl": True})
    assert with_ddl["ddl"] == _call(server, "link_schema", {"question": QUESTION})
    assert "CREATE TABLE" in with_ddl["ddl"]


def test_server_over_a_linker_source(store_snapshot):
    server = create_server(LinkerSource(Linker(build_graph([store_snapshot]))))
    detail = _call(server, "find_join_path", {"from_table": "customer", "to_table": "shipment"})
    assert "public.order_items" in detail["paths"][0]["tables"]


# ------------------------------------------------------------------ HTTP transport


async def _list_and_link(url: str) -> tuple[set[str], str]:
    """Over a real MCP HTTP session: the tool names and ``link_schema``'s DDL."""
    fastmcp = pytest.importorskip("fastmcp")
    async with fastmcp.Client(url) as client:
        tools = {tool.name for tool in await client.list_tools()}
        result = await client.call_tool("link_schema", {"question": QUESTION, "max_tables": 4})
    return tools, result.content[0].text


def test_serve_http_round_trip(store_engine):
    pytest.importorskip("fastmcp")
    with serve_http(create_server(store_engine)) as url:
        assert url.startswith("http://127.0.0.1:") and url.endswith("/mcp")
        tools, ddl = asyncio.run(_list_and_link(url))
    assert {"link_schema", "link_schema_json", "get_table", "find_join_path"} <= tools
    assert ddl == store_engine.link(QUESTION, max_tables=4).ddl


def test_fastapi_app_serves_mcp_at_slash_mcp(store_engine, tmp_path):
    pytest.importorskip("fastmcp")
    app = create_app(store_engine, web_dist=tmp_path / "no_dist")
    assert not any(path.startswith("/mcp") for path in app.openapi()["paths"])
    with serve_asgi(app) as base_url:
        tools, ddl = asyncio.run(_list_and_link(base_url + "/mcp"))
    assert "link_schema" in tools and "CREATE TABLE" in ddl


# Seconds ``/api/health`` may take while a ``/mcp`` link waits on the Engine lock.
HEALTH_WHILE_LOCKED_SECONDS = 1.0


async def _health_while_a_link_waits(base_url: str, engine: Engine) -> float:
    """With the Engine lock held, start a ``/mcp`` link and time ``/api/health`` while it waits."""
    fastmcp = pytest.importorskip("fastmcp")
    async with fastmcp.Client(base_url + "/mcp") as client:
        engine._lock.acquire()  # a reload or build in progress
        try:
            link = asyncio.create_task(client.call_tool("link_schema", {"question": QUESTION}))
            await asyncio.sleep(0.3)  # the link is now blocked on the lock
            assert not link.done()
            async with httpx.AsyncClient() as http:
                start = time.monotonic()
                response = await http.get(base_url + "/api/health")
                elapsed = time.monotonic() - start
        finally:
            engine._lock.release()
        assert response.status_code == 200
        result = await link
    assert "CREATE TABLE" in result.content[0].text
    return elapsed


def test_a_waiting_mcp_link_does_not_block_the_api(store_engine, tmp_path):
    pytest.importorskip("fastmcp")
    app = create_app(store_engine, web_dist=tmp_path / "no_dist")
    with serve_asgi(app) as base_url:
        elapsed = asyncio.run(_health_while_a_link_waits(base_url, store_engine))
    assert elapsed < HEALTH_WHILE_LOCKED_SECONDS


def test_loopback_mcp_rejects_a_foreign_host_header(store_engine, tmp_path):
    app = create_app(store_engine, web_dist=tmp_path / "no_dist")
    initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    headers = {"Host": "evil.example", "Accept": "application/json, text/event-stream"}
    with serve_asgi(app) as base_url:
        response = httpx.post(base_url + "/mcp", json=initialize, headers=headers)
    assert response.status_code == 421


def test_a_non_loopback_host_logs_a_warning(store_engine, caplog):
    with caplog.at_level("WARNING", logger="schemagraph.mcp"):
        create_server(store_engine)
    assert not caplog.records
    with caplog.at_level("WARNING", logger="schemagraph.mcp"):
        create_server(store_engine, host="0.0.0.0")
    assert "no DNS-rebinding protection" in caplog.text


@pytest.mark.parametrize(
    ("transport", "expected"),
    [("http", "http"), ("streamable-http", "http"), ("sse", "sse"), ("stdio", "stdio")],
)
def test_mcp_cli_transports(tmp_path, monkeypatch, transport, expected):
    from mcp.server.fastmcp import FastMCP

    from schemagraph.mcp import http

    served: list[tuple] = []
    monkeypatch.setattr(http, "run_http", lambda server, host, port: served.append(("http", port)))

    def run(server, transport="stdio"):
        served.append((transport, server.settings.port))

    monkeypatch.setattr(FastMCP, "run", run)
    arguments = ["mcp", "--home", str(tmp_path), "--transport", transport, "--port", "8799"]
    result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 0, result.output
    assert served[0][0] == expected
    if expected != "stdio":
        assert served[0][1] == 8799


def test_mcp_cli_rejects_an_unknown_transport(tmp_path):
    result = CliRunner().invoke(cli_app, ["mcp", "--home", str(tmp_path), "--transport", "websocket"])
    assert result.exit_code == 2 and "'websocket' is not one of" in result.output
