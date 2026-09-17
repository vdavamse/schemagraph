from fastapi.testclient import TestClient

from schemagraph.api.app import create_app
from schemagraph.engine import Engine
from schemagraph.mcp.server import create_server
from tests.conftest import STORE_DDL


def test_engine_persist_and_reload(tmp_path):
    eng = Engine(tmp_path, llm=None)
    snap = eng.add_connection("store", "ddl", {"ddl": STORE_DDL, "dialect": "postgres", "default_schema": "public"})
    assert len(snap.tables) == 7
    eng.upsert_term(__import__("schemagraph.model", fromlist=["BusinessTerm"]).BusinessTerm(name="revenue", targets=["public.orders.total_amount"]))
    eng.close()
    eng2 = Engine(tmp_path, llm=None)
    assert eng2.stats()["tables"] == 7 and "revenue" in eng2.graph.terms
    r = eng2.link("revenue by customer")
    assert any(t.fqn == "public.orders" for t in r.tables)
    assert eng2.connections()[0]["built"] is True
    eng2.close()


def test_env_substitution_in_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_DDL", "CREATE TABLE t (id INT PRIMARY KEY);")
    eng = Engine(tmp_path, llm=None)
    snap = eng.add_connection("env", "ddl", {"ddl": "${MY_DDL}"})
    assert snap.tables[0].name == "t"
    assert eng.connections()[0]["config"]["ddl"] == "${MY_DDL}"  # stored unexpanded
    eng.close()


def test_api_roundtrip(tmp_path):
    app = create_app(Engine(tmp_path, llm=None), web_dist=tmp_path / "nope")
    c = TestClient(app)
    r = c.post("/api/ddl", json={"name": "store", "ddl": STORE_DDL, "dialect": "postgres", "default_schema": "public"})
    assert r.status_code == 200 and r.json()["tables"] == 7
    assert any(ct["type"] == "collibra" for ct in c.get("/api/connector-types").json())
    assert c.get("/api/graph/tables").json()[0]["fqn"]
    r = c.post("/api/link", json={"question": "orders shipped by carrier for customers in texas"})
    body = r.json()
    assert r.status_code == 200 and body["tables"] and "CREATE TABLE" in body["ddl"]
    r = c.get("/api/graph/path", params={"a": "customer", "b": "shipment"})
    assert r.json()["paths"][0] == ["public.customer", "public.orders", "public.order_items", "public.shipment"]
    r = c.post("/api/join-hints", json={"from_table": "public.orders", "to_table": "public.audit_log", "from_columns": ["id"], "to_columns": ["id"]})
    assert r.status_code == 200
    assert any(e["kind"] == "join_hint" for e in c.get("/api/graph/edges").json())
    r = c.post("/api/glossary", json={"name": "GMV", "targets": ["public.order_items.line_total"], "synonyms": ["gross merchandise value"]})
    assert r.status_code == 200 and any(t["name"] == "gmv" for t in c.get("/api/glossary").json())
    assert c.delete("/api/connections/store").status_code == 200
    assert c.get("/api/graph/stats").json()["tables"] == 2  # only the two stubs referenced by the join hint remain
    assert c.post("/api/connections", json={"name": "bad", "type": "nope", "config": {}}).status_code == 400


def test_mcp_tools_registered(tmp_path):
    eng = Engine(tmp_path, llm=None)
    eng.add_connection("store", "ddl", {"ddl": STORE_DDL, "dialect": "postgres", "default_schema": "public"})
    server = create_server(eng)
    import anyio

    tools = anyio.run(server.list_tools)
    names = {t.name for t in tools}
    assert {"link_schema", "link_schema_json", "get_table", "find_join_path", "search_tables"} <= names
    out = anyio.run(server.call_tool, "find_join_path", {"from_table": "customer", "to_table": "shipment"})
    assert "public.order_items" in str(out)


def test_engine_embed_auto_and_explicit(tmp_path):
    from schemagraph.engine import Engine

    eng = Engine(tmp_path / "h", llm=None, embed=False)
    assert eng.has_embed is False and eng.stats()["embed"] is False
    eng2 = Engine(tmp_path / "h2", llm=None, embed="auto")
    try:
        import model2vec  # noqa: F401

        assert eng2.has_embed is True
    except ImportError:
        assert eng2.has_embed is False
