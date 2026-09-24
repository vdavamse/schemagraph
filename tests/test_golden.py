"""Golden regression oracle for the readability refactor (issue #8).

Pins, byte for byte, what the unmodified code (origin/main @ 1db6bfe) produces on offline fixtures:
every connector's snapshot, the merged graph (node/edge insertion order and attributes), the lexical
index, ``LinkResult`` + DDL for a question set under a dozen ``LinkOptions`` variants, ``explain()``,
and the frozen public surfaces (OpenAPI, MCP tools, connector config schemas, CLI, LinkOptions and
bench Row fields). Floats are compared through ``json.dumps`` (``repr``, exact round trip), so a
reordered floating-point sum shows up.

Regenerate only on purpose: ``SCHEMAGRAPH_UPDATE_GOLDEN=1 uv run pytest -q tests/test_golden.py``.
The golden files were generated before the refactor; a refactoring commit must never regenerate them
(except ``surfaces_*`` description-only changes, which the test already ignores).

Callers pass renamed parameters positionally (``schema_graph`` -> ``schema_graph``) so this file survives the rename.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path

import duckdb
import httpx
import pytest

from schemagraph.connectors.collibra import CollibraClient, CollibraConfig, introspect_collibra
from schemagraph.connectors.dbt import DbtConfig, parse_manifest, parse_project
from schemagraph.connectors.ddl import DDLConfig, parse_ddl
from schemagraph.connectors.duckdb_conn import DuckDBConfig, introspect_duckdb
from schemagraph.connectors.glue import GlueConfig, introspect_glue
from schemagraph.connectors.spider2 import Spider2Config, introspect_spider2
from schemagraph.connectors.unity import UnityClient, UnityConfig, introspect_unity
from schemagraph.graph.build import SchemaGraph, build_graph
from schemagraph.graph.infer import with_inferred_edges
from schemagraph.linking.linker import Linker, LinkOptions
from schemagraph.linking.render import render_ddl
from schemagraph.model import BusinessTerm, Column, Edge, SchemaSnapshot, Table
from tests.test_catalog_connectors import collibra_handler, unity_handler
from tests.test_spider2_and_infer import _write_table

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("SCHEMAGRAPH_UPDATE_GOLDEN") == "1"


# ----------------------------------------------------------------------------- comparison


def _to_jsonable(obj):
    """Graph attribute values -> JSON (Edge lists inside ``relations``)."""
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", by_alias=True)
    return obj


def check_golden(name: str, payload) -> None:
    text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
    path = GOLDEN_DIR / f"{name}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.exists(), f"missing golden {path}; generate it on the unmodified code"
    expected = path.read_text(encoding="utf-8")
    if text != expected:
        got, want = text.splitlines(), expected.splitlines()
        i = next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), min(len(got), len(want)))
        ctx = "\n".join(f"  want: {w}\n  got:  {g}" for w, g in zip(want[i : i + 3], got[i : i + 3], strict=False))
        pytest.fail(f"{name}: first difference at line {i + 1}\n{ctx}")


# ----------------------------------------------------------------------------- inputs

STORE_DDL = (REPO / "examples" / "store.sql").read_text(encoding="utf-8")

EXTRA_DDL = """
CREATE TABLE ops.warehouse (id INT, code VARCHAR NOT NULL, region VARCHAR, CONSTRAINT pk_w PRIMARY KEY (id));
CREATE TABLE ops.stock (warehouse_id INT, product_id INT, qty INT, PRIMARY KEY (warehouse_id, product_id));
ALTER TABLE ops.stock ADD FOREIGN KEY (warehouse_id) REFERENCES ops.warehouse(id);
ALTER TABLE ops.stock ADD CONSTRAINT fk_p FOREIGN KEY (product_id) REFERENCES products(id);
COMMENT ON COLUMN warehouse.region IS 'sales region of the warehouse';
COMMENT ON TABLE ops.stock IS 'On-hand stock per warehouse and product';
CREATE VIEW ops.v_stock AS SELECT * FROM ops.stock;
"""

LONG_VALUE = "awaiting confirmation from the payment provider today"  # > MAX_VALUE_TOKENS words


def snap_store() -> SchemaSnapshot:
    snap = parse_ddl(DDLConfig(ddl=STORE_DDL, dialect="postgres", default_schema="public"), "store")
    snap.table("public.customer").column("state").sample_values = ["California", "Texas", "New York", "U.S."]
    snap.table("public.shipment").column("carrier").sample_values = ["UPS", "FedEx", "DHL"]
    snap.table("public.orders").column("status").sample_values = ["paid", "refunded", LONG_VALUE, "10%", "A++"]
    return snap


def snap_extra_ddl() -> SchemaSnapshot:
    return parse_ddl(DDLConfig(ddl=EXTRA_DDL, dialect="postgres", default_schema="public"), "ops_ddl")


def snap_user() -> SchemaSnapshot:
    return SchemaSnapshot(
        source="user",
        source_type="user",
        priority=0,
        terms=[
            BusinessTerm(name="Revenue", description="money in", synonyms=["sales", "gross revenue"], targets=["public.orders.total_amount"]),
            BusinessTerm(name="buyer", targets=["customer"]),
            BusinessTerm(name="courier", synonyms=["carrier company"], targets=["public.shipment.carrier", "public.stores"]),
        ],
        edges=[
            Edge(kind="join_hint", from_table="public.orders", to_table="public.stores", from_columns=["store_id"], to_columns=["id"], description="orders to stores"),
            Edge(kind="join_hint", from_table="public.audit_log", to_table="public.customer", from_columns=["actor"], to_columns=["email"]),
        ],
    ).stamp()


def snap_duckdb(tmp: Path) -> SchemaSnapshot:
    path = tmp / "g.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA public")
    con.execute("CREATE TABLE public.customer (id INTEGER PRIMARY KEY, name VARCHAR, state VARCHAR, loyalty_tier VARCHAR)")
    con.execute("COMMENT ON TABLE public.customer IS 'customers from the operational db'")
    con.execute("COMMENT ON COLUMN public.customer.loyalty_tier IS 'bronze, silver or gold'")
    # fewer distinct values per text column than the sample cap (25), so DISTINCT ... LIMIT returns
    # all of them; 23 names still exceed the graph's 20-sample merge cap
    vals = ", ".join(f"({i}, 'n{i % 23}', 'State {i % 7}', '{['bronze', 'silver', 'gold'][i % 3]}')" for i in range(60))
    con.execute(f"INSERT INTO public.customer VALUES {vals}")
    con.execute("CREATE TABLE public.stores (id INTEGER PRIMARY KEY, city VARCHAR, manager VARCHAR)")
    con.execute("INSERT INTO public.stores VALUES (1, 'Austin', 'Ann'), (2, 'Boston', 'Bob')")
    con.execute("CREATE TABLE public.visits (id INTEGER, store_id INTEGER REFERENCES public.stores(id), customer_id INTEGER REFERENCES public.customer(id))")
    con.execute("CREATE VIEW public.v_visits AS SELECT * FROM public.visits")
    con.close()
    snap = introspect_duckdb(DuckDBConfig(path=str(path), sample_values=25), "warehouse_db")
    # SELECT DISTINCT has no defined order in DuckDB; pin it so the golden is deterministic
    for table in snap.tables:
        for column in table.columns:
            column.sample_values.sort()
    return snap


MANIFEST = {
    "nodes": {
        "model.p.orders": {"resource_type": "model", "name": "orders", "schema": "analytics", "description": "orders model", "tags": ["core"], "columns": {"id": {"name": "id", "constraints": [{"type": "primary_key"}]}, "customer_id": {"name": "customer_id", "description": "buyer", "constraints": [{"type": "foreign_key", "to": "ref('customers')", "to_columns": ["id"]}]}}, "depends_on": {"nodes": ["source.p.raw.raw_orders", "model.p.customers"]}, "constraints": [{"type": "foreign_key", "to": "ref('customers')", "columns": ["customer_id"], "to_columns": ["id"]}]},
        "model.p.customers": {"resource_type": "model", "name": "customers", "alias": "dim_customers", "schema": "analytics", "columns": {"id": {"name": "id", "data_type": "int"}}, "depends_on": {"nodes": []}},
        "seed.p.country_codes": {"resource_type": "seed", "name": "country_codes", "schema": "analytics", "columns": {"code": {"name": "code"}}},
        "test.p.rel": {"resource_type": "test", "test_metadata": {"name": "relationships", "kwargs": {"to": "ref('customers')", "field": "id", "column_name": "customer_id"}}, "attached_node": "model.p.orders", "depends_on": {"nodes": ["model.p.orders", "model.p.customers"]}},
        "test.p.rel2": {"resource_type": "test", "test_metadata": {"name": "relationships", "kwargs": {"to": "source('raw', 'raw_orders')", "field": "id", "column_name": "id"}}, "depends_on": {"nodes": ["model.p.orders", "source.p.raw.raw_orders"]}},
        "test.p.uniq": {"resource_type": "test", "test_metadata": {"name": "unique", "kwargs": {"column_name": "code"}}, "attached_node": "seed.p.country_codes"},
    },
    "sources": {"source.p.raw.raw_orders": {"resource_type": "source", "name": "raw_orders", "source_name": "raw", "schema": "raw", "identifier": "raw_orders", "columns": {"id": {"name": "id"}}}},
    "semantic_models": {"sm.orders": {"name": "orders_sm", "depends_on": {"nodes": ["model.p.orders"]}, "entities": [{"name": "order", "type": "primary", "expr": "id"}], "measures": [{"name": "order_total", "agg": "sum", "expr": "total"}, {"name": "order_count", "agg": "count", "description": "number of orders"}]}},
}


def snap_manifest(tmp: Path) -> SchemaSnapshot:
    path = tmp / "manifest.json"
    path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    return parse_manifest(DbtConfig(manifest_path=str(path)), "dbt_manifest")


def snap_dbt_project() -> SchemaSnapshot:
    return parse_project(DbtConfig(project_dir=str(FIXTURES / "dbt_airbnb")), "airbnb")


def snap_collibra() -> SchemaSnapshot:
    cfg = CollibraConfig(host="https://c.example", token="t")
    return introspect_collibra(cfg, "collibra", CollibraClient(cfg, transport=httpx.MockTransport(collibra_handler)))


def snap_unity() -> SchemaSnapshot:
    cfg = UnityConfig(host="https://u.example", token="tok", lineage=True)
    return introspect_unity(cfg, "unity", UnityClient(cfg, transport=httpx.MockTransport(unity_handler)))


class _Paginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **kw):
        return iter(self.pages)


class _FakeGlue:
    def get_paginator(self, name):
        if name == "get_databases":
            return _Paginator([{"DatabaseList": [{"Name": "lake"}, {"Name": "skip_me"}]}])
        return _Paginator([{"TableList": [
            {"Name": "events", "Description": "clickstream", "Owner": "web", "TableType": "EXTERNAL_TABLE", "StorageDescriptor": {"Columns": [{"Name": "user_id", "Type": "bigint", "Parameters": {"pii": "no"}}, {"Name": "url", "Type": "string", "Comment": "page"}], "Location": "s3://b/events"}, "PartitionKeys": [{"Name": "dt", "Type": "string"}], "Parameters": {"classification": "parquet", "ignored": "x"}},
            {"Name": "v_daily", "TableType": "VIRTUAL_VIEW", "StorageDescriptor": {"Columns": [{"Name": "dt", "Type": "string"}]}, "Parameters": {"comment": "daily view"}},
        ]}])


class _FakeLF:
    def get_resource_lf_tags(self, **kw):
        if kw["Resource"]["Table"]["Name"] == "v_daily":
            raise RuntimeError("AccessDenied")
        return {"LFTagsOnTable": [{"TagKey": "domain", "TagValues": ["web"]}], "LFTagsOnColumns": [{"Name": "user_id", "LFTags": [{"TagKey": "pii", "TagValues": ["low"]}]}]}


def snap_glue() -> SchemaSnapshot:
    return introspect_glue(GlueConfig(region="eu-west-1", databases=["lake"], lf_tags=True), "glue", glue_client=_FakeGlue(), lf_client=_FakeLF())


def snap_spider2(tmp: Path) -> SchemaSnapshot:
    root = tmp / "databases"
    ds = root / "bigquery" / "shop" / "proj.shop"
    for day in ("20210101", "20210102", "20210103"):
        _write_table(ds, f"proj.shop.events_{day}", ["event_name", "user_id", "page"], ["STRING", "INT64", "STRING"], descs=["name of the event", "", ""], rows=[{"event_name": "purchase", "user_id": 1, "page": "/cart"}, {"event_name": "view_item", "page": "/p"}], nested=["event_name", "user_id", "page", "params.key"])
    for year in ("2018", "2019"):
        _write_table(ds, f"proj.shop.county_{year}_1yr", ["geo_id", "median_income", "population"], ["STRING", "FLOAT64", "INT64"])
    _write_table(ds, "proj.shop.imaging_level2_metadata_r1", ["series_id", "modality"], ["STRING", "STRING"])
    _write_table(ds, "proj.shop.imaging_level2_metadata_r2", ["series_id", "modality"], ["STRING", "STRING"])
    _write_table(ds, "proj.shop.imaging_level4_metadata_r1", ["series_id", "modality"], ["STRING", "STRING"])
    _write_table(ds, "proj.shop.imaging_level4_metadata_r2", ["series_id", "modality"], ["STRING", "STRING"])
    _write_table(ds, "proj.shop.users", ["user_id", "country", "signup_date"], ["INT64", "STRING", "DATE"], rows=[{"country": "Portugal"}, {"country": "Czech Republic"}])
    _write_table(ds, "proj.shop.orders", ["order_id", "user_id", "product_code", "amount"], ["INT64", "INT64", "STRING", "NUMERIC"])
    _write_table(ds, "proj.shop.products", ["product_code", "title"], ["STRING", "STRING"])
    snap = introspect_spider2(Spider2Config(root=str(root), dialect="bigquery", db="shop"), "spider2:shop")
    return with_inferred_edges(snap)


def snap_spider2_sqlite(tmp: Path) -> SchemaSnapshot:
    sq = tmp / "databases" / "sqlite" / "Shop"
    _write_table(sq, "orders", ["id", "customer_id"], ["INTEGER", "INTEGER"])
    _write_table(sq, "customers", ["id", "name"], [None, "TEXT"])
    (sq / "DDL.csv").write_text('table_name,DDL\norders,"CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, FOREIGN KEY (customer_id) REFERENCES customers(id));"\ncustomers,"CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"\n', encoding="utf-8")
    return introspect_spider2(Spider2Config(root=str(tmp / "databases"), dialect="sqlite", db="Shop"), "spider2:Shop")


def snap_dump(snap: SchemaSnapshot) -> dict:
    return snap.model_dump(mode="json", by_alias=True, exclude={"created_at"})


# ----------------------------------------------------------------------------- graph / linker payloads


def graph_dump(schema_graph: SchemaGraph) -> dict:
    nxg = schema_graph.graph
    return {
        "nodes": [[n, _to_jsonable(d)] for n, d in nxg.nodes(data=True)],
        "edges": [[u, v, _to_jsonable(d)] for u, v, d in nxg.edges(data=True)],
        "tables": {k: t.model_dump(mode="json", by_alias=True) for k, t in schema_graph.tables.items()},
        "terms": {k: t.model_dump(mode="json") for k, t in schema_graph.terms.items()},
        "sources": dict(schema_graph.sources),
        "stats": schema_graph.stats(),
        "table_graph_edges": [[u, v, d["weight"]] for u, v, d in schema_graph.table_graph().edges(data=True)],
        "lineage": {k: list(schema_graph.lineage(k)) for k in schema_graph.tables},
    }


class FakeLLM:
    """Deterministic anchor picker: first candidate is the source, second the destination."""

    def anchor_tables(self, question, candidates):
        names = [t.fqn for t in candidates]
        return names[:1] + ["not.a.table"], names[1:2]


class FailingLLM:
    def anchor_tables(self, question, candidates):
        raise RuntimeError("boom")


VARIANTS: dict[str, dict] = {
    "default": {},
    "tight": {"max_tables": 3, "anchor_k": 2, "bypass_if_fits": False},
    "tight_paths": {"max_tables": 4, "anchor_k": 3, "bypass_if_fits": False, "path_extra": 1.5, "prune_top_k": 1},
    "no_paths": {"max_tables": 4, "anchor_k": 3, "bypass_if_fits": False, "paths": False},
    "ppr": {"ranker": "ppr", "bypass_if_fits": False, "max_tables": 5},
    "bm25": {"ranker": "bm25", "bypass_if_fits": False, "max_tables": 5},
    "sum_noidf_nostop": {"agg": "sum", "idf": False, "ngram_stop": False, "min_numeric_len": 1},
    "routing_seed": {"schema_routing": 0.5, "seed_bm25": True, "bypass_if_fits": False, "max_tables": 6},
    "affinity": {"ppr_edge_attr": "affinity", "bypass_if_fits": False, "max_tables": 5},
    "columns": {"columns": "all", "max_columns_per_table": 2, "columns_top_uncapped": 0, "bypass_if_fits": False},
    "columns_capped": {"max_columns_per_table": 1, "columns_top_uncapped": 2, "bypass_if_fits": False, "max_tables": 4},
    "adaptive": {"large_threshold": 3, "max_tables": 2, "max_tables_large": 5, "anchor_k": 1, "anchor_k_large": 3, "bypass_if_fits": False},
    "small_bypass": {"small_schema_bypass": 1000},
    "llm": {"use_llm": True, "bypass_if_fits": False, "max_tables": 5},
    "llm_failing": {"use_llm": True, "bypass_if_fits": False, "max_tables": 5},
    "ranking_limited": {"ranking_limit": 3},
}

COMMON_QUESTIONS = [
    "hello there",  # no schema vocabulary: empty activation
    "top 5 of everything in 2023",
]

QUESTIONS: dict[str, list[str]] = {
    "store": [
        "revenue by product category for customers in california",
        "which orders were shipped by UPS",
        "total sales per customer state",
        "gross revenue by courier",
        "avg qty of order items per prod",
        "list all categories and their segments",
        "shipmnts per carrier company",
        "date of birth of customers",
        f"orders {LONG_VALUE}",
        "audit log actions by actor and buyer email",
        "warehouse stock by region for each product",
        "which stores have the most orders",
    ],
    "airbnb": [
        "average review score per host",
        "listings with full moon reviews",
        "monthly reviews sentiment for superhosts",
        "minimum nights and price of listings by host name",
        "week over week review counts",
    ],
    "merged": [
        "revenue by customer loyalty tier",
        "orders per customer from raw orders",
        "clickstream page urls per user",
        "store visits by city and manager",
        "order total measure per customer",
        "number of orders",
        "customers in State 3",
    ],
    "spider2": [
        "purchase events per user country",
        "median income by county in 2019",
        "imaging level4 metadata modality",
        "order amount by product title for users in Portugal",
    ],
}


def link_payload(linker: Linker, graph_name: str) -> dict:
    out: dict = {}
    for variant, kw in VARIANTS.items():
        opts = LinkOptions(**{"debug": True, "ranking_limit": 0, **kw})
        linker.llm = FakeLLM() if variant == "llm" else FailingLLM() if variant == "llm_failing" else None
        rows = []
        for question in QUESTIONS[graph_name] + COMMON_QUESTIONS:
            result = linker.link(question, opts).model_dump(mode="json")
            result["stats"].pop("ms", None)
            rows.append(result)
        out[variant] = rows
    linker.llm = None
    out["explain"] = [linker.explain(q) for q in QUESTIONS[graph_name] + COMMON_QUESTIONS]
    first = linker.link(QUESTIONS[graph_name][0], LinkOptions(render=False))
    out["ddl_no_samples"] = render_ddl(linker_graph(linker), first, samples=False)
    return out


def linker_graph(linker: Linker) -> SchemaGraph:
    return linker.schema_graph


def index_dump(linker: Linker) -> dict:
    return _to_jsonable(dataclasses.asdict(linker.index))


def check_graph_and_links(name: str, schema_graph: SchemaGraph) -> None:
    check_golden(f"graph_{name}", graph_dump(schema_graph))
    linker = Linker(schema_graph)
    check_golden(f"index_{name}", {"index": index_dump(linker), "graph_with_tokens": graph_dump(schema_graph)})
    check_golden(f"link_{name}", link_payload(linker, name))


# ----------------------------------------------------------------------------- tests


def test_golden_connector_snapshots(tmp_path):
    snaps = {
        "ddl_store": snap_store(),
        "ddl_extra": snap_extra_ddl(),
        "dbt_project": snap_dbt_project(),
        "dbt_manifest": snap_manifest(tmp_path),
        "duckdb": snap_duckdb(tmp_path),
        "collibra": snap_collibra(),
        "unity": snap_unity(),
        "glue": snap_glue(),
        "spider2_bq": snap_spider2(tmp_path),
        "spider2_sqlite": snap_spider2_sqlite(tmp_path),
    }
    payload = {k: snap_dump(s) for k, s in snaps.items()}
    assert str(tmp_path) not in json.dumps(payload), "golden must not embed tmp paths"
    check_golden("snapshots", payload)


def test_golden_store():
    check_graph_and_links("store", build_graph([snap_store(), snap_extra_ddl(), snap_user()]))


def test_golden_airbnb():
    check_graph_and_links("airbnb", build_graph([snap_dbt_project()]))


def test_golden_merged(tmp_path):
    snaps = [snap_store(), snap_duckdb(tmp_path), snap_manifest(tmp_path), snap_collibra(), snap_unity(), snap_glue(), snap_user()]
    snaps[1].priority = 5  # a connection priority that outranks collibra
    check_graph_and_links("merged", build_graph(snaps))


def test_golden_spider2(tmp_path):
    check_graph_and_links("spider2", build_graph([snap_spider2(tmp_path)]))


def test_golden_incremental(tmp_path):
    """The add_snapshot path: stubs created by early edges, replaced by later tables, terms re-resolved."""
    schema_graph = SchemaGraph()
    schema_graph.add_snapshot(snap_user())
    schema_graph.add_snapshot(snap_store())
    schema_graph.add_snapshot(snap_duckdb(tmp_path))
    schema_graph.add_snapshot(SchemaSnapshot(source="echo", source_type="duckdb", tables=[Table(name="t", properties={"stub": "true"}, columns=[Column(name="a")])]))
    schema_graph.add_snapshot(snap_extra_ddl())
    check_golden("graph_incremental", graph_dump(schema_graph))


# ----------------------------------------------------------------------------- frozen surfaces


def _strip_schema_descriptions(schema: dict) -> dict:
    """Drop the class-docstring ``description`` at the root of each schema object (docstrings may be
    added to pydantic classes); per-property descriptions stay compared (the UI masks secrets by them)."""
    out = {k: v for k, v in schema.items() if k != "description"}
    for key in ("$defs", "definitions"):
        if key in out:
            out[key] = {n: {k: v for k, v in s.items() if k != "description"} for n, s in out[key].items()}
    return out


def test_golden_surfaces(tmp_path):
    from typer.main import get_command

    from schemagraph.api.app import create_app
    from schemagraph.bench import spider1, spider2_lite
    from schemagraph.cli import app as cli_app
    from schemagraph.connectors import config_schema, connector_types
    from schemagraph.engine import Engine
    from schemagraph.mcp.server import create_server

    engine = Engine(tmp_path / "home", llm=None, embed=False)
    openapi = create_app(engine, web_dist=tmp_path / "no_dist").openapi()
    openapi["components"]["schemas"] = {n: _strip_schema_descriptions(s) for n, s in openapi["components"]["schemas"].items()}
    tools = asyncio.run(create_server(engine).list_tools())
    command = get_command(cli_app)
    cli = {
        name: [[p.name, list(getattr(p, "opts", [])), repr(p.default), getattr(p.type, "name", str(p.type))] for p in cmd.params]
        for name, cmd in sorted(command.commands.items())
    }
    check_golden("surfaces", {
        "openapi": openapi,
        "mcp_tools": [t.model_dump(mode="json", exclude_none=True) for t in tools],
        "connector_config_schemas": {t: _strip_schema_descriptions(config_schema(t)) for t in connector_types()},
        "cli": cli,
        "link_options": [[f.name, repr(f.default)] for f in dataclasses.fields(LinkOptions)],
        "spider2_row_fields": [f.name for f in dataclasses.fields(spider2_lite.Row)],
        "spider1_row_fields": [f.name for f in dataclasses.fields(spider1.Row)],
        "stats_keys": sorted(engine.stats()),
    })
    engine.close()


def test_golden_llm_request():
    """The Claude anchor-picker request (system prompt, compact schema, output schema) is LLM-visible: frozen."""
    from types import SimpleNamespace

    from schemagraph.llm.anchors import ClaudeAnchorPicker

    sent: list[dict] = []

    def create(**kw):
        sent.append(kw)
        text = json.dumps({"source_tables": ["customer", "nope"], "destination_tables": ["PUBLIC.ORDERS"], "rationale": "x"})
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=2), model="m", stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)), messages=SimpleNamespace(create=create))
    schema_graph = build_graph([snap_store()])
    candidates = list(schema_graph.tables.values())
    picks = [ClaudeAnchorPicker(model="m", client=client, fallbacks=fb).anchor_tables("revenue by state", candidates) for fb in (True, False)]
    check_golden("llm_request", {"requests": sent, "picks": picks})
