"""Catalog connectors tested against canned API payloads (no network)."""

import json

import httpx

from schemagraph.connectors.collibra import CollibraClient, CollibraConfig, introspect_collibra
from schemagraph.connectors.unity import UnityClient, UnityConfig, introspect_unity

# ----------------------------------------------------------------------------- Unity Catalog

UNITY_ROUTES = {
    "/api/2.1/unity-catalog/catalogs": {"catalogs": [{"name": "sales"}, {"name": "system"}]},
    "/api/2.1/unity-catalog/schemas": {"schemas": [{"name": "core"}, {"name": "information_schema"}]},
    "/api/2.1/unity-catalog/tables": {
        "tables": [
            {
                "name": "orders",
                "table_type": "MANAGED",
                "comment": "Orders fact",
                "owner": "data-eng",
                "columns": [{"name": "id", "type_text": "bigint", "position": 0}, {"name": "customer_id", "type_text": "bigint", "position": 1, "comment": "FK to customers"}],
                "table_constraints": [
                    {"primary_key_constraint": {"name": "pk", "child_columns": ["id"]}},
                    {"foreign_key_constraint": {"name": "fk_c", "child_columns": ["customer_id"], "parent_table": "sales.core.customers", "parent_columns": ["id"]}},
                ],
            },
            {"name": "customers", "table_type": "MANAGED", "columns": [{"name": "id", "type_text": "bigint", "position": 0}], "table_constraints": []},
            {"name": "v_orders", "table_type": "VIEW", "columns": [{"name": "id", "type_text": "bigint"}]},
        ]
    },
    "/api/2.0/lineage-tracking/table-lineage": {"upstreams": [{"tableInfo": {"catalog_name": "sales", "schema_name": "core", "name": "customers"}}], "downstreams": []},
}


def unity_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers.get("authorization") == "Bearer tok"
    return httpx.Response(200, json=UNITY_ROUTES[request.url.path])


def test_unity_tables_constraints_and_lineage():
    cfg = UnityConfig(host="https://ws.example", token="tok", lineage=True)
    client = UnityClient(cfg, transport=httpx.MockTransport(unity_handler))
    snap = introspect_unity(cfg, "uc", client=client)
    fq = {t.fqn for t in snap.tables}
    assert fq == {"sales.core.orders", "sales.core.customers", "sales.core.v_orders"}
    orders = snap.table("sales.core.orders")
    assert orders.primary_key == ["id"] and orders.owner == "data-eng"
    fks = [e for e in snap.edges if e.kind == "foreign_key"]
    assert fks[0].to_table == "sales.core.customers" and fks[0].from_columns == ["customer_id"]
    assert any(e.kind == "lineage" and e.to_table == "sales.core.orders" for e in snap.edges)
    assert snap.table("sales.core.v_orders").kind == "view"


# ----------------------------------------------------------------------------- Collibra

T_TABLE, T_COLUMN, T_SCHEMA, T_TERM = "t-table", "t-col", "t-schema", "t-term"
RT_CONTAINS, RT_REF, RT_REPR = "rt-contains", "rt-ref", "rt-repr"

ASSETS = {
    T_SCHEMA: [{"id": "s1", "name": "sales", "displayName": "sales"}],
    T_TABLE: [{"id": "tb1", "name": "sales > orders", "displayName": "orders", "domain": {"name": "Sales"}}, {"id": "tb2", "name": "sales > customers", "displayName": "customers", "domain": {"name": "Sales"}}],
    T_COLUMN: [
        {"id": "c1", "name": "sales > orders > customer_id", "displayName": "customer_id"},
        {"id": "c2", "name": "sales > customers > id", "displayName": "id"},
        {"id": "c3", "name": "sales > orders > amount", "displayName": "amount"},
    ],
    T_TERM: [{"id": "bt1", "name": "Revenue", "displayName": "Revenue"}],
}
RELATIONS = [
    {"id": "r1", "source": {"id": "s1"}, "target": {"id": "tb1"}, "type": {"id": RT_CONTAINS}},
    {"id": "r2", "source": {"id": "s1"}, "target": {"id": "tb2"}, "type": {"id": RT_CONTAINS}},
    {"id": "r3", "source": {"id": "tb1"}, "target": {"id": "c1"}, "type": {"id": RT_CONTAINS}},
    {"id": "r4", "source": {"id": "tb1"}, "target": {"id": "c3"}, "type": {"id": RT_CONTAINS}},
    {"id": "r5", "source": {"id": "tb2"}, "target": {"id": "c2"}, "type": {"id": RT_CONTAINS}},
    {"id": "r6", "source": {"id": "c1"}, "target": {"id": "c2"}, "type": {"id": RT_REF}},
    {"id": "r7", "source": {"id": "bt1"}, "target": {"id": "c3"}, "type": {"id": RT_REPR}},
]
ATTRS = {
    "tb1": [{"type": {"name": "Description"}, "value": "Customer orders"}, {"type": {"name": "Data Classification"}, "value": "Internal"}],
    "c3": [{"type": {"name": "Description"}, "value": "Order amount in EUR"}, {"type": {"name": "Data Type"}, "value": "DECIMAL"}],
    "bt1": [{"type": {"name": "Description"}, "value": "Money we made"}],
}


def collibra_handler(request: httpx.Request) -> httpx.Response:
    p = request.url.path
    q = dict(request.url.params)
    if p.endswith("/assetTypes"):
        name = q["name"]
        ids = {"Table": T_TABLE, "Column": T_COLUMN, "Schema": T_SCHEMA, "Business Term": T_TERM}
        return httpx.Response(200, json={"results": [{"id": ids[name], "name": name}] if name in ids else []})
    if p.endswith("/relationTypes"):
        return httpx.Response(200, json={"results": [{"id": RT_CONTAINS, "role": "contains", "coRole": "is part of"}, {"id": RT_REF, "role": "references", "coRole": "is referenced by"}, {"id": RT_REPR, "role": "represents", "coRole": "is represented by"}]})
    if p.endswith("/assets"):
        return httpx.Response(200, json={"results": ASSETS.get(q["typeIds"], [])})
    if p.endswith("/attributes"):
        return httpx.Response(200, json={"results": ATTRS.get(q["assetId"], [])})
    if p.endswith("/relations"):
        if "sourceId" in q:
            res = [r for r in RELATIONS if r["source"]["id"] == q["sourceId"]]
        else:
            res = [r for r in RELATIONS if r["target"]["id"] == q["targetId"]]
        return httpx.Response(200, json={"results": res})
    return httpx.Response(404, json={"error": p})


def test_collibra_tables_columns_fk_terms():
    cfg = CollibraConfig(host="https://acme.collibra.com", username="u", password="p")
    client = CollibraClient(cfg, transport=httpx.MockTransport(collibra_handler))
    assert client.http.headers["authorization"].startswith("Basic ")
    snap = introspect_collibra(cfg, "collibra", client=client)
    orders = snap.table("sales.orders")
    assert orders is not None and orders.description == "Customer orders" and "Data Classification=Internal" in orders.tags
    assert {c.name for c in orders.columns} == {"customer_id", "amount"}
    assert orders.column("amount").data_type == "DECIMAL"
    fk = [e for e in snap.edges if e.kind == "foreign_key"]
    assert len(fk) == 1 and fk[0].from_table == "sales.orders" and fk[0].to_table == "sales.customers" and fk[0].from_columns == ["customer_id"] and fk[0].to_columns == ["id"]
    assert snap.terms[0].name == "Revenue" and snap.terms[0].targets == ["sales.orders.amount"] and snap.terms[0].description == "Money we made"


# ----------------------------------------------------------------------------- Glue


def test_glue_with_stubbed_client():
    from schemagraph.connectors.glue import GlueConfig, introspect_glue

    class FakePaginator:
        def __init__(self, pages):
            self.pages = pages

        def paginate(self, **kw):
            return iter(self.pages)

    class FakeGlue:
        def get_paginator(self, name):
            if name == "get_databases":
                return FakePaginator([{"DatabaseList": [{"Name": "lake"}, {"Name": "skip_me"}]}])
            return FakePaginator(
                [
                    {
                        "TableList": [
                            {
                                "Name": "events",
                                "Description": "clickstream",
                                "Owner": "web",
                                "TableType": "EXTERNAL_TABLE",
                                "StorageDescriptor": {"Columns": [{"Name": "user_id", "Type": "bigint"}, {"Name": "url", "Type": "string", "Comment": "page"}], "Location": "s3://b/events"},
                                "PartitionKeys": [{"Name": "dt", "Type": "string"}],
                                "Parameters": {"classification": "parquet"},
                            }
                        ]
                    }
                ]
            )

    snap = introspect_glue(GlueConfig(region="eu-west-1", databases=["lake"]), "glue", glue_client=FakeGlue())
    assert [t.fqn for t in snap.tables] == ["lake.events"]
    t = snap.tables[0]
    assert t.kind == "external" and t.properties["location"] == "s3://b/events" and t.properties["classification"] == "parquet"
    assert [c.name for c in t.columns] == ["user_id", "url", "dt"] and "partition_key" in t.columns[-1].tags
    assert snap.edges == []
    json.dumps(snap.model_dump(mode="json"))  # serialisable
