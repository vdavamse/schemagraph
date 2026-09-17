import json

from schemagraph.bench.spider1 import gold_tables, run, snapshot_from_spider
from schemagraph.graph import build_graph

DB = {
    "db_id": "shop",
    "table_names_original": ["customer", "orders", "order_items", "products"],
    "table_names": ["customer", "orders", "order items", "products"],
    "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [1, "id"], [1, "customer_id"], [2, "id"], [2, "order_id"], [2, "product_id"], [3, "id"], [3, "name"]],
    "column_names": [[-1, "*"], [0, "id"], [0, "customer name"], [1, "id"], [1, "customer id"], [2, "id"], [2, "order id"], [2, "product id"], [3, "id"], [3, "product name"]],
    "column_types": ["text", "number", "text", "number", "number", "number", "number", "number", "number", "text"],
    "primary_keys": [1, 3, 5, 8],
    "foreign_keys": [[4, 1], [6, 3], [7, 8]],
}


def test_snapshot_from_spider_and_gold_tables():
    snap = snapshot_from_spider(DB)
    assert [t.name for t in snap.tables] == ["customer", "orders", "order_items", "products"]
    assert snap.table("orders").primary_key == ["id"] and snap.table("customer").column("name").properties["business_name"] == "customer name"
    assert {(e.from_table, e.to_table) for e in snap.edges} == {("orders", "customer"), ("order_items", "orders"), ("order_items", "products")}
    assert gold_tables("WITH c AS (SELECT * FROM customer) SELECT T1.name FROM c AS T1 JOIN orders AS T2 ON T1.id = T2.customer_id") == {"customer", "orders"}
    assert gold_tables("SELECT count(*) FROM products EXCEPT SELECT id FROM order_items") == {"products", "order_items"}
    sg = build_graph([snap])
    assert sg.relations("order_items", "products")[0].kind == "foreign_key"


def test_run_scores_bridge_tables(tmp_path):
    tables = tmp_path / "tables.json"
    tables.write_text(json.dumps([DB]))
    qs = tmp_path / "q.json"
    qs.write_text(json.dumps([{"db_id": "shop", "instance_id": "q1", "question": "names of customers and the products they bought", "query": "SELECT c.name, p.name FROM customer c JOIN orders o ON c.id = o.customer_id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id"}]))
    res = run(tables, qs, max_tables=4, anchor_k=2, out_dir=tmp_path, tag="t")
    row = res["rows"][0]
    assert row.n_gold == 4 and row.strict == 1 and row.n_bridge >= 1 and row.bridge_recovered == row.n_bridge
    assert res["summary"]["bridge_recall"] == 100.0
    off = run(tables, qs, max_tables=2, anchor_k=2, paths=False)
    assert off["rows"][0].strict == 0  # two anchors, no path union: the bridge tables are gone
    assert (tmp_path / "spider1_t.json").exists()
