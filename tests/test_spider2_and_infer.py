import json

from schemagraph.connectors.spider2 import Spider2Config, canonical_table, introspect_spider2
from schemagraph.graph import build_graph
from schemagraph.graph.infer import infer_edges
from schemagraph.linking import Linker
from schemagraph.model import Column, SchemaSnapshot, Table


def _write_table(folder, fullname, cols, types, descs=None, rows=None, nested=None):
    folder.mkdir(parents=True, exist_ok=True)
    name = fullname.split(".")[-1]
    d = {"table_name": name, "table_fullname": fullname, "column_names": cols, "column_types": types, "description": descs or ["" for _ in cols], "sample_rows": rows or []}
    if nested:
        d["nested_column_names"] = nested
        d["nested_column_types"] = ["STRING" for _ in nested]
    (folder / f"{name}.json").write_text(json.dumps(d))


def test_canonical_collapses_date_shards():
    assert canonical_table("bigquery-public-data.ga4.events_20210101") == "bigquery-public-data.ga4.events_*"
    assert canonical_table("cms.inpatient_charges_2014") == "cms.inpatient_charges_*"  # year families too
    assert canonical_table("noaa.gsod2019") == "noaa.gsod_*"
    assert canonical_table("GITHUB_REPOS_DATE.DAY._20230118") == "github_repos_date.day._*"
    assert canonical_table("bookings") == "bookings"
    assert canonical_table("x.covid19") == "x.covid19"  # two digits are not a year
    assert canonical_table("x.table_1000") == "x.table_1000"


def test_spider2_connector_families_ddl_keys(tmp_path):
    root = tmp_path / "databases"
    ds = root / "bigquery" / "ga4" / "bigquery-public-data.ga4"
    for day in ("20210101", "20210102", "20210103"):
        _write_table(ds, f"bigquery-public-data.ga4.events_{day}", ["event_name", "user_pseudo_id"], ["STRING", "STRING"], rows=[{"event_name": "purchase", "user_pseudo_id": "x"}], nested=["event_name", "user_pseudo_id", "event_params.key"])
    _write_table(ds, "bigquery-public-data.ga4.items", ["item_id", "item_name"], ["STRING", "STRING"])
    snap = introspect_spider2(Spider2Config(root=str(root), dialect="bigquery", db="ga4"), "t")
    names = sorted(t.fqn for t in snap.tables)
    assert names == ["bigquery-public-data.ga4.events_*", "bigquery-public-data.ga4.items"]
    fam = snap.table("bigquery-public-data.ga4.events_*")
    assert fam.properties["members"].count(",") == 2 and "family of 3" in fam.description
    assert fam.column("event_name").sample_values == ["purchase"]
    assert fam.column("event_params.key") is not None and "nested" in fam.column("event_params.key").tags

    # sqlite db with DDL.csv declaring a FK
    sq = root / "sqlite" / "Shop"
    _write_table(sq, "orders", ["id", "customer_id"], ["INTEGER", "INTEGER"])
    _write_table(sq, "customers", ["id", "name"], ["INTEGER", "TEXT"])
    (sq / "DDL.csv").write_text('table_name,DDL\norders,"CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, FOREIGN KEY (customer_id) REFERENCES customers(id));"\ncustomers,"CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"\n')
    snap = introspect_spider2(Spider2Config(root=str(root), dialect="sqlite", db="Shop"), "t")
    assert snap.table("orders").primary_key == ["id"]
    assert len(snap.edges) == 1 and snap.edges[0].kind == "foreign_key" and snap.edges[0].to_table == "customers"


def test_infer_edges_rules():
    snap = SchemaSnapshot(
        source="x",
        source_type="ddl",
        tables=[
            Table(name="orders", schema="s", columns=[Column(name="order_id", is_primary_key=True), Column(name="customer_id"), Column(name="id")]),
            Table(name="customers", schema="s", columns=[Column(name="id", is_primary_key=True), Column(name="name")]),
            Table(name="order_items", schema="s", columns=[Column(name="order_id"), Column(name="product_id")]),
            Table(name="products", schema="s", columns=[Column(name="product_id", is_primary_key=True)]),
            Table(name="unrelated", schema="other", columns=[Column(name="customer_id")]),
        ],
    )
    edges = infer_edges(snap)
    pairs = {(e.from_table, e.from_columns[0], e.to_table, e.to_columns[0]) for e in edges}
    assert ("s.orders", "customer_id", "s.customers", "id") in pairs  # rule 1: customer_id -> customers.id
    assert ("s.order_items", "order_id", "s.orders", "order_id") in pairs  # rule 2 shared key with pk holder
    assert ("s.order_items", "product_id", "s.products", "product_id") in pairs
    assert not any("unrelated" in e.from_table or "unrelated" in e.to_table for e in edges)  # different schema
    assert all(e.kind == "inferred" for e in edges)


def test_inferred_edges_enable_bridge_paths():
    snap = SchemaSnapshot(
        source="x",
        source_type="ddl",
        tables=[
            Table(name="customers", columns=[Column(name="customer_id", is_primary_key=True), Column(name="state", sample_values=["Texas"])]),
            Table(name="orders", columns=[Column(name="order_id", is_primary_key=True), Column(name="customer_id")]),
            Table(name="order_items", columns=[Column(name="order_id"), Column(name="product_id"), Column(name="quantity")]),
            Table(name="products", columns=[Column(name="product_id", is_primary_key=True), Column(name="category")]),
        ],
    )
    snap.edges.extend(infer_edges(snap))
    r = Linker(build_graph([snap])).link("quantity by product category for customers in Texas")
    assert {"customers", "orders", "order_items", "products"} <= {t.fqn for t in r.tables}
    assert any(len(p.tables) == 4 for p in r.join_paths)


def test_family_collapse_keeps_clusters_with_the_same_signature(tmp_path):
    # imaging_level2_metadata_r2..r5 and imaging_level4_metadata_r3..r5 share the digit-run signature
    # imaging_level*_metadata_r* but have different columns: two families, both must survive
    root = tmp_path / "databases"
    ds = root / "snowflake" / "HTAN" / "HTAN.V"
    for r in ("2", "3", "4", "5"):
        _write_table(ds, f"HTAN.V.imaging_level2_metadata_r{r}", ["file_id", "channel", "pixel_size"], ["STRING"] * 3)
    for r in ("3", "4", "5"):
        _write_table(ds, f"HTAN.V.imaging_level4_metadata_r{r}", ["file_id", "cell_type", "marker", "score"], ["STRING"] * 4)
    _write_table(ds, "HTAN.V.imaging_level1_metadata_r5", ["file_id", "raw_path", "instrument", "vendor", "lens"], ["STRING"] * 5)
    snap = introspect_spider2(Spider2Config(root=str(root), dialect="snowflake", db="HTAN"), "t")
    names = sorted(t.fqn for t in snap.tables)
    assert names == ["HTAN.V.imaging_level1_metadata_r5", "HTAN.V.imaging_level2_metadata_r*", "HTAN.V.imaging_level4_metadata_r*"]
    l2 = snap.table("HTAN.V.imaging_level2_metadata_r*")
    assert l2.properties["members"].count(",") == 3 and "imaging_level2_metadata_r5" in l2.properties["members"]
    assert snap.table("HTAN.V.imaging_level4_metadata_r*").properties["members"].count(",") == 2
