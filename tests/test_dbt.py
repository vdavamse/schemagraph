import json

from schemagraph.connectors.dbt import DbtConfig, parse_manifest, parse_project


def test_parse_project_lineage_and_relationships(dbt_airbnb_dir):
    snap = parse_project(DbtConfig(project_dir=str(dbt_airbnb_dir)), "airbnb")
    names = {t.properties.get("dbt_name"): t for t in snap.tables}
    assert {"hosts", "listings", "reviews"} <= {k for k, t in names.items() if t.kind == "source"}
    assert "fct_reviews" in names and "dim_listings_hosts" in names
    kinds = {e.kind for e in snap.edges}
    assert "lineage" in kinds and "relationship_test" in kinds
    rel = [e for e in snap.edges if e.kind == "relationship_test" and e.from_table.endswith("fct_reviews")]
    assert any(e.to_table.endswith("dim_listings") and e.from_columns == ["LISTING_ID"] and e.to_columns == ["LISTING_ID"] for e in rel)
    lin = [e for e in snap.edges if e.kind == "lineage" and e.to_table.endswith("fct_reviews")]
    assert lin, "fct_reviews must have upstream lineage from its ref()/source() calls"
    src_lin = [e for e in snap.edges if e.kind == "lineage" and e.from_table == "main.raw_hosts"]
    assert src_lin and src_lin[0].to_table.endswith("src_hosts")  # source() -> staging model
    assert any("dim_listings_hosts" in w for w in snap.warnings) or names["dim_listings_hosts"].columns  # declared in yml only
    # unique+not_null -> primary key hint
    assert names["dim_hosts"].column("HOST_ID").is_primary_key
    # accepted_values -> sample values
    assert "positive" in names["fct_reviews"].column("REVIEW_SENTIMENT").sample_values
    assert names["dim_hosts"].schema_name == "main"  # from profiles.yml


def test_parse_manifest_minimal(tmp_path):
    manifest = {
        "nodes": {
            "model.p.orders": {"resource_type": "model", "name": "orders", "schema": "analytics", "description": "orders", "columns": {"id": {"name": "id"}, "customer_id": {"name": "customer_id", "constraints": [{"type": "foreign_key", "to": "ref('customers')", "to_columns": ["id"]}]}}, "depends_on": {"nodes": ["source.p.raw.raw_orders", "model.p.customers"]}},
            "model.p.customers": {"resource_type": "model", "name": "customers", "schema": "analytics", "columns": {"id": {"name": "id"}}, "depends_on": {"nodes": []}},
            "test.p.rel": {"resource_type": "test", "test_metadata": {"name": "relationships", "kwargs": {"to": "ref('customers')", "field": "id", "column_name": "customer_id"}}, "attached_node": "model.p.orders", "depends_on": {"nodes": ["model.p.orders", "model.p.customers"]}},
        },
        "sources": {"source.p.raw.raw_orders": {"resource_type": "source", "name": "raw_orders", "source_name": "raw", "schema": "raw", "identifier": "raw_orders", "columns": {}}},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    snap = parse_manifest(DbtConfig(manifest_path=str(p)), "m")
    fq = {t.fqn for t in snap.tables}
    assert {"analytics.orders", "analytics.customers", "raw.raw_orders"} == fq
    kinds = sorted(e.kind for e in snap.edges)
    assert kinds == ["foreign_key", "lineage", "lineage", "relationship_test"]
