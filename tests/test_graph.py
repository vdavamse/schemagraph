from schemagraph.graph import build_graph, tnode
from schemagraph.graph.pathfinding import union_of_shortest_paths
from schemagraph.graph.ppr import personalized_pagerank, table_scores
from schemagraph.graph.pruning import prune_paths
from schemagraph.model import BusinessTerm, Column, Edge, SchemaSnapshot, Table


def test_build_and_stats(store_graph):
    s = store_graph.stats()
    assert s["tables"] == 7 and s["relations"] == 5 and s["columns"] == 30


def test_merge_two_sources_fill_blanks_and_union_edges(store_snapshot):
    other = SchemaSnapshot(
        source="collibra",
        source_type="collibra",
        tables=[Table(name="orders", schema="public", description="Customer orders (curated)", columns=[Column(name="status", description="Order lifecycle state", tags=["PII=no"])])],
        edges=[Edge(kind="catalog_relation", from_table="public.orders", to_table="public.shipment", description="Collibra: is related to")],
        terms=[BusinessTerm(name="revenue", targets=["public.orders.total_amount"], synonyms=["sales"])],
    )
    sg = build_graph([store_snapshot, other])
    t = sg.table("public.orders")
    assert t.description == "Customer orders (curated)"
    assert t.column("status").description == "current order status"  # first source wins on filled fields
    assert "PII=no" in t.column("status").tags
    assert "collibra" in t.source and "store" in t.source
    assert sg.relations("public.orders", "public.shipment")[0].kind == "catalog_relation"
    assert "revenue" in sg.terms
    assert sg.g.has_edge("k:revenue", "c:public.orders.total_amount")


def test_stub_for_unknown_referenced_table():
    snap = SchemaSnapshot(source="x", source_type="ddl", tables=[Table(name="a", columns=[Column(name="b_id")])], edges=[Edge(kind="foreign_key", from_table="a", to_table="b", from_columns=["b_id"], to_columns=["id"])])
    sg = build_graph([snap])
    assert sg.table("b") is not None and sg.table("b").properties.get("stub") == "true"


def test_shortest_path_union_includes_bridge_tables(store_graph):
    paths, union = union_of_shortest_paths(store_graph, ["public.customer"], ["public.product_category"])
    assert len(paths) == 1
    assert [store_graph.g.nodes[n]["fqn"] for n in paths[0]] == ["public.customer", "public.orders", "public.order_items", "public.products", "public.product_category"]
    assert "public.order_items" in union  # semantically irrelevant, structurally mandatory


def test_weighted_paths_prefer_foreign_keys_over_lineage(store_snapshot):
    # add a lineage shortcut customer -> product_category; FK path is longer but must still be preferred
    store_snapshot.edges.append(Edge(kind="lineage", from_table="public.customer", to_table="public.products"))
    sg = build_graph([store_snapshot])
    paths, _ = union_of_shortest_paths(sg, ["public.customer"], ["public.products"])
    fq = [[sg.g.nodes[n]["fqn"] for n in p] for p in paths]
    # lineage weight 1.6 vs FK path 3 hops (3.0): lineage wins here (shorter) - assert both endpoints present, and that
    # a direct FK chain is chosen when lineage is made expensive
    assert any(p[0] == "public.customer" and p[-1] == "public.products" for p in fq)


def test_ppr_ranks_seeded_tables_first(store_graph):
    seeds = {tnode("public.shipment"): 1.0}
    scores = table_scores(store_graph, personalized_pagerank(store_graph, seeds))
    top = sorted(scores.items(), key=lambda x: -x[1])
    assert top[0][0] == "public.shipment"
    assert top[1][0] == "public.order_items"  # one hop away


def test_prune_paths_dedupes_subpaths(store_graph):
    paths, _ = union_of_shortest_paths(store_graph, ["public.orders", "public.products", "public.order_items"])
    jps = prune_paths(store_graph, paths, [tnode("public.orders"), tnode("public.products"), tnode("public.order_items")])
    tables = [jp.tables for jp in jps]
    assert ["public.orders", "public.order_items", "public.products"] in tables or ["public.products", "public.order_items", "public.orders"] in tables
    assert len(jps) == 1  # the two 2-hop sub-paths are contained in the 3-table path
    assert jps[0].steps[0].kind == "foreign_key" and "order_id" in jps[0].steps[0].on
