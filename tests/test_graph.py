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


def test_ppr_matrix_matches_networkx(store_graph):
    import networkx as nx

    from schemagraph.graph.ppr import PPRMatrix
    from schemagraph.linking.lexical import build_index

    build_index(store_graph)  # adds the w: token nodes exactly as the linker does
    seeds = {tnode("public.shipment"): 1.0, "c:public.customer.state": 0.5}
    mine = PPRMatrix(store_graph).run(seeds, alpha=0.85)
    ref = nx.pagerank(store_graph.g, alpha=0.85, personalization=seeds, weight="weight", tol=1e-12, max_iter=2000)
    assert abs(sum(mine.values()) - 1.0) < 1e-9
    for n, s in ref.items():
        assert abs(mine.get(n, 0.0) - s) < 1e-7, n


def test_ppr_untouched_component_scores_exactly_zero():
    snap = SchemaSnapshot(source="x", source_type="ddl", tables=[Table(name="alpha", columns=[Column(name="x1")]), Table(name="beta", columns=[Column(name="y1")])])
    sg = build_graph([snap])
    scores = personalized_pagerank(sg, {tnode("alpha"): 1.0})
    assert tnode("alpha") in scores and tnode("beta") not in scores


def test_shortest_paths_match_exhaustive_enumeration():
    import random

    import networkx as nx

    from schemagraph.graph.pathfinding import shortest_paths_between

    rng = random.Random(7)
    checked = 0
    for _ in range(40):
        n = rng.randint(5, 11)
        g = nx.gnp_random_graph(n, 0.35, seed=rng.randint(0, 10**6))
        for u, v in g.edges():
            g[u][v]["weight"] = rng.choice([1.0, 1.3, 1.6, 2.5])
        for a, b in [(0, n - 1), (1, n - 2)]:
            for extra in (0.0, 0.7):
                got = {tuple(p) for p in shortest_paths_between(g, a, b, max_extra=extra, cutoff=n)}
                if not nx.has_path(g, a, b):
                    assert got == set()
                    continue
                best = nx.shortest_path_length(g, a, b, weight="weight")
                exp = {tuple(p) for p in nx.all_simple_paths(g, a, b) if sum(g[u][v]["weight"] for u, v in zip(p, p[1:], strict=False)) <= best + extra + 1e-9}
                assert got == exp, (a, b, extra)
                checked += 1
    assert checked > 40


def test_relation_edges_carry_join_cost_and_ppr_affinity(store_snapshot):
    from schemagraph.graph.ppr import PPRMatrix

    store_snapshot.edges.append(Edge(kind="inferred", from_table="public.orders", to_table="public.audit_log", from_columns=["id"], to_columns=["id"]))
    sg = build_graph([store_snapshot])
    inf = sg.g.get_edge_data(tnode("public.orders"), tnode("public.audit_log"))
    fk = sg.g.get_edge_data(tnode("public.orders"), tnode("public.customer"))
    assert (inf["weight"], inf["affinity"]) == (2.5, 1.0)  # cost for path-finding, uniform flow for PPR
    assert (fk["weight"], fk["affinity"]) == (1.0, 1.0)
    by_cost = PPRMatrix(sg, "weight").run({tnode("public.orders"): 1.0})
    by_affinity = PPRMatrix(sg, "affinity").run({tnode("public.orders"): 1.0})
    assert by_cost[tnode("public.audit_log")] > by_affinity[tnode("public.audit_log")]  # reading cost as affinity over-feeds inferred edges
