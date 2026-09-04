from schemagraph.graph import build_graph
from schemagraph.linking import Linker, LinkOptions
from schemagraph.linking.lexical import activate, build_index, tokenize
from schemagraph.model import BusinessTerm, SchemaSnapshot, Table


def test_tokenize_splits_snake_and_camel():
    assert tokenize("orderItems total_amount") == ["order", "items", "total", "amount"]
    assert tokenize("show me the top orders") == ["orders"]  # stopwords and query noise dropped


def test_activation_value_and_glossary(store_snapshot):
    store_snapshot.terms.append(BusinessTerm(name="revenue", synonyms=["turnover"], targets=["public.orders.total_amount"]))
    sg = build_graph([store_snapshot])
    idx = build_index(sg)
    act = activate(sg, idx, "turnover for customers in California")
    assert "turnover" in act.matched_terms
    assert "california" in act.matched_values
    assert act.seeds.get("c:public.customer.state", 0) > 1.0
    assert act.seeds.get("c:public.orders.total_amount", 0) > 1.0


def test_link_store_question(store_graph):
    r = Linker(store_graph).link("total revenue and qty by product category for recent orders from customers in california", LinkOptions(bypass_if_fits=False))
    fqns = [t.fqn for t in r.tables]
    assert "public.customer" in r.anchors
    assert "public.product_category" in r.anchors
    assert "public.order_items" in fqns and "public.products" in fqns  # bridge tables
    assert "public.audit_log" not in fqns  # unrelated noise stays out
    longest = max(r.join_paths, key=lambda p: len(p.tables))
    assert set(longest.tables) >= {"public.customer", "public.orders", "public.order_items", "public.products", "public.product_category"}
    assert "CREATE TABLE public.orders" in r.ddl and "Join paths" in r.ddl
    cust = next(t for t in r.tables if t.fqn == "public.customer")
    assert any(c.name == "state" and c.reason and "value" in c.reason for c in cust.columns)


def test_link_keeps_join_keys_on_bridge_tables(store_graph):
    r = Linker(store_graph).link("carrier for each customer", LinkOptions(anchor_k=2))
    oi = next((t for t in r.tables if t.fqn == "public.order_items"), None)
    assert oi is not None
    names = {c.name for c in oi.columns}
    assert {"id", "order_id"} <= names  # PK + join key kept even though not mentioned


def test_small_schema_bypass(store_graph):
    r = Linker(store_graph).link("anything", LinkOptions(small_schema_bypass=10))
    assert len(r.tables) == 7 and r.stats.get("bypass")


def test_bypass_if_fits_keeps_anchors_and_paths(store_graph):
    r = Linker(store_graph).link("orders shipped by carrier for customers in texas")
    assert len(r.tables) == 7  # 7 tables <= max_tables 12: everything returned
    assert r.anchors and r.join_paths  # but the ranking/paths are still computed for the DDL
    r2 = Linker(store_graph).link("orders shipped by carrier for customers in texas", LinkOptions(max_tables=5))
    assert len(r2.tables) <= 5


def test_no_match_returns_empty(store_graph):
    r = Linker(store_graph).link("zzzz qqqq")
    assert r.tables == [] and r.anchors == []  # nothing activated: bypass does not fire either


def test_llm_anchor_picker_is_used_when_enabled(store_graph):
    class FakeLLM:
        def anchor_tables(self, question, candidates):
            return ["public.customer"], ["public.shipment"]

    r = Linker(store_graph, llm=FakeLLM()).link("shipments per customer", LinkOptions(use_llm=True))
    assert r.anchors == ["public.customer", "public.shipment"]
    assert r.stats["llm"] == "yes"
    assert any(set(p.tables) >= {"public.customer", "public.orders", "public.order_items", "public.shipment"} for p in r.join_paths)


def test_disconnected_anchors_still_returned():
    snap = SchemaSnapshot(source="x", source_type="ddl", tables=[Table(name="alpha_metrics"), Table(name="beta_events")])
    sg = build_graph([snap])
    r = Linker(sg).link("alpha metrics and beta events")
    assert {t.fqn for t in r.tables} == {"alpha_metrics", "beta_events"}
    assert r.join_paths == []
