import sys
from dataclasses import replace

import pytest

from schemagraph.graph import build_graph
from schemagraph.linking import Linker, LinkOptions
from schemagraph.linking.lexical import activate, build_index, tokenize
from schemagraph.model import BusinessTerm, Column, Edge, SchemaSnapshot, Table


def test_tokenize_splits_snake_and_camel():
    assert tokenize("orderItems total_amount") == ["order", "items", "total", "amount"]
    assert tokenize("show me the top orders") == ["orders"]  # stopwords and query noise dropped


def test_activation_value_and_glossary(store_snapshot):
    store_snapshot.terms.append(BusinessTerm(name="revenue", synonyms=["turnover"], targets=["public.orders.total_amount"]))
    schema_graph = build_graph([store_snapshot])
    idx = build_index(schema_graph)
    act = activate(schema_graph, idx, "turnover for customers in California")
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


def test_no_match_on_small_schema_returns_everything(store_graph):
    # nothing activated but the schema fits the budget: the bypass still fires ("Death of Schema
    # Linking"), with every column, because there is no evidence to select columns by
    r = Linker(store_graph).link("zzzz qqqq")
    assert len(r.tables) == 7 and r.anchors == [] and r.join_paths == []
    assert all(len(t.columns) == len(store_graph.table(t.fqn).columns) for t in r.tables)


def test_no_match_on_large_schema_returns_empty(store_graph):
    r = Linker(store_graph).link("zzzz qqqq", LinkOptions(max_tables=5))
    assert r.tables == [] and r.anchors == []


def test_debug_ranking_limit(store_graph):
    r = Linker(store_graph).link("orders shipped by carrier", LinkOptions(debug=True, ranking_limit=2))
    assert len(r.ranking) == 2
    r = Linker(store_graph).link("orders shipped by carrier", LinkOptions(debug=True, ranking_limit=0))
    assert len(r.ranking) > 2


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
    schema_graph = build_graph([snap])
    r = Linker(schema_graph).link("alpha metrics and beta events")
    assert {t.fqn for t in r.tables} == {"alpha_metrics", "beta_events"}
    assert r.join_paths == []


def test_value_matching_is_whole_word_and_punctuation_tolerant(store_snapshot):
    store_snapshot.table("public.customer").column("city").sample_values = ["St. Louis", "New York", "iPhone City", "Bo"]
    schema_graph = build_graph([store_snapshot])
    idx = build_index(schema_graph)
    act = activate(schema_graph, idx, "orders shipped to st louis or new york")
    assert {"st. louis", "new york"} <= set(act.matched_values)
    assert act.seeds.get("c:public.customer.city", 0) >= 3.0  # two values at 1.5 each
    assert "texas" not in activate(schema_graph, idx, "texasvalues in the list").matched_values  # whole words only
    assert "california" in activate(schema_graph, idx, "customers in California!").matched_values
    assert "iphone city" in activate(schema_graph, idx, "iPhone City stores").matched_values
    assert "bo" not in idx.values  # too short to be evidence
    store_snapshot.table("public.customer").column("state").sample_values = ["A++", "10%", "$50", "1,000", "10:30", "2023/01/15", "U.S.", "E.U."]
    schema_graph = build_graph([store_snapshot])
    idx = build_index(schema_graph)
    for q in ("a list of deals", "top 10 customers", "more than 50 orders", "top 1,000 customers", "at 10:30 on 2023/01/15"):  # punctuation residue is not a value
        assert activate(schema_graph, idx, q).matched_values == [], q
    assert set(activate(schema_graph, idx, "total sales in the U.S. and the E.U.").matched_values) == {"u.s.", "e.u."}  # dotted abbreviations are evidence


def test_ngram_matches_names_containing_stopwords():
    snap = SchemaSnapshot(source="x", source_type="ddl", tables=[Table(name="person", columns=[Column(name="date_of_birth"), Column(name="first_name"), Column(name="number_of_employees"), Column(name="birth_date")])])
    schema_graph = build_graph([snap])
    idx = build_index(schema_graph)
    act = activate(schema_graph, idx, "first name and date of birth by number of employees")
    for col in ("date_of_birth", "first_name", "number_of_employees"):
        reasons = act.reasons.get(f"c:person.{col}", [])
        assert sum(r.startswith("n-gram") for r in reasons) == 1, (col, reasons)  # one piece of evidence per name
    assert not any(r.startswith("n-gram") for r in act.reasons.get("c:person.birth_date", []))
    off = activate(schema_graph, idx, "first name and date of birth", ngram_stop=False)
    assert not any(r.startswith("n-gram") for r in off.reasons.get("c:person.date_of_birth", []))


def test_explain_ranking_matches_link(store_graph):
    lk = Linker(store_graph)
    q = "total revenue by product category for customers in california"
    r = lk.link(q, LinkOptions(debug=True, ranking_limit=0))
    ex = lk.explain(q)
    assert [f for f, _ in ex["tables"]] == [f for f, _ in r.ranking][:20]


def test_bm25_and_rrf_rankers(store_graph):
    from schemagraph.linking.bm25 import bm25_scores, build_bm25, query_terms

    idx = build_bm25(store_graph)
    assert idx.n_docs == 7
    q = "shipped date and carrier for each customer"
    assert {"shipped", "carrier", "customer"} <= set(query_terms(q))
    scores = bm25_scores(idx, q)
    assert max(scores, key=scores.get) == "public.shipment"
    lk = Linker(store_graph)
    for ranker in ("bm25", "rrf"):
        r = lk.link(q, LinkOptions(ranker=ranker, debug=True, ranking_limit=0, bypass_if_fits=False))
        assert r.ranking[0][0] == "public.shipment", ranker
        assert "public.shipment" in r.anchors and any(t.fqn == "public.customer" for t in r.tables)


def test_bm25_backend_default_is_bm25f(store_graph):
    from schemagraph.linking.bm25 import bm25_scores, build_bm25

    q = "shipped date and carrier for each customer"
    lk = Linker(store_graph)
    assert lk._sparse_scores(q, LinkOptions()) == bm25_scores(build_bm25(store_graph), q)
    lk.link(q, LinkOptions(ranker="rrf", seed_bm25=True))
    assert lk._bm25 is not None and lk._bm25s == {}  # bm25s is never built unless asked for


@pytest.mark.parametrize("method", ["lucene", "bm25+"])
def test_bm25s_backend_ranks_tables(store_graph, method):
    pytest.importorskip("bm25s")
    from schemagraph.linking.bm25 import query_terms
    from schemagraph.linking.bm25s_backend import FIELD_REPEAT, BM25SIndex

    assert FIELD_REPEAT == {"name": 3, "columns": 3, "business": 3, "tags": 1, "desc": 1}
    q = "shipped date and carrier for each customer"
    idx = BM25SIndex(store_graph, method)
    scores = idx.scores(q)
    assert max(scores, key=scores.get) == "public.shipment"
    assert all(score > 0 for score in scores.values())
    # only tables holding some query term are scored (bm25+ gives every table a floor)
    held = {fqn for t in query_terms(q) for fqn in (idx.fqns[i] for i in idx.postings.get(t, []))}
    assert set(scores) <= held and len(scores) < len(idx.fqns)
    assert BM25SIndex(store_graph, method).scores(q) == scores  # deterministic
    assert idx.scores("zzzz qqqq") == {} and idx.scores("the of") == {}  # OOV / empty query
    # each weight group (surface forms 1.0, expansions such as the lemma "customer" 0.8) is one
    # get_scores call, summed with its weight
    q2 = "carrier for customers"
    groups: dict[float, list[str]] = {}
    for token, weight in query_terms(q2).items():
        if token in idx.postings:
            groups.setdefault(weight, []).append(token)
    assert set(groups) == {1.0, 0.8}
    for fqn, score in idx.scores(q2).items():
        i = idx.fqns.index(fqn)
        expected = sum(w * idx.retriever.get_scores(toks)[i] for w, toks in groups.items())
        assert score == pytest.approx(expected), fqn
    opts = LinkOptions(bm25_backend="bm25s", bm25_method=method, debug=True, ranking_limit=0,
                       bypass_if_fits=False)
    lk = Linker(store_graph)
    for ranker in ("bm25", "rrf"):
        r = lk.link(q, replace(opts, ranker=ranker))
        assert r.ranking[0][0] == "public.shipment", (method, ranker)
    assert list(lk._bm25s) == [method]  # one cached index per method
    assert lk._bm25 is None  # the ranking came from bm25s, BM25F was never built


def test_bm25s_backend_empty_vocabulary():
    pytest.importorskip("bm25s")
    from schemagraph.linking.bm25s_backend import BM25SIndex

    # one-letter names give no tokens at all; bm25s cannot index an empty vocabulary
    table = Table(name="t", columns=[Column(name="a"), Column(name="b")])
    schema_graph = build_graph([SchemaSnapshot(source="x", source_type="ddl", tables=[table])])
    assert BM25SIndex(schema_graph).scores("t a") == {}
    Linker(schema_graph).link("t a", LinkOptions(bm25_backend="bm25s"))  # no crash


def test_bm25_backend_errors(store_graph, monkeypatch):
    with pytest.raises(ValueError, match="bm25_backend"):
        Linker(store_graph).link("carrier", LinkOptions(bm25_backend="tantivy"))
    with pytest.raises(ValueError, match="bm25_method"):
        Linker(store_graph).link("carrier", LinkOptions(bm25_backend="bm25s", bm25_method="nope"))
    monkeypatch.setitem(sys.modules, "bm25s", None)
    with pytest.raises(ImportError, match="uv sync --extra bm25s"):
        Linker(store_graph).link("carrier", LinkOptions(bm25_backend="bm25s"))


def test_lineage_is_context_not_a_join_path(store_snapshot):
    from schemagraph.graph import tnode
    from schemagraph.graph.pathfinding import union_of_shortest_paths

    store_snapshot.edges.append(Edge(kind="lineage", from_table="public.audit_log", to_table="public.shipment"))
    schema_graph = build_graph([store_snapshot])
    paths, _ = union_of_shortest_paths(schema_graph, ["public.audit_log"], ["public.shipment"])
    assert paths == []  # lineage alone is not a join path
    assert not schema_graph.table_graph().has_edge(tnode("public.audit_log"), tnode("public.shipment"))
    assert schema_graph.table_graph(kinds=None).has_edge(tnode("public.audit_log"), tnode("public.shipment"))
    assert schema_graph.lineage("public.shipment") == (["public.audit_log"], []) and schema_graph.lineage("public.audit_log") == ([], ["public.shipment"])
    ddl = Linker(schema_graph).link("carrier for each shipment").ddl
    assert "built from public.audit_log" in ddl and "feeds public.shipment" in ddl
    # a foreign-key path is unaffected
    paths, _ = union_of_shortest_paths(schema_graph, ["public.customer"], ["public.shipment"])
    assert paths and [schema_graph.graph.nodes[n]["fqn"] for n in paths[0]] == ["public.customer", "public.orders", "public.order_items", "public.shipment"]


def test_embedding_activator_seeds_paraphrases(store_graph, embed_model):
    from schemagraph.linking.embed import EmbeddingActivator, phrases, question_part

    assert question_part("q\n\n" + "d" * 500) == "q" and question_part("short\n\nquestion") == "short\n\nquestion"
    assert "shipped date" in phrases("shipped date by carrier")
    emb = EmbeddingActivator(store_graph)
    assert EmbeddingActivator(store_graph).model is emb.model  # loaded once, shared across rebuilds
    seeds = dict((n, (w, why)) for n, w, why in emb.activate("which delivery company shipped each order"))
    assert "c:public.shipment.carrier" in seeds or "t:public.shipment" in seeds
    assert all(w <= 0.8 for w, _ in seeds.values())
    lk = Linker(store_graph)
    r = lk.link("which delivery company shipped each order", LinkOptions(embed=True, debug=True, ranking_limit=0))
    assert any(t.fqn == "public.shipment" for t in r.tables) and r.ranking


def test_rank_tiered_column_cap():
    wide = Table(name="wide", columns=[Column(name="id", is_primary_key=True)] + [Column(name=f"metric_{i}") for i in range(30)])
    other = Table(name="narrow", columns=[Column(name="id", is_primary_key=True)] + [Column(name=f"attr_{i}") for i in range(30)])
    schema_graph = build_graph([SchemaSnapshot(source="x", source_type="ddl", tables=[wide, other])])
    lk = Linker(schema_graph)
    r = lk.link("wide metric", LinkOptions(max_columns_per_table=5, columns_top_uncapped=1, debug=True, ranking_limit=0))
    by = {t.fqn: t for t in r.tables}
    assert r.ranking[0][0] == "wide"
    assert len(by["wide"].columns) == 31  # rank 1 keeps every column
    assert len(by["narrow"].columns) == 5 and by["narrow"].columns[0].name == "id"  # capped, keys first
    r0 = lk.link("wide metric", LinkOptions(max_columns_per_table=5, columns_top_uncapped=0))
    assert all(len(t.columns) == 5 for t in r0.tables)  # columns_top_uncapped=0: the cap applies to every table
    r1 = lk.link("wide metric", LinkOptions(max_columns_per_table=5))
    assert len(next(t for t in r1.tables if t.fqn == "wide").columns) == 31  # default keeps the top-ranked table whole


def test_load_model_uses_cache_first_and_heals_a_broken_cache(monkeypatch):
    import pytest

    pytest.importorskip("model2vec")
    from model2vec import StaticModel

    from schemagraph.linking import embed

    calls = []

    def fake(name, force_download=True):
        calls.append(force_download)
        if not force_download and name == "broken/model":
            raise FileNotFoundError("Could not find expected model files")
        return object()

    from model2vec.persistence import hf

    monkeypatch.setattr(StaticModel, "from_pretrained", staticmethod(fake))
    monkeypatch.setattr(hf, "maybe_get_cached_model_path", lambda name: None if name == "missing/model" else "snapshot")
    embed.load_model.cache_clear()
    try:
        embed.load_model("good/model")
        assert calls == [False]  # cached model: the Hub is never asked
        calls.clear()
        embed.load_model("broken/model")
        assert calls == [False, True]  # partial snapshot: falls back to a real download
        calls.clear()

        def offline(name, force_download=True):
            calls.append(force_download)
            raise OSError("hub unreachable")

        monkeypatch.setattr(StaticModel, "from_pretrained", staticmethod(offline))
        with pytest.raises(OSError):
            embed.load_model("missing/model")
        assert calls == [False]  # nothing cached: the Hub is not asked a second time
    finally:
        embed.load_model.cache_clear()
