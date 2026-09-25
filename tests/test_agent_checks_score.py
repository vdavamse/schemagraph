"""Deterministic checks and the score formula (core dependencies only)."""

from __future__ import annotations

import pytest

from schemagraph.agent.checks import result_checks, static_checks
from schemagraph.agent.execute import DuckDBExecutor, SQLiteExecutor
from schemagraph.agent.guard import guard_sql
from schemagraph.agent.results import (
    CheckReport,
    ExecResult,
    Finding,
    Judgement,
    RubricBase,
    rubric_type,
)
from schemagraph.agent.score import combine, feedback
from schemagraph.connectors.duckdb_conn import DuckDBConfig, introspect_duckdb
from schemagraph.graph.build import JOIN_KINDS, SchemaGraph, build_graph
from schemagraph.graph.pathfinding import union_of_shortest_paths
from schemagraph.model import Edge


class GraphLookups:
    """The plain lookups ``static_checks`` takes, backed by a schema graph."""

    def __init__(self, schema_graph: SchemaGraph):
        self.schema_graph = schema_graph

    def join_relations(self, a: str, b: str) -> list[Edge]:
        relations = self.schema_graph.relations(a, b)
        return [edge for edge in relations if edge.kind in JOIN_KINDS]

    def join_path(self, a: str, b: str) -> list[list[str]]:
        paths, _ = union_of_shortest_paths(self.schema_graph, [a], [b])
        nodes = self.schema_graph.graph.nodes
        return [[nodes[node]["fqn"] for node in path] for path in paths]


@pytest.fixture
def lookups(store_duckdb):
    snap = introspect_duckdb(DuckDBConfig(path=str(store_duckdb)), "db")
    return GraphLookups(build_graph([snap]))


@pytest.fixture
def duck(store_duckdb, lookups):
    """The store database; introspected first, as DuckDB refuses a second, differently configured
    connection to an open file."""
    executor = DuckDBExecutor(store_duckdb)
    yield executor
    executor.close()


@pytest.fixture
def check(duck, lookups):
    """Run the static checks on DuckDB SQL against the store database and its graph."""

    def run(sql: str) -> CheckReport:
        return static_checks(
            guard_sql(sql, "duckdb"),
            duck.catalog(),
            duck.resolve_table,
            lookups.join_relations,
            lookups.join_path,
        )

    return run


def codes(report):
    return [finding.code for finding in report.findings]


CLEAN = [
    "select pc.name, sum(o.total_amount) from orders o join customer c on o.customer_id = c.id "
    "join order_items oi on oi.order_id = o.id join products p on p.id = oi.product_id "
    "join product_category pc on pc.id = p.category_id where c.state = 'CA' "
    "group by pc.name order by 2 desc",
    "select state, count(*) as n from customer group by state order by n desc",
    "with t as (select customer_id, sum(total_amount) total from orders group by customer_id) "
    "select c.name, t.total from t join customer c on c.id = t.customer_id",
    "select * from orders join order_items using (id)",
    "select id, sum(total_amount) over (order by id) from orders",
    "select x.total from (select sum(total_amount) as total from orders) x",
    "select c.name from orders o, customer c where o.customer_id = c.id",
    "select c.name from customer c where exists "
    "(select 1 from orders o where o.customer_id = c.id)",
    "select o.id, t.m from orders o cross join (select max(total_amount) m from orders) t",
    # an alias reused in a subquery
    "select o.id from orders t, orders o where o.id = t.id "
    "and exists (select 1 from customer t where t.state = 'CA')",
    # percent of total
    "select total_amount * 1.0 / (select sum(total_amount) from orders) from orders",
    "select c.name, (select count(*) from orders o where o.customer_id = c.id) from customer c",
    "select state, count(*) filter (where name like 'A%') from customer group by state",
    "select state, name, count(*) from customer group by all",
    "select state, count(*) from customer group by rollup (state)",
]


@pytest.mark.parametrize("sql", CLEAN)
def test_correct_queries_have_no_penalty(check, sql):
    report = check(sql)
    assert report.det == 1.0, report.findings


def test_unknown_names_with_suggestions(check):
    report = check("select o.cust_id from orders o")
    assert codes(report) == ["unknown_column"]
    assert "customer_id" in report.findings[0].message
    assert report.det == pytest.approx(0.7)

    assert codes(check("select totl from orders")) == ["unknown_column"]

    report = check("select * from ordrs")
    assert codes(report) == ["unknown_table"]
    assert "`orders`" in report.findings[0].message
    assert report.tables == []


def test_cartesian_and_join_off_graph(check):
    assert codes(check("select * from orders, customer")) == ["cartesian"]
    report = check("select * from orders o join customer c on o.id = c.id")
    assert codes(report) == ["join_off_graph"]
    assert report.det == 1.0  # reported, not penalised until measured
    assert "customer_id" in report.findings[0].message


def test_join_without_a_relation_suggests_the_shortest_path(duck, lookups):
    guarded = guard_sql("select * from orders o join products p on o.id = p.id", "duckdb")
    report = static_checks(
        guarded,
        duck.catalog(),
        duck.resolve_table,
        lookups.join_relations,
        lookups.join_path,
    )
    assert codes(report) == ["join_off_graph"]
    assert report.findings[0].message == (
        "no known relation between `main.orders` and `main.products`; "
        "shortest known path: main.orders -> main.order_items -> main.products"
    )

    without_path = static_checks(
        guarded,
        duck.catalog(),
        duck.resolve_table,
        lookups.join_relations,
    )
    assert without_path.findings[0].message == (
        "no known relation between `main.orders` and `main.products`"
    )


def test_join_check_is_off_without_relations(duck):
    guarded = guard_sql("select * from orders o join customer c on o.id = c.id", "duckdb")
    assert static_checks(guarded, duck.catalog(), duck.resolve_table).findings == []


def test_ungrouped_column(check):
    report = check("select c.name, c.state, count(*) from customer c group by c.state")
    assert codes(report) == ["ungrouped_column"]
    assert "c.name" in report.findings[0].message


def test_result_checks(duck):
    guarded = guard_sql("select * from orders o, order_items i, customer", "duckdb")
    result = duck.execute(guarded.sql)
    base = {table: duck.row_count(table) for table in ("orders", "order_items", "customer")}
    exploded = result_checks(result, base, count_cap=100_000, guarded=guarded)
    assert codes(CheckReport(parsed=True, findings=exploded)) == ["row_explosion"]

    empty = duck.execute("select * from orders where id < 0")
    assert [f.code for f in result_checks(empty, {}, count_cap=100)] == ["empty_result"]

    nulls = ExecResult(ok=True, columns=["a", "b"], rows=[[1, None], [2, None]], row_count=2)
    assert [f.code for f in result_checks(nulls, {}, count_cap=100)] == ["null_column"]

    timeout = ExecResult(ok=False, error="x", error_kind="timeout")
    assert [f.code for f in result_checks(timeout, {}, count_cap=1)] == ["timeout"]

    recursive = guard_sql(
        "with recursive r(n) as (select 1 union all select n + 1 from r where n < 100) "
        "select * from r",
        "duckdb",
    )
    generated = duck.execute(recursive.sql)
    assert result_checks(generated, {"orders": 3}, count_cap=100_000, guarded=recursive) == []


def test_a_capped_count_is_an_explosion_only_below_the_cap(duck):
    guarded = guard_sql("select * from orders o, order_items i, customer", "duckdb")
    capped = ExecResult(ok=True, columns=["a"], rows=[[1]], row_count=10, row_count_capped=True)
    below = result_checks(capped, {"orders": 3}, count_cap=10, guarded=guarded)
    assert [f.message for f in below] == [
        "10+ rows, more than 2x the largest table read (3); a join is probably missing a key"
    ]
    # the limit (2 x 4) reaches the count cap, so the capped count says nothing
    assert result_checks(capped, {"orders": 4}, count_cap=8, guarded=guarded) == []


def judgement(mean, missing=None):
    return Judgement(
        model="m",
        fields=dict.fromkeys(RubricBase.model_fields, mean),
        missing=missing or {},
        mean=mean,
    )


def test_combine():
    ok = ExecResult(ok=True, row_count=1)
    clean = CheckReport(parsed=True)
    assert combine(CheckReport(parsed=False), None, None)[0] == 0.0
    assert combine(clean, ExecResult(ok=False, error_kind="guard"), None)[0] == 0.0
    assert combine(clean, ExecResult(ok=False, error_kind="runtime"), None)[0] == 0.05
    assert combine(clean, ok, judgement(1.0))[0] == pytest.approx(1.0)
    # the early-stop line
    assert combine(clean, ok, judgement(0.804))[0] >= 0.9 > combine(clean, ok, judgement(0.8))[0]
    warning = Finding(code="empty_result", severity="warn", message="m", penalty=0.3)
    warned = CheckReport(parsed=True, findings=[warning], det=0.7)
    assert combine(warned, ok, judgement(1.0))[0] < 0.9  # one warning cannot early-stop
    assert combine(clean, ok, None)[0] == pytest.approx(1.0)  # no judge: det alone
    for det in (0.0, 0.5, 1.0):
        for mean in (0.0, 0.5, 1.0):
            reward, _ = combine(CheckReport(parsed=True, det=det), ok, judgement(mean))
            assert 0.15 <= reward <= 1.0  # anything that executed beats a failure


def test_combine_parts():
    ok = ExecResult(ok=True, row_count=1)
    reward, parts = combine(CheckReport(parsed=True, det=0.5), ok, judgement(1.0))
    assert parts == {"det": 0.5, "judge": 1.0, "x": pytest.approx(0.4 * 0.5 + 0.6)}
    assert reward == pytest.approx(0.15 + 0.85 * parts["x"])
    failed = ExecResult(ok=False, error_kind="runtime")
    assert combine(CheckReport(parsed=True, det=0.5), failed, None)[1] == {
        "det": 0.5,
        "judge": None,
        "x": 0.0,
    }


def test_feedback_order():
    report = CheckReport(
        parsed=True,
        findings=[
            Finding(code="join_off_graph", severity="info", message="info"),
            Finding(code="unknown_column", severity="error", message="err"),
        ],
    )
    judged = judgement(0.9, {"main.products": 0.8, "main.x": 0.2})
    judged.fields["right_filters"] = 0.2
    lines = feedback(report, judged)
    assert lines[:2] == ["err", "info"]
    assert "reviewer doubts" in lines[2]
    assert lines[3] == "may need table(s): main.products (p=0.80)"


def test_rubric_type():
    # Jev: a Literal needs 2 options
    assert rubric_type(()) is RubricBase
    assert rubric_type(("a",)) is RubricBase
    many = rubric_type(tuple(f"t{i}" for i in range(300)))
    properties = many.model_json_schema()["properties"]
    assert len(properties["missing"]["items"]["enum"]) == 255
    assert properties["answers_question"]["maximum"] == 1
    assert properties["answers_question"]["minimum"] == 0
    assert rubric_type(("a", "b")) is rubric_type(("a", "b"))


def test_sqlite_specifics(store_sqlite):
    lite = SQLiteExecutor(store_sqlite)

    def check_as(sql, dialect):
        return static_checks(guard_sql(sql, dialect), lite.catalog(), lite.resolve_table)

    implicit = [
        "select c.name from main.customer c",
        "select rowid, name from customer",
        "select c.rowid from customer c",
    ]
    for sql in implicit:
        report = check_as(sql, "sqlite")
        assert report.det == 1.0, (sql, report.findings)
    # SQLite's bare-column rule with one max(); elsewhere it is ambiguous
    bare = "select state, max(id), name from customer group by state"
    assert check_as(bare, "sqlite").det == 1.0
    assert codes(check_as(bare, "duckdb")) == ["ungrouped_column"]
    # double-quoted literals are unknown columns in SQLite, capped at 0.6 per code
    many = 'select "a", "b", "c", "d", "e" from customer'
    report = check_as(many, "sqlite")
    assert codes(report).count("unknown_column") == 5
    assert report.det == pytest.approx(0.4)
    assert "silently becomes a string" in report.findings[0].message
    lite.close()
