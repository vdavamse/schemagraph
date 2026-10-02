"""Plan work estimates and the cost gate's refusal, on fixed plan text (no database).

Core dependencies only: runs without the ``agent`` extra.
"""

from __future__ import annotations

import json

import pytest

from schemagraph.agent.cost import (
    DUCKDB_MAX_JOIN_ROWS,
    SQLITE_MAX_LOOP_ROWS,
    SqliteNames,
    cost_refusal,
    duckdb_plan_cost,
    sqlite_plan_cost,
)
from schemagraph.agent.results import PlanInfo


def _scan(table: str, rows: int) -> dict:
    return {
        "name": "SEQ_SCAN",
        "children": [],
        "extra_info": {"Table": table, "Estimated Cardinality": str(rows)},
    }


def _node(name: str, children: list[dict], **extra) -> dict:
    return {"name": name, "children": children, "extra_info": extra}


# ------------------------------------------------------------------------------------- DuckDB


def test_duckdb_hash_join_cost_is_its_estimate_with_the_condition():
    join = _node(
        "HASH_JOIN",
        [_scan("big", 2_000_000), _scan("big", 2_000_000)],
        **{"Conditions": "k = k", "Estimated Cardinality": "571428571428"},
    )
    plan = json.dumps([_node("PROJECTION", [join], **{"Estimated Cardinality": "571428571428"})])
    rows, reason = duckdb_plan_cost(plan)
    assert rows == pytest.approx(5.71428571428e11)
    assert reason.startswith("HASH_JOIN estimated to emit 5.7e+11 rows") and "(k = k)" in reason


def test_duckdb_cross_product_without_an_estimate_multiplies_its_inputs():
    cross = _node("CROSS_PRODUCT", [_scan("big", 2_000_000), _scan("small", 1000)])
    rows, reason = duckdb_plan_cost(json.dumps([cross]))
    assert rows == 2e9
    assert reason == "CROSS_PRODUCT estimated to compare 2.0e+09 rows (no join condition)"


def test_duckdb_worst_join_wins_and_derived_rows_flow_upward():
    inner = _node(
        "HASH_JOIN",
        [_scan("a", 1000), _scan("b", 1000)],
        **{"Conditions": ["x = x", "y = y"], "Estimated Cardinality": "1000"},
    )
    ordered = _node("ORDER_BY", [inner])  # no estimate: the largest input
    loop = _node("NESTED_LOOP_JOIN", [ordered, _scan("c", 500)], **{"Conditions": "p < q"})
    rows, reason = duckdb_plan_cost(json.dumps(loop))  # a single root object is accepted too
    assert rows == 500_000 and reason.startswith("NESTED_LOOP_JOIN estimated to compare")


def test_duckdb_scans_and_aggregates_alone_cost_nothing():
    aggregate = _node(
        "HASH_GROUP_BY", [_scan("big", 10**12)], **{"Estimated Cardinality": str(10**12)}
    )
    assert duckdb_plan_cost(json.dumps([aggregate])) == (0.0, "")


def test_duckdb_inequality_join_emits_at_least_a_third_of_its_input_pairs():
    # a.id < b.id over 200k rows each: DuckDB estimates 1.4e7 rows, the join emits 2e10
    join = _node(
        "PIECEWISE_MERGE_JOIN",
        [_scan("big", 200_000), _scan("big", 200_000)],
        **{"Conditions": "id < id", "Estimated Cardinality": "13861042"},
    )
    rows, reason = duckdb_plan_cost(json.dumps([_node("UNGROUPED_AGGREGATE", [join])]))
    assert rows == pytest.approx(200_000 * 200_000 / 3)
    assert rows > DUCKDB_MAX_JOIN_ROWS
    assert reason == "PIECEWISE_MERGE_JOIN estimated to emit 1.3e+10 rows (id < id)"
    selective = _node(
        "IE_JOIN",
        [_scan("a", 30), _scan("b", 30)],
        **{"Conditions": ["x < x", "y > y"], "Estimated Cardinality": "600"},
    )
    assert duckdb_plan_cost(json.dumps([selective]))[0] == 600  # its own estimate is larger


def test_duckdb_blockwise_join_reads_its_singular_condition():
    join = _node(
        "BLOCKWISE_NL_JOIN", [_scan("a", 10), _scan("b", 20)], Condition="((id + id) = 3)"
    )
    rows, reason = duckdb_plan_cost(json.dumps([join]))
    assert rows == 200 and reason.endswith("(((id + id) = 3))")


@pytest.mark.parametrize("text", ["", "not json", "42", json.dumps([{"children": 3}])])
def test_duckdb_unreadable_plans_fail_open(text):
    assert duckdb_plan_cost(text) == (None, "")


# ------------------------------------------------------------------------------------- SQLite

SIZES = {"a": 1000, "b": 20_000, "c": 500}


def _table_rows(name: str) -> int | None:
    return SIZES.get(name.lower())


NAMES = SqliteNames(table_rows=_table_rows)


def test_sqlite_nested_scans_multiply():
    rows = [(3, 0, 0, "SCAN a"), (5, 0, 0, "SCAN b")]
    work, reason = sqlite_plan_cost(rows, NAMES)
    assert work == 1000 + 1000 * 20_000
    assert reason == "nested loop: SCAN a (1,000 rows) × SCAN b (20,000 rows)"


def test_sqlite_indexed_search_and_automatic_index():
    rows = [
        (3, 0, 0, "SCAN a"),
        (7, 0, 0, "BLOOM FILTER ON b (g=?)"),
        (17, 0, 0, "SEARCH b USING AUTOMATIC COVERING INDEX (g=?)"),
        (20, 0, 0, "USE TEMP B-TREE FOR ORDER BY"),
    ]
    work, _ = sqlite_plan_cost(rows, NAMES)
    # scan a, build the index over b, one lookup per row of a, sort the rows of a
    assert work == 1000 + 20_000 + 1000 + 1000


def test_sqlite_correlated_subquery_runs_per_outer_row():
    rows = [(2, 0, 0, "SCAN a"), (6, 0, 0, "CORRELATED SCALAR SUBQUERY 1"), (11, 6, 0, "SCAN b")]
    work, reason = sqlite_plan_cost(rows, NAMES)
    assert work == 1000 + 1000 * 20_000
    assert reason == "correlated subquery run 1.0e+03 times: SCAN b (20,000 rows)"


def test_sqlite_sub_plans_run_once_and_unknown_tables_count_as_one_row():
    rows = [
        (2, 0, 0, "MERGE (UNION)"),
        (4, 2, 0, "LEFT"),
        (7, 4, 0, "SCAN a"),
        (22, 2, 0, "RIGHT"),
        (25, 22, 0, "SCAN c"),
        (30, 0, 0, "SCAN cte_alias"),  # unknown size: a factor of one
        (31, 0, 0, "SCAN CONSTANT ROW"),
    ]
    work, reason = sqlite_plan_cost(rows, NAMES)
    assert work == 1000 + 500 + 1
    assert reason == "SCAN a (1,000 rows)"


def test_sqlite_materialised_results_are_sized_by_the_scans_that_fill_them():
    # select count(*) from (select k, count(*) from b group by id) x, (... ) y
    rows = [
        (2, 0, 0, "CO-ROUTINE x"),
        (8, 2, 0, "SCAN b"),
        (10, 2, 0, "USE TEMP B-TREE FOR GROUP BY"),
        (47, 0, 0, "MATERIALIZE y"),
        (54, 47, 0, "SCAN b"),
        (56, 47, 0, "USE TEMP B-TREE FOR GROUP BY"),
        (95, 0, 0, "SCAN x"),
        (100, 0, 0, "SCAN y"),
    ]
    work, reason = sqlite_plan_cost(rows, NAMES)
    fill = 20_000 + 20_000  # scan b and group it, per subquery
    assert work == 2 * fill + 20_000 + 20_000 * 20_000
    assert reason == "nested loop: SCAN x (20,000 rows) × SCAN y (20,000 rows)"


def test_sqlite_a_cte_scanned_under_two_aliases_multiplies_by_its_size_each_time():
    # with c as materialized (select * from a) select count(*) from c c1, c c2
    rows = [(3, 0, 0, "MATERIALIZE c"), (6, 3, 0, "SCAN a"), (20, 0, 0, "SCAN c1"),
            (22, 0, 0, "SCAN c2")]  # fmt: skip
    names = SqliteNames(table_rows=_table_rows, source={"c1": "c", "c2": "c"}.get)
    work, reason = sqlite_plan_cost(rows, names)
    assert work == 1000 + 1000 + 1000 * 1000
    assert reason == "nested loop: SCAN c1 (1,000 rows) × SCAN c2 (1,000 rows)"


def test_sqlite_a_bound_in_the_sql_caps_a_materialised_result():
    # hi: ... LIMIT 1; mx: select max(x) from b (one row); both cross-joined with b
    rows = [
        (2, 0, 0, "MATERIALIZE hi"),
        (5, 2, 0, "SCAN b"),
        (9, 0, 0, "MATERIALIZE mx"),
        (12, 9, 0, "SCAN b"),
        (20, 0, 0, "SCAN hi"),
        (22, 0, 0, "SCAN mx"),
        (24, 0, 0, "SCAN b"),
    ]
    names = SqliteNames(table_rows=_table_rows, max_rows={"hi": 1, "mx": 1}.get)
    work, _ = sqlite_plan_cost(rows, names)
    assert work == 20_000 + 20_000 + 1 + 1 + 20_000


def test_sqlite_flattened_cte_scans_its_base_table():
    # with c as (select * from b) select count(*) from c c1, c c2: SQLite inlines c
    work, _ = sqlite_plan_cost([(4, 0, 0, "SCAN b"), (6, 0, 0, "SCAN b")], NAMES)
    assert work == 20_000 + 20_000 * 20_000


def test_sqlite_plan_before_3_36_has_no_estimate():
    old_format = [(3, 0, 0, "SCAN TABLE a"), (5, 0, 0, "SCAN TABLE b")]
    assert sqlite_plan_cost(old_format, NAMES) == (None, "")


def test_sqlite_empty_plan_costs_nothing():
    assert sqlite_plan_cost([], NAMES) == (0.0, "")


# ------------------------------------------------------------------------------------ refusal


def test_cost_refusal_names_the_operator_and_fails_open():
    heavy = PlanInfo(ok=True, rows=2e11, reason="nested loop: SCAN a × SCAN b")
    message = cost_refusal(heavy, SQLITE_MAX_LOOP_ROWS)
    assert message is not None
    assert message.startswith("the plan would process about 2.0e+11 rows (nested loop:")
    assert "add join conditions or filters" in message
    assert cost_refusal(PlanInfo(ok=True, rows=1e10), DUCKDB_MAX_JOIN_ROWS) is None  # at the cap
    assert cost_refusal(PlanInfo(ok=True, rows=None), 1.0) is None  # no estimate
    assert cost_refusal(PlanInfo(ok=False, rows=1e20, error="x"), 1.0) is None  # failed plan
