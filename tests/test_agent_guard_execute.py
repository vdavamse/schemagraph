"""Read-only execution: the guard, the engine statement check and the hardened connections.

Core dependencies only (sqlglot, duckdb, sqlite3): runs without the ``agent`` extra.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest
import sqlglot

from schemagraph.agent import execute
from schemagraph.agent.checks import static_checks
from schemagraph.agent.execute import (
    AgentError,
    DuckDBExecutor,
    Executor,
    SQLiteExecutor,
    executor_for_connection,
    executor_for_path,
)
from schemagraph.agent.guard import (
    GuardedSQL,
    GuardError,
    bounded_sql,
    guard_sql,
    normalize,
    row_limit,
)
from schemagraph.agent.results import ExecResult, PlanInfo
from schemagraph.connectors.duckdb_conn import DuckDBConfig, introspect_duckdb
from schemagraph.engine import Engine

from .conftest import STORE_DDL

ACCEPT = [
    "select 1",
    "with a as (select 1 x) select * from a",
    "select 1 union select 2",
    "(select 1)",
    "select * from orders;",
    "```sql\nselect 1\n```",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
]
REJECT = [
    "insert into orders values (1)",
    "update orders set id = 1",
    "delete from orders",
    "create table x (a int)",
    "drop table orders",
    "alter table orders add c int",
    "attach 'x.db' as y",
    "detach y",
    "pragma table_info(orders)",
    "set threads = 1",
    "copy orders to 'x.csv'",
    "install httpfs",
    "load httpfs",
    "call pragma_version()",
    "select * into x from orders",
    "describe orders",
    "select 1; select 2",
    "select 1; drop table orders",
    "select * from read_csv('/etc/passwd')",
    "select * from read_parquet('a.parquet')",
    "select * from 'data.csv'",
    'select * from "/etc/hosts"',
    "select load_extension('x')",
    "select getenv('HOME')",
    "",
    "select from where",
]

# Statements a hardened DuckDB connection refuses even when the guard is bypassed.
DUCKDB_UNGUARDED_ATTACKS = [
    "select * from read_csv('/etc/passwd')",
    "attach ':memory:' as m",
    "copy orders to 'x.csv'",
    "set enable_external_access = true",
    "install httpfs",
    "create table x (a int)",
]


def sqlite_unguarded_attacks(target: Path) -> list[str]:
    """Statements the SQLite authorizer refuses even when the guard is bypassed."""
    return [
        f"attach '{target}' as n",
        "insert into orders values (9, 1, 1, '2024-01-01')",
        "pragma table_info(orders)",
        "select * from pragma_table_info('orders')",
        "select load_extension('x')",
        "select 1; select 2",
    ]


@pytest.mark.parametrize("dialect", ["duckdb", "sqlite"])
@pytest.mark.parametrize("sql", ACCEPT)
def test_guard_accepts_queries(sql, dialect):
    guarded = guard_sql(sql, dialect)
    assert not guarded.sql.endswith(";")
    assert "```" not in guarded.sql


@pytest.mark.parametrize("dialect", ["duckdb", "sqlite"])
@pytest.mark.parametrize("sql", REJECT)
def test_guard_rejects_everything_else(sql, dialect):
    with pytest.raises(GuardError):
        guard_sql(sql, dialect)


def test_guard_runs_the_original_text():
    assert normalize("  ```sql\nSELECT  a FROM t;\n```  ") == "SELECT  a FROM t"
    # not sqlglot's regenerated SQL
    assert guard_sql("SELECT  a  FROM t", "duckdb").sql == "SELECT  a  FROM t"


@pytest.fixture(params=["duckdb", "sqlite"])
def executor(request, store_duckdb, store_sqlite):
    if request.param == "duckdb":
        opened: Executor = DuckDBExecutor(store_duckdb)
    else:
        opened = SQLiteExecutor(store_sqlite)
    yield opened
    opened.close()


def test_execute_preview_count_and_caps(executor):
    result = executor.execute("select * from orders order by id", limit=2)
    assert result.ok
    assert result.columns == ["id", "customer_id", "total_amount", "order_date"]
    assert len(result.rows) == 2
    assert result.row_count == 3
    assert result.truncated and not result.row_count_capped

    capped = executor.execute("select * from order_items", limit=1, count_cap=2)
    assert capped.row_count == 2 and capped.row_count_capped

    full = executor.execute("select * from order_items", limit=None, raw=True)
    assert full.ok and len(full.rows) == 4 and not full.truncated


def test_full_result_beyond_the_eval_cap_is_a_row_cap_error(executor, monkeypatch):
    monkeypatch.setattr(execute, "EVAL_MAX_ROWS", 3)
    result = executor.execute("select * from order_items", limit=None, raw=True)
    assert not result.ok and result.error_kind == "row_cap"
    assert result.row_count == 3 and result.row_count_capped
    assert executor.execute("select * from orders", limit=None).ok  # exactly the cap


def test_execute_error_kinds(executor):
    assert executor.execute("drop table orders").error_kind == "guard"
    assert executor.execute("select nope from orders").error_kind == "syntax"
    assert executor.plan("select id from orders").ok
    assert executor.plan("select nope from orders").error_kind == "syntax"
    assert executor.plan("drop table orders").error_kind == "guard"
    assert executor.plan("explain select 1").error_kind == "guard"  # the model cannot EXPLAIN


def test_timeout(executor):
    if executor.dialect == "duckdb":
        endless = "select count(*) from range(10000000000)"
    else:
        endless = (
            "with recursive r(n) as (select 1 union all select n + 1 from r) "
            "select count(*) from r"
        )
    assert executor.execute(endless, timeout_s=0.3).error_kind == "timeout"


def test_catalog_and_row_counts(executor):
    catalog = executor.catalog()
    assert catalog["orders"] == ["id", "customer_id", "total_amount", "order_date"]
    assert executor.row_count("orders") == 3
    assert executor.row_count("nope") is None
    # one canonical key per table
    canonical = "main.orders" if executor.dialect == "duckdb" else "orders"
    assert executor.resolve_table("ORDERS") == canonical
    assert executor.resolve_table("main.orders") == executor.resolve_table("orders")


def test_quoted_name_comes_from_the_catalog(executor):
    key = executor.resolve_table("orders")
    expected = '"main"."orders"' if executor.dialect == "duckdb" else '"orders"'
    assert executor.quoted_name(key) == expected


def test_duckdb_connection_is_hardened_without_the_guard(store_duckdb):
    duck = DuckDBExecutor(store_duckdb)
    try:
        for sql in DUCKDB_UNGUARDED_ATTACKS:
            # PermissionException / CatalogException / InvalidInputException
            with pytest.raises(duckdb.Error):
                duck._con.cursor().execute(sql)
        assert duck._statement_ok(duck._con.cursor(), "select 1; drop table orders") is not None
    finally:
        duck.close()


def test_sqlite_connection_is_hardened_without_the_guard(store_sqlite, tmp_path):
    lite = SQLiteExecutor(store_sqlite)
    target = tmp_path / "created.db"
    for sql in sqlite_unguarded_attacks(target):
        result = lite._run(sql, 5, lambda cursor, started: ExecResult(ok=True))
        assert result.error_kind == "guard", sql
    assert not target.exists()  # mode=ro alone would have created it
    count = sqlite3.connect(store_sqlite).execute("select count(*) from orders").fetchone()[0]
    assert count == 3


def test_executor_factories(store_duckdb, store_sqlite, tmp_path):
    assert executor_for_path(store_sqlite).dialect == "sqlite"
    assert executor_for_path(store_duckdb).dialect == "duckdb"
    renamed = tmp_path / "store.db"
    renamed.write_bytes(Path(store_sqlite).read_bytes())
    assert executor_for_path(renamed).dialect == "sqlite"  # sniffed from the header
    with pytest.raises(FileNotFoundError):
        executor_for_path(tmp_path / "missing.duckdb")
    assert not (tmp_path / "missing.duckdb").exists()  # never created


def test_executor_for_connection(tmp_path, store_duckdb):
    engine = Engine(tmp_path / "home", llm=None, embed=False)
    engine.add_connection("ddl", "ddl", {"ddl": STORE_DDL, "dialect": "postgres"})
    engine.add_connection("db", "duckdb", {"path": str(store_duckdb)})
    duck = executor_for_connection(engine, "db")
    assert duck.dialect == "duckdb"
    assert duck.execute("select count(*) from orders").rows == [[3]]
    duck.close()
    with pytest.raises(AgentError):
        executor_for_connection(engine, "ddl")
    with pytest.raises(AgentError):
        executor_for_connection(engine, None)
    with pytest.raises(AgentError, match="unknown connection 'missing'"):
        executor_for_connection(engine, "missing")
    engine.close()


def test_duckdb_connector_introspects(store_duckdb):
    """Regression: duckdb_tables() has no table_type column in DuckDB >= 1.1."""
    snap = introspect_duckdb(DuckDBConfig(path=str(store_duckdb)), "db")
    assert {table.fqn for table in snap.tables} == {
        "main.customer",
        "main.orders",
        "main.order_items",
        "main.products",
        "main.product_category",
    }
    assert any(
        edge.kind == "foreign_key" and edge.from_table == "main.orders" for edge in snap.edges
    )
    assert snap.table("main.orders").row_count == 3


def test_exact_cap_is_not_capped(executor):
    exact = executor.execute("select * from order_items", limit=2, count_cap=4)
    assert exact.row_count == 4 and not exact.row_count_capped and exact.truncated
    over = executor.execute("select * from order_items", limit=2, count_cap=3)
    assert over.row_count == 3 and over.row_count_capped and len(over.rows) == 2


def test_sqlite_value_length_is_bounded(store_sqlite):
    lite = SQLiteExecutor(store_sqlite)
    huge = "select length(zeroblob(400000000) || zeroblob(400000000))"
    result = lite.execute(huge, timeout_s=5)
    # SQLITE_LIMIT_LENGTH: one allocation cannot reach gigabytes
    assert not result.ok and "too big" in result.error


def test_duckdb_explain_checks_statements(store_duckdb, monkeypatch):
    """Layer 2 alone: the guard is bypassed, so only the engine statement check can reject this."""

    def no_guard(sql, dialect):
        return GuardedSQL(sql, sqlglot.parse_one("select 1"), dialect)

    monkeypatch.setattr(execute, "guard_sql", no_guard)
    duck = DuckDBExecutor(store_duckdb)
    try:
        for run in (duck.plan, duck.execute):
            result = run("select 1; create temp table z (a int)")
            assert result.error_kind == "guard" and "one SELECT" in result.error
    finally:
        duck.close()


def test_duckdb_resolves_like_the_search_path(tmp_path):
    """A bare name in two schemas is main's (DuckDB's search path), not unknown."""
    path = tmp_path / "two.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "create schema raw; "
        "create table main.orders (id int, amount int); "
        "create table raw.orders (id int, amt_cents int); "
        "insert into main.orders values (1, 5)"
    )
    con.close()
    duck = DuckDBExecutor(path)
    try:
        assert duck.resolve_table("orders") == "main.orders"
        assert duck.resolve_table("raw.orders") == "raw.orders"
        assert duck.resolve_table("two.main.orders") == "main.orders"
        assert duck.row_count("orders") == 1
        guarded = guard_sql("select sum(amount) from orders", "duckdb")
        report = static_checks(guarded, duck.catalog(), duck.resolve_table)
        assert report.det == 1.0 and report.tables == ["main.orders"]
    finally:
        duck.close()
    # the connection's schemas limit what the agent is shown
    scoped = DuckDBExecutor(path, schemas=["main"])
    try:
        assert "raw.orders" not in scoped.catalog()
        assert scoped.resolve_table("orders") == "main.orders"
    finally:
        scoped.close()


def test_sqlite_placeholder_is_a_syntax_error_not_a_guard_reject(store_sqlite):
    lite = SQLiteExecutor(store_sqlite)
    assert lite.execute("select * from orders where id = ?").error_kind == "syntax"


def test_sqlite_catalog_skips_a_broken_view(tmp_path):
    path = tmp_path / "broken.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    con.execute("CREATE VIEW v AS SELECT a, b FROM t")
    con.execute("DROP TABLE t")  # v now names a column that is gone
    con.execute("CREATE TABLE t (a INTEGER)")
    con.commit()
    con.close()
    executor = SQLiteExecutor(path)
    try:
        assert executor.catalog() == {"t": ["a"]}
        assert executor.execute("SELECT a FROM t").ok
    finally:
        executor.close()


def test_join_key_counts_quote_reserved_names_and_leave_out_nulls(tmp_path):
    from types import SimpleNamespace

    from schemagraph.agent.answer import Answerer

    path = tmp_path / "keys.sqlite"
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE "order" (id INTEGER, coupon INTEGER)')
    con.executemany('INSERT INTO "order" VALUES (?, ?)', [(1, 7), (2, None), (3, None), (4, 8)])
    con.commit()
    con.close()
    executor = SQLiteExecutor(path)
    try:
        host = SimpleNamespace(executor=executor, _key_counts={})
        # coupon is unique among the rows that can match a join; the NULLs match nothing
        assert Answerer._key_stats(host, executor.resolve_table("order"), "coupon") == (2, 2)
        assert Answerer._key_stats(host, executor.resolve_table("order"), "id") == (4, 4)
    finally:
        executor.close()


# ------------------------------------------------------------------ row cap in the SQL text

# (dialect, query, text after bounded_sql(..., 21)); None means unchanged.
BOUNDED = [
    ("sqlite", "select a from t", "select a from t\nLIMIT 21"),
    ("duckdb", "select a from t", "select a from t\nLIMIT 21"),
    ("sqlite", "with x as (select a from t) select a from x order by a",
     "with x as (select a from t) select a from x order by a\nLIMIT 21"),
    ("duckdb", "select a from t union select b from u", "select a from t union select b from u\nLIMIT 21"),
    ("sqlite", "select a from t union all select b from u",
     "select a from t union all select b from u\nLIMIT 21"),
    ("duckdb", "from t", "from t\nLIMIT 21"),
    ("sqlite", "select a from t -- trailing comment", "select a from t -- trailing comment\nLIMIT 21"),
    ("duckdb", "select a from t limit 5", None),
    ("sqlite", "select a from t limit 5, 10", None),  # SQLite's LIMIT offset, count
    # a literal above the cap: its digits are replaced in place, the rest runs as written
    ("sqlite", "select a from t limit 50 offset 5", "select a from t limit 21 offset 5"),
    ("sqlite", "select a from t limit 5, 500", "select a from t limit 5, 21"),
    ("duckdb", "select a from t fetch first 30 rows only", "select a from t fetch first 21 rows only"),
    ("duckdb", "select a /* keep */ from t\n  LIMIT   50 -- note",
     "select a /* keep */ from t\n  LIMIT   21 -- note"),
    ("duckdb", "select a from t limit +500", "select a from t limit +21"),
    ("sqlite", "select a from t limit 18446744073709551616", None),  # an error as written
    ("duckdb", "select a from t offset 5", None),  # OFFSET alone: no LIMIT to add after it
    ("duckdb", "select a from t limit 10%", None),  # a percentage: the cap would change it
    ("duckdb", "select a from t limit 1 + 100", None),
]  # fmt: skip


@pytest.mark.parametrize(("dialect", "sql", "expected"), BOUNDED)
def test_bounded_sql_caps_the_text_and_passes_the_guard_again(dialect, sql, expected):
    guarded = guard_sql(sql, dialect)
    text = bounded_sql(guarded, 21)
    assert text == (expected or guarded.sql)
    capped = guard_sql(text, dialect)
    if expected is not None:
        assert row_limit(capped.tree) == 21


@pytest.fixture(params=["duckdb", "sqlite"])
def numbers(request, tmp_path):
    """An executor over one table of 100 rows: nums(id, k) with k = id % 7."""
    rows = ", ".join(f"({i}, {i % 7})" for i in range(100))
    if request.param == "duckdb":
        path = tmp_path / "nums.duckdb"
        con = duckdb.connect(str(path))
        con.execute(f"CREATE TABLE nums (id INT, k INT); INSERT INTO nums VALUES {rows}")
        con.close()
        opened: Executor = DuckDBExecutor(path)
    else:
        path = tmp_path / "nums.sqlite"
        con = sqlite3.connect(path)
        con.executescript(f"CREATE TABLE nums (id INT, k INT); INSERT INTO nums VALUES {rows};")
        con.close()
        opened = SQLiteExecutor(path)
    yield opened
    opened.close()


def _spy_runs(executor, monkeypatch) -> list[str]:
    """Record the SQL text of every query the executor runs."""
    ran: list[str] = []
    run = executor._run

    def spy(sql, timeout_s, then):
        ran.append(sql)
        return run(sql, timeout_s, then)

    monkeypatch.setattr(executor, "_run", spy)
    return ran


def test_execute_puts_the_count_cap_in_the_sql(numbers, monkeypatch):
    ran = _spy_runs(numbers, monkeypatch)
    probe = numbers.execute("select id from nums", limit=20, count_cap=20)
    assert ran == ["select id from nums\nLIMIT 21"]
    assert probe.row_count == 20 and probe.row_count_capped and len(probe.rows) == 20
    assert probe.plan_rows is None  # not gated
    exact = numbers.execute("select id from nums limit 20", limit=20, count_cap=20)
    assert exact.row_count == 20 and not exact.row_count_capped  # its own LIMIT runs as written
    full = numbers.execute("select id from nums", limit=None)
    assert ran[-1] == "select id from nums" and full.row_count == 100  # evaluation: never capped


def test_the_cap_keeps_duplicate_column_names(numbers):
    result = numbers.execute("select a.id, b.id from nums a join nums b on a.id = b.id", limit=5)
    assert result.ok and result.columns == ["id", "id"] and result.row_count == 100


def test_cost_gate_refuses_an_expensive_plan_unrun(numbers, monkeypatch):
    cross = "select count(*) from nums a, nums b, nums c"
    monkeypatch.setattr(numbers, "max_plan_rows", 1e5)
    ran = _spy_runs(numbers, monkeypatch)
    refused = numbers.execute(cross, limit=20, cost_gate=True)
    assert not refused.ok and refused.error_kind == "cost"
    assert refused.plan_rows is not None and refused.plan_rows > 1e5
    assert "the plan would process about" in refused.error and "rows" in refused.error
    assert not any(sql.startswith("select count") for sql in ran)  # only the EXPLAIN ran
    assert numbers.execute(cross, limit=20).rows == [[1_000_000]]  # ungated, it runs
    assert numbers.execute(cross, limit=None, cost_gate=True).ok  # evaluation is never gated
    cheap = numbers.execute("select id from nums a", limit=5, cost_gate=True)
    assert cheap.ok and cheap.plan_rows is not None and cheap.plan_rows <= 1e5


def test_plan_estimates_resolve_aliases_and_are_memoised(numbers, monkeypatch):
    calls: list[str] = []
    explain = numbers._explain

    def counting(guarded, timeout_s, *, estimate):
        calls.append(guarded.sql)
        return explain(guarded, timeout_s, estimate=estimate)

    monkeypatch.setattr(numbers, "_explain", counting)
    plan = numbers.plan("select x.id from nums x, nums y")
    # SQLite: the scans of x and y, sized through their aliases; DuckDB: the cross product
    assert plan.ok and plan.rows == (100 + 100 * 100 if numbers.dialect == "sqlite" else 100 * 100)
    assert plan.reason
    assert numbers.plan("select x.id from nums x, nums y") is plan
    assert numbers.plan("select x.id from nums x, nums y;") is plan  # the same guarded text
    assert len(calls) == 1
    assert numbers.plan("select nope from nums").error_kind == "syntax"
    cte = numbers.plan("with c as (select id from nums) select * from c c1, c c2")
    assert cte.ok and cte.rows is not None


def test_a_capped_probe_runs_the_text_as_written_apart_from_the_limit(numbers, monkeypatch):
    ran = _spy_runs(numbers, monkeypatch)
    probe = numbers.execute(
        "select id /* ids */ from nums\n  limit   50 -- note", limit=20, count_cap=20
    )
    assert ran == ["select id /* ids */ from nums\n  limit   21 -- note"]
    assert probe.row_count == 20 and probe.row_count_capped
    commented = numbers.execute("select id from nums -- trailing comment", limit=20, count_cap=20)
    assert ran[-1] == "select id from nums -- trailing comment\nLIMIT 21"
    assert commented.ok and commented.row_count == 20 and commented.row_count_capped


def test_sqlite_still_rejects_syntax_it_would_reject_as_written(tmp_path):
    path = tmp_path / "nums.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE nums (id INT)")
    con.close()
    executor = SQLiteExecutor(path)
    try:
        for sql in (
            "select id from nums fetch first 500 rows only",  # not SQLite syntax
            "from nums select id limit 500",  # DuckDB's FROM-first
        ):
            result = executor.execute(sql, limit=20, count_cap=20)
            assert not result.ok and result.error_kind == "syntax", sql
    finally:
        executor.close()


@pytest.mark.parametrize(
    "capped",
    [
        "select 1; select 2",
        "select * from read_csv('x.csv')",
        "select id from nums\nLIMIT 500",  # a limit, but not the cap
    ],
)
def test_capped_text_that_fails_re_admission_runs_the_original(numbers, monkeypatch, capped):
    monkeypatch.setattr(execute, "bounded_sql", lambda guarded, max_rows: capped)
    ran = _spy_runs(numbers, monkeypatch)
    result = numbers.execute("select id from nums", limit=20, count_cap=20)
    assert ran == ["select id from nums"]
    assert result.ok and result.row_count == 20 and result.row_count_capped  # the fetch cap


def test_plan_memo_keeps_plans_not_timeouts_and_drops_the_oldest(numbers, monkeypatch):
    calls: list[str] = []
    explain = numbers._explain
    outcome = {"fail": "timeout"}

    def flaky(guarded, timeout_s, *, estimate):
        calls.append(guarded.sql)
        if outcome["fail"]:
            return PlanInfo(ok=False, error="boom", error_kind=outcome["fail"])
        return explain(guarded, timeout_s, estimate=estimate)

    monkeypatch.setattr(numbers, "_explain", flaky)
    monkeypatch.setattr(execute, "PLAN_CACHE_SIZE", 2)
    assert numbers.plan("select 1").error_kind == "timeout"
    outcome["fail"] = "runtime"
    assert numbers.plan("select 1").error_kind == "runtime"
    outcome["fail"] = None
    assert numbers.plan("select 1").ok
    assert calls == ["select 1"] * 3  # neither failure was kept
    numbers.plan("select 1")
    assert len(calls) == 3  # the plan was
    numbers.plan("select 2")
    numbers.plan("select 3")  # evicts "select 1", the oldest
    numbers.plan("select 1")
    assert calls[3:] == ["select 2", "select 3", "select 1"]


def test_planning_honours_the_callers_timeout(numbers, monkeypatch):
    timeouts: list[float] = []
    explain = numbers._explain

    def recording(guarded, timeout_s, *, estimate):
        timeouts.append(timeout_s)
        return explain(guarded, timeout_s, estimate=estimate)

    monkeypatch.setattr(numbers, "_explain", recording)
    numbers.execute("select id from nums", limit=5, timeout_s=2.0, cost_gate=True)
    numbers.execute("select k from nums", limit=5, timeout_s=60.0, cost_gate=True)
    assert timeouts == [2.0, execute.EXPLAIN_TIMEOUT_S]


def test_sqlite_counts_tables_only_for_an_estimate(tmp_path, monkeypatch):
    path = tmp_path / "nums.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE nums (id INT)")
    con.close()
    executor = SQLiteExecutor(path)
    counted: list[str] = []
    monkeypatch.setattr(
        executor, "_count", lambda schema, table, timeout_s: counted.append(table) or 10
    )
    try:
        bind_only = executor.plan("select id from nums", estimate=False)
        assert bind_only.ok and bind_only.rows is None and counted == []
        assert executor.plan("select id from nums").rows == 10 and counted == ["nums"]
    finally:
        executor.close()


def test_sqlite_count_timeout_falls_back_to_max_rowid(tmp_path, monkeypatch):
    path = tmp_path / "gaps.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (a INT)")
    con.executemany("INSERT INTO t (rowid, a) VALUES (?, ?)", [(1, 1), (2, 2), (50, 3)])
    con.execute("CREATE VIEW v AS SELECT * FROM t")
    con.commit()
    con.close()
    # every count is interrupted at its first progress check
    monkeypatch.setattr(execute, "_SQLITE_PROGRESS_OPS", 1)
    monkeypatch.setattr(execute, "_deadline_handler", lambda timeout_s: lambda: 1)
    executor = SQLiteExecutor(path)
    try:
        assert executor.row_count("t") == 50  # an upper bound of the 3 rows
        assert executor.row_count("v") is None  # a view has no rowid: unknown
    finally:
        executor.close()


def test_key_stats_skip_tables_over_the_row_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from schemagraph.agent import answer
    from schemagraph.agent.answer import Answerer

    path = tmp_path / "keys.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()
    monkeypatch.setattr(answer, "KEY_STATS_MAX_ROWS", 2)
    executor = SQLiteExecutor(path)
    ran = _spy_runs(executor, monkeypatch)
    try:
        host = SimpleNamespace(executor=executor, _key_counts={})
        assert Answerer._key_stats(host, "t", "id") is None
        assert ran == [] and host._key_counts == {("t", "id"): None}  # memoised, never counted
    finally:
        executor.close()


def test_an_alias_naming_different_tables_is_sized_as_unknown(tmp_path):
    path = tmp_path / "alias.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE big (id INT)")
    con.execute("CREATE TABLE small (id INT)")
    con.executemany("INSERT INTO big VALUES (?)", [(i,) for i in range(1000)])
    con.executemany("INSERT INTO small VALUES (?)", [(i,) for i in range(3)])
    con.commit()
    con.close()
    tree = guard_sql("select * from small t where exists (select 1 from big t)", "sqlite")
    assert execute._names_by_alias(tree) == {"t": None}
    executor = SQLiteExecutor(path)
    try:
        # both scans of `t` count as one row, whichever table the plan meant
        plan = executor.plan("select count(*) from big t where exists (select 1 from small t)")
        assert plan.ok and plan.rows is not None and plan.rows < 10
    finally:
        executor.close()


def test_sqlite_sizes_materialised_results_by_what_fills_them(tmp_path):
    path = tmp_path / "big.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE big (id INT, k INT)")
    con.executemany("INSERT INTO big VALUES (?, ?)", [(i, i % 7) for i in range(2000)])
    con.commit()
    con.close()
    executor = SQLiteExecutor(path)
    grouped = "(select k, count(*) n from big group by id)"
    try:
        for sql in (
            f"select count(*) from {grouped} x, {grouped} y",  # GROUP BY subqueries
            "with c as materialized (select * from big) select count(*) from c c1, c c2",
        ):
            assert executor.plan(sql).rows >= 2000 * 2000, sql
        # what the SQL states bounds the size: LIMIT 1, and one row for an ungrouped aggregate
        bounded = (
            "with hi as (select id from big order by k desc limit 1), "
            "mx as (select max(id) m from big) "
            "select count(*) from hi, mx, big where big.id > hi.id"
        )
        assert executor.plan(bounded).rows < 20_000
    finally:
        executor.close()


def test_duckdb_refuses_a_large_inequality_join(tmp_path):
    path = tmp_path / "big.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE big AS SELECT range AS id FROM range(200000)")
    con.close()
    executor = DuckDBExecutor(path)
    try:
        # DuckDB estimates ~1.4e7 rows for this piecewise merge join; it emits 2e10
        result = executor.execute(
            "select count(*) from big a join big b on a.id < b.id", limit=20, cost_gate=True
        )
        assert result.error_kind == "cost" and "PIECEWISE_MERGE_JOIN" in result.error
    finally:
        executor.close()
