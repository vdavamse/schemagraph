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
from schemagraph.agent.guard import GuardedSQL, GuardError, guard_sql, normalize
from schemagraph.agent.results import ExecResult
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
    assert executor.explain("select id from orders").ok
    assert executor.explain("select nope from orders").error_kind == "syntax"
    assert executor.explain("drop table orders").error_kind == "guard"


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
        for run in (duck.explain, duck.execute):
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
