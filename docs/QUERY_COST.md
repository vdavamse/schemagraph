# Query cost: bounding what the agent runs on the database

Issue #19 asked whether the generator's read-only probe query ("limit of 20 rows") can put a
high compute load on the database, and whether `EXPLAIN` is the answer. It can, and `EXPLAIN`
is part of the answer: the agent layer now **bounds** every query it runs with a `LIMIT` in the
SQL text and **gates** it with an `EXPLAIN`-based cost estimate before it reaches the database.
Execution itself stays: the search needs real results (empty-result, NULL-column and
row-explosion checks, result fingerprints, the judge's preview, refinement feedback).

Code: `src/schemagraph/agent/cost.py` (estimates and the refusal), `execute.py`
(`Executor.plan`, the cap and the gate), `guard.py` (`bounded_sql`), `agents.py` (the probe
tool and the output validator).

## Why "20 rows" did not bound the work

Before this change the probe (`run_query`) ran the model's SQL as written and fetched rows on
the client: it kept 20 for the model but kept fetching, up to the 100,001-row count cap, to report
a row count. Nothing put a `LIMIT` in the SQL.

Even a `LIMIT` in the SQL bounds only what the query *returns*. Whether the database can stop
early depends on the plan:

| Plan shape | Stops early under LIMIT? |
|---|---|
| Scan, filter, projection, nested-loop join driven by an index | Yes: the operators are pipelined (PostgreSQL's `Limit` node, Oracle's `COUNT STOPKEY`, DuckDB's streaming result) |
| `ORDER BY` without a usable index | No, except as a top-N sort (DuckDB `TOP_N`, PostgreSQL top-N heapsort), which still reads every input row |
| `GROUP BY`, aggregates, `DISTINCT`, window functions | No: every input row is read before the first output row |
| Hash join | The build side is read in full |
| A selective filter without an index | Scans until enough rows match, possibly the whole table |

Measured on DuckDB 1.5.5 over `range(3e8)`: a plain scan with 20 rows fetched took 0.03 s, a
`GROUP BY` 4.3 s and an `ORDER BY` 20.4 s. And a client-side "fetch 20" is not a bound on
most drivers at all: psycopg's default cursor, for example, materialises the whole result on
the client before the first `fetchmany`. So the cap has to be in the SQL text.

## What runs now

Every call through `Executor.execute` with a `limit`:

1. **Admit**: the guard (one read-only query) and the engine's statement check, as before.
2. **Gate** (`cost_gate=True`): plan the query with `EXPLAIN`, estimate its work, and refuse it
   unrun (`error_kind` `cost`) when the estimate is over the engine's threshold. The refusal
   names the operator and sizes, e.g. *"the plan would process about 4.2e+11 rows (nested loop:
   SCAN word_list (373,804 rows) × SCAN word_list (373,804 rows)); add join conditions or
   filters, filter or aggregate before joining, and avoid correlated subqueries over large
   tables"*. Plans are memoised by query text (the output validator and scoring plan the same
   query); a timed-out `EXPLAIN` or a runtime error is not memoised, while a plan or a binding
   (syntax) error is. Planning is bounded by the smaller of
   `EXPLAIN_TIMEOUT_S` (10 s) and the call's own timeout.
3. **Cap**: append `LIMIT max(count_cap, limit) + 1` to the SQL text (`guard.bounded_sql`) and
   admit the capped text again. The one extra row is how the collector tells "exactly the cap"
   from "more", so `row_count` and `row_count_capped` mean what they did. A query with its own
   literal `LIMIT` / `FETCH FIRST` at or below the cap runs as written; for one above it, the
   digits of the literal are replaced where they stand (the literal's token span; `1_000_000`
   too), and so is a count that spells "no limit" (SQLite's `LIMIT -1`, minus sign included;
   DuckDB's `LIMIT ALL` / `LIMIT NULL`; DuckDB rejects `LIMIT -1`, so it stays), so the rest
   of the text runs byte for byte and the database still rejects what it would reject as
   written (SQLite refuses `FETCH FIRST 500 ROWS ONLY` either way). The capped text must parse
   back to a top-level limit of exactly the cap, else the query runs as written. `OFFSET` alone,
   a percentage, an expression limit or a literal too large for a 64-bit integer is left as
   written: capping would change the result or what the database accepts.
   Replayed on the same 805 stored candidates as the calibration below, the cap was appended
   to 659 and left out of 146 (their own `LIMIT` was smaller); none had a literal above the
   cap and none fell back. All 395 distinct correct candidates returned the same rows and
   counts with the cap as without it.
4. **Run** under the existing wall-clock timeout, DuckDB memory/thread limits and SQLite
   value-length limits.

| Caller | Cap in the SQL | Gated |
|---|---|---|
| `run_query` probe | `LIMIT 21` (the model sees "20+ rows" for anything larger) | yes |
| `sample_values` (fixed `SELECT DISTINCT … LIMIT n`) | its own `LIMIT n` | no |
| Generator output validator | — (plans only) | yes: sent back to the model, accepted on the last output retry so scoring records a `cost` candidate |
| Candidate scoring (`Answerer.score`) | `LIMIT 100,001` | yes |
| Judge join-key statistics (`COUNT`, `COUNT DISTINCT`) | appended, harmless | no; skipped for tables over `KEY_STATS_MAX_ROWS` (10M) |
| Benchmark EX (`limit=None`) | none | no |
| Trusted catalog counts (`row_count`) | none | no |

A refused query scores like any other execution failure (`ScoreWeights.exec_fail`), with a
`cost` finding whose message goes into the refinement prompt. `AgentConfig.cost_gate` turns the
gate off from Python; there is no CLI flag. Exec benchmark runs with the gate on (the default,
and the only choice from the CLI) get a new config hash, so they do not resume rows recorded
before the gate existed. `cost_gate=False` is the legacy value in the bench's `_LEGACY_VALUES`,
which only keeps those old rows resumable for a Python caller that passes `cost_gate=False`.

## The estimates

`EXPLAIN` never runs the query. What it gives differs per engine.

**DuckDB** (`EXPLAIN (FORMAT json)`): every operator carries an `Estimated Cardinality`. The gate
reads only joins, the operators that can grow far past the data: a hash or AsOf join's
estimate is the rows it emits; a nested-loop, blockwise-NL join or cross product compares every
pair of input rows, so its work is the product of its inputs. The inequality joins
(piecewise-merge, IE) emit their estimate but at least a third of their input pairs
(`DUCKDB_INEQUALITY_SELECTIVITY`, System R's default selectivity for a range predicate):
DuckDB estimated `a.id < b.id` over two 200k-row inputs at 1.4e7 rows, and it emits 2e10.
Operators without an estimate take their inputs' product (loop joins) or largest input. Scans
and aggregates alone are never refused. The estimates are rough (a `GROUP BY` estimated at ~500k rows returned 7; a
hash join estimated at 1.6M rows returned 2M) but they are right about orders of magnitude on
catastrophic plans: a self-join on a 7-value key over 2M rows is estimated at 5.7e11 rows.
Threshold `DUCKDB_MAX_JOIN_ROWS = 1e10`, not yet calibrated (Spider 2.0's local execution tasks
are all SQLite); `plan_rows` is recorded on every gated result and in the candidates file for
recalibration.

**SQLite** (`EXPLAIN QUERY PLAN`, the format of SQLite 3.36 and later; an older `SCAN TABLE t`
plan is not read and the gate fails open): operators only, no row estimates. The work is read
from the plan's shape and the base tables' row counts (the executor's trusted, memoised
`count(*)`, sized only when the gate is on; the counts share what is left of the planning
timeout, and a count that runs out falls back to `max(rowid)`, an upper bound that reads one
b-tree path. The stand-in is memoised for later estimates only, so a slow table is counted
once; `row_count` (the judge, the row-explosion check) memoises exact counts only and never
returns it):

* rows under one parent are loop levels, outer to inner; each full `SCAN` multiplies the
  running product by the table's size and visits that many rows;
* an indexed `SEARCH` and a `USE TEMP B-TREE` visit the running product once more; an automatic
  index also reads its table once to build;
* a `CORRELATED … SUBQUERY` runs once per outer row (scaled by the running product); other
  sub-plans (`MATERIALIZE`, `CO-ROUTINE`, compound-query sides, `IN` lists) run once;
* a materialised CTE or subquery (`MATERIALIZE c`, `CO-ROUTINE x`) is sized by the scan product
  of the nest that fills it, capped by what its SQL states (a literal `LIMIT`; one row for an
  aggregate without `GROUP BY`), so each later `SCAN` of it multiplies by that size: a CTE
  referenced twice, or two `GROUP BY` subqueries cross-joined, cost what they read. The scan
  product is an upper bound only for a nest of scans: an index `SEARCH`'s rows per lookup are
  not in the plan and are not counted, so a low-selectivity equi-join through an index (a
  self-join on a 7-value key) is left to the timeout. A `GROUP BY` or `DISTINCT` inside returns
  fewer rows than the plan can say; a CTE SQLite flattens shows up as scans of its base tables;
* plan names are aliases, resolved through the query's own parse tree; an alias that names
  different tables in different scopes, a view that could not be counted and a recursive
  CTE count as one row, since a false refusal costs more than a slow query the timeout still
  stops.

Threshold `SQLITE_MAX_LOOP_ROWS = 1e11`.

### Calibration (SQLite)

805 stored candidates of the Spider 2.0-Lite local exec runs (`bench_results/*_candidates.jsonl`),
re-planned with the final code against the local SQLite databases (planning: 0.8 ms median,
43 ms max). "Correct" is a candidate whose result matched the gold (`ex = 1`); "timeouts" are
the 9 candidates that hit the 30 s execution timeout.

| Threshold | Refused | Correct refused | Timeouts caught |
|---|---|---|---|
| 1e7 | 19 | 4 | 8 / 9 |
| 1e8 | 16 | 2 | 8 / 9 |
| 1e9 | 9 | 1 | 8 / 9 |
| 1e10 | 3 | 1 (estimate 8.4e10, runs in 4.3 s) | 2 / 9 |
| **1e11** | **2** | **0** | **2 / 9** |
| 1e12 | 0 | 0 | 0 / 9 |

The estimate is about two orders of magnitude off the real work, so the gate refuses only
catastrophic plans. The 7 timeouts it misses are one task's recursive CTEs (`local169`): sizing
materialised results lifted their estimates from ~1e5 to 4e9–8e9, still under the threshold,
since recursion depth is invisible to a plan; the timeout still bounds them.

Sizing a materialised result by its scan product alone (without the `LIMIT` / ungrouped
aggregate bounds) refused 8 candidates at 1e11, 6 of them wrong answers that ran in 0.1 s or
less: `hi` / `lo` CTEs with `LIMIT 1` over a `GROUP BY` of a `DISTINCT` × `DISTINCT` grid
(estimated 1e15), and a `MAX()` CTE cross-joined with its 101k-row source (1.2e11). The
bounds the SQL states are exact upper bounds, so they never raise the estimate while removing
those; no correct candidate was refused either way.

## What stays timeout-bound

The gate is a coarse filter, not a guarantee. Recursive CTEs, expensive scalar functions,
regular expressions over large text columns, and anything the estimate gets wrong are still
stopped by the wall-clock timeout (`TOOL_QUERY_TIMEOUT_S` = 10 s for probes,
`AgentConfig.exec_timeout_s` = 30 s for candidates), DuckDB's `max_memory` / `threads`, and
SQLite's value-length limit. A probe refused by the gate still spends the node's probe budget.

## Warehouses (not implemented)

Executors exist only for DuckDB and SQLite files. A warehouse executor would keep the same three
parts — a `LIMIT` in the text, a plan-based gate, and server-side limits — with each engine's own
tools:

* **PostgreSQL**: `EXPLAIN (FORMAT JSON)` gives `Total Cost` and `Plan Rows` per node;
  `statement_timeout` and `default_transaction_read_only` per session; a server-side (named)
  cursor so the driver does not materialise the result.
* **Snowflake**: `EXPLAIN USING JSON` (partitions and bytes to scan);
  `STATEMENT_TIMEOUT_IN_SECONDS` per session; resource monitors for credit budgets.
* **BigQuery**: a dry run returns the bytes a query would process, before it runs;
  `maximumBytesBilled` makes the service refuse anything larger.
* **Databricks**: `EXPLAIN COST` gives size and row statistics; SQL warehouses cancel a
  statement past `STATEMENT_TIMEOUT`; the query watchdog is a setting of classic clusters, not
  of SQL warehouses.
* **Oracle**: `EXPLAIN PLAN` into `PLAN_TABLE` (`CARDINALITY`, `COST`); Resource Manager's
  `MAX_EST_EXEC_TIME` is a plan-based gate like this one (it refuses a call whose optimizer
  estimate exceeds the limit, before it runs), and its other directives limit a session's CPU
  and elapsed time.

Row-level security, budgets and PII stay the caller's responsibility, as for the rest of the
agent layer.
