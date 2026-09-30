"""Work estimates from a query plan, and the cost gate that refuses a catastrophic one.

A ``LIMIT`` bounds what a query returns, not what the database does to produce it: a sort, an
aggregate, a ``DISTINCT`` or a hash join reads all of its input before the first row comes out,
so a 20-row probe can still scan and join whole tables. The executor therefore plans every query
the agent runs with ``EXPLAIN`` first and refuses one whose plan is estimated far beyond the
execution timeout (``error_kind`` ``cost``), before it reaches the database. The rules are
deliberately coarse, so they refuse only plans that could never finish:

* DuckDB: the rows a join is estimated to emit (hash and merge joins) or to compare (nested
  loops and cross products), from ``EXPLAIN (FORMAT json)``'s cardinality estimates; an
  inequality join emits at least a fixed share of its input pairs.
* SQLite: its ``EXPLAIN QUERY PLAN`` has operators but no estimates, so the work is the running
  product of the full scans in each loop nest, sized by the base tables' row counts and, for a
  materialised CTE or subquery, by the scans that fill it.

Pure functions over the plan text (core dependencies only); :mod:`schemagraph.agent.execute`
runs the ``EXPLAIN``. ``docs/QUERY_COST.md`` explains the choices and the calibration.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from schemagraph.agent.results import PlanInfo

# A SQLite plan is refused above this many estimated row visits: the running product of
# nested full scans over base tables and materialised results. Calibrated on 805 stored Spider2
# candidates: 0 correct ones above it; 1e10 refuses a correct 4.3 s query (estimate 8.4e10).
SQLITE_MAX_LOOP_ROWS = 1e11
# A DuckDB plan is refused when a join is estimated to emit (hash/merge) or compare
# (nested loop, cross product) more rows than this; at DuckDB's join throughput that is well
# past DEFAULT_TIMEOUT_S. Uncalibrated: Spider2's local exec tasks are SQLite only.
DUCKDB_MAX_JOIN_ROWS = 1e10
# Share of input pairs an inequality join (``a.x < b.y``) is assumed to emit at least: System R's
# default selectivity of a range predicate (Selinger et al., 1979). DuckDB's own estimate for
# these joins is far lower (1.4e7 for ``a.id < b.id`` over 200k rows each, which emits 2e10).
DUCKDB_INEQUALITY_SELECTIVITY = 1 / 3

# DuckDB joins whose estimate is the rows they emit.
_DUCKDB_JOIN_OPERATORS = frozenset({"HASH_JOIN", "ASOF_JOIN"})
# DuckDB inequality joins: they emit their estimate, but at least a share of the input pairs.
_DUCKDB_INEQUALITY_OPERATORS = frozenset({"PIECEWISE_MERGE_JOIN", "IE_JOIN"})
# DuckDB joins that compare every pair of input rows: their work is the product of the inputs.
_DUCKDB_LOOP_OPERATORS = frozenset({"NESTED_LOOP_JOIN", "BLOCKWISE_NL_JOIN", "CROSS_PRODUCT"})

_DUCKDB_ESTIMATE = "Estimated Cardinality"
# Where a join's condition is printed: most joins use the plural, BLOCKWISE_NL_JOIN the singular.
_DUCKDB_CONDITION_KEYS = ("Conditions", "Condition")
# Characters of a join condition quoted in a refusal.
_CONDITION_CHARS = 120

# One SQLite ``EXPLAIN QUERY PLAN`` row: (id, parent id, unused, detail).
SqlitePlanRow = Sequence[Any]


def _unknown(_name: str) -> None:
    """Know nothing about a name."""
    return None


@dataclass(frozen=True)
class SqliteNames:
    """What the SQLite estimate knows about a name as the plan prints it (often an alias).

    Attributes:
        table_rows: Rows of the base table or view behind the name, or None when unknown.
        source: The CTE or subquery the name reads (``c`` for ``c1`` in ``FROM c AS c1``),
            or None when it reads a table or the name is itself the source.
        max_rows: At most how many rows the named CTE or subquery returns, as its SQL says
            (a literal ``LIMIT``, one row for an aggregate without ``GROUP BY``), or None.
    """

    table_rows: Callable[[str], int | None]
    source: Callable[[str], str | None] = _unknown
    max_rows: Callable[[str], int | None] = _unknown

# Plan details from SQLite before 3.36 (``SCAN TABLE t``); the walk reads the newer format only.
_SQLITE_OLD_FORMAT = ("SCAN TABLE ", "SEARCH TABLE ")
# Sub-plans that fill a named temporary result, which a later ``SCAN <name>`` reads.
_SQLITE_NAMED_RESULTS = ("MATERIALIZE ", "CO-ROUTINE ")


def cost_refusal(plan: PlanInfo, max_rows: float) -> str | None:
    """Return why a planned query is refused, or None when it may run.

    A plan that failed or has no estimate is never refused (the gate fails open): running the
    query reports the real error, and the timeout still bounds it.

    Args:
        plan: The query's plan.
        max_rows: The engine's threshold (:data:`DUCKDB_MAX_JOIN_ROWS` or
            :data:`SQLITE_MAX_LOOP_ROWS`).

    Returns:
        The refusal message, which names the dominant operator so the model can rewrite it.
    """
    if not plan.ok or plan.rows is None or plan.rows <= max_rows:
        return None
    return (
        f"the plan would process about {plan.rows:.1e} rows ({plan.reason}); add join "
        "conditions or filters, filter or aggregate before joining, and avoid correlated "
        "subqueries over large tables"
    )


# --------------------------------------------------------------------------------------- DuckDB
def duckdb_plan_cost(plan_json: str) -> tuple[float | None, str]:
    """Return the largest join estimate of a DuckDB ``EXPLAIN (FORMAT json)`` plan.

    Each node's rows are its own estimate; operators without one (cross product, order by,
    top-N, union) take the product of their inputs for a loop join, else the largest input.
    An inequality join (piecewise merge, IE join) emits at least
    :data:`DUCKDB_INEQUALITY_SELECTIVITY` of its input pairs.
    Scans and aggregates alone are never the cost: only joins can grow far past the data.

    Args:
        plan_json: The second column of the ``EXPLAIN (FORMAT json)`` row.

    Returns:
        (rows, reason), e.g. ``(5.7e11, "HASH_JOIN estimated 5.7e+11 rows (k = k)")``;
        ``(0.0, "")`` for a plan without joins and ``(None, "")`` when the text cannot be read.
    """
    try:
        roots = json.loads(plan_json)
    except (TypeError, ValueError):
        return None, ""
    if isinstance(roots, dict):
        roots = [roots]
    if not isinstance(roots, list):
        return None, ""
    worst: list[tuple[float, str]] = [(0.0, "")]
    try:
        for root in roots:
            _duckdb_rows(root, worst)
    except (AttributeError, TypeError):
        return None, ""
    return max(worst, key=lambda join: join[0])


def _duckdb_rows(node: dict[str, Any], joins: list[tuple[float, str]]) -> float:
    """Return a DuckDB plan node's estimated output rows; append each join's (rows, reason).

    Args:
        node: One node of the JSON plan (``name``, ``children``, ``extra_info``).
        joins: Receives the work of every join operator under and at ``node``.
    """
    inputs = [_duckdb_rows(child, joins) for child in node.get("children") or []]
    name = str(node.get("name", "")).strip().upper()
    extra = node.get("extra_info") or {}
    estimate = _number(extra.get(_DUCKDB_ESTIMATE))
    product = _product(inputs)
    if estimate is not None:
        rows = estimate
    elif name in _DUCKDB_LOOP_OPERATORS:
        rows = product
    else:
        rows = max(inputs, default=0.0)
    if name in _DUCKDB_INEQUALITY_OPERATORS:
        rows = max(rows, product * DUCKDB_INEQUALITY_SELECTIVITY)
    if name in _DUCKDB_LOOP_OPERATORS:
        joins.append((product, _join_reason(name, "compare", product, extra)))
    elif name in _DUCKDB_JOIN_OPERATORS | _DUCKDB_INEQUALITY_OPERATORS:
        joins.append((rows, _join_reason(name, "emit", rows, extra)))
    return rows


def _join_reason(name: str, verb: str, rows: float, extra: dict[str, Any]) -> str:
    """Describe one join for a refusal: ``HASH_JOIN estimated to emit 5.7e+11 rows (k = k)``."""
    conditions = next((extra[key] for key in _DUCKDB_CONDITION_KEYS if extra.get(key)), None)
    if isinstance(conditions, list):
        conditions = " AND ".join(str(condition) for condition in conditions)
    on = f" ({str(conditions)[:_CONDITION_CHARS]})" if conditions else " (no join condition)"
    return f"{name} estimated to {verb} {rows:.1e} rows{on}"


def _number(value: Any) -> float | None:
    """Parse a DuckDB estimate (a string in the JSON plan), or None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _product(values: Iterable[float]) -> float:
    """Multiply ``values``, each at least 1 (an empty input multiplies as 1)."""
    result = 1.0
    for value in values:
        result *= max(1.0, value)
    return result


# --------------------------------------------------------------------------------------- SQLite
def sqlite_plan_cost(
    plan_lines: Iterable[SqlitePlanRow], names: SqliteNames
) -> tuple[float | None, str]:
    """Estimate the row visits of a SQLite ``EXPLAIN QUERY PLAN``.

    SQLite plans have no cardinalities, so the estimate reads the plan's shape. The rows under
    one parent are loop levels, outer to inner. In each nest a full ``SCAN`` multiplies the
    running product of rows by the table's size and visits that many rows; an indexed
    ``SEARCH`` and a temp B-tree visit the running product once more, and an automatic index
    also reads its table once to build. A correlated subquery runs once per outer row, so its
    work is scaled by the running product; other sub-plans (materialised CTEs, co-routines,
    both sides of a compound query, ``IN`` lists) run once.

    A materialised CTE or subquery (``MATERIALIZE c``, ``CO-ROUTINE x``) is sized by the scan
    product of the nest that fills it, or by the smaller bound its SQL states (a literal
    ``LIMIT``; one row for an aggregate without ``GROUP BY``), so each later ``SCAN`` of it
    multiplies by that size. Both are upper bounds: a ``GROUP BY`` or ``DISTINCT`` inside
    returns fewer rows, which the plan does not say. A CTE that SQLite flattens into the query
    shows up as scans of its base tables instead.
    A table whose size is still unknown (a view that could not be counted, a recursive CTE)
    counts as one row: a false refusal costs more than a slow query, which the timeout still
    bounds.

    The walk reads the plan format of SQLite 3.36 and later (``SCAN t``); an older plan
    (``SCAN TABLE t``) has no estimate, so the gate fails open on it.

    Args:
        plan_lines: The ``(id, parent, unused, detail)`` rows.
        names: Sizes of the tables and CTEs behind the names the plan prints.

    Returns:
        (estimated row visits, the dominant term, e.g. ``"nested loop: SCAN T2 (540,800 rows)
        × SCAN T3 (1,000 rows)"``), or ``(None, "")`` for a plan in the old format.
    """
    children: dict[Any, list[tuple[Any, str]]] = {}
    for line in plan_lines:
        detail = str(line[3])
        if detail.startswith(_SQLITE_OLD_FORMAT):
            return None, ""
        children.setdefault(line[1], []).append((line[0], detail))
    walk = _SqliteWalk(children, names)
    nest = walk.nest(0, 1.0)
    return nest.work, nest.reason


class _Nest(NamedTuple):
    """What one SQLite loop nest costs.

    Attributes:
        work: Estimated row visits of the nest and its sub-plans, times the runs.
        reason: The largest single term, for the refusal message.
        rows: Upper bound of the rows one run of the nest yields: the product of its scans,
            else the sum of its sub-plans (the arms of a compound query), else 1.
    """

    work: float
    reason: str
    rows: float


class _SqliteWalk:
    """One walk over a SQLite plan, remembering the size of each named temporary result.

    Attributes:
        children: Plan rows by parent id, as (id, detail).
        names: See :func:`sqlite_plan_cost`.
        filled: Upper-bound rows of each ``MATERIALIZE`` / ``CO-ROUTINE`` result seen so far,
            by lowercase name. SQLite lists a result's sub-plan before the scans that read it.
    """

    def __init__(self, children: dict[Any, list[tuple[Any, str]]], names: SqliteNames) -> None:
        self.children = children
        self.names = names
        self.filled: dict[str, float] = {}

    def size(self, name: str) -> float:
        """Return the rows behind a name as the plan prints it; 1 when unknown."""
        source = (self.names.source(name) or name).lower()
        if source in self.filled:
            return self.filled[source]
        return float(self.names.table_rows(name) or 1)

    def fill(self, name: str, rows: float) -> None:
        """Record the size of a materialised result: its nest's rows, or its SQL's bound."""
        bound = self.names.max_rows(name)
        self.filled[name.lower()] = rows if bound is None else min(rows, max(1.0, bound))

    def nest(self, parent: Any, outer: float) -> _Nest:
        """Return the cost of the loop nest under ``parent``.

        Args:
            parent: The id whose rows form this nest.
            outer: How many times the nest runs (a correlated subquery's outer rows), else 1.
        """
        product = outer
        scanned = 1.0  # the product of this nest's own scans, without the outer runs
        scans: list[str] = []
        sub_rows = 0.0
        work = 0.0
        largest = (0.0, "")
        for node_id, detail in self.children.get(parent, []):
            words = detail.split()
            name = words[1] if len(words) > 1 else ""
            term = (0.0, "")
            if detail.startswith("SCAN ") and "CONSTANT ROW" not in detail:
                size = self.size(name)
                product *= max(1.0, size)
                scanned *= max(1.0, size)
                scans.append(f"SCAN {name} ({size:,.0f} rows)")
                term = (product, _loop_reason(scans, outer))
            elif detail.startswith("SEARCH "):
                if "AUTOMATIC" in detail:
                    work += self.size(name)  # building the automatic index
                term = (product, f"{detail} once per outer row ({product:.1e} lookups)")
            elif detail.startswith("USE TEMP B-TREE"):
                term = (product, f"{detail} over {product:.1e} rows")
            elif "BLOOM FILTER" in detail:
                continue
            elif detail.startswith("CORRELATED "):
                sub = self.nest(node_id, product)
                term = (sub.work, sub.reason)
            else:  # MATERIALIZE, CO-ROUTINE, MERGE, LEFT/RIGHT, LIST/SCALAR SUBQUERY, ...
                sub = self.nest(node_id, 1.0)
                if detail.startswith(_SQLITE_NAMED_RESULTS):
                    self.fill(name, sub.rows)
                sub_rows += sub.rows
                term = (sub.work, sub.reason)
            work += term[0]
            largest = max(largest, term, key=lambda item: item[0])
        rows = scanned if scans else (sub_rows or 1.0)
        return _Nest(work, largest[1], rows)


def _loop_reason(scans: list[str], outer: float) -> str:
    """Describe a nest's scans so far: ``nested loop: SCAN a (10 rows) × SCAN b (20 rows)``."""
    loops = " × ".join(scans)
    if outer > 1:
        return f"correlated subquery run {outer:.1e} times: {loops}"
    return f"nested loop: {loops}" if len(scans) > 1 else loops
