"""Deterministic checks on a candidate query and its result (sqlglot and plain lookups, no LLM).

Each finding carries a penalty and a message that goes verbatim into the refine prompt, so the
generator gets "`orders.cust_id` does not exist; closest is `customer_id`" rather than a number.
Checks use the live catalog of the database, not the graph, so a stale snapshot cannot turn a
correct column into an "unknown" one. Relations for the join check come from caller-supplied
lookups, so the checks stay pure and synchronous. Penalties are v1 guesses to be refit on the
judge pool; a join that is not a known relation is reported (``info``) but not penalised until it
is measured.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable

from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from schemagraph.agent.guard import GuardedSQL, func_name
from schemagraph.agent.results import CheckReport, ExecResult, Finding
from schemagraph.model import Edge

# Penalty per finding, by code (v1 guesses; to be refit on the judge pool).
PENALTY = {
    "unknown_table": 0.4,
    "unknown_column": 0.3,
    "cartesian": 0.3,
    "ungrouped_column": 0.2,
    "empty_result": 0.3,
    "null_column": 0.15,
    "row_explosion": 0.3,
}
# Most one all-NULL-columns finding can cost, however many columns it names.
NULL_COLUMN_MAX = 0.3
# Most one code can cost in total: eleven quoted literals are one mistake, not eleven.
PENALTY_CAP = {
    "unknown_column": 0.6,
    "unknown_table": 0.8,
    "ungrouped_column": 0.4,
    "cartesian": 0.3,
    "null_column": NULL_COLUMN_MAX,
}
# Columns every SQLite table has without declaring them.
SQLITE_IMPLICIT = frozenset({"rowid", "oid", "_rowid_"})
# A result more than this many times the largest table read is a row explosion.
EXPLOSION_FACTOR = 2

# Table-valued functions that produce rows without a base table behind them.
_GENERATORS = {"range", "generate_series", "unnest", "json_each", "json_tree"}
# How many close names a suggestion lists, and how close they must be (difflib ratio).
_SUGGESTIONS = 2
_SUGGESTION_CUTOFF = 0.6
# How many known relations a join_off_graph message lists.
_KNOWN_RELATIONS_SHOWN = 3

JoinRelations = Callable[[str, str], list[Edge]]
JoinPath = Callable[[str, str], list[list[str]]]
# Alias -> catalog key within one SELECT scope; None for CTEs, derived tables, table functions.
_ScopeKeys = dict[str, str | None]


def _norm(name: str) -> str:
    return name.strip().strip('`"[]').lower()


def _closest(word: str, options: Iterable[str]) -> str:
    """Return up to two close names from ``options`` as a backquoted list, or ``""``."""
    hits = difflib.get_close_matches(
        word.lower(),
        [option.lower() for option in options],
        n=_SUGGESTIONS,
        cutoff=_SUGGESTION_CUTOFF,
    )
    return ", ".join(f"`{hit}`" for hit in hits)


def _with_suggestion(message: str, closest: str) -> str:
    return message + (f"; closest: {closest}" if closest else "")


def _table_name(table: exp.Table) -> str:
    return ".".join(_norm(part) for part in (table.catalog, table.db, table.name) if part)


def _under(
    node: exp.Expression,
    kinds: tuple[type[exp.Expression], ...],
    stop: exp.Expression | None = None,
) -> bool:
    """Return whether ``node`` has an ancestor of ``kinds``, up to and including ``stop``."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, kinds):
            return True
        if parent is stop:
            return False
        parent = parent.parent
    return False


def det_of(findings: list[Finding]) -> float:
    """Return 1 minus the penalties, each code capped by :data:`PENALTY_CAP`."""
    per_code: dict[str, float] = {}
    for finding in findings:
        per_code[finding.code] = per_code.get(finding.code, 0.0) + finding.penalty
    capped = sum(min(total, PENALTY_CAP.get(code, total)) for code, total in per_code.items())
    return max(0.0, 1.0 - capped)


def static_checks(
    guarded: GuardedSQL,
    catalog: dict[str, list[str]],
    resolve_table: Callable[[str], str | None],
    join_relations: JoinRelations | None = None,
    join_path: JoinPath | None = None,
) -> CheckReport:
    """Check a guarded query against the live catalog, before looking at its result.

    Aliases are resolved per SELECT scope (walking out for correlated references), so ``t`` may
    name a different table in a subquery.

    Args:
        guarded: The query that passed the guard.
        catalog: Lowercase table key -> column names (``Executor.catalog``).
        resolve_table: Maps a raw table name to its catalog key, or None
            (``Executor.resolve_table``).
        join_relations: Join-capable relations between two catalog keys. Without it, joins are not
            compared with known relations.
        join_path: Shortest known join paths between two catalog keys, for the suggestion when
            two joined tables have no direct relation. Without it, there is no suggestion.

    Returns:
        The findings, the catalog keys the query reads (in order) and the deterministic score.
    """
    tree = guarded.tree
    sqlite = guarded.dialect == "sqlite"
    cte_names = {_norm(cte.alias_or_name) for cte in tree.find_all(exp.CTE)}
    columns_of = {key: {column.lower() for column in columns} for key, columns in catalog.items()}

    findings, used = _table_findings(tree, cte_names, catalog, resolve_table)
    scopes = _scopes(tree)
    keys_of = {id(scope): _scope_keys(scope, cte_names, resolve_table) for scope in scopes}
    reported: set[tuple[str, str]] = set()
    findings += _qualified_column_findings(
        scopes, keys_of, columns_of, catalog, sqlite=sqlite, reported=reported
    )
    if used and not _reads_table_function(tree):
        findings += _unqualified_column_findings(
            tree, used, columns_of, catalog, sqlite=sqlite, reported=reported
        )
    for scope in scopes:
        findings += _scope_checks(scope, keys_of[id(scope)], columns_of, join_relations, join_path)
    findings += _ungrouped(tree, guarded.dialect)
    return CheckReport(parsed=True, findings=findings, tables=used, det=det_of(findings))


def tables_read(guarded: GuardedSQL, resolve_table: Callable[[str], str | None]) -> list[str]:
    """Return the catalog keys a query reads in first-use order, as :func:`static_checks` does.

    CTE names and names ``resolve_table`` does not know are left out. Callers use it to fetch the
    join lookups for exactly these tables before running the checks.
    """
    tree = guarded.tree
    cte_names = {_norm(cte.alias_or_name) for cte in tree.find_all(exp.CTE)}
    _, used = _table_findings(tree, cte_names, {}, resolve_table)
    return used


def join_keys(
    guarded: GuardedSQL, resolve_table: Callable[[str], str | None]
) -> list[tuple[str, str, str, str]]:
    """Return the ``a.x = b.y`` join conditions between two base tables of a query.

    Each is ``(table_a, column_a, table_b, column_b)`` in catalog keys, deduplicated in first-use
    order. Only conditions of ``ON`` clauses whose two sides are columns qualified by the alias
    of a base table (not a CTE or a derived table) count: those are the joins whose key
    uniqueness can be measured on the database.
    """
    tree = guarded.tree
    cte_names = {_norm(cte.alias_or_name) for cte in tree.find_all(exp.CTE)}
    found: list[tuple[str, str, str, str]] = []
    for select in tree.find_all(exp.Select):
        aliases: dict[str, str] = {}
        for table in select.find_all(exp.Table):
            if table.find_ancestor(exp.Select) is not select or _table_name(table) in cte_names:
                continue
            key = resolve_table(_table_name(table))
            if key:
                aliases[_norm(table.alias_or_name)] = key
        for join in select.args.get("joins") or []:
            condition = join.args.get("on")
            if condition is None:
                continue
            for equality in condition.find_all(exp.EQ):
                left, right = equality.this, equality.expression
                if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                    continue
                table_a, table_b = aliases.get(_norm(left.table)), aliases.get(_norm(right.table))
                if table_a and table_b and table_a != table_b:
                    pair = (table_a, left.name, table_b, right.name)
                    if pair not in found:
                        found.append(pair)
    return found


def _reads_table_function(tree: exp.Expression) -> bool:
    """Return whether the query reads a table function or a row generator in FROM/JOIN.

    Such a source has columns the catalog does not know, so unqualified columns cannot be checked.
    """
    if any(isinstance(table.this, exp.Func) for table in tree.find_all(exp.Table)):
        return True
    return any(
        func_name(function) in _GENERATORS and _under(function, (exp.From, exp.Join))
        for function in tree.find_all(exp.Func)
    )


def _table_findings(
    tree: exp.Expression,
    cte_names: set[str],
    catalog: dict[str, list[str]],
    resolve_table: Callable[[str], str | None],
) -> tuple[list[Finding], list[str]]:
    """Report unknown tables; return the findings and the catalog keys read, in first-use order."""
    findings: list[Finding] = []
    used: list[str] = []
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Func) or not table.name:
            continue
        raw = _table_name(table)
        if raw in cte_names:
            continue
        key = resolve_table(raw)
        if key is None:
            closest = _closest(raw.split(".")[-1], sorted(catalog))
            findings.append(
                Finding(
                    code="unknown_table",
                    severity="error",
                    message=_with_suggestion(f"table `{raw}` does not exist", closest),
                    penalty=PENALTY["unknown_table"],
                )
            )
        elif key not in used:
            used.append(key)
    return findings, used


def _scopes(tree: exp.Expression) -> list[Scope]:
    """Return the query's SELECT scopes, or none when sqlglot cannot scope this construct."""
    try:
        return traverse_scope(tree)
    except Exception:  # the scope-level checks are skipped then
        return []


def _scope_keys(
    scope: Scope,
    cte_names: set[str],
    resolve_table: Callable[[str], str | None],
) -> _ScopeKeys:
    """Map this scope's source aliases to catalog keys; None for non-base-table sources."""
    keys: _ScopeKeys = {}
    for alias, source in scope.sources.items():
        is_base_table = (
            isinstance(source, exp.Table)
            and not isinstance(source.this, exp.Func)
            and _table_name(source) not in cte_names
        )
        keys[_norm(alias)] = resolve_table(_table_name(source)) if is_base_table else None
    return keys


def _qualifier_key(scope: Scope, qualifier: str, keys_of: dict[int, _ScopeKeys]) -> str | None:
    """Resolve a column qualifier in ``scope``, then outwards (correlated subqueries)."""
    current: Scope | None = scope
    while current is not None:
        keys = keys_of.get(id(current), {})
        if qualifier in keys:
            return keys[qualifier]
        current = current.parent
    return None


def _qualified_column_findings(
    scopes: list[Scope],
    keys_of: dict[int, _ScopeKeys],
    columns_of: dict[str, set[str]],
    catalog: dict[str, list[str]],
    *,
    sqlite: bool,
    reported: set[tuple[str, str]],
) -> list[Finding]:
    """Report ``alias.column`` references whose base table has no such column.

    Adds each reported ``(key, column)`` to ``reported`` so it is reported once.
    """
    findings: list[Finding] = []
    for scope in scopes:
        for column in scope.columns:
            name = _norm(column.name)
            qualifier = _norm(column.table) if column.table else ""
            if not qualifier or not name or isinstance(column.this, exp.Star):
                continue
            key = _qualifier_key(scope, qualifier, keys_of)
            if key is None:
                continue  # a CTE, a derived table, a table function or an unknown table
            valid = columns_of.get(key, set()) | (SQLITE_IMPLICIT if sqlite else set())
            if name in valid or (key, name) in reported:
                continue
            reported.add((key, name))
            closest = _closest(name, catalog.get(key, []))
            message = f"`{qualifier}.{name}` does not exist in `{key}`"
            findings.append(
                Finding(
                    code="unknown_column",
                    severity="error",
                    message=_with_suggestion(message, closest),
                    penalty=PENALTY["unknown_column"],
                )
            )
    return findings


def _derived_names(tree: exp.Expression) -> set[str]:
    """Return names a query may reference without a base table: aliases and alias column lists."""
    names = {_norm(alias.alias) for alias in tree.find_all(exp.Alias)}
    for table_alias in tree.find_all(exp.TableAlias):
        names.update(_norm(column.name) for column in table_alias.columns)
    return names


def _unqualified_column_findings(
    tree: exp.Expression,
    used: list[str],
    columns_of: dict[str, set[str]],
    catalog: dict[str, list[str]],
    *,
    sqlite: bool,
    reported: set[tuple[str, str]],
) -> list[Finding]:
    """Report bare columns that no table the query reads has and no alias defines.

    Adds each reported ``("", column)`` to ``reported`` so it is reported once.
    """
    base_columns = set().union(*(columns_of.get(key, set()) for key in used))
    if sqlite:
        base_columns |= SQLITE_IMPLICIT
    derived = _derived_names(tree)
    tables = ", ".join(f"`{key}`" for key in used)
    candidates = sorted({column for key in used for column in catalog.get(key, [])})
    findings: list[Finding] = []
    for column in tree.find_all(exp.Column):
        name = _norm(column.name)
        if column.table or not name or isinstance(column.this, exp.Star):
            continue
        if name in base_columns or name in derived or ("", name) in reported:
            continue
        reported.add(("", name))
        closest = _closest(name, candidates)
        message = _with_suggestion(f"column `{name}` is not in {tables}", closest)
        if sqlite and column.this.args.get("quoted"):
            message += " (in SQLite a double-quoted unknown name silently becomes a string)"
        findings.append(
            Finding(
                code="unknown_column",
                severity="error",
                message=message,
                penalty=PENALTY["unknown_column"],
            )
        )
    return findings


class _Components:
    """Union-find over a scope's source aliases: which sources a predicate connects."""

    def __init__(self, aliases: list[str]) -> None:
        self._parent = {alias: alias for alias in aliases}

    def __contains__(self, alias: str) -> bool:
        return alias in self._parent

    def find(self, alias: str) -> str:
        """Return the representative of ``alias``'s component (with path halving)."""
        parent = self._parent
        while parent[alias] != alias:
            parent[alias] = parent[parent[alias]]
            alias = parent[alias]
        return alias

    def union(self, alias: str, other: str) -> None:
        """Put ``alias`` and ``other`` in one component."""
        self._parent[self.find(alias)] = self.find(other)

    def groups(self) -> list[list[str]]:
        """Return the components, each in alias order, ordered by their first alias."""
        groups: dict[str, list[str]] = {}
        for alias in self._parent:
            groups.setdefault(self.find(alias), []).append(alias)
        return list(groups.values())


def _join_predicates(
    select: exp.Select,
    keys: _ScopeKeys,
    components: _Components,
) -> list[exp.Expression]:
    """Return the WHERE and ON predicates; connect sources joined by USING, NATURAL or CROSS.

    A USING/NATURAL join, or a cross join onto a derived table, is treated as intentional: it
    connects the joined source to every source before it.
    """
    predicates: list[exp.Expression] = []
    where = select.args.get("where")
    if where is not None:
        predicates.append(where.this)
    previous: list[str] = []
    source = select.args.get("from_") or select.args.get("from")
    if source is not None:
        previous.append(_norm(source.this.alias_or_name))
    for join in select.args.get("joins") or []:
        joined = _norm(join.this.alias_or_name)
        if join.args.get("on") is not None:
            predicates.append(join.args["on"])
        crosses_to_derived = join.kind == "CROSS" and keys.get(joined) is None
        if join.args.get("using") or join.method or crosses_to_derived:
            for earlier in previous:
                if earlier in components and joined in components:
                    components.union(joined, earlier)
        previous.append(joined)
    return predicates


def _column_owners(
    column: exp.Column,
    keys: _ScopeKeys,
    columns_of: dict[str, set[str]],
) -> list[str]:
    """Return the aliases a column can come from: its qualifier, else every source that has it."""
    qualifier = _norm(column.table) if column.table else ""
    if qualifier:
        return [qualifier] if qualifier in keys else []
    name = _norm(column.name)
    return [alias for alias, key in keys.items() if key and name in columns_of.get(key, set())]


def _scope_checks(
    scope: Scope,
    keys: _ScopeKeys,
    columns_of: dict[str, set[str]],
    join_relations: JoinRelations | None,
    join_path: JoinPath | None,
) -> list[Finding]:
    """Report cartesian products and joins that are not known relations, within one scope."""
    select = scope.expression
    if not isinstance(select, exp.Select) or len(keys) < 2:
        return []
    components = _Components(list(keys))
    predicates = _join_predicates(select, keys, components)
    for predicate in predicates:
        conjuncts = predicate.flatten() if isinstance(predicate, exp.And) else [predicate]
        for conjunct in conjuncts:
            owners = sorted(
                {
                    owner
                    for column in conjunct.find_all(exp.Column)
                    for owner in _column_owners(column, keys, columns_of)
                }
            )
            for alias, other in zip(owners, owners[1:], strict=False):
                components.union(alias, other)

    findings: list[Finding] = []
    with_base = [group for group in components.groups() if any(keys[alias] for alias in group)]
    if len(with_base) > 1:
        described = " | ".join(
            ", ".join(keys[alias] or alias for alias in group) for group in with_base
        )
        findings.append(
            Finding(
                code="cartesian",
                severity="warn",
                message=(
                    "no join predicate connects these groups of tables (cartesian product): "
                    f"{described}"
                ),
                penalty=PENALTY["cartesian"],
            )
        )
    if join_relations is not None:
        findings += _join_off_graph(predicates, keys, join_relations, join_path)
    return findings


def _describe_relation(edge: Edge) -> str:
    source = f"{edge.from_table}.{', '.join(edge.from_columns)}"
    target = f"{edge.to_table}.{', '.join(edge.to_columns)}"
    return f"`{source} -> {target}` ({edge.kind})"


def _join_off_graph(
    predicates: list[exp.Expression],
    keys: _ScopeKeys,
    join_relations: JoinRelations,
    join_path: JoinPath | None,
) -> list[Finding]:
    """Report ``a.x = b.y`` equalities between two base tables that no known relation backs."""
    findings: list[Finding] = []
    for predicate in predicates:
        for equality in predicate.find_all(exp.EQ):
            left, right = equality.this, equality.expression
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            if not (left.table and right.table):
                continue
            left_key, right_key = keys.get(_norm(left.table)), keys.get(_norm(right.table))
            if not left_key or not right_key or left_key == right_key:
                continue
            left_column, right_column = _norm(left.name), _norm(right.name)
            relations = join_relations(left_key, right_key)
            pairs = {
                (source.lower(), target.lower())
                for edge in relations
                for source, target in zip(edge.from_columns, edge.to_columns, strict=False)
            }
            pairs |= {(target, source) for source, target in pairs}
            if (left_column, right_column) in pairs:
                continue
            if relations:
                known = "; ".join(_describe_relation(e) for e in relations[:_KNOWN_RELATIONS_SHOWN])
                message = (
                    f"join `{left_key}.{left_column} = {right_key}.{right_column}` "
                    f"is not a known relation; known: {known}"
                )
            else:
                message = f"no known relation between `{left_key}` and `{right_key}`"
                paths = join_path(left_key, right_key) if join_path is not None else []
                if paths:
                    message += f"; shortest known path: {' -> '.join(paths[0])}"
            findings.append(
                Finding(code="join_off_graph", severity="info", message=message, penalty=0.0)
            )
    return findings


def _own_select(node: exp.Expression) -> exp.Expression | None:
    """Return the innermost SELECT that contains ``node``."""
    parent = node.parent
    while parent is not None and not isinstance(parent, exp.Select):
        parent = parent.parent
    return parent


def _ungrouped(tree: exp.Expression, dialect: str = "") -> list[Finding]:
    """Report projected columns neither aggregated nor grouped (SQLite returns an arbitrary row).

    Aggregates inside scalar subqueries do not make the outer SELECT an aggregate; GROUP BY ALL is
    DuckDB's "group by every non-aggregate", so it is never flagged.
    """
    findings: list[Finding] = []
    for select in tree.find_all(exp.Select):
        findings += _ungrouped_in_select(select, dialect)
    return findings


def _own_aggregates(select: exp.Select) -> list[exp.AggFunc]:
    """Return the aggregates that aggregate this SELECT (not a subquery's, not a window's)."""
    return [
        aggregate
        for projection in select.expressions
        for aggregate in projection.find_all(exp.AggFunc)
        if _own_select(aggregate) is select and not _under(aggregate, (exp.Window,), select)
    ]


def _ungrouped_in_select(select: exp.Select, dialect: str) -> list[Finding]:
    """Report the ungrouped projections of one SELECT: at most one finding per projection."""
    group = select.args.get("group")
    if group is not None and group.args.get("all"):
        return []
    aggregates = _own_aggregates(select)
    single_extreme = len(aggregates) == 1 and isinstance(aggregates[0], (exp.Min, exp.Max))
    if dialect == "sqlite" and single_extreme:
        return []  # SQLite's bare-column rule: with one min()/max(), bare columns come from its row
    if group is None and not aggregates:
        return []
    projections = select.expressions
    # ROLLUP / CUBE / GROUPING SETS included
    group_exprs = [node for node in group.walk() if node is not group] if group is not None else []
    group_sql = {expression.sql().lower() for expression in group_exprs}
    group_names = _grouped_names(projections, group_exprs, group_sql)
    ordinals = {
        int(expression.name)
        for expression in (group.expressions if group is not None else [])
        if isinstance(expression, exp.Literal) and expression.is_int
    }
    findings: list[Finding] = []
    for position, projection in enumerate(projections, 1):
        inner = projection.this if isinstance(projection, exp.Alias) else projection
        grouped_alias = isinstance(projection, exp.Alias) and _norm(projection.alias) in group_names
        if position in ordinals or inner.sql().lower() in group_sql or grouped_alias:
            continue
        column = _first_ungrouped_column(inner, select, group_names)
        if column is not None:
            findings.append(
                Finding(
                    code="ungrouped_column",
                    severity="warn",
                    message=f"`{column.sql()}` is selected but neither aggregated nor in GROUP BY",
                    penalty=PENALTY["ungrouped_column"],
                )
            )
    return findings


def _grouped_names(
    projections: list[exp.Expression],
    group_exprs: list[exp.Expression],
    group_sql: set[str],
) -> set[str]:
    """Return the grouped column names plus the names of projections that are grouped."""
    names = {_norm(node.name) for node in group_exprs if isinstance(node, exp.Column)}
    for projection in projections:
        grouped = projection.sql().lower() in group_sql or (
            isinstance(projection, exp.Alias) and projection.this.sql().lower() in group_sql
        )
        if grouped:
            names.add(_norm(projection.alias_or_name))
    return names


def _first_ungrouped_column(
    inner: exp.Expression,
    select: exp.Select,
    group_names: set[str],
) -> exp.Column | None:
    """Return the first bare column of a projection that is not grouped, or None."""
    for column in inner.find_all(exp.Column):
        aggregated = _under(column, (exp.AggFunc, exp.Window, exp.Subquery, exp.Filter), inner)
        if aggregated or isinstance(column.this, exp.Star) or _own_select(column) is not select:
            continue
        if _norm(column.name) not in group_names:
            return column
    return None


def result_checks(
    result: ExecResult,
    base_rows: dict[str, int | None],
    *,
    count_cap: int,
    guarded: GuardedSQL | None = None,
) -> list[Finding]:
    """Check an execution result.

    Args:
        result: What the query returned.
        base_rows: Row count of every table the query reads (None when unknown).
        count_cap: The row-count cap the query ran with; a capped count above the explosion
            limit is only reported when the limit is below the cap.
        guarded: The query, for the row-explosion check; without it that check is skipped.

    Returns:
        A timeout/database-error finding for a failed query, else empty-result, all-NULL-column
        and row-explosion findings.
    """
    if not result.ok:
        return [_failure_finding(result)]
    if result.row_count == 0:
        return [
            Finding(
                code="empty_result",
                severity="warn",
                message=(
                    "the query returned no rows; check filters, join keys and literal values "
                    "(sample_values shows real values)"
                ),
                penalty=PENALTY["empty_result"],
            )
        ]
    findings: list[Finding] = []
    if result.rows:
        findings += _null_columns(result)
    sizes = [rows for rows in base_rows.values() if rows]
    if sizes and guarded is not None:
        findings += _row_explosion(result, max(sizes), guarded, count_cap)
    return findings


def _failure_finding(result: ExecResult) -> Finding:
    if result.error_kind == "timeout":
        message = f"{result.error}; avoid cross joins and add join predicates or filters"
        return Finding(code="timeout", severity="error", message=message)
    code = "guard" if result.error_kind == "guard" else "exec_error"
    return Finding(code=code, severity="error", message=f"database error: {result.error}")


def _null_columns(result: ExecResult) -> list[Finding]:
    """Report the columns that are NULL in every previewed row, as one finding."""
    nulls = [
        column
        for index, column in enumerate(result.columns)
        if all(row[index] is None for row in result.rows)
    ]
    if not nulls:
        return []
    names = ", ".join(f"`{column}`" for column in nulls)
    return [
        Finding(
            code="null_column",
            severity="warn",
            message=f"column(s) {names} are NULL in every previewed row",
            penalty=min(NULL_COLUMN_MAX, PENALTY["null_column"] * len(nulls)),
        )
    ]


def _row_explosion(
    result: ExecResult,
    largest: int,
    guarded: GuardedSQL,
    count_cap: int,
) -> list[Finding]:
    """Report a result far larger than the largest table read, unless rows are generated."""
    tree = guarded.tree
    recursive = any(with_.args.get("recursive") for with_ in tree.find_all(exp.With))
    generator = any(func_name(function) in _GENERATORS for function in tree.find_all(exp.Func))
    limit = EXPLOSION_FACTOR * largest
    if recursive or generator or result.row_count <= limit:
        return []
    if result.row_count_capped and limit >= count_cap:
        return []  # the count stopped at the cap, so it says nothing about the real size
    plus = "+" if result.row_count_capped else ""
    message = (
        f"{result.row_count:,}{plus} rows, more than {EXPLOSION_FACTOR}x the largest table read "
        f"({largest:,}); a join is probably missing a key"
    )
    return [
        Finding(
            code="row_explosion",
            severity="warn",
            message=message,
            penalty=PENALTY["row_explosion"],
        )
    ]
