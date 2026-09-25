"""Read-only, bounded execution against DuckDB and SQLite files.

Three layers, each sufficient against the common cases and together against parser differentials:

1. :func:`~schemagraph.agent.guard.guard_sql` (one query, no writes, no file access);
2. the engine's own statement check (DuckDB ``extract_statements`` must be one SELECT; SQLite's
   ``execute`` refuses more than one statement);
3. a hardened connection. DuckDB: ``read_only`` plus ``enable_external_access=false`` and
   ``lock_configuration=true`` (read-only alone still lets ``read_csv('/etc/passwd')`` through).
   SQLite: ``mode=ro`` plus an authorizer that allows only SELECT/READ/FUNCTION/RECURSIVE
   (``mode=ro`` alone still lets ``ATTACH`` create a file) and ``SQLITE_LIMIT_ATTACHED=0``.

Every call is bounded: a wall-clock timeout (DuckDB ``interrupt()``, SQLite progress handler), a
preview limit and a row-count cap. Catalog introspection (``catalog()``, ``row_count()``) runs fixed
SQL over a separate trusted path, never model-written text.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import duckdb

from schemagraph.agent.guard import GuardedSQL, GuardError, guard_sql
from schemagraph.agent.results import ErrorKind, ExecResult
from schemagraph.store import substitute_env

if TYPE_CHECKING:
    from schemagraph.engine import Engine

# Rows kept and counted when the caller asks for the full result (``limit=None``, evaluation).
EVAL_MAX_ROWS = 1_000_000
# Default preview rows, query timeout and row-count cap of :meth:`Executor.execute`.
DEFAULT_PREVIEW_ROWS = 1000
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_COUNT_CAP = 100_000
# Default timeout of :meth:`Executor.explain`: binding a query is cheap.
EXPLAIN_TIMEOUT_S = 10.0
# Timeout of the trusted ``count(*)`` behind the row-explosion check; a view can be arbitrarily
# expensive, so the count is None on timeout.
COUNT_TIMEOUT_S = 10.0
# Bytes per SQLite string/blob value: one zeroblob() cannot allocate gigabytes.
SQLITE_MAX_LENGTH = 10_000_000
# Longest SQL text SQLite accepts.
SQLITE_MAX_SQL = 1_000_000
# Characters kept per previewed text cell and per error message.
CELL_CHARS = 200
ERROR_CHARS = 500
# DuckDB resources per connection; SCHEMAGRAPH_EXEC_MAX_MEMORY overrides the memory limit.
DUCKDB_THREADS = 4
DUCKDB_MAX_MEMORY = "2GB"

_FETCH_BATCH_ROWS = 1000
_SQLITE_PROGRESS_OPS = 10_000  # virtual-machine instructions between two timeout checks
_SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db3"}
_SQLITE_HEADER = b"SQLite format 3\x00"
_NOT_ONE_SELECT = "the database parsed this as something other than one SELECT statement"


class AgentError(RuntimeError):
    """The agent cannot run: no executable connection or database file."""


class Executor(Protocol):
    """Read-only, bounded access to one database file.

    Attributes:
        dialect: The sqlglot dialect of the database: ``duckdb`` or ``sqlite``.
        name: A short display name (the file stem).
    """

    dialect: str
    name: str

    def execute(
        self,
        sql: str,
        *,
        limit: int | None = DEFAULT_PREVIEW_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        count_cap: int = DEFAULT_COUNT_CAP,
        raw: bool = False,
    ) -> ExecResult:
        """Guard and run one read-only query.

        Args:
            sql: Model-written SQL.
            limit: Rows to keep in the result; None keeps the full result, up to
                :data:`EVAL_MAX_ROWS`, and fails with ``row_cap`` beyond it.
            timeout_s: Wall-clock limit in seconds.
            count_cap: Rows to count past the preview before reporting ``row_count_capped``.
            raw: Keep cell values as the driver returns them instead of JSON-safe previews.

        Returns:
            The result, or an error result with ``error_kind`` set; this never raises.
        """
        ...

    def explain(self, sql: str, *, timeout_s: float = EXPLAIN_TIMEOUT_S) -> ExecResult:
        """Guard and plan the query without running it: catches unknown names and type errors."""
        ...

    def catalog(self) -> dict[str, list[str]]:
        """Map lowercase ``schema.table`` (and unambiguous bare ``table``) keys to column names."""
        ...

    def resolve_table(self, name: str) -> str | None:
        """Return the catalog key a raw table name resolves to, or None."""
        ...

    def quoted_name(self, key: str) -> str:
        """Return the table behind a catalog key as a quoted identifier."""
        ...

    def row_count(self, table: str) -> int | None:
        """Return a table's row count (memoised), or None when unknown or the count timed out."""
        ...

    def close(self) -> None:
        """Release the connections."""
        ...


class _Cursor(Protocol):
    """The DB-API cursor surface the result collector reads."""

    @property
    def description(self) -> Any: ...

    def fetchmany(self, size: int) -> Sequence[Sequence[Any]]: ...


class _Refused(Exception):
    """Internal: a query was refused before it ran; ``result`` is the error to return."""

    def __init__(self, result: ExecResult) -> None:
        super().__init__(result.error)
        self.result = result


def json_safe(value: Any) -> Any:
    """Return a JSON-serialisable preview of one cell: long text is cut, blobs are summarised."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= CELL_CHARS else value[:CELL_CHARS] + "…"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<blob {len(bytes(value))} bytes>"
    if isinstance(value, (Decimal, date, datetime, dtime)):
        return str(value)
    return json_safe(str(value))


def _collect(
    cursor: _Cursor,
    *,
    limit: int | None,
    count_cap: int,
    raw: bool,
    started: float,
) -> ExecResult:
    """Fetch a query's result, keeping a preview and counting rows up to a cap.

    Args:
        cursor: A cursor that has executed the query.
        limit: Rows to keep; None keeps and counts up to :data:`EVAL_MAX_ROWS`.
        count_cap: Rows to count when ``limit`` is set (at least ``limit``).
        raw: Keep driver values instead of :func:`json_safe` previews.
        started: ``time.perf_counter()`` when the query started.

    Returns:
        The result; with ``limit=None``, a ``row_cap`` error beyond :data:`EVAL_MAX_ROWS` rows.
    """
    columns = [column[0] for column in (cursor.description or [])]
    keep = EVAL_MAX_ROWS if limit is None else max(0, limit)
    cap = EVAL_MAX_ROWS if limit is None else max(count_cap, keep)
    rows: list[list[Any]] = []
    seen = 0
    while seen <= cap:  # fetch one row past the cap to tell "exactly cap rows" from "more"
        batch = cursor.fetchmany(min(_FETCH_BATCH_ROWS, cap + 1 - seen))
        if not batch:
            break
        for row in batch:
            if len(rows) < keep:
                rows.append(list(row) if raw else [json_safe(value) for value in row])
        seen += len(batch)
    elapsed_ms = _elapsed_ms(started)
    if limit is None and seen > EVAL_MAX_ROWS:
        return ExecResult(
            ok=False,
            error=f"result has more than {EVAL_MAX_ROWS} rows",
            error_kind="row_cap",
            columns=columns,
            row_count=EVAL_MAX_ROWS,
            row_count_capped=True,
            elapsed_ms=elapsed_ms,
        )
    capped = seen > cap
    count = cap if capped else seen
    kept = rows[:count]
    return ExecResult(
        ok=True,
        columns=columns,
        rows=kept,
        row_count=count,
        row_count_capped=capped,
        truncated=count > len(kept),
        elapsed_ms=elapsed_ms,
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _fail(kind: ErrorKind, message: str, started: float | None = None) -> ExecResult:
    """Return an error result; ``started`` (a ``perf_counter`` reading) sets the elapsed time."""
    return ExecResult(
        ok=False,
        error=message[:ERROR_CHARS],
        error_kind=kind,
        elapsed_ms=_elapsed_ms(started) if started else 0.0,
    )


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _qualified(schema: str | None, table: str) -> str:
    """Return ``"schema"."table"``, or ``"table"`` without a schema."""
    return f"{_quote(schema)}.{_quote(table)}" if schema else _quote(table)


# One introspected table: (schema or None, table name, column names).
_TableColumns = tuple[str | None, str, list[str]]


class _CatalogExecutor:
    """Catalog, name resolution and row counts shared by both executors.

    Subclasses provide the trusted introspection and count queries, the engine statement check
    and the query runner.

    Attributes:
        default_schema: Where the engine resolves a bare name (DuckDB's search path: ``main``).
    """

    dialect: str
    name: str
    default_schema: str | None = None

    def __init__(self) -> None:
        self._catalog: dict[str, list[str]] | None = None
        self._parts: dict[str, tuple[str | None, str]] = {}  # key -> (schema, table) as stored
        self._counts: dict[str, int | None] = {}

    def _introspect(self) -> list[_TableColumns]:
        raise NotImplementedError

    def _count(self, schema: str | None, table: str) -> int | None:
        raise NotImplementedError

    def _admit(self, sql: str) -> GuardedSQL:
        """Run the guard (layer 1) and the engine statement check (layer 2), or raise _Refused."""
        try:
            return guard_sql(sql, self.dialect)
        except GuardError as error:
            raise _Refused(_fail("guard", str(error))) from error

    def catalog(self) -> dict[str, list[str]]:
        """Map lowercase ``schema.table`` (and unambiguous bare ``table``) keys to column names."""
        if self._catalog is None:
            self._catalog = self._build_catalog(self._introspect())
        return self._catalog

    def _build_catalog(self, tables: list[_TableColumns]) -> dict[str, list[str]]:
        """Key every table by ``schema.table``, plus its bare name when that is unambiguous.

        Records each key's stored ``(schema, table)`` in ``self._parts``.
        """
        catalog: dict[str, list[str]] = {}
        by_bare_name: dict[str, list[_TableColumns]] = {}
        for schema, table, columns in tables:
            if schema:
                key = f"{schema}.{table}".lower()
                catalog[key] = columns
                self._parts[key] = (schema, table)
            by_bare_name.setdefault(table.lower(), []).append((schema, table, columns))
        for bare, hits in by_bare_name.items():
            if len(hits) == 1 and bare not in catalog:
                schema, table, columns = hits[0]
                catalog[bare] = columns
                self._parts[bare] = (schema, table)
        return catalog

    def resolve_table(self, name: str) -> str | None:
        """Return the catalog key for a raw table name, resolved like the engine does.

        In order: an exact key, ``catalog.schema.table`` without the catalog, a bare name in the
        default schema, else a unique ``.name`` suffix. None when nothing or several tables match.
        """
        catalog = self.catalog()
        lowered = name.strip().strip('`"').lower()
        if lowered in catalog:
            return self._canonical(lowered)
        parts = lowered.split(".")
        without_catalog = ".".join(parts[1:])
        if len(parts) == 3 and without_catalog in catalog:
            return without_catalog
        last = parts[-1]
        in_default = f"{self.default_schema}.{last}"
        if len(parts) == 1 and self.default_schema and in_default in catalog:
            return in_default  # `orders` with main.orders and raw.orders: main wins
        hits = [key for key in catalog if key == last or key.endswith("." + last)]
        if len({self._parts[key] for key in hits}) != 1:
            return None
        return max(hits, key=lambda key: key.count("."))  # SQLite's `main.t` -> key `t`

    def _canonical(self, key: str) -> str:
        """Return one key per table: ``schema.table`` when the catalog has schemas.

        A bare and a qualified reference to the same table then never count as two tables.
        """
        schema, table = self._parts[key]
        full = f"{schema}.{table}".lower() if schema else key
        return full if full in self.catalog() else key

    def quoted_name(self, key: str) -> str:
        """Return the table behind a catalog key as a quoted identifier (never from free text)."""
        schema, table = self._parts[key]
        return _qualified(schema, table)

    def row_count(self, table: str) -> int | None:
        """Return a table's row count (memoised), or None when unknown or the count failed."""
        key = self.resolve_table(table)
        if key is None:
            return None
        if key not in self._counts:
            schema, name = self._parts[key]
            try:
                self._counts[key] = self._count(schema, name)
            except Exception:
                self._counts[key] = None
        return self._counts[key]


class DuckDBExecutor(_CatalogExecutor):
    """Read-only DuckDB file with external access off and the configuration locked.

    Attributes:
        path: The database file.
        schemas: Lowercase schemas the agent is shown, or None for all of them.
    """

    dialect = "duckdb"
    default_schema = "main"

    def __init__(self, path: str | Path, *, schemas: list[str] | None = None):
        """Open ``path`` read-only.

        Args:
            path: An existing DuckDB file; it is never created.
            schemas: Limits what the agent is shown (catalog, suggestions, checks) to the schemas
                the connection introspects; queries still run against the whole file, read-only.

        Raises:
            FileNotFoundError: ``path`` is not a file.
        """
        super().__init__()
        self.schemas = {schema.lower() for schema in schemas} if schemas else None
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.name = self.path.stem
        # every setting is fixed at connect time; lock_configuration then refuses SET
        self._con = duckdb.connect(
            str(self.path),
            read_only=True,
            config={
                "enable_external_access": False,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
                "allow_community_extensions": False,
                "max_memory": os.environ.get("SCHEMAGRAPH_EXEC_MAX_MEMORY", DUCKDB_MAX_MEMORY),
                "threads": DUCKDB_THREADS,
                "lock_configuration": True,
            },
        )

    def _statement_ok(self, cursor: duckdb.DuckDBPyConnection, sql: str) -> str | None:
        """Return why DuckDB does not see exactly one SELECT in ``sql``, or None."""
        statements = cursor.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            return _NOT_ONE_SELECT
        return None

    def _admit(self, sql: str) -> GuardedSQL:
        guarded = super()._admit(sql)
        cursor = self._con.cursor()
        try:
            problem = self._statement_ok(cursor, guarded.sql)
        except duckdb.Error as error:
            raise _Refused(_fail("syntax", str(error))) from error
        finally:
            cursor.close()
        if problem:
            raise _Refused(_fail("guard", problem))
        return guarded

    def _run(
        self,
        sql: str,
        timeout_s: float,
        then: Callable[[duckdb.DuckDBPyConnection, float], ExecResult],
    ) -> ExecResult:
        """Run admitted ``sql`` on a fresh cursor, interrupted after ``timeout_s`` seconds."""
        started = time.perf_counter()
        cursor = self._con.cursor()  # cursors share the locked configuration; one per call
        timer = threading.Timer(timeout_s, cursor.interrupt)
        try:
            timer.start()
            cursor.execute(sql)
            return then(cursor, started)
        except duckdb.InterruptException:
            return _fail("timeout", f"query exceeded {timeout_s:g}s", started)
        except (duckdb.ParserException, duckdb.BinderException, duckdb.CatalogException) as error:
            return _fail("syntax", str(error), started)
        except duckdb.Error as error:
            return _fail("runtime", str(error), started)
        finally:
            timer.cancel()
            cursor.close()

    def execute(
        self,
        sql: str,
        *,
        limit: int | None = DEFAULT_PREVIEW_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        count_cap: int = DEFAULT_COUNT_CAP,
        raw: bool = False,
    ) -> ExecResult:
        """Guard and run one read-only query (see :meth:`Executor.execute`)."""
        try:
            guarded = self._admit(sql)
        except _Refused as refused:
            return refused.result

        def fetch(cursor: duckdb.DuckDBPyConnection, started: float) -> ExecResult:
            return _collect(cursor, limit=limit, count_cap=count_cap, raw=raw, started=started)

        return self._run(guarded.sql, timeout_s, fetch)

    def explain(self, sql: str, *, timeout_s: float = EXPLAIN_TIMEOUT_S) -> ExecResult:
        """Bind the query without running it: catches unknown tables/columns and type errors."""
        try:
            guarded = self._admit(sql)  # DuckDB would run every statement after EXPLAIN
        except _Refused as refused:
            return refused.result

        def planned(_cursor: duckdb.DuckDBPyConnection, started: float) -> ExecResult:
            return ExecResult(ok=True, elapsed_ms=_elapsed_ms(started))

        return self._run("EXPLAIN " + guarded.sql, timeout_s, planned)

    def _introspect(self) -> list[_TableColumns]:
        cursor = self._con.cursor()
        try:
            rows = cursor.execute(_DUCKDB_COLUMNS_SQL).fetchall()
        finally:
            cursor.close()
        tables: dict[tuple[str, str], list[str]] = {}
        for schema, table, column in rows:
            if self.schemas is None or schema.lower() in self.schemas:
                tables.setdefault((schema, table), []).append(column)
        return [(schema, table, columns) for (schema, table), columns in tables.items()]

    def _count(self, schema: str | None, table: str) -> int | None:
        cursor = self._con.cursor()
        timer = threading.Timer(COUNT_TIMEOUT_S, cursor.interrupt)
        try:
            timer.start()
            # the name comes from the catalog, quoted
            row = cursor.execute(f"SELECT count(*) FROM {_qualified(schema, table)}").fetchone()
            return int(row[0]) if row else None
        finally:
            timer.cancel()
            cursor.close()

    def close(self) -> None:
        """Close the connection."""
        self._con.close()


_DUCKDB_COLUMNS_SQL = (
    "SELECT schema_name, table_name, column_name FROM duckdb_columns() WHERE NOT internal "
    "ORDER BY schema_name, table_name, column_index"
)
_SQLITE_TABLES_SQL = (
    "SELECT name FROM sqlite_schema WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
    "ORDER BY name"
)
_SQLITE_COLUMNS_SQL = "SELECT name FROM pragma_table_info(?) ORDER BY cid"

# SQLite authorizer: everything else (ATTACH/DETACH, PRAGMA and pragma_* table functions, writes,
# DDL, transactions) is denied. SQLITE_RECURSIVE is 33 on Pythons that do not export it.
_SQLITE_ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}
_SQLITE_DENIED_FUNCS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


def _sqlite_authorizer(
    action: int,
    _arg1: str | None,
    arg2: str | None,
    _db: str | None,
    _trigger: str | None,
) -> int:
    """Allow only reads and non-file functions (``arg2`` is the function name for FUNCTION)."""
    if action not in _SQLITE_ALLOWED:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _SQLITE_DENIED_FUNCS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _deadline_handler(timeout_s: float) -> Callable[[], int]:
    """Return a SQLite progress handler that aborts the statement after ``timeout_s`` seconds."""
    deadline = time.monotonic() + timeout_s
    return lambda: 1 if time.monotonic() > deadline else 0


def _sqlite_error_kind(error: sqlite3.Error) -> ErrorKind:
    """Classify a SQLite error; a refusal by the authorizer or the one-statement rule is a guard."""
    message = str(error)
    if "interrupted" in message:
        return "timeout"
    if "not authorized" in message or "one statement" in message:
        return "guard"
    if isinstance(error, sqlite3.ProgrammingError):
        return "syntax"  # e.g. a stray ? / :x placeholder: a broken query, not a write attempt
    if any(marker in message for marker in ("syntax error", "no such", "ambiguous", "misuse")):
        return "syntax"
    return "runtime"


class SQLiteExecutor(_CatalogExecutor):
    """Read-only SQLite file behind an authorizer, with a fresh hardened connection per query.

    Attributes:
        path: The resolved database file.
    """

    dialect = "sqlite"

    def __init__(self, path: str | Path):
        """Prepare read-only access to ``path``.

        Raises:
            FileNotFoundError: ``path`` is not a file (it is never created).
        """
        super().__init__()
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.name = self.path.stem
        self._uri = self.path.as_uri() + "?mode=ro"
        self._trusted: sqlite3.Connection | None = None
        self._trusted_lock = threading.Lock()  # one trusted connection, shared by worker threads

    def _connect(self, timeout_s: float) -> sqlite3.Connection:
        """Open a fresh hardened connection per call: cheap, and thread-safe by construction."""
        con = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        con.execute("PRAGMA query_only = ON")  # before the authorizer, which denies every PRAGMA
        con.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        # the progress handler cannot stop one huge allocation
        con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, SQLITE_MAX_LENGTH)
        con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, SQLITE_MAX_SQL)
        con.set_authorizer(_sqlite_authorizer)
        con.set_progress_handler(_deadline_handler(timeout_s), _SQLITE_PROGRESS_OPS)
        return con

    def _run(
        self,
        sql: str,
        timeout_s: float,
        then: Callable[[sqlite3.Cursor, float], ExecResult],
    ) -> ExecResult:
        """Run ``sql`` on a hardened connection; SQLite's ``execute`` refuses a second statement."""
        started = time.perf_counter()
        con = self._connect(timeout_s)
        try:
            return then(con.execute(sql), started)
        except sqlite3.Error as error:
            kind = _sqlite_error_kind(error)
            message = f"query exceeded {timeout_s:g}s" if kind == "timeout" else str(error)
            return _fail(kind, message, started)
        finally:
            con.close()

    def execute(
        self,
        sql: str,
        *,
        limit: int | None = DEFAULT_PREVIEW_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        count_cap: int = DEFAULT_COUNT_CAP,
        raw: bool = False,
    ) -> ExecResult:
        """Guard and run one read-only query (see :meth:`Executor.execute`)."""
        try:
            guarded = self._admit(sql)
        except _Refused as refused:
            return refused.result

        def fetch(cursor: sqlite3.Cursor, started: float) -> ExecResult:
            return _collect(cursor, limit=limit, count_cap=count_cap, raw=raw, started=started)

        return self._run(guarded.sql, timeout_s, fetch)

    def explain(self, sql: str, *, timeout_s: float = EXPLAIN_TIMEOUT_S) -> ExecResult:
        """Plan the query without running it: catches unknown tables/columns."""
        try:
            guarded = self._admit(sql)
        except _Refused as refused:
            return refused.result

        def planned(cursor: sqlite3.Cursor, started: float) -> ExecResult:
            cursor.fetchall()
            return ExecResult(ok=True, elapsed_ms=_elapsed_ms(started))

        return self._run("EXPLAIN QUERY PLAN " + guarded.sql, timeout_s, planned)

    def _trusted_con(self) -> sqlite3.Connection:
        """Return the trusted connection for fixed catalog SQL (no authorizer; call under lock)."""
        if self._trusted is None:
            self._trusted = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
            self._trusted.execute("PRAGMA query_only = ON")
        return self._trusted

    def _introspect(self) -> list[_TableColumns]:
        with self._trusted_lock:
            con = self._trusted_con()
            names = [row[0] for row in con.execute(_SQLITE_TABLES_SQL)]
            tables: list[_TableColumns] = []
            for name in names:
                columns = [row[0] for row in con.execute(_SQLITE_COLUMNS_SQL, (name,))]
                tables.append((None, name, columns))
            return tables

    def _count(self, schema: str | None, table: str) -> int | None:
        with self._trusted_lock:
            con = self._trusted_con()
            con.set_progress_handler(_deadline_handler(COUNT_TIMEOUT_S), _SQLITE_PROGRESS_OPS)
            try:
                # the name comes from the catalog, quoted
                row = con.execute(f"SELECT count(*) FROM {_quote(table)}").fetchone()
            finally:
                con.set_progress_handler(None, 0)
            return int(row[0]) if row else None

    def close(self) -> None:
        """Close the trusted connection; query connections are closed after every call."""
        with self._trusted_lock:
            if self._trusted is not None:
                self._trusted.close()
                self._trusted = None


def executor_for_path(path: str | Path) -> Executor:
    """Open a database file by suffix, else by its header.

    ``.duckdb`` is DuckDB and ``.sqlite``/``.sqlite3``/``.db3`` are SQLite; any other file is SQLite
    when it starts with the SQLite header, else DuckDB.

    Raises:
        FileNotFoundError: ``path`` is missing; executors never create databases.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(file)
    suffix = file.suffix.lower()
    if suffix == ".duckdb":
        return DuckDBExecutor(file)
    if suffix in _SQLITE_SUFFIXES:
        return SQLiteExecutor(file)
    with file.open("rb") as handle:
        header = handle.read(len(_SQLITE_HEADER))
    return SQLiteExecutor(file) if header == _SQLITE_HEADER else DuckDBExecutor(file)


def executor_for_connection(
    engine: Engine,
    connection: str | None,
    *,
    db: str | Path | None = None,
) -> Executor:
    """Return the executor for a database file, or behind a registered ``duckdb`` connection.

    Args:
        engine: The engine whose store holds the connection.
        connection: A registered connection name; only ``duckdb`` connections execute.
        db: A database file that takes precedence over ``connection``.

    Raises:
        AgentError: Neither is given, or the connection is not registered or not a ``duckdb``
            one.
    """
    if db is not None:
        return executor_for_path(db)
    if connection is None:
        raise AgentError("pass a duckdb connection or a database file")
    row = engine.store.connection(connection)
    if row is None:
        raise AgentError(f"unknown connection {connection!r}")
    type_name, config = row
    if type_name != "duckdb":
        raise AgentError(
            f"connection {connection!r} is {type_name}; "
            "execution needs a duckdb connection, or pass --db PATH"
        )
    cfg = substitute_env(config)
    return DuckDBExecutor(cfg["path"], schemas=cfg.get("schemas"))
