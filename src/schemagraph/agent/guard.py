"""Read-only SQL guard, the first of three read-only layers.

The other two layers are the engine's statement check and the hardened connection, both in
:mod:`schemagraph.agent.execute`.

The guard accepts exactly one query (``SELECT``, ``WITH … SELECT``, set operations, DuckDB
``FROM t``) and rejects anything that writes, changes session state or reaches outside the
database: DDL/DML, ``ATTACH``, ``PRAGMA``, ``SET``, ``COPY``, ``INSTALL``/``LOAD``,
``SELECT … INTO``, file-reading table functions and file-like table names (DuckDB replacement
scans). The text that passes is executed as written, not sqlglot's regenerated SQL; a parser
differential is caught by the engine-side checks. The one edit is the row cap the executor adds
(:func:`bounded_sql`), which appends a ``LIMIT`` or rewrites a literal one (or a count that
means "no limit") in place; the capped text goes through the guard and the engine check again.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel
from sqlglot.tokens import TokenType

# Longest query the guard parses; model-written SQL is far shorter, anything longer is abuse.
MAX_SQL_CHARS = 50_000

# Characters of a parser error kept in the guard message.
_PARSE_ERROR_CHARS = 300

# sqlglot warns "falling back to parsing as a Command" for LOAD/CALL/SHOW; the guard rejects those
# anyway, and model-written SQL would otherwise fill the log.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Node types that never belong in a read-only query. They are resolved by name so that a sqlglot
# upgrade that renames one cannot break the import: it just stops being checked by name, and the
# rule that the root must be a query still holds.
_DENY_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
    "Command", "Pragma", "Attach", "Detach", "Set", "Copy", "Use", "Transaction", "Commit",
    "Rollback", "Into", "Install", "LoadData", "Describe", "Summarize", "Show", "Analyze",
    "Cache", "Uncache", "Refresh", "Kill", "Grant", "Revoke", "Lock", "ReadCSV", "ReadParquet",
)  # fmt: skip


def _deny_types() -> tuple[type[exp.Expression], ...]:
    """Return the sqlglot classes behind :data:`_DENY_NAMES` that this sqlglot version defines."""
    types = []
    for name in _DENY_NAMES:
        node_type = getattr(exp, name, None)
        if isinstance(node_type, type):
            types.append(node_type)
    return tuple(types)


DENY_TYPES: tuple[type[exp.Expression], ...] = _deny_types()

# Functions that read files, other databases or the environment (DuckDB and SQLite spellings).
FILE_FUNCS = frozenset({
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json", "read_json_auto",
    "read_json_objects", "read_ndjson", "read_ndjson_auto", "read_ndjson_objects", "read_text",
    "read_blob", "glob", "sniff_csv", "iceberg_scan", "delta_scan", "st_read", "sqlite_scan",
    "sqlite_attach", "postgres_scan", "postgres_attach", "mysql_scan", "load_extension", "readfile",
    "writefile", "getenv", "query", "query_table", "edit", "fts3_tokenizer",
})  # fmt: skip

# A table name ending in one of these is a DuckDB replacement scan of a file.
FILE_SUFFIXES = (
    ".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson", ".txt", ".db", ".sqlite",
    ".sqlite3", ".duckdb", ".gz", ".zst", ".xlsx",
)  # fmt: skip

# Largest integer literal SQLite and DuckDB accept as a LIMIT; a larger one is an error as written.
_MAX_LIMIT_LITERAL = 2**63 - 1

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class GuardError(ValueError):
    """A query the guard refuses to run; the message says why."""


@dataclass(frozen=True)
class GuardedSQL:
    """A query that passed the guard.

    Attributes:
        sql: The normalised original text. This, not sqlglot's regenerated SQL, is what runs;
            the executor's row cap (:func:`bounded_sql`) only appends a ``LIMIT`` or
            replaces the count of an existing one.
        tree: The parsed query.
        dialect: The sqlglot dialect it was parsed as.
    """

    sql: str
    tree: exp.Query
    dialect: str


def normalize(sql: str) -> str:
    """Strip a markdown fence, surrounding whitespace and one trailing semicolon."""
    fence = _FENCE.search(sql)
    text = (fence.group(1) if fence else sql).strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def func_name(node: exp.Func) -> str:
    """Return the lowercase SQL name of a function call, including unknown (anonymous) ones."""
    name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
    return name.lower()


def guard_sql(sql: str, dialect: str) -> GuardedSQL:
    """Accept exactly one read-only query, or raise.

    Args:
        sql: Model-written SQL, possibly fenced and with a trailing semicolon.
        dialect: The sqlglot dialect to parse it as (``duckdb`` or ``sqlite``).

    Returns:
        The normalised text with its parse tree.

    Raises:
        GuardError: The text is empty, too long, unparseable, not exactly one query, or contains
            a write, a session change or file access.
    """
    text = normalize(sql)
    if not text:
        raise GuardError("empty query")
    if len(text) > MAX_SQL_CHARS:
        raise GuardError(f"query longer than {MAX_SQL_CHARS} characters")
    tree = _parse_one(text, dialect)
    if not isinstance(tree, exp.Query):
        raise GuardError(f"only SELECT queries are allowed, got {type(tree).__name__.upper()}")
    for node in tree.walk():
        _check_node(node)
    return GuardedSQL(text, tree, dialect)


def _parse_one(text: str, dialect: str) -> exp.Expression:
    """Parse ``text`` and return its single statement, or raise :class:`GuardError`."""
    try:
        parsed = sqlglot.parse(text, read=dialect, error_level=ErrorLevel.RAISE)
    except Exception as error:  # sqlglot raises ParseError / TokenError / assorted ValueErrors
        first_line = str(error).splitlines()[0][:_PARSE_ERROR_CHARS]
        raise GuardError(f"could not parse as {dialect}: {first_line}") from error
    trees = [tree for tree in parsed if tree is not None]
    if not trees:
        raise GuardError("empty query")
    if len(trees) != 1:
        raise GuardError("exactly one statement allowed")
    return trees[0]


def _check_node(node: exp.Expression) -> None:
    """Raise :class:`GuardError` if ``node`` writes, changes state or reaches a file."""
    if isinstance(node, DENY_TYPES):
        raise GuardError(f"{type(node).__name__.upper()} is not allowed in a read-only query")
    if isinstance(node, exp.Func) and func_name(node) in FILE_FUNCS:
        raise GuardError(f"function {func_name(node)}() is not allowed")
    if isinstance(node, exp.Table) and _looks_like_file(node.name.lower()):
        raise GuardError(f"table name {node.name!r} looks like a file")


def _looks_like_file(name: str) -> bool:
    """Return whether a lowercase table name is a path, a URL or a data-file name."""
    return "/" in name or "\\" in name or "://" in name or name.endswith(FILE_SUFFIXES)


def row_limit(tree: exp.Query) -> int | None:
    """Return the query's top-level ``LIMIT`` / ``FETCH FIRST`` row count when it is a literal.

    None when there is none, or when it is an expression, a percentage or ``WITH TIES``, whose
    row count the text does not say.
    """
    count = _limit_literal(tree)
    return None if count is None else int(count.this)


def _limit_literal(tree: exp.Query) -> exp.Literal | None:
    """Return the integer literal of the top-level ``LIMIT`` / ``FETCH FIRST``, or None."""
    node = tree.args.get("limit")
    if isinstance(node, exp.Fetch):
        count = node.args.get("count")
        options = node.args.get("limit_options")
        if options is not None and (options.args.get("percent") or options.args.get("with_ties")):
            return None
    elif isinstance(node, exp.Limit):
        count = node.expression
        options = node.args.get("limit_options")
        if options is not None and options.args.get("percent"):
            return None
    else:
        return None
    if isinstance(count, exp.Literal) and not count.is_string and count.this.isdigit():
        return count
    return None


def bounded_sql(guarded: GuardedSQL, max_rows: int) -> str:
    """Return the query's text with its result capped at ``max_rows`` rows.

    The cap goes into the SQL text, not only the fetch: a client-side fetch of the first rows
    still lets many drivers materialise the whole result, and a pipelined plan (scan, filter,
    nested-loop join) stops early only when the database sees the ``LIMIT``. The rest of the
    text runs byte for byte as written, so the database still rejects what it would reject.

    * No top-level ``LIMIT``, ``FETCH`` or ``OFFSET``: ``LIMIT max_rows`` is appended to the
      text as written, on a new line so that a trailing ``--`` comment cannot swallow it.
    * A literal limit of at most ``max_rows``: unchanged.
    * A literal limit above it (``LIMIT`` or ``FETCH FIRST``, with any ``OFFSET``; digit
      separators as in ``1_000_000`` too): its digits are replaced by ``max_rows`` where they
      stand in the text (the literal's token span).
    * A count that spells "no limit": SQLite's negative literal (``LIMIT -1``) and DuckDB's
      ``LIMIT ALL`` / ``LIMIT NULL`` are replaced by ``max_rows`` the same way. DuckDB
      rejects a negative limit, so there it stays, an error as written.
    * Anything else (``OFFSET`` alone, an expression, a percentage, ``WITH TIES``), a
      literal too large for a 64-bit integer (an error as written) or one whose span cannot
      be located: unchanged, because the cap cannot be added without changing the result or
      what the database accepts.

    Callers must pass the returned text through the guard again before running it.

    Args:
        guarded: A query that passed :func:`guard_sql`.
        max_rows: The most rows the query may return.

    Returns:
        The text to run.
    """
    tree, text = guarded.tree, guarded.sql
    span = _no_limit_span(guarded) or _over_cap_literal_span(tree, text, max_rows)
    if span is not None:
        start, end = span
        return text[:start] + str(max_rows) + text[end + 1 :]
    if tree.args.get("limit") is None and tree.args.get("offset") is None:
        return f"{text}\nLIMIT {max_rows}"
    return text


def _over_cap_literal_span(tree: exp.Query, text: str, max_rows: int) -> tuple[int, int] | None:
    """Return the span of the top-level literal limit when it exceeds ``max_rows``, or None."""
    count = _limit_literal(tree)
    if count is None or not max_rows < int(count.this) <= _MAX_LIMIT_LITERAL:
        return None
    return _literal_span(text, count)


def _literal_span(text: str, literal: exp.Literal) -> tuple[int, int] | None:
    """Return the inclusive span of ``literal`` in ``text``, or None when it is off.

    sqlglot records the source span (``start``/``end``, inclusive) of the token a literal was
    parsed from in its ``meta``; the span must still spell the literal's digits, digit
    separators aside (``1_000_000`` parses as ``1000000``).
    """
    start, end = literal.meta.get("start"), literal.meta.get("end")
    if start is None or end is None or text[start : end + 1].replace("_", "") != literal.this:
        return None
    return start, end


def _no_limit_span(guarded: GuardedSQL) -> tuple[int, int] | None:
    """Return the inclusive span of a top-level ``LIMIT`` count that means "no limit", or None."""
    if guarded.dialect == "sqlite":
        return _sqlite_negative_limit(guarded)
    if guarded.dialect == "duckdb":
        return _duckdb_no_limit_keyword(guarded.sql)
    return None


def _sqlite_negative_limit(guarded: GuardedSQL) -> tuple[int, int] | None:
    """Return the span of SQLite's negative ``LIMIT`` literal, minus sign included, or None.

    ``LIMIT -1`` and ``LIMIT 5, -1`` mean no limit; ``-0`` is a limit of zero rows, and a
    literal below the smallest 64-bit integer is an error as written.
    """
    node = guarded.tree.args.get("limit")
    count = node.expression if isinstance(node, exp.Limit) else None
    if not (isinstance(count, exp.Neg) and isinstance(count.this, exp.Literal)):
        return None
    literal = count.this
    if literal.is_string or not literal.this.isdigit():
        return None
    if not 0 < int(literal.this) <= _MAX_LIMIT_LITERAL + 1:
        return None
    span = _literal_span(guarded.sql, literal)
    if span is None:
        return None
    before = guarded.sql[: span[0]].rstrip()
    return (len(before) - 1, span[1]) if before.endswith("-") else None


def _duckdb_no_limit_keyword(text: str) -> tuple[int, int] | None:
    """Return the span of ``ALL`` / ``NULL`` right after DuckDB's top-level ``LIMIT``, or None."""
    tokens = sqlglot.Dialect.get_or_raise("duckdb").tokenize(text)
    depth = 0
    last_limit = None
    for index, token in enumerate(tokens):
        if token.token_type == TokenType.L_PAREN:
            depth += 1
        elif token.token_type == TokenType.R_PAREN:
            depth -= 1
        elif token.token_type == TokenType.LIMIT and depth == 0:
            last_limit = index
    if last_limit is None or last_limit + 1 >= len(tokens):
        return None
    keyword = tokens[last_limit + 1]
    if keyword.token_type not in (TokenType.ALL, TokenType.NULL):
        return None
    return keyword.start, keyword.end
