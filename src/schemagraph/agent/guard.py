"""Read-only SQL guard, the first of three read-only layers.

The other two layers are the engine's statement check and the hardened connection, both in
:mod:`schemagraph.agent.execute`.

The guard accepts exactly one query (``SELECT``, ``WITH … SELECT``, set operations, DuckDB
``FROM t``) and rejects anything that writes, changes session state or reaches outside the
database: DDL/DML, ``ATTACH``, ``PRAGMA``, ``SET``, ``COPY``, ``INSTALL``/``LOAD``,
``SELECT … INTO``, file-reading table functions and file-like table names (DuckDB replacement
scans). The text that passes is executed as written, not sqlglot's regenerated SQL; a parser
differential is caught by the engine-side checks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel

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

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class GuardError(ValueError):
    """A query the guard refuses to run; the message says why."""


@dataclass(frozen=True)
class GuardedSQL:
    """A query that passed the guard.

    Attributes:
        sql: The normalised original text. This, not sqlglot's regenerated SQL, is what runs.
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
