"""Spider 2.0-Lite / -Snow schema-linking benchmark: gold-table recall of ``link_schema``.

No execution, no credentials, no LLM (unless ``use_llm``). For each of the 547 tasks:
build the graph of the task's database from the shipped schema files, link the
question, and compare the returned tables with ``methods/gold-tables``.

Metrics per instance (names canonicalised so a GA4 ``events_*`` family equals its
daily shards):

* recall      |gold ∩ pred| / |gold|
* precision   |gold ∩ pred| / |pred|
* strict      1 if gold ⊆ pred else 0   (EviLink's SRR; the number that tracks EX)
* anchor_hit  1 if gold ⊆ anchors        (how good the PPR ranking alone is)
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from schemagraph.bench._output import blank_if_none, write_outputs
from schemagraph.bench.gold_sql import gold_columns
from schemagraph.connectors.spider2 import (
    DIALECT_FOR_PREFIX,
    Spider2Config,
    canonical_table,
    introspect_spider2,
)
from schemagraph.graph.build import SchemaGraph, build_graph
from schemagraph.graph.infer import with_inferred_edges
from schemagraph.linking.lexical import DESC_WEIGHT, build_index
from schemagraph.linking.linker import Linker, LinkOptions
from schemagraph.linking.render import LINKED_TABLES_MARKER
from schemagraph.model import LinkResult


@dataclass(frozen=True)
class Suite:
    """Where a Spider 2.0 variant keeps its tasks, gold tables, schema files and documents.

    Paths are relative to the Spider2 clone root.

    Attributes:
        name: Suite name (``lite`` or ``snow``).
        tasks: JSONL file of the tasks.
        gold: JSONL file of the gold tables per task.
        databases: Folder of the per-database schema files.
        documents: Folder of the external-knowledge documents.
        question_key: Task field holding the question.
        db_key: Task field holding the database name.
        flat: True: ``databases/<db>`` (Snow); False: ``databases/<dialect>/<db>`` (Lite).
        gold_sql: Folder with ``<instance_id>.sql`` for the tasks whose gold SQL is public.
    """

    name: str
    tasks: str
    gold: str
    databases: str
    documents: str
    question_key: str
    db_key: str
    flat: bool
    gold_sql: str = ""

    def db_dir(self, root: Path, dialect: str, db: str) -> Path:
        """Schema folder of one database under the clone ``root``."""
        base = root / self.databases
        return base / db if self.flat else base / dialect / db


SUITES: dict[str, Suite] = {
    "lite": Suite(
        name="lite",
        tasks="spider2-lite/spider2-lite.jsonl",
        gold="methods/gold-tables/spider2-lite-gold-tables.jsonl",
        databases="spider2-lite/resource/databases",
        documents="spider2-lite/resource/documents",
        question_key="question",
        db_key="db",
        flat=False,
        gold_sql="spider2-lite/evaluation_suite/gold/sql",
    ),
    "snow": Suite(
        name="snow",
        tasks="spider2-snow/spider2-snow.jsonl",
        gold="methods/gold-tables/spider2-snow-gold-tables.jsonl",
        databases="spider2-snow/resource/databases",
        documents="spider2-snow/resource/documents",
        question_key="instruction",
        db_key="db_id",
        flat=True,
        gold_sql="spider2-snow/evaluation_suite/gold/sql",
    ),
}

# Instance-id prefixes that name a dialect (keys of DIALECT_FOR_PREFIX), tried in this order.
_INSTANCE_PREFIXES = ("local", "ga", "bq", "sf")

# DBCC buckets by raw column count of the database (Liu et al. 2026, arXiv 2606.28601), as
# half-open ranges ``low <= n_cols < high`` (None = unbounded). "cols>5k" means n_cols > 5000,
# i.e. n_cols >= 5001 for an integer count. Buckets overlap; order is the output order.
_COLUMN_BUCKETS: dict[str, tuple[int | None, int | None]] = {
    "cols<1k": (None, 1000),
    "cols1k-10k": (1000, 10000),
    "cols>=10k": (10000, None),
    "cols>5k": (5001, None),
}

# Columns of the CSV written next to the JSON results, in order.
_CSV_HEADER = (
    "instance_id db dialect n_gold n_pred n_tables_db hit recall precision strict anchor_hit "
    "max_gold_rank ms n_cols_db n_cols_graph ddl_tokens n_gold_cols n_pred_cols col_recall "
    "col_precision col_strict unresolved_cols missed missed_cols"
).split()

# Summary keys of the table-level markdown table printed by format_table.
_TABLE_COLUMNS = (
    "n recall precision f1 strict_recall anchor_hit gold_in_top10 gold_in_top20 "
    "avg_pred_tables avg_db_tables p50_ms"
).split()
# Summary keys of the DBCC-protocol markdown table printed by format_table.
_DBCC_COLUMNS = (
    "n avg_db_cols p50_ddl_tokens avg_ddl_tokens n_colgold col_strict col_recall col_precision "
    "avg_gold_cols avg_pred_cols"
).split()
_DBCC_CAPTION = (
    "DBCC protocol (column-level where gold SQL is public; "
    "tokens = o200k count of the rendered DDL context):"
)


class _Encoder(Protocol):
    """A tokenizer such as tiktoken's ``Encoding``: only ``encode`` is used."""

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text``."""
        ...


@dataclass
class Instance:
    """One benchmark task with its gold tables and, where public, its gold SQL.

    Attributes:
        instance_id: Spider 2.0 task id.
        db: Database name.
        dialect: ``bigquery``, ``snowflake`` or ``sqlite``.
        question: The natural-language question.
        doc: Text of the task's external-knowledge document, if any.
        gold: Gold tables, name-canonical (date/year shards collapsed).
        gold_raw: Gold tables as listed in gold-tables, lowercased.
        gold_sql: Public gold SQL, if any.
    """

    instance_id: str
    db: str
    dialect: str
    question: str
    doc: str | None
    gold: set[str]
    gold_raw: set[str] = field(default_factory=set)
    gold_sql: str | None = None


@dataclass
class Row:
    """Scores of one instance: one JSON row and one CSV line of the output.

    Attributes:
        instance_id: Spider 2.0 task id.
        db: Database name.
        dialect: ``bigquery``, ``snowflake`` or ``sqlite``.
        n_gold: Number of canonical gold tables.
        n_pred: Number of canonical returned tables.
        n_tables_db: Tables in the database's graph.
        hit: Gold tables returned.
        recall: ``hit / n_gold``.
        precision: ``hit / n_pred`` (0.0 when nothing was returned).
        strict: 1 if every gold table was returned.
        anchor_hit: 1 if every gold table was an anchor.
        ms: Link latency in milliseconds, rounded to 0.1.
        missed: Gold tables not returned, sorted.
        max_gold_rank: Worst rank of any gold table in the candidate ranking (None = unranked).
        gold_in_top10: 1 if every gold table ranks in the top 10.
        gold_in_top20: 1 if every gold table ranks in the top 20.
        n_cols_db: Raw column count of the database (DBCC protocol).
        n_cols_graph: Columns in the database's graph (after family collapsing).
        ddl_tokens: Tokens of the rendered context (None when not rendered).
        n_gold_cols: Gold columns resolved from the public gold SQL. This and the other ``col_*``
            / ``n_pred_cols`` fields are None without gold SQL, when sqlglot cannot parse it, or
            when it resolves to no known column.
        n_pred_cols: Columns in the returned DDL.
        col_hit: Gold columns returned.
        col_recall: ``col_hit / n_gold_cols``.
        col_precision: ``col_hit / n_pred_cols`` (0.0 when no column was returned).
        col_strict: 1 if every gold column was returned.
        unresolved_cols: Gold SQL column names that matched no base table (None without gold SQL).
        sql_parsed: Whether sqlglot parsed the gold SQL (None without gold SQL).
        missed_cols: Gold columns not in the returned DDL, as ``table.column``.
    """

    instance_id: str
    db: str
    dialect: str
    n_gold: int
    n_pred: int
    n_tables_db: int
    hit: int
    recall: float
    precision: float
    strict: int
    anchor_hit: int
    ms: float
    missed: list[str] = field(default_factory=list)
    max_gold_rank: int | None = None
    gold_in_top10: int = 0
    gold_in_top20: int = 0
    n_cols_db: int = 0
    n_cols_graph: int = 0
    ddl_tokens: int | None = None
    n_gold_cols: int | None = None
    n_pred_cols: int | None = None
    col_hit: int | None = None
    col_recall: float | None = None
    col_precision: float | None = None
    col_strict: int | None = None
    unresolved_cols: int | None = None
    sql_parsed: bool | None = None
    missed_cols: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- loading
def _prefix(instance_id: str) -> str:
    """Dialect prefix of an instance id (its first two characters when no known prefix fits)."""
    for prefix in _INSTANCE_PREFIXES:
        if instance_id.startswith(prefix):
            return prefix
    return instance_id[:2]


def _load_gold(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Read the gold-tables JSONL.

    Returns:
        ``(canonical, raw)``: per instance id, the name-canonical gold tables and the listed ones
        lowercased. A repeated instance id keeps its last line.
    """
    gold: dict[str, set[str]] = {}
    gold_raw: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            gold[record["instance_id"]] = {canonical_table(t) for t in record["gold_tables"]}
            gold_raw[record["instance_id"]] = {t.strip().lower() for t in record["gold_tables"]}
    return gold, gold_raw


def _read_doc(spider2_root: Path, suite_spec: Suite, task: dict) -> str | None:
    """Text of the task's external-knowledge document; None if it has none or the file is gone."""
    if task.get("external_knowledge"):
        path = spider2_root / suite_spec.documents / task["external_knowledge"]
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def _has_min_tables(folder: Path, min_db_tables: int) -> bool:
    """Whether a database folder exists and holds at least ``min_db_tables`` table files."""
    return folder.is_dir() and sum(1 for _ in folder.rglob("*.json")) >= min_db_tables


def _read_gold_sql(spider2_root: Path, suite_spec: Suite, instance_id: str) -> str | None:
    """Public gold SQL of one instance, or None when the suite or the task has none."""
    if suite_spec.gold_sql:
        path = spider2_root / suite_spec.gold_sql / f"{instance_id}.sql"
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def load_instances(
    spider2_root: Path,
    *,
    limit: int | None = None,
    dialects: set[str] | None = None,
    only: set[str] | None = None,
    min_db_tables: int = 0,
    suite: str = "lite",
) -> list[Instance]:
    """Load the tasks of a suite that have gold tables, in task-file order.

    Args:
        spider2_root: The Spider2 clone.
        limit: Stop after this many instances (None or 0: all).
        dialects: Keep only these dialects (None or empty: all).
        only: Keep only these instance ids (None or empty: all).
        min_db_tables: Keep only tasks whose database folder has at least this many table files.
        suite: Key of :data:`SUITES`.

    Returns:
        The selected instances.
    """
    suite_spec = SUITES[suite]
    gold, gold_raw = _load_gold(spider2_root / suite_spec.gold)
    instances: list[Instance] = []
    for line in (spider2_root / suite_spec.tasks).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        instance_id = task["instance_id"]
        if only and instance_id not in only:
            continue
        dialect = DIALECT_FOR_PREFIX[_prefix(instance_id)]
        if dialects and dialect not in dialects:
            continue
        if instance_id not in gold or not gold[instance_id]:
            continue
        doc = _read_doc(spider2_root, suite_spec, task)
        if min_db_tables:
            folder = suite_spec.db_dir(spider2_root, dialect, task[suite_spec.db_key])
            if not _has_min_tables(folder, min_db_tables):
                continue
        instances.append(
            Instance(
                instance_id,
                task[suite_spec.db_key],
                dialect,
                task[suite_spec.question_key],
                doc,
                gold[instance_id],
                gold_raw[instance_id],
                _read_gold_sql(spider2_root, suite_spec, instance_id),
            )
        )
        if limit and len(instances) >= limit:
            break
    return instances


# ---------------------------------------------------------------- graph cache
def _member_map(schema_graph: SchemaGraph) -> dict[str, str]:
    """Map each lowercased partition-family member to the lowercased fqn of its representative."""
    member_map: dict[str, str] = {}
    for table in schema_graph.tables.values():
        for member in table.properties.get("members", "").split(","):
            if member:
                member_map[member.strip().lower()] = table.fqn.lower()
    return member_map


def _count_raw_columns(folder: Path) -> int:
    """Count the columns listed across a database's per-table JSON files (before collapsing).

    Unreadable files count as zero columns.
    """
    n_columns = 0
    for json_file in folder.rglob("*.json"):
        if json_file.name == "DDL.json":
            continue
        try:
            table_doc = json.loads(json_file.read_text(encoding="utf-8"))
            n_columns += len(table_doc.get("column_names") or [])
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
    return n_columns


class _GraphCache:
    """Per-database linkers, built on first use and kept for the whole run.

    Attributes:
        root: The Spider2 clone.
        suite: The suite whose database folders are read.
        cache: Per ``(dialect, db)``: the linker, its table count, the introspection warnings and
            the partition-family member map.
        raw_cols: Raw column count per ``(dialect, db)``.
        schema: Per ``(dialect, db)``: canonical table name -> lowercased column names.
    """

    def __init__(
        self,
        spider2_root: Path,
        suite: Suite,
        *,
        infer: bool,
        sample_values: int,
        collapse_families: bool = True,
        desc_weight: float = DESC_WEIGHT,
    ):
        self.root = spider2_root
        self.suite = suite
        self.infer = infer
        self.sample_values = sample_values
        self.collapse_families = collapse_families
        self.desc_weight = desc_weight
        self.cache: dict[tuple[str, str], tuple[Linker, int, list[str], dict[str, str]]] = {}
        self.raw_cols: dict[tuple[str, str], int] = {}
        self.schema: dict[tuple[str, str], dict[str, set[str]]] = {}

    def db_dir(self, dialect: str, db: str) -> Path:
        """Schema folder of one database."""
        return self.suite.db_dir(self.root, dialect, db)

    def get(self, dialect: str, db: str) -> tuple[Linker, int, list[str], dict[str, str]]:
        """Return the linker of one database, building and caching it on first use.

        Returns:
            ``(linker, n_tables, warnings, member_map)``: the linker, the number of tables in its
            graph, the introspection warnings and the partition-family member map.
        """
        key = (dialect, db)
        if key not in self.cache:
            folder = self.db_dir(dialect, db)
            config = Spider2Config(
                root=str(folder.parent),
                dialect=dialect,
                db=db,
                path=str(folder),
                sample_values=self.sample_values,
                collapse_families=self.collapse_families,
            )
            snap = introspect_spider2(config, f"spider2:{db}")
            if self.infer:
                with_inferred_edges(snap)
            schema_graph = build_graph([snap])
            member_map = _member_map(schema_graph)
            index = build_index(schema_graph, desc_weight=self.desc_weight)
            self.cache[key] = (
                Linker(schema_graph, index),
                len(schema_graph.tables),
                snap.warnings,
                member_map,
            )
            self.schema[key] = {
                canon(table.fqn, member_map): {column.name.lower() for column in table.columns}
                for table in schema_graph.tables.values()
            }
            self.raw_cols[key] = _count_raw_columns(folder)
        return self.cache[key]


def canon(name: str, member_map: dict[str, str]) -> str:
    """Map a table name to its family representative if the db collapsed it.

    Otherwise return the name-based canonical form.
    """
    lower_name = name.strip().lower()
    if lower_name in member_map:
        return member_map[lower_name]
    return canonical_table(lower_name)


# ---------------------------------------------------------------- scoring
def _question_text(instance: Instance, use_docs: bool, doc_chars: int) -> str:
    """The text to link: the question, then the document's head when docs are on.

    The head of an external-knowledge document is where table/column names are usually explained.
    """
    if use_docs and instance.doc:
        return f"{instance.question}\n\n{instance.doc[:doc_chars]}"
    return instance.question


def _gold_rank_of(gold: set[str], result: LinkResult, member_map: dict[str, str]) -> int | None:
    """Worst rank of any gold table in the candidate ranking, None if one is unranked.

    Ranks are 1-based; a canonical name ranked more than once keeps its first (best) rank.
    """
    rank_of = {
        canon(fqn, member_map): i + 1
        for i, (fqn, _) in reversed(list(enumerate(result.ranking)))
    }
    ranks = [rank_of.get(table) for table in gold]
    return None if any(rank is None for rank in ranks) else max(ranks)


def _ddl_tokens(ddl: str, encoder: _Encoder | None) -> int:
    """Count the tokens of the rendered context, without the question header.

    The ``-- Question: ...`` header carries the external document too, so only what follows
    the linked-tables marker counts. Without an encoder, whitespace-separated words are counted.
    """
    body = ddl.split(LINKED_TABLES_MARKER, 1)[-1]
    return len(encoder.encode(body)) if encoder else len(body.split())


def _schema_key_resolver(
    schema: dict[str, set[str]],
    member_map: dict[str, str],
) -> Callable[[str], str | None]:
    """Build the function mapping a raw table name in gold SQL to its key in ``schema``."""

    def resolve(raw: str) -> str | None:
        key = canon(raw, member_map)
        if key in schema:
            return key
        # gold may name a shard member or omit the catalog: match on the trailing parts
        tail = key.split(".")[-2:]
        hits = [s for s in schema if s.split(".")[-2:] == tail] or [
            s for s in schema if s.split(".")[-1] == tail[-1]
        ]
        return hits[0] if len(hits) == 1 else None

    return resolve


def _column_metrics(
    instance: Instance,
    schema: dict[str, set[str]],
    member_map: dict[str, str],
    result: LinkResult,
) -> dict:
    """Score the returned columns against the columns of the public gold SQL.

    Args:
        instance: The task; must have ``gold_sql``.
        schema: Canonical table name -> lowercased column names of the task's database.
        member_map: Partition-family member map of the database.
        result: What the linker returned.

    Returns:
        Keyword arguments for :class:`Row`: every column field when the gold SQL parsed and named
        known columns, else only ``sql_parsed`` and ``unresolved_cols``.
    """
    resolve = _schema_key_resolver(schema, member_map)
    gold = gold_columns(instance.gold_sql, instance.dialect, schema, resolve)
    pred_cols = {
        (canon(table.fqn, member_map), column.name.lower())
        for table in result.tables
        for column in table.columns
    }
    if not (gold.parsed and gold.columns):
        return {"sql_parsed": gold.parsed, "unresolved_cols": len(gold.unresolved)}
    col_hit = len(gold.columns & pred_cols)
    return {
        "n_gold_cols": len(gold.columns),
        "n_pred_cols": len(pred_cols),
        "col_hit": col_hit,
        "col_recall": col_hit / len(gold.columns),
        "col_precision": col_hit / len(pred_cols) if pred_cols else 0.0,
        "col_strict": int(gold.columns <= pred_cols),
        "unresolved_cols": len(gold.unresolved),
        "sql_parsed": True,
        "missed_cols": sorted(
            f"{table.split('.')[-1]}.{column}" for table, column in gold.columns - pred_cols
        ),
    }


def _score_instance(
    instance: Instance,
    result: LinkResult,
    ms: float,
    cache: _GraphCache,
    encoder: _Encoder | None,
    render: bool,
) -> Row:
    """Score one linked instance against its gold tables (and gold columns where public).

    Args:
        instance: The task.
        result: What the linker returned (with ``ranking``, i.e. ``debug=True``).
        ms: Link latency in milliseconds.
        cache: The graph cache holding the task's database.
        encoder: Tokenizer for ``ddl_tokens``, or None to count words.
        render: Whether DDL was rendered (``ddl_tokens`` stays None otherwise).

    Returns:
        The row of the instance.
    """
    key = (instance.dialect, instance.db)
    linker, n_tables, _warnings, member_map = cache.get(instance.dialect, instance.db)
    gold = {canon(table, member_map) for table in instance.gold_raw}
    pred = {canon(table.fqn, member_map) for table in result.tables}
    anchors = {canon(anchor, member_map) for anchor in result.anchors}
    hit = len(gold & pred)
    max_rank = _gold_rank_of(gold, result, member_map)
    ddl_tokens = _ddl_tokens(result.ddl, encoder) if render and result.ddl else None
    column_metrics: dict = {}
    if instance.gold_sql:
        column_metrics = _column_metrics(instance, cache.schema[key], member_map, result)
    return Row(
        instance_id=instance.instance_id,
        db=instance.db,
        dialect=instance.dialect,
        n_gold=len(gold),
        n_pred=len(pred),
        n_tables_db=n_tables,
        hit=hit,
        recall=hit / len(gold),
        precision=hit / len(pred) if pred else 0.0,
        strict=int(gold <= pred),
        anchor_hit=int(gold <= anchors),
        ms=round(ms, 1),
        missed=sorted(gold - pred),
        max_gold_rank=max_rank,
        gold_in_top10=int(max_rank is not None and max_rank <= 10),
        gold_in_top20=int(max_rank is not None and max_rank <= 20),
        n_cols_db=cache.raw_cols[key],
        n_cols_graph=sum(len(table.columns) for table in linker.schema_graph.tables.values()),
        ddl_tokens=ddl_tokens,
        **column_metrics,
    )


# ---------------------------------------------------------------- output
def _default_tag(
    max_tables: int,
    anchor_k: int,
    use_docs: bool,
    infer: bool,
    use_llm: bool,
) -> str:
    """Output file tag derived from the main run settings."""
    docs = "docs" if use_docs else "nodocs"
    inferred = "infer" if infer else "noinfer"
    llm = "_llm" if use_llm else ""
    return f"mt{max_tables}_k{anchor_k}_{docs}_{inferred}{llm}"


def _csv_row(row: Row) -> list:
    """One CSV line of a row, in ``_CSV_HEADER`` order."""
    return [
        row.instance_id,
        row.db,
        row.dialect,
        row.n_gold,
        row.n_pred,
        row.n_tables_db,
        row.hit,
        f"{row.recall:.3f}",
        f"{row.precision:.3f}",
        row.strict,
        row.anchor_hit,
        blank_if_none(row.max_gold_rank),
        row.ms,
        row.n_cols_db,
        row.n_cols_graph,
        blank_if_none(row.ddl_tokens),
        blank_if_none(row.n_gold_cols),
        blank_if_none(row.n_pred_cols),
        blank_if_none(row.col_recall, "{:.3f}"),
        blank_if_none(row.col_precision, "{:.3f}"),
        blank_if_none(row.col_strict),
        blank_if_none(row.unresolved_cols),
        ";".join(row.missed),
        ";".join(row.missed_cols),
    ]


# ---------------------------------------------------------------- run
def run(
    spider2_root: str | Path,
    *,
    max_tables: int = 12,
    anchor_k: int = 4,
    use_docs: bool = True,
    doc_chars: int = 4000,
    infer: bool = True,
    sample_values: int = 5,
    limit: int | None = None,
    dialects: set[str] | None = None,
    only: set[str] | None = None,
    min_db_tables: int = 0,
    collapse_families: bool = True,
    use_llm: bool = False,
    llm=None,
    out_dir: str | Path | None = None,
    progress=None,
    tag: str | None = None,
    suite: str = "lite",
    render: bool = True,
    desc_weight: float = DESC_WEIGHT,
    **link_kwargs,
) -> dict:
    """Link every selected task of a suite and score the result.

    Args:
        spider2_root: The Spider2 clone.
        max_tables: ``LinkOptions.max_tables``.
        anchor_k: ``LinkOptions.anchor_k``.
        use_docs: Append the task's external-knowledge document to the question.
        doc_chars: Characters of the document to append.
        infer: Add name-based inferred edges (the catalogs declare no FKs).
        sample_values: Sample values read per column.
        limit: Stop after this many instances.
        dialects: Keep only these dialects.
        only: Keep only these instance ids.
        min_db_tables: Keep only tasks whose database has at least this many table files.
        collapse_families: Collapse partition families into one table.
        use_llm: Let Claude pick the anchor tables.
        llm: Anchor picker to set on every linker, or None to keep each linker's own.
        out_dir: Folder for ``spider2_<suite>_<tag>.json`` / ``.csv``; None writes nothing.
        progress: Called as ``progress(done, total, row)`` after each instance.
        tag: Output file tag; derived from the settings when None.
        suite: Key of :data:`SUITES`.
        render: Render DDL (needed for ``ddl_tokens``).
        desc_weight: Index weight of a token found only in a description.
        **link_kwargs: Any other ``LinkOptions`` fields.

    Returns:
        ``{"summary": ..., "rows": [Row, ...]}``; the summary is :func:`summarize` plus the run
        ``config``.
    """
    root = Path(spider2_root)
    suite_spec = SUITES[suite]
    instances = load_instances(
        root,
        limit=limit,
        dialects=dialects,
        only=only,
        min_db_tables=min_db_tables,
        suite=suite,
    )
    cache = _GraphCache(
        root,
        suite_spec,
        infer=infer,
        sample_values=sample_values,
        collapse_families=collapse_families,
        desc_weight=desc_weight,
    )
    opts = LinkOptions(
        max_tables=max_tables,
        anchor_k=anchor_k,
        render=render,
        use_llm=use_llm,
        debug=True,
        **{"ranking_limit": 0, **link_kwargs},
    )
    encoder = _tokenizer() if render else None
    rows: list[Row] = []
    skipped: list[str] = []
    for i, instance in enumerate(instances):
        if not cache.db_dir(instance.dialect, instance.db).is_dir():
            skipped.append(instance.instance_id)
            continue
        linker = cache.get(instance.dialect, instance.db)[0]
        if llm is not None:
            linker.llm = llm
        question = _question_text(instance, use_docs, doc_chars)
        t0 = time.perf_counter()
        result = linker.link(question, opts)
        ms = (time.perf_counter() - t0) * 1000
        rows.append(_score_instance(instance, result, ms, cache, encoder, render))
        if progress:
            progress(i + 1, len(instances), rows[-1])
    summary = summarize(rows)
    # Key order of the config dict is part of the output format.
    summary["config"] = {
        "render": render,
        "max_tables": max_tables,
        "anchor_k": anchor_k,
        "use_docs": use_docs,
        "doc_chars": doc_chars,
        "infer": infer,
        "sample_values": sample_values,
        "use_llm": use_llm,
        "min_db_tables": min_db_tables,
        "collapse_families": collapse_families,
        "link_kwargs": link_kwargs,
        "suite": suite,
        "n": len(rows),
        "skipped_missing_schema": skipped,
    }
    if out_dir:
        tag = tag or _default_tag(max_tables, anchor_k, use_docs, infer, use_llm)
        write_outputs(
            Path(out_dir),
            f"spider2_{suite}_{tag}",
            summary,
            rows,
            _CSV_HEADER,
            _csv_row,
        )
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------- summary
def _percent(count: int, total: int) -> float:
    """``count`` as a percentage of ``total``, rounded to 2 decimals."""
    return round(100 * count / total, 2)


def _aggregate(rows: list[Row]) -> dict:
    """Summary metrics of a group of rows (empty dict for no rows).

    Token keys appear only when some row has ``ddl_tokens``, column keys only when some row was
    scored on gold columns.
    """
    if not rows:
        return {}
    recall = statistics.mean(row.recall for row in rows)
    precision = statistics.mean(row.precision for row in rows)
    f1 = round(200 * recall * precision / (recall + precision), 2) if recall + precision else 0.0
    summary = {
        "n": len(rows),
        "recall": round(recall * 100, 2),
        "precision": round(precision * 100, 2),
        "f1": f1,
        "strict_recall": _percent(sum(row.strict for row in rows), len(rows)),
        "anchor_hit": _percent(sum(row.anchor_hit for row in rows), len(rows)),
        "gold_in_top10": _percent(sum(row.gold_in_top10 for row in rows), len(rows)),
        "gold_in_top20": _percent(sum(row.gold_in_top20 for row in rows), len(rows)),
        "avg_pred_tables": round(statistics.mean(row.n_pred for row in rows), 2),
        "avg_db_tables": round(statistics.mean(row.n_tables_db for row in rows), 1),
        "avg_db_cols": round(statistics.mean(row.n_cols_db for row in rows), 0),
        "p50_ms": round(statistics.median(row.ms for row in rows), 1),
    }
    tokens = [row.ddl_tokens for row in rows if row.ddl_tokens is not None]
    if tokens:
        summary["p50_ddl_tokens"] = int(statistics.median(tokens))
        summary["avg_ddl_tokens"] = int(statistics.mean(tokens))
    scored = [row for row in rows if row.col_strict is not None]
    if scored:
        col_recall = statistics.mean(row.col_recall for row in scored)
        col_precision = statistics.mean(row.col_precision for row in scored)
        summary["n_colgold"] = len(scored)
        summary["col_recall"] = round(100 * col_recall, 2)
        summary["col_precision"] = round(100 * col_precision, 2)
        summary["col_strict"] = _percent(sum(row.col_strict for row in scored), len(scored))
        summary["avg_gold_cols"] = round(statistics.mean(row.n_gold_cols for row in scored), 1)
        summary["avg_pred_cols"] = round(statistics.mean(row.n_pred_cols for row in scored), 1)
    return summary


def _in_range(n_cols: int, low: int | None, high: int | None) -> bool:
    """Whether ``low <= n_cols < high``, a None bound being unbounded."""
    return (low is None or n_cols >= low) and (high is None or n_cols < high)


def summarize(rows: list[Row]) -> dict:
    """Aggregate rows overall, per dialect (sorted) and per DBCC column-count bucket.

    Returns:
        ``{"overall": ..., "by_dialect": {dialect: ...}, "by_bucket": {bucket: ...}}``; empty
        buckets are left out.
    """
    by_dialect: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_dialect[row.dialect].append(row)
    by_bucket = {
        bucket: [row for row in rows if _in_range(row.n_cols_db, low, high)]
        for bucket, (low, high) in _COLUMN_BUCKETS.items()
    }
    return {
        "overall": _aggregate(rows),
        "by_dialect": {
            dialect: _aggregate(dialect_rows)
            for dialect, dialect_rows in sorted(by_dialect.items())
        },
        "by_bucket": {
            bucket: _aggregate(bucket_rows)
            for bucket, bucket_rows in by_bucket.items()
            if bucket_rows
        },
    }


def _markdown_table(splits: list[tuple[str, dict]], columns: list[str]) -> list[str]:
    """Lines of a markdown table with one row per split and a blank cell for a missing metric."""
    lines = [
        "| split | " + " | ".join(columns) + " |",
        "|---|" + "|".join("---" for _ in columns) + "|",
    ]
    for name, split in splits:
        lines.append(f"| {name} | " + " | ".join(str(split.get(c, "")) for c in columns) + " |")
    return lines


def format_table(summary: dict) -> str:
    """Render a summary as markdown: table-level metrics, then DBCC metrics when present."""
    splits = [
        ("overall", summary["overall"]),
        *summary["by_dialect"].items(),
        *summary.get("by_bucket", {}).items(),
    ]
    lines = _markdown_table(splits, _TABLE_COLUMNS)
    if any("col_strict" in split or "p50_ddl_tokens" in split for _, split in splits):
        lines += ["", _DBCC_CAPTION, *_markdown_table(splits, _DBCC_COLUMNS)]
    return "\n".join(lines)


def _tokenizer() -> _Encoder | None:
    """Return the o200k tokenizer, or None when tiktoken is not installed."""
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # pragma: no cover - tiktoken is a dev extra; fall back to whitespace tokens
        return None
