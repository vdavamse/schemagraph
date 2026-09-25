"""Spider-format schema-linking benchmark (Spider 1.0 dev, LinkAlign's Spider subset, EHRSQL, ...).

Loads a Spider ``tables.json`` (declared foreign keys and primary keys) as one snapshot per database
and a questions file (``[{db_id, question, query}]``), takes the gold tables from the gold SQL with
sqlglot, and scores ``link`` like the Spider 2.0 bench. Databases here are small and their FK graphs
real, so this is where anchors, path union and pruning are measurable: run with a tight table budget
and ``bypass_if_fits=false`` and read ``bridge_recall`` (gold tables that were not anchors and were
still returned) next to strict recall.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp

from schemagraph.graph.build import build_graph
from schemagraph.graph.infer import with_inferred_edges
from schemagraph.linking.linker import Linker, LinkOptions
from schemagraph.model import Column, Edge, LinkResult, SchemaSnapshot, Table

# Fallback gold-table extraction when sqlglot cannot parse the gold query.
_FROM_RE = re.compile(r"(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", re.I)

# Columns of the CSV written next to the JSON results, in order.
_CSV_HEADER = [
    "instance_id",
    "db",
    "n_gold",
    "n_pred",
    "n_tables_db",
    "hit",
    "recall",
    "precision",
    "strict",
    "anchor_hit",
    "n_bridge",
    "bridge_recovered",
    "max_gold_rank",
    "ms",
    "missed",
]

# Summary keys of the markdown table printed by format_table.
TABLE_COLUMNS = [
    "n",
    "recall",
    "precision",
    "strict_recall",
    "anchor_hit",
    "bridge_tasks",
    "bridge_recall",
    "strict_on_bridge_tasks",
    "avg_pred_tables",
    "avg_db_tables",
    "p50_ms",
]


# ---------------------------------------------------------------- snapshots
def _spider_tables(db: dict) -> list[Table]:
    """Column-less tables of a ``tables.json`` entry.

    A readable name that is not just the original with spaces for underscores becomes
    ``business_name``.
    """
    names = db["table_names_original"]
    readable = db.get("table_names") or names
    tables: list[Table] = []
    for i, name in enumerate(names):
        properties = (
            {"business_name": readable[i]}
            if i < len(readable) and readable[i].lower() != name.lower().replace("_", " ")
            else {}
        )
        tables.append(Table(name=name, properties=properties))
    return tables


def _attach_spider_columns(db: dict, tables: list[Table]) -> None:
    """Append every column of a ``tables.json`` entry to its table (skipping the ``*`` column).

    Mutates ``tables`` in place.
    """
    columns = db["column_names_original"]
    readable_columns = db.get("column_names") or columns
    column_types = db.get("column_types") or []
    for column_index, (table_index, column_name) in enumerate(columns):
        if table_index < 0:
            continue
        readable = (
            readable_columns[column_index][1] if column_index < len(readable_columns) else None
        )
        properties = (
            {"business_name": readable}
            if readable and readable.lower() != column_name.lower().replace("_", " ")
            else {}
        )
        data_type = column_types[column_index] if column_index < len(column_types) else None
        tables[table_index].columns.append(
            Column(name=column_name, data_type=data_type, properties=properties)
        )


def _apply_spider_primary_keys(db: dict, tables: list[Table]) -> None:
    """Mark the primary-key columns (single or composite) of a ``tables.json`` entry.

    Mutates ``tables`` in place.
    """
    columns = db["column_names_original"]
    for pk in db.get("primary_keys") or []:
        for column_index in pk if isinstance(pk, list) else [pk]:
            table_index, column_name = columns[column_index]
            column = tables[table_index].column(column_name)
            if column is not None:
                column.is_primary_key = True
                if column_name not in tables[table_index].primary_key:
                    tables[table_index].primary_key.append(column_name)


def _spider_fk_edges(db: dict) -> list[Edge]:
    """One ``foreign_key`` edge per column pair in ``foreign_keys``."""
    names = db["table_names_original"]
    columns = db["column_names_original"]
    edges: list[Edge] = []
    for from_index, to_index in db.get("foreign_keys") or []:
        from_table, from_column = columns[from_index]
        to_table, to_column = columns[to_index]
        edges.append(
            Edge(
                kind="foreign_key",
                from_table=names[from_table],
                to_table=names[to_table],
                from_columns=[from_column],
                to_columns=[to_column],
            )
        )
    return edges


def snapshot_from_spider(db: dict, source: str | None = None) -> SchemaSnapshot:
    """One Spider ``tables.json`` entry -> SchemaSnapshot.

    FKs become ``foreign_key`` edges and readable names ``business_name`` properties.
    """
    tables = _spider_tables(db)
    _attach_spider_columns(db, tables)
    _apply_spider_primary_keys(db, tables)
    edges = _spider_fk_edges(db)
    return SchemaSnapshot(
        source=source or f"spider:{db['db_id']}",
        source_type="spider",
        tables=tables,
        edges=edges,
    )


def gold_tables(sql: str, dialect: str = "sqlite") -> set[str]:
    """Base tables referenced by a gold query (lowercase; CTE names excluded)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return {match.lower() for match in _FROM_RE.findall(sql)}
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    return {
        table.name.lower()
        for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in ctes
    }


@dataclass
class Row:
    """Scores of one question: one JSON row and one CSV line of the output.

    Attributes:
        instance_id: The question's ``instance_id``, else its position in the questions file.
        db: Database id.
        n_gold: Gold tables (those of the gold SQL that exist in the database).
        n_pred: Returned tables.
        n_tables_db: Tables in the database.
        hit: Gold tables returned.
        recall: ``hit / n_gold``.
        precision: ``hit / n_pred`` (0.0 when nothing was returned).
        strict: 1 if every gold table was returned.
        anchor_hit: 1 if every gold table was an anchor.
        n_bridge: Gold tables that were not anchors.
        bridge_recovered: ... of which the returned set still contains.
        ms: Link latency in milliseconds, rounded to 0.1.
        missed: Gold tables not returned, sorted.
        max_gold_rank: Worst rank of any gold table in the candidate ranking (None = unranked).
    """

    instance_id: str
    db: str
    n_gold: int
    n_pred: int
    n_tables_db: int
    hit: int
    recall: float
    precision: float
    strict: int
    anchor_hit: int
    n_bridge: int
    bridge_recovered: int
    ms: float
    missed: list[str] = field(default_factory=list)
    max_gold_rank: int | None = None


# ---------------------------------------------------------------- scoring
def _gold_rank_of(gold: set[str], result: LinkResult) -> int | None:
    """Worst 1-based rank of any gold table in the candidate ranking, None if one is unranked.

    A name ranked more than once keeps its last rank.
    """
    rank_of = {fqn.lower(): rank + 1 for rank, (fqn, _) in enumerate(result.ranking)}
    ranks = [rank_of.get(table) for table in gold]
    return None if any(rank is None for rank in ranks) else max(ranks)


def _score_question(
    instance_id: str,
    db_id: str,
    gold: set[str],
    result: LinkResult,
    ms: float,
    n_tables_db: int,
) -> Row:
    """Score one linked question against its gold tables.

    Args:
        instance_id: Row id of the question.
        db_id: Database id.
        gold: Lowercased gold tables, all present in the database.
        result: What the linker returned (with ``ranking``, i.e. ``debug=True``).
        ms: Link latency in milliseconds.
        n_tables_db: Tables in the database.

    Returns:
        The row of the question.
    """
    pred = {table.fqn.lower() for table in result.tables}
    anchors = {anchor.lower() for anchor in result.anchors}
    bridge = gold - anchors
    return Row(
        instance_id=instance_id,
        db=db_id,
        n_gold=len(gold),
        n_pred=len(pred),
        n_tables_db=n_tables_db,
        hit=len(gold & pred),
        recall=len(gold & pred) / len(gold),
        precision=len(gold & pred) / len(pred) if pred else 0.0,
        strict=int(gold <= pred),
        anchor_hit=int(gold <= anchors),
        n_bridge=len(bridge),
        bridge_recovered=len(bridge & pred),
        ms=round(ms, 1),
        missed=sorted(gold - pred),
        max_gold_rank=_gold_rank_of(gold, result),
    )


# ---------------------------------------------------------------- output
def _csv_row(row: Row) -> list:
    """One CSV line of a row, in ``_CSV_HEADER`` order."""
    return [
        row.instance_id,
        row.db,
        row.n_gold,
        row.n_pred,
        row.n_tables_db,
        row.hit,
        f"{row.recall:.3f}",
        f"{row.precision:.3f}",
        row.strict,
        row.anchor_hit,
        row.n_bridge,
        row.bridge_recovered,
        "" if row.max_gold_rank is None else row.max_gold_rank,
        row.ms,
        ";".join(row.missed),
    ]


def _write_outputs(out_dir: Path, stem: str, summary: dict, rows: list[Row]) -> None:
    """Write ``<stem>.json`` (summary and rows) and ``<stem>.csv`` (one line per row)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": [asdict(row) for row in rows]}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for row in rows:
            writer.writerow(_csv_row(row))


# ---------------------------------------------------------------- run
def run(
    tables_json: str | Path,
    questions_json: str | Path,
    *,
    max_tables: int = 4,
    anchor_k: int = 2,
    infer: bool = False,
    limit: int | None = None,
    dialect: str = "sqlite",
    out_dir: str | Path | None = None,
    tag: str | None = None,
    **link_kwargs,
) -> dict:
    """Link every question of a Spider-format dataset and score the result.

    Questions whose database is unknown or whose gold SQL names no table of it are skipped.

    Args:
        tables_json: Spider ``tables.json`` (declared FKs and PKs).
        questions_json: Questions as ``[{db_id, question, query}]`` (``sql`` accepted for query).
        max_tables: ``LinkOptions.max_tables``.
        anchor_k: ``LinkOptions.anchor_k``.
        infer: Add name-based inferred edges on top of the declared FKs.
        limit: Score only the first this many questions.
        dialect: sqlglot dialect of the gold SQL.
        out_dir: Folder for ``spider1_<tag>.json`` / ``.csv``; None writes nothing.
        tag: Output file tag; ``mt<max_tables>_k<anchor_k>`` when None.
        **link_kwargs: Any other ``LinkOptions`` fields (``bypass_if_fits`` defaults to False).

    Returns:
        ``{"summary": ..., "rows": [Row, ...]}``; the summary is :func:`summarize` plus the run
        ``config``.
    """
    dbs = {db["db_id"]: db for db in json.loads(Path(tables_json).read_text(encoding="utf-8"))}
    questions = json.loads(Path(questions_json).read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    opts = LinkOptions(
        max_tables=max_tables,
        anchor_k=anchor_k,
        render=False,
        debug=True,
        **{"ranking_limit": 0, "bypass_if_fits": False, **link_kwargs},
    )
    linkers: dict[str, Linker] = {}
    rows: list[Row] = []
    skipped = 0
    for i, question in enumerate(questions):
        db_id = question["db_id"]
        if db_id not in dbs:
            skipped += 1
            continue
        if db_id not in linkers:
            snap = snapshot_from_spider(dbs[db_id])
            if infer:
                with_inferred_edges(snap)
            linkers[db_id] = Linker(build_graph([snap]))
        linker = linkers[db_id]
        gold_sql = question.get("query") or question.get("sql") or ""
        gold = {
            table
            for table in gold_tables(gold_sql, dialect)
            if table in linker.schema_graph.tables
        }
        if not gold:
            skipped += 1
            continue
        t0 = time.perf_counter()
        result = linker.link(question["question"], opts)
        ms = (time.perf_counter() - t0) * 1000
        rows.append(
            _score_question(
                str(question.get("instance_id", i)),
                db_id,
                gold,
                result,
                ms,
                len(linker.schema_graph.tables),
            )
        )
    summary = summarize(rows)
    summary["config"] = {
        "max_tables": max_tables,
        "anchor_k": anchor_k,
        "infer": infer,
        "dialect": dialect,
        "link_kwargs": link_kwargs,
        "n": len(rows),
        "skipped": skipped,
        "tables_json": str(tables_json),
        "questions_json": str(questions_json),
    }
    if out_dir:
        tag = tag or f"mt{max_tables}_k{anchor_k}"
        _write_outputs(Path(out_dir), f"spider1_{tag}", summary, rows)
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------- summary
def summarize(rows: list[Row]) -> dict:
    """Aggregate rows into percentages (2 decimals), bridge metrics, averages and median latency.

    ``bridge_recall`` is None when no gold table was a bridge.
    """
    if not rows:
        return {"n": 0}
    n_bridge = sum(row.n_bridge for row in rows)
    bridge_tasks = sum(1 for row in rows if row.n_bridge)
    bridge_recall = (
        round(100 * sum(row.bridge_recovered for row in rows) / n_bridge, 2) if n_bridge else None
    )
    strict_on_bridge = round(
        100 * sum(row.strict for row in rows if row.n_bridge) / max(1, bridge_tasks),
        2,
    )
    return {
        "n": len(rows),
        "recall": round(100 * statistics.mean(row.recall for row in rows), 2),
        "precision": round(100 * statistics.mean(row.precision for row in rows), 2),
        "strict_recall": round(100 * sum(row.strict for row in rows) / len(rows), 2),
        "anchor_hit": round(100 * sum(row.anchor_hit for row in rows) / len(rows), 2),
        "bridge_tasks": bridge_tasks,
        "bridge_recall": bridge_recall,
        "strict_on_bridge_tasks": strict_on_bridge,
        "avg_pred_tables": round(statistics.mean(row.n_pred for row in rows), 2),
        "avg_db_tables": round(statistics.mean(row.n_tables_db for row in rows), 1),
        "p50_ms": round(statistics.median(row.ms for row in rows), 1),
    }


def format_table(summary: dict) -> str:
    """Render a summary as a one-row markdown table."""
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"
    values = "| " + " | ".join(str(summary.get(c, "")) for c in TABLE_COLUMNS) + " |"
    return f"{header}\n{rule}\n{values}"
