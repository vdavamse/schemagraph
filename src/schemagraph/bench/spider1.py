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
from schemagraph.model import Column, Edge, SchemaSnapshot, Table


def snapshot_from_spider(db: dict, source: str | None = None) -> SchemaSnapshot:
    """One Spider ``tables.json`` entry -> SchemaSnapshot (FKs as ``foreign_key`` edges, readable names as ``business_name``)."""
    names = db["table_names_original"]
    readable = db.get("table_names") or names
    cols = db["column_names_original"]
    creadable = db.get("column_names") or cols
    ctypes = db.get("column_types") or []
    tables: list[Table] = []
    for i, n in enumerate(names):
        props = {"business_name": readable[i]} if i < len(readable) and readable[i].lower() != n.lower().replace("_", " ") else {}
        tables.append(Table(name=n, properties=props))
    for ci, (ti, cname) in enumerate(cols):
        if ti < 0:
            continue
        rd = creadable[ci][1] if ci < len(creadable) else None
        props = {"business_name": rd} if rd and rd.lower() != cname.lower().replace("_", " ") else {}
        tables[ti].columns.append(Column(name=cname, data_type=ctypes[ci] if ci < len(ctypes) else None, properties=props))
    for pk in db.get("primary_keys") or []:
        for ci in pk if isinstance(pk, list) else [pk]:
            ti, cname = cols[ci]
            c = tables[ti].column(cname)
            if c is not None:
                c.is_primary_key = True
                if cname not in tables[ti].primary_key:
                    tables[ti].primary_key.append(cname)
    edges: list[Edge] = []
    for a, b in db.get("foreign_keys") or []:
        ta, ca = cols[a]
        tb, cb = cols[b]
        edges.append(Edge(kind="foreign_key", from_table=names[ta], to_table=names[tb], from_columns=[ca], to_columns=[cb]))
    return SchemaSnapshot(source=source or f"spider:{db['db_id']}", source_type="spider", tables=tables, edges=edges)


_FROM_RE = re.compile(r"(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", re.I)


def gold_tables(sql: str, dialect: str = "sqlite") -> set[str]:
    """Base tables referenced by a gold query (lowercase; CTE names excluded)."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return {m.lower() for m in _FROM_RE.findall(sql)}
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}


@dataclass
class Row:
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
    n_bridge: int  # gold tables that were not anchors
    bridge_recovered: int  # ... of which the returned set still contains
    ms: float
    missed: list[str] = field(default_factory=list)
    max_gold_rank: int | None = None


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
    dbs = {d["db_id"]: d for d in json.loads(Path(tables_json).read_text(encoding="utf-8"))}
    questions = json.loads(Path(questions_json).read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    opts = LinkOptions(max_tables=max_tables, anchor_k=anchor_k, render=False, debug=True, **{"ranking_limit": 0, "bypass_if_fits": False, **link_kwargs})
    linkers: dict[str, Linker] = {}
    rows: list[Row] = []
    skipped = 0
    for i, q in enumerate(questions):
        db_id = q["db_id"]
        if db_id not in dbs:
            skipped += 1
            continue
        if db_id not in linkers:
            snap = snapshot_from_spider(dbs[db_id])
            if infer:
                with_inferred_edges(snap)
            linkers[db_id] = Linker(build_graph([snap]))
        linker = linkers[db_id]
        gold = {g for g in gold_tables(q.get("query") or q.get("sql") or "", dialect) if g in linker.sg.tables}
        if not gold:
            skipped += 1
            continue
        t0 = time.perf_counter()
        res = linker.link(q["question"], opts)
        ms = (time.perf_counter() - t0) * 1000
        pred = {t.fqn.lower() for t in res.tables}
        anchors = {a.lower() for a in res.anchors}
        rank_of = {f.lower(): r + 1 for r, (f, _) in enumerate(res.ranking)}
        ranks = [rank_of.get(g) for g in gold]
        bridge = gold - anchors
        rows.append(
            Row(
                instance_id=str(q.get("instance_id", i)),
                db=db_id,
                n_gold=len(gold),
                n_pred=len(pred),
                n_tables_db=len(linker.sg.tables),
                hit=len(gold & pred),
                recall=len(gold & pred) / len(gold),
                precision=len(gold & pred) / len(pred) if pred else 0.0,
                strict=int(gold <= pred),
                anchor_hit=int(gold <= anchors),
                n_bridge=len(bridge),
                bridge_recovered=len(bridge & pred),
                ms=round(ms, 1),
                missed=sorted(gold - pred),
                max_gold_rank=None if any(r is None for r in ranks) else max(ranks),
            )
        )
    summary = summarize(rows)
    summary["config"] = {"max_tables": max_tables, "anchor_k": anchor_k, "infer": infer, "dialect": dialect, "link_kwargs": link_kwargs, "n": len(rows), "skipped": skipped, "tables_json": str(tables_json), "questions_json": str(questions_json)}
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        tag = tag or f"mt{max_tables}_k{anchor_k}"
        (od / f"spider1_{tag}.json").write_text(json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2), encoding="utf-8")
        with (od / f"spider1_{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["instance_id", "db", "n_gold", "n_pred", "n_tables_db", "hit", "recall", "precision", "strict", "anchor_hit", "n_bridge", "bridge_recovered", "max_gold_rank", "ms", "missed"])
            for r in rows:
                w.writerow([r.instance_id, r.db, r.n_gold, r.n_pred, r.n_tables_db, r.hit, f"{r.recall:.3f}", f"{r.precision:.3f}", r.strict, r.anchor_hit, r.n_bridge, r.bridge_recovered, "" if r.max_gold_rank is None else r.max_gold_rank, r.ms, ";".join(r.missed)])
    return {"summary": summary, "rows": rows}


def summarize(rows: list[Row]) -> dict:
    if not rows:
        return {"n": 0}
    n_bridge = sum(r.n_bridge for r in rows)
    return {
        "n": len(rows),
        "recall": round(100 * statistics.mean(r.recall for r in rows), 2),
        "precision": round(100 * statistics.mean(r.precision for r in rows), 2),
        "strict_recall": round(100 * sum(r.strict for r in rows) / len(rows), 2),
        "anchor_hit": round(100 * sum(r.anchor_hit for r in rows) / len(rows), 2),
        "bridge_tasks": sum(1 for r in rows if r.n_bridge),
        "bridge_recall": round(100 * sum(r.bridge_recovered for r in rows) / n_bridge, 2) if n_bridge else None,
        "strict_on_bridge_tasks": round(100 * sum(r.strict for r in rows if r.n_bridge) / max(1, sum(1 for r in rows if r.n_bridge)), 2),
        "avg_pred_tables": round(statistics.mean(r.n_pred for r in rows), 2),
        "avg_db_tables": round(statistics.mean(r.n_tables_db for r in rows), 1),
        "p50_ms": round(statistics.median(r.ms for r in rows), 1),
    }


def format_table(summary: dict) -> str:
    cols = ["n", "recall", "precision", "strict_recall", "anchor_hit", "bridge_tasks", "bridge_recall", "strict_on_bridge_tasks", "avg_pred_tables", "avg_db_tables", "p50_ms"]
    return "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n| " + " | ".join(str(summary.get(c, "")) for c in cols) + " |"
