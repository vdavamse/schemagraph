"""Spider 2.0-Lite schema-linking benchmark: gold-table recall of ``link_schema``.

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

import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from schemagraph.connectors.spider2 import (
    DIALECT_FOR_PREFIX,
    Spider2Config,
    canonical_table,
    introspect_spider2,
)
from schemagraph.graph.build import build_graph
from schemagraph.graph.infer import with_inferred_edges
from schemagraph.linking.linker import Linker, LinkOptions


@dataclass
class Instance:
    instance_id: str
    db: str
    dialect: str
    question: str
    doc: str | None
    gold: set[str]  # name-canonical (date/year shards collapsed)
    gold_raw: set[str] = field(default_factory=set)  # as listed in gold-tables, lowercased


@dataclass
class Row:
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
    max_gold_rank: int | None = None  # worst rank of any gold table in the candidate ranking (None = unranked)
    gold_in_top10: int = 0
    gold_in_top20: int = 0


def _prefix(instance_id: str) -> str:
    for p in ("local", "ga", "bq", "sf"):
        if instance_id.startswith(p):
            return p
    return instance_id[:2]


def load_instances(spider2_root: Path, *, limit: int | None = None, dialects: set[str] | None = None, only: set[str] | None = None, min_db_tables: int = 0) -> list[Instance]:
    lite = spider2_root / "spider2-lite"
    gold: dict[str, set[str]] = {}
    gold_raw: dict[str, set[str]] = {}
    for line in (spider2_root / "methods" / "gold-tables" / "spider2-lite-gold-tables.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            gold[d["instance_id"]] = {canonical_table(t) for t in d["gold_tables"]}
            gold_raw[d["instance_id"]] = {t.strip().lower() for t in d["gold_tables"]}
    out: list[Instance] = []
    for line in (lite / "spider2-lite.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        iid = d["instance_id"]
        if only and iid not in only:
            continue
        dialect = DIALECT_FOR_PREFIX[_prefix(iid)]
        if dialects and dialect not in dialects:
            continue
        if iid not in gold or not gold[iid]:
            continue
        doc = None
        if d.get("external_knowledge"):
            p = lite / "resource" / "documents" / d["external_knowledge"]
            if p.exists():
                doc = p.read_text(encoding="utf-8", errors="replace")
        if min_db_tables:
            folder = lite / "resource" / "databases" / dialect / d["db"]
            if not folder.is_dir() or sum(1 for _ in folder.rglob("*.json")) < min_db_tables:
                continue
        out.append(Instance(iid, d["db"], dialect, d["question"], doc, gold[iid], gold_raw[iid]))
        if limit and len(out) >= limit:
            break
    return out


class _GraphCache:
    def __init__(self, databases_root: Path, *, infer: bool, sample_values: int, collapse_families: bool = True):
        self.root = databases_root
        self.infer = infer
        self.sample_values = sample_values
        self.collapse_families = collapse_families
        self.cache: dict[tuple[str, str], tuple[Linker, int, list[str], dict[str, str]]] = {}

    def get(self, dialect: str, db: str) -> tuple[Linker, int, list[str], dict[str, str]]:
        key = (dialect, db)
        if key not in self.cache:
            snap = introspect_spider2(Spider2Config(root=str(self.root), dialect=dialect, db=db, sample_values=self.sample_values, collapse_families=self.collapse_families), f"spider2:{db}")
            if self.infer:
                with_inferred_edges(snap)
            sg = build_graph([snap])
            member_map: dict[str, str] = {}
            for t in sg.tables.values():
                for m in t.properties.get("members", "").split(","):
                    if m:
                        member_map[m.strip().lower()] = t.fqn.lower()
            self.cache[key] = (Linker(sg), len(sg.tables), snap.warnings, member_map)
        return self.cache[key]


def canon(name: str, member_map: dict[str, str]) -> str:
    """Map a table name to its family representative if the db collapsed it, else the name-based canonical form."""
    ln = name.strip().lower()
    if ln in member_map:
        return member_map[ln]
    return canonical_table(ln)


def _doc_excerpt(doc: str, limit: int) -> str:
    # keep the head of the document; that's where table/column names are usually explained
    return doc[:limit]


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
    **link_kwargs,
) -> dict:
    root = Path(spider2_root)
    instances = load_instances(root, limit=limit, dialects=dialects, only=only, min_db_tables=min_db_tables)
    cache = _GraphCache(root / "spider2-lite" / "resource" / "databases", infer=infer, sample_values=sample_values, collapse_families=collapse_families)
    opts = LinkOptions(max_tables=max_tables, anchor_k=anchor_k, render=False, use_llm=use_llm, debug=True, **link_kwargs)
    rows: list[Row] = []
    skipped: list[str] = []
    for i, inst in enumerate(instances):
        if not (cache.root / inst.dialect / inst.db).is_dir():
            skipped.append(inst.instance_id)
            continue
        linker, n_tables, _warn, member_map = cache.get(inst.dialect, inst.db)
        gold = {canon(g, member_map) for g in inst.gold_raw}
        if llm is not None:
            linker.llm = llm
        q = inst.question
        if use_docs and inst.doc:
            q = f"{inst.question}\n\n{_doc_excerpt(inst.doc, doc_chars)}"
        t0 = time.perf_counter()
        res = linker.link(q, opts)
        ms = (time.perf_counter() - t0) * 1000
        pred = {canon(t.fqn, member_map) for t in res.tables}
        anchors = {canon(a, member_map) for a in res.anchors}
        hit = len(gold & pred)
        rank_of = {canon(f, member_map): i + 1 for i, (f, _) in reversed(list(enumerate(res.ranking)))}
        ranks = [rank_of.get(g) for g in gold]
        max_rank = None if any(r is None for r in ranks) else max(ranks)
        rows.append(
            Row(
                instance_id=inst.instance_id,
                db=inst.db,
                dialect=inst.dialect,
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
            )
        )
        if progress:
            progress(i + 1, len(instances), rows[-1])
    summary = summarize(rows)
    summary["config"] = {"max_tables": max_tables, "anchor_k": anchor_k, "use_docs": use_docs, "doc_chars": doc_chars, "infer": infer, "sample_values": sample_values, "use_llm": use_llm, "min_db_tables": min_db_tables, "collapse_families": collapse_families, "link_kwargs": link_kwargs, "n": len(rows), "skipped_missing_schema": skipped}
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        tag = tag or f"mt{max_tables}_k{anchor_k}_{'docs' if use_docs else 'nodocs'}_{'infer' if infer else 'noinfer'}{'_llm' if use_llm else ''}"
        (od / f"spider2_lite_{tag}.json").write_text(json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2), encoding="utf-8")
        with (od / f"spider2_lite_{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["instance_id", "db", "dialect", "n_gold", "n_pred", "n_tables_db", "hit", "recall", "precision", "strict", "anchor_hit", "max_gold_rank", "ms", "missed"])
            for r in rows:
                w.writerow([r.instance_id, r.db, r.dialect, r.n_gold, r.n_pred, r.n_tables_db, r.hit, f"{r.recall:.3f}", f"{r.precision:.3f}", r.strict, r.anchor_hit, r.max_gold_rank if r.max_gold_rank is not None else "", r.ms, ";".join(r.missed)])
    return {"summary": summary, "rows": rows}


def summarize(rows: list[Row]) -> dict:
    def agg(rs: list[Row]) -> dict:
        if not rs:
            return {}
        rec = statistics.mean(r.recall for r in rs)
        prec = statistics.mean(r.precision for r in rs)
        return {
            "n": len(rs),
            "recall": round(rec * 100, 2),
            "precision": round(prec * 100, 2),
            "f1": round(200 * rec * prec / (rec + prec), 2) if rec + prec else 0.0,
            "strict_recall": round(100 * sum(r.strict for r in rs) / len(rs), 2),
            "anchor_hit": round(100 * sum(r.anchor_hit for r in rs) / len(rs), 2),
            "gold_in_top10": round(100 * sum(r.gold_in_top10 for r in rs) / len(rs), 2),
            "gold_in_top20": round(100 * sum(r.gold_in_top20 for r in rs) / len(rs), 2),
            "avg_pred_tables": round(statistics.mean(r.n_pred for r in rs), 2),
            "avg_db_tables": round(statistics.mean(r.n_tables_db for r in rs), 1),
            "p50_ms": round(statistics.median(r.ms for r in rs), 1),
        }

    by_dialect: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_dialect[r.dialect].append(r)
    return {"overall": agg(rows), "by_dialect": {d: agg(rs) for d, rs in sorted(by_dialect.items())}}


def format_table(summary: dict) -> str:
    cols = ["n", "recall", "precision", "f1", "strict_recall", "anchor_hit", "gold_in_top10", "gold_in_top20", "avg_pred_tables", "avg_db_tables", "p50_ms"]
    lines = ["| split | " + " | ".join(cols) + " |", "|---|" + "|".join("---" for _ in cols) + "|"]
    for name, s in [("overall", summary["overall"]), *summary["by_dialect"].items()]:
        lines.append(f"| {name} | " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)
