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

import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from schemagraph.bench.gold_sql import gold_columns
from schemagraph.connectors.spider2 import (
    DIALECT_FOR_PREFIX,
    Spider2Config,
    canonical_table,
    introspect_spider2,
)
from schemagraph.graph.build import build_graph
from schemagraph.graph.infer import with_inferred_edges
from schemagraph.linking.linker import Linker, LinkOptions


@dataclass(frozen=True)
class Suite:
    """Where a Spider 2.0 variant keeps its tasks, gold tables, schema files and documents."""

    name: str
    tasks: str
    gold: str
    databases: str
    documents: str
    question_key: str
    db_key: str
    flat: bool  # True: databases/<db> (Snow); False: databases/<dialect>/<db> (Lite)
    gold_sql: str = ""  # folder with <instance_id>.sql for the tasks whose gold SQL is public

    def db_dir(self, root: Path, dialect: str, db: str) -> Path:
        base = root / self.databases
        return base / db if self.flat else base / dialect / db


SUITES: dict[str, Suite] = {
    "lite": Suite("lite", "spider2-lite/spider2-lite.jsonl", "methods/gold-tables/spider2-lite-gold-tables.jsonl", "spider2-lite/resource/databases", "spider2-lite/resource/documents", "question", "db", False, "spider2-lite/evaluation_suite/gold/sql"),
    "snow": Suite("snow", "spider2-snow/spider2-snow.jsonl", "methods/gold-tables/spider2-snow-gold-tables.jsonl", "spider2-snow/resource/databases", "spider2-snow/resource/documents", "instruction", "db_id", True, "spider2-snow/evaluation_suite/gold/sql"),
}


@dataclass
class Instance:
    instance_id: str
    db: str
    dialect: str
    question: str
    doc: str | None
    gold: set[str]  # name-canonical (date/year shards collapsed)
    gold_raw: set[str] = field(default_factory=set)  # as listed in gold-tables, lowercased
    gold_sql: str | None = None


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
    # DBCC-protocol extras: raw column count of the database, tokens of the rendered context, and
    # column-level scores where the gold SQL is public (None otherwise)
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
    missed_cols: list[str] = field(default_factory=list)  # gold columns not in the returned DDL, as table.column


def _prefix(instance_id: str) -> str:
    for p in ("local", "ga", "bq", "sf"):
        if instance_id.startswith(p):
            return p
    return instance_id[:2]


def load_instances(spider2_root: Path, *, limit: int | None = None, dialects: set[str] | None = None, only: set[str] | None = None, min_db_tables: int = 0, suite: str = "lite") -> list[Instance]:
    st = SUITES[suite]
    gold: dict[str, set[str]] = {}
    gold_raw: dict[str, set[str]] = {}
    for line in (spider2_root / st.gold).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            gold[d["instance_id"]] = {canonical_table(t) for t in d["gold_tables"]}
            gold_raw[d["instance_id"]] = {t.strip().lower() for t in d["gold_tables"]}
    out: list[Instance] = []
    for line in (spider2_root / st.tasks).read_text(encoding="utf-8").splitlines():
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
            p = spider2_root / st.documents / d["external_knowledge"]
            if p.exists():
                doc = p.read_text(encoding="utf-8", errors="replace")
        if min_db_tables:
            folder = st.db_dir(spider2_root, dialect, d[st.db_key])
            if not folder.is_dir() or sum(1 for _ in folder.rglob("*.json")) < min_db_tables:
                continue
        gsql = None
        if st.gold_sql:
            gp = spider2_root / st.gold_sql / f"{iid}.sql"
            if gp.exists():
                gsql = gp.read_text(encoding="utf-8", errors="replace")
        out.append(Instance(iid, d[st.db_key], dialect, d[st.question_key], doc, gold[iid], gold_raw[iid], gsql))
        if limit and len(out) >= limit:
            break
    return out


class _GraphCache:
    def __init__(self, spider2_root: Path, suite: Suite, *, infer: bool, sample_values: int, collapse_families: bool = True):
        self.root = spider2_root
        self.suite = suite
        self.infer = infer
        self.sample_values = sample_values
        self.collapse_families = collapse_families
        self.cache: dict[tuple[str, str], tuple[Linker, int, list[str], dict[str, str]]] = {}
        self.raw_cols: dict[tuple[str, str], int] = {}
        self.schema: dict[tuple[str, str], dict[str, set[str]]] = {}

    def db_dir(self, dialect: str, db: str) -> Path:
        return self.suite.db_dir(self.root, dialect, db)

    def get(self, dialect: str, db: str) -> tuple[Linker, int, list[str], dict[str, str]]:
        key = (dialect, db)
        if key not in self.cache:
            folder = self.db_dir(dialect, db)
            snap = introspect_spider2(Spider2Config(root=str(folder.parent), dialect=dialect, db=db, path=str(folder), sample_values=self.sample_values, collapse_families=self.collapse_families), f"spider2:{db}")
            if self.infer:
                with_inferred_edges(snap)
            sg = build_graph([snap])
            member_map: dict[str, str] = {}
            for t in sg.tables.values():
                for m in t.properties.get("members", "").split(","):
                    if m:
                        member_map[m.strip().lower()] = t.fqn.lower()
            self.cache[key] = (Linker(sg), len(sg.tables), snap.warnings, member_map)
            self.schema[key] = {canon(t.fqn, member_map): {c.name.lower() for c in t.columns} for t in sg.tables.values()}
            n = 0
            for jf in folder.rglob("*.json"):
                if jf.name == "DDL.json":
                    continue
                try:
                    n += len(json.loads(jf.read_text(encoding="utf-8")).get("column_names") or [])
                except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                    pass
            self.raw_cols[key] = n
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
    suite: str = "lite",
    render: bool = True,
    **link_kwargs,
) -> dict:
    root = Path(spider2_root)
    st = SUITES[suite]
    instances = load_instances(root, limit=limit, dialects=dialects, only=only, min_db_tables=min_db_tables, suite=suite)
    cache = _GraphCache(root, st, infer=infer, sample_values=sample_values, collapse_families=collapse_families)
    opts = LinkOptions(max_tables=max_tables, anchor_k=anchor_k, render=render, use_llm=use_llm, debug=True, **{"ranking_limit": 0, **link_kwargs})
    enc = _tokenizer() if render else None
    rows: list[Row] = []
    skipped: list[str] = []
    for i, inst in enumerate(instances):
        if not cache.db_dir(inst.dialect, inst.db).is_dir():
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
        ddl_tokens = None
        if render and res.ddl:
            # context only: drop the "-- Question: ..." header (it carries the external doc too)
            body = res.ddl.split("-- Linked tables:", 1)[-1]
            ddl_tokens = len(enc.encode(body)) if enc else len(body.split())
        colm: dict = {}
        if inst.gold_sql:
            schema = cache.schema[(inst.dialect, inst.db)]

            def resolve(raw: str, _schema=schema, _mm=member_map) -> str | None:
                k = canon(raw, _mm)
                if k in _schema:
                    return k
                # gold may name a shard member or omit the catalog: match on the trailing parts
                tail = k.split(".")[-2:]
                hits = [s for s in _schema if s.split(".")[-2:] == tail] or [s for s in _schema if s.split(".")[-1] == tail[-1]]
                return hits[0] if len(hits) == 1 else None

            gc = gold_columns(inst.gold_sql, inst.dialect, schema, resolve)
            pred_cols = {(canon(t.fqn, member_map), c.name.lower()) for t in res.tables for c in t.columns}
            if gc.parsed and gc.columns:
                chit = len(gc.columns & pred_cols)
                colm = {"n_gold_cols": len(gc.columns), "n_pred_cols": len(pred_cols), "col_hit": chit, "col_recall": chit / len(gc.columns), "col_precision": chit / len(pred_cols) if pred_cols else 0.0, "col_strict": int(gc.columns <= pred_cols), "unresolved_cols": len(gc.unresolved), "sql_parsed": True, "missed_cols": sorted(f"{t.split('.')[-1]}.{c}" for t, c in gc.columns - pred_cols)}
            else:
                colm = {"sql_parsed": gc.parsed, "unresolved_cols": len(gc.unresolved)}
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
                n_cols_db=cache.raw_cols[(inst.dialect, inst.db)],
                n_cols_graph=sum(len(t.columns) for t in linker.sg.tables.values()),
                ddl_tokens=ddl_tokens,
                **colm,
            )
        )
        if progress:
            progress(i + 1, len(instances), rows[-1])
    summary = summarize(rows)
    summary["config"] = {"render": render, "max_tables": max_tables, "anchor_k": anchor_k, "use_docs": use_docs, "doc_chars": doc_chars, "infer": infer, "sample_values": sample_values, "use_llm": use_llm, "min_db_tables": min_db_tables, "collapse_families": collapse_families, "link_kwargs": link_kwargs, "suite": suite, "n": len(rows), "skipped_missing_schema": skipped}
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        tag = tag or f"mt{max_tables}_k{anchor_k}_{'docs' if use_docs else 'nodocs'}_{'infer' if infer else 'noinfer'}{'_llm' if use_llm else ''}"
        (od / f"spider2_{suite}_{tag}.json").write_text(json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2), encoding="utf-8")
        with (od / f"spider2_{suite}_{tag}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["instance_id", "db", "dialect", "n_gold", "n_pred", "n_tables_db", "hit", "recall", "precision", "strict", "anchor_hit", "max_gold_rank", "ms", "n_cols_db", "n_cols_graph", "ddl_tokens", "n_gold_cols", "n_pred_cols", "col_recall", "col_precision", "col_strict", "unresolved_cols", "missed", "missed_cols"])

            def f(v, fmt="{}"):
                return "" if v is None else fmt.format(v)

            for r in rows:
                w.writerow([r.instance_id, r.db, r.dialect, r.n_gold, r.n_pred, r.n_tables_db, r.hit, f"{r.recall:.3f}", f"{r.precision:.3f}", r.strict, r.anchor_hit, f(r.max_gold_rank), r.ms, r.n_cols_db, r.n_cols_graph, f(r.ddl_tokens), f(r.n_gold_cols), f(r.n_pred_cols), f(r.col_recall, "{:.3f}"), f(r.col_precision, "{:.3f}"), f(r.col_strict), f(r.unresolved_cols), ";".join(r.missed), ";".join(r.missed_cols)])
    return {"summary": summary, "rows": rows}


def summarize(rows: list[Row]) -> dict:
    def agg(rs: list[Row]) -> dict:
        if not rs:
            return {}
        rec = statistics.mean(r.recall for r in rs)
        prec = statistics.mean(r.precision for r in rs)
        out = {
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
            "avg_db_cols": round(statistics.mean(r.n_cols_db for r in rs), 0),
            "p50_ms": round(statistics.median(r.ms for r in rs), 1),
        }
        toks = [r.ddl_tokens for r in rs if r.ddl_tokens is not None]
        if toks:
            out["p50_ddl_tokens"] = int(statistics.median(toks))
            out["avg_ddl_tokens"] = int(statistics.mean(toks))
        cs = [r for r in rs if r.col_strict is not None]
        if cs:
            crec = statistics.mean(r.col_recall for r in cs)
            cprec = statistics.mean(r.col_precision for r in cs)
            out["n_colgold"] = len(cs)
            out["col_recall"] = round(100 * crec, 2)
            out["col_precision"] = round(100 * cprec, 2)
            out["col_strict"] = round(100 * sum(r.col_strict for r in cs) / len(cs), 2)
            out["avg_gold_cols"] = round(statistics.mean(r.n_gold_cols for r in cs), 1)
            out["avg_pred_cols"] = round(statistics.mean(r.n_pred_cols for r in cs), 1)
        return out

    by_dialect: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_dialect[r.dialect].append(r)
    # DBCC buckets by raw column count of the database (Liu et al. 2026, arXiv 2606.28601)
    by_bucket = {
        "cols<1k": [r for r in rows if r.n_cols_db < 1000],
        "cols1k-10k": [r for r in rows if 1000 <= r.n_cols_db < 10000],
        "cols>=10k": [r for r in rows if r.n_cols_db >= 10000],
        "cols>5k": [r for r in rows if r.n_cols_db > 5000],
    }
    return {"overall": agg(rows), "by_dialect": {d: agg(rs) for d, rs in sorted(by_dialect.items())}, "by_bucket": {b: agg(rs) for b, rs in by_bucket.items() if rs}}


def format_table(summary: dict) -> str:
    cols = ["n", "recall", "precision", "f1", "strict_recall", "anchor_hit", "gold_in_top10", "gold_in_top20", "avg_pred_tables", "avg_db_tables", "p50_ms"]
    splits = [("overall", summary["overall"]), *summary["by_dialect"].items(), *summary.get("by_bucket", {}).items()]
    lines = ["| split | " + " | ".join(cols) + " |", "|---|" + "|".join("---" for _ in cols) + "|"]
    for name, s in splits:
        lines.append(f"| {name} | " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
    if any("col_strict" in s or "p50_ddl_tokens" in s for _, s in splits):
        cols2 = ["n", "avg_db_cols", "p50_ddl_tokens", "avg_ddl_tokens", "n_colgold", "col_strict", "col_recall", "col_precision", "avg_gold_cols", "avg_pred_cols"]
        lines += ["", "DBCC protocol (column-level where gold SQL is public; tokens = o200k count of the rendered DDL context):", "| split | " + " | ".join(cols2) + " |", "|---|" + "|".join("---" for _ in cols2) + "|"]
        for name, s in splits:
            lines.append(f"| {name} | " + " | ".join(str(s.get(c, "")) for c in cols2) + " |")
    return "\n".join(lines)


def _tokenizer():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # pragma: no cover - tiktoken is a dev extra; fall back to whitespace tokens
        return None
