"""Spider 2.0-Lite execution accuracy on the 135 ``local*`` SQLite tasks.

The agentic answer loop (``schemagraph.agent``) runs at equal budget per strategy and is scored
with the official comparison (:mod:`schemagraph.bench.spider2_eval`). Needs the ``agent`` extra,
model keys, and Spider2's ``local_sqlite.zip`` unpacked into
``spider2-lite/resource/databases/spider2-localdb/`` (one ``<db>.sqlite`` per task ``db``).

Each database's linker is served by its own in-process MCP server
(:func:`~schemagraph.mcp.http.serve_http_async`) while that database's tasks run, so the agents
read the schema exactly as they do from ``schemagraph ask``.

Per task: EX of the final pick, EX of the top-score candidate, oracle EX (any candidate correct),
table recall of the final SQL against ``methods/gold-tables``, nodes, tokens and latency per
role. :mod:`schemagraph.bench.spider2_judge` measures the judge on the same tasks.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import statistics
import threading
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from schemagraph.bench.spider2_eval import evaluate_rows
from schemagraph.bench.spider2_lite import SUITES, Instance, _GraphCache, canon, load_instances

if TYPE_CHECKING:
    from schemagraph.agent.answer import Answerer
    from schemagraph.agent.execute import SQLiteExecutor
    from schemagraph.agent.models import AgentModels
    from schemagraph.agent.results import AgentConfig, AnswerResult, Candidate
    from schemagraph.linking.linker import Linker

LOCALDB = "spider2-lite/resource/databases/spider2-localdb"
GOLD_DIR = "spider2-lite/evaluation_suite/gold/exec_result"
EVAL_STANDARD = "spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl"
# Timeout of the full (unpreviewed) run of a query that is compared with the gold result.
EVAL_TIMEOUT_S = 120.0
# External-knowledge characters passed with a task.
DOC_CHARS = 4000
# Sample values per column in the linked graphs.
SAMPLE_VALUES = 5
# Each task's seed is the run seed plus a hash of its id modulo this.
TASK_SEED_SPREAD = 100_000
# Characters of a task-level error message.
TASK_ERROR_CHARS = 500
# Columns of the per-task CSV, and of the summary table.
_CSV_COLUMNS = [
    "instance_id", "db", "ex", "ex_by_score", "oracle", "chosen_by", "score", "nodes",
    "stopped_early", "table_recall", "tokens_in", "tokens_out", "tokens_reasoning", "cost_usd",
    "ms", "error",
]  # fmt: skip
_SUMMARY_COLUMNS = [
    "n", "ex", "ex_by_score", "oracle", "table_recall", "avg_nodes", "avg_refinements",
    "early_stop", "avg_tokens_in", "avg_tokens_out", "avg_cost_usd", "p90_cost_usd", "p50_s",
    "errors",
]  # fmt: skip
# Per-database summary keys.
_DB_SUMMARY_KEYS = {"n", "ex", "oracle"}
# Usage fields a row keeps per role.
_ROW_USAGE_FIELDS = {
    "calls", "requests", "input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens",
    "tool_calls", "cost_usd", "unpriced", "ms",
}  # fmt: skip
# Decimals of a USD cost in rows and summaries.
COST_DECIMALS = 6

TaskProgress = Callable[[int, int, dict], None]


# --------------------------------------------------------------------------- tasks and databases
def _key(name: str) -> str:
    return name.lower().replace("-", "_")


def resolve_schema_dir(base: Path, db: str) -> Path:
    """Return the schema folder of a task db: exact, else the unique one equal up to case and -/_.

    ``Db-IMDB`` finds ``DB_IMDB`` and ``sqlite-sakila`` finds ``SQLITE_SAKILA``.

    Raises:
        FileNotFoundError: No folder, or several, match.
    """
    exact = base / db
    if exact.is_dir():
        return exact
    hits = []
    if base.is_dir():
        hits = [path for path in base.iterdir() if path.is_dir() and _key(path.name) == _key(db)]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"no unique schema folder for {db!r} under {base} "
            f"(candidates: {[path.name for path in hits]})"
        )
    return hits[0]


def sqlite_path(root: Path, db: str) -> Path:
    """Return the SQLite file of a task db: exact, else the unique one equal up to case and -/_.

    Raises:
        FileNotFoundError: No file matches, with a hint to unpack ``local_sqlite.zip``.
    """
    folder = root / LOCALDB
    path = folder / f"{db}.sqlite"
    if path.is_file():
        return path
    hits = list(folder.glob("*.sqlite")) if folder.is_dir() else []
    hits = [hit for hit in hits if _key(hit.stem) == _key(db)]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(
        f"{path} not found: download local_sqlite.zip (spider2-lite/README.md) and unzip the "
        f".sqlite files into {folder}"
    )


class _ExecGraphCache(_GraphCache):
    """The Lite bench's graph cache with an alias-tolerant folder lookup and no families.

    The Lite bench keeps its exact lookup, so its numbers cannot move; families are off because
    the generator must see real table names.
    """

    def db_dir(self, dialect: str, db: str) -> Path:
        """Return the schema folder of one database, found up to case and -/_."""
        base = self.suite.db_dir(self.root, dialect, "")
        try:
            return resolve_schema_dir(base, db)
        except FileNotFoundError:
            return base / db


def load_tasks(
    root: Path,
    *,
    limit: int | None = None,
    only: set[str] | None = None,
) -> tuple[list[Instance], dict[str, dict]]:
    """Load the ``local*`` SQLite tasks and the evaluation standard (instance id -> settings)."""
    standard = {}
    for line in (root / EVAL_STANDARD).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            standard[entry["instance_id"]] = entry
    instances = load_instances(root, dialects={"sqlite"}, only=only, suite="lite")
    tasks = [task for task in instances if task.instance_id.startswith("local")]
    return (tasks[:limit] if limit else tasks), standard


def _average_ranks(scores: list[float]) -> list[float]:
    """Return 1-based ranks of ``scores``, ties sharing their average rank."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Return the AUROC of ``scores`` against 0/1 ``labels``; None when a class is empty.

    Mann-Whitney U / (n_pos * n_neg), with average ranks for ties.
    """
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = _average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return round((rank_sum - positives * (positives + 1) / 2) / (positives * negatives), 4)


class Runner:
    """Shared state of one benchmark run: linkers, executors and the EX cache.

    Tasks run concurrently with ``--concurrency`` > 1, so each cache has its own lock: opening
    an executor or reading the EX cache never waits for another database's graph build. The
    EX of a (task, sql) pair is computed outside the lock; two tasks racing on the same pair
    may both run it, and the first stored result wins.
    """

    def __init__(self, root: Path, *, use_docs: bool = True, doc_chars: int = DOC_CHARS):
        self.root = root
        self.use_docs = use_docs
        self.doc_chars = doc_chars
        self.graphs = _ExecGraphCache(
            root,
            SUITES["lite"],
            infer=True,
            sample_values=SAMPLE_VALUES,
            collapse_families=False,
        )
        self.gold_dir = root / GOLD_DIR
        self._executors: dict[str, SQLiteExecutor] = {}
        self._matches: dict[tuple[str, str], tuple[int, str | None]] = {}
        self._graphs_lock = threading.Lock()
        self._executors_lock = threading.Lock()
        self._matches_lock = threading.Lock()

    def executor(self, db: str) -> SQLiteExecutor:
        """Return the database's executor, opened on first use."""
        from schemagraph.agent.execute import SQLiteExecutor

        with self._executors_lock:
            if db not in self._executors:
                self._executors[db] = SQLiteExecutor(sqlite_path(self.root, db))
            return self._executors[db]

    def linker(self, db: str) -> Linker:
        """Return the database's linker, built on first use (blocking)."""
        with self._graphs_lock:
            linker, _, _, _ = self.graphs.get("sqlite", db)
        return linker

    def evidence(self, task: Instance) -> str | None:
        """Return the task's external knowledge, when documents are on."""
        return task.doc[: self.doc_chars] if self.use_docs and task.doc else None

    def execution_match(
        self,
        task: Instance,
        sql: str | None,
        standard: dict,
    ) -> tuple[int, str | None]:
        """Return (EX, error) of one SQL, run in full (not the preview), cached per (task, sql)."""
        if not sql:
            return 0, "no sql"
        key = (task.instance_id, sql)
        with self._matches_lock:
            cached = self._matches.get(key)
        if cached is not None:
            return cached
        match = self._execution_match(task, sql, standard)
        with self._matches_lock:
            return self._matches.setdefault(key, match)

    def _execution_match(self, task: Instance, sql: str, standard: dict) -> tuple[int, str | None]:
        """Run ``sql`` in full and compare it with the gold result (blocking, uncached)."""
        result = self.executor(task.db).execute(sql, limit=None, raw=True, timeout_s=EVAL_TIMEOUT_S)
        if not result.ok:
            return 0, f"{result.error_kind}: {result.error}"
        return evaluate_rows(
            task.instance_id,
            result.columns,
            result.rows,
            self.gold_dir,
            standard.get(task.instance_id, {}),
        )

    def close(self) -> None:
        """Close every executor."""
        for executor in self._executors.values():
            executor.close()


class McpServers:
    """One MCP server per database, each over that database's linker.

    A server starts when a task of its database first asks for it and stops after the last of
    the database's ``pending`` tasks is done, or at :meth:`aclose`. Each database has its own
    lock, so a task whose server is up never waits for another database; graph builds run one
    at a time (``Runner._graphs_lock``).
    """

    def __init__(self, runner: Runner, pending: Counter[str]):
        self._runner = runner
        self._pending = pending
        self._servers: dict[str, tuple[AsyncExitStack, str]] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def url(self, db: str) -> str:
        """Return the MCP URL of the database's server, starting it if needed."""
        from schemagraph.mcp import LinkerSource, create_server, serve_http_async

        async with self._locks[db]:
            if db not in self._servers:
                linker = await asyncio.to_thread(self._runner.linker, db)
                server = create_server(LinkerSource(linker, dialect="sqlite"))
                stack = AsyncExitStack()
                url = await stack.enter_async_context(serve_http_async(server))
                self._servers[db] = (stack, url)
            return self._servers[db][1]

    async def task_done(self, db: str) -> None:
        """Count one of the database's tasks as done; stop its server after the last one."""
        self._pending[db] -= 1
        if self._pending[db] <= 0:
            await self._stop(db)

    async def _stop(self, db: str) -> None:
        async with self._locks[db]:
            entry = self._servers.pop(db, None)
        if entry is not None:
            await entry[0].aclose()

    async def aclose(self) -> None:
        """Stop every server still running."""
        for db in list(self._servers):
            await self._stop(db)


def new_answerer(
    url: str,
    executor: SQLiteExecutor,
    cfg: AgentConfig,
    models: AgentModels | None,
) -> Answerer:
    """Return an Answerer that reads the schema from the MCP server at ``url``."""
    from schemagraph.agent.answer import Answerer
    from schemagraph.agent.schema_client import SchemaClient

    return Answerer(SchemaClient(url), executor, cfg, models)


# --------------------------------------------------------------------------- results files
def read_rows(path: Path) -> list[dict]:
    """Rows of a jsonl results file, the last row per instance winning (a resumed task appends)."""
    if not path.exists():
        return []
    by_id: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_id[row["instance_id"]] = row
    return list(by_id.values())


def done_ids(path: Path, config_hash: str | None = None) -> set[str]:
    """Tasks finished without an error under ``config_hash`` (failed tasks are retried)."""
    return {
        row["instance_id"]
        for row in read_rows(path)
        if not row.get("error") and (config_hash is None or row.get("config_hash") == config_hash)
    }


def config_hash(config: dict) -> str:
    """Hash what a row's numbers depend on (models, strategy, budget, AgentConfig, docs, seed).

    ``concurrency`` is left out: it changes speed, not the configuration.
    """
    hashed = {key: value for key, value in config.items() if key != "concurrency"}
    text = json.dumps(hashed, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def append_record(path: Path, record: dict) -> None:
    """Append one JSON record to a jsonl file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _run_config(
    cfg: AgentConfig,
    models: AgentModels,
    *,
    seed: int,
    use_docs: bool,
    concurrency: int,
) -> dict[str, Any]:
    """Return the run's recorded configuration, ``config_hash`` included.

    ``mcp_url`` is left out of ``agent_config``: the benchmark always serves each database
    itself, so the field would be None-valued noise, and leaving it out keeps the hash of rows
    written before the field existed. ``trace`` is left out too: it records, it changes no
    answer.
    """
    agent_config = {
        key: value
        for key, value in asdict(cfg).items()
        if key not in {"weights", "mcp_url", "trace"}
    }
    config: dict[str, Any] = {
        "strategy": cfg.strategy,
        "budget": cfg.budget,
        "batch_size": cfg.batch_size,
        "models": models.names,
        "seed": seed,
        "use_docs": use_docs,
        "concurrency": concurrency,
        "agent_config": agent_config,
        "weights": asdict(cfg.weights),
    }
    from schemagraph.agent.models import reasoning_level

    if any(name.startswith("openrouter:") for name in models.names.values()):
        # the reasoning effort changes the answers; recorded only when a model reads it, so the
        # hash of runs on other providers is unchanged
        config["reasoning"] = reasoning_level()
    config["config_hash"] = config_hash(config)
    return config


def _resume(
    rows_path: Path, per_task_paths: tuple[Path, ...], chash: str, *, resume: bool
) -> set[str]:
    """Prepare the results files and return the tasks already done under ``chash``.

    ``per_task_paths`` are the files keyed by ``instance_id`` (candidates, transcripts); their
    records of tasks that run again are dropped.

    Raises:
        ValueError: The rows file holds rows of another configuration; the summary would
            describe numbers it did not produce.
    """
    if not resume:
        for path in (rows_path, *per_task_paths):
            path.unlink(missing_ok=True)
    other = {row.get("config_hash") for row in read_rows(rows_path)} - {chash}
    if other:
        raise ValueError(
            f"{rows_path} holds rows from another configuration "
            f"({', '.join(sorted(map(str, other)))}); use a different --tag or --no-resume"
        )
    done = done_ids(rows_path, chash)
    for path in per_task_paths:  # drop records of tasks that run again (a crash, an error row)
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        keep = [line for line in lines if line.strip() and json.loads(line)["instance_id"] in done]
        path.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    return done


def _write_outputs(
    out_dir: Path,
    tag: str,
    summary: dict,
    config: dict,
    rows: list[dict],
) -> None:
    """Write the run's JSON (summary, config, rows) and per-task CSV."""
    report = {"summary": summary, "config": config, "rows": rows}
    (out_dir / f"spider2_exec_{tag}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    with (out_dir / f"spider2_exec_{tag}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for row in rows:
            writer.writerow([row.get(column, "") for column in _CSV_COLUMNS])


# --------------------------------------------------------------------------- execution accuracy
def run(
    spider2_root: str | Path,
    *,
    cfg: AgentConfig | None = None,
    models: AgentModels | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
    tag: str | None = None,
    out_dir: str | Path = "bench_results",
    seed: int = 0,
    use_docs: bool = True,
    resume: bool = True,
    concurrency: int = 1,
    progress: TaskProgress | None = None,
) -> dict:
    """Answer every task with ``cfg`` and score it.

    Args:
        spider2_root: The Spider2 clone.
        cfg: The agent settings; ``cfg.mcp_url`` is ignored, each database is served here.
        models: Ready models (tests); None resolves them from ``cfg``.
        limit: Only the first this many tasks.
        only: Only these instance ids.
        tag: Output file tag (default ``<strategy>_n<budget>``).
        out_dir: Where the results go.
        seed: The run seed; each task adds a hash of its id.
        use_docs: Pass each task's external-knowledge document.
        resume: Skip tasks already done under the same configuration.
        concurrency: Tasks in parallel. Above 1 it is faster, but AB-MCTS then shares
            TreeQuest's global RNG across tasks, so runs are not reproducible.
        progress: Called with (finished, total, row) after each task.

    Returns:
        The summary, the configuration and the rows.
    """
    from schemagraph.agent.models import AgentModels
    from schemagraph.agent.results import AgentConfig

    root, out = Path(spider2_root), Path(out_dir)
    cfg = cfg or AgentConfig()
    models = models or AgentModels.resolve(cfg)
    tasks, standard = load_tasks(root, limit=limit, only=only)
    tag = tag or f"{cfg.strategy}_n{cfg.budget}"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"spider2_exec_{tag}.rows.jsonl"
    candidates_path = out / f"spider2_exec_{tag}_candidates.jsonl"
    messages_path = out / f"spider2_exec_{tag}_messages.jsonl"
    config = _run_config(cfg, models, seed=seed, use_docs=use_docs, concurrency=concurrency)
    done = _resume(
        rows_path, (candidates_path, messages_path), config["config_hash"], resume=resume
    )
    bench = _ExecBench(
        runner=Runner(root, use_docs=use_docs),
        cfg=cfg,
        models=models,
        standard=standard,
        rows_path=rows_path,
        candidates_path=candidates_path,
        messages_path=messages_path,
        config_hash=config["config_hash"],
        seed=seed,
        total=len(tasks),
        finished=len(done),
        progress=progress,
    )
    todo = [task for task in tasks if task.instance_id not in done]
    asyncio.run(bench.run(todo, concurrency))
    ids = {task.instance_id for task in tasks}
    rows = [row for row in read_rows(rows_path) if row["instance_id"] in ids]
    summary = summarize(rows)
    _write_outputs(out, tag, summary, config, rows)
    return {"summary": summary, "config": config, "rows": rows}


@dataclass
class _ExecBench:
    """One execution-accuracy run: answers tasks and appends their rows and candidates.

    Attributes:
        runner: Linkers, executors and the EX cache.
        cfg: The agent settings.
        models: The resolved models.
        standard: The evaluation standard per instance id.
        rows_path: The rows file (one row per task attempt).
        candidates_path: The candidates file (one record per candidate).
        messages_path: The transcripts file (one record per model call attempt), written when
            ``cfg.trace`` is on.
        config_hash: Stamped on every row.
        seed: The run seed.
        total: Tasks in the run, done ones included.
        finished: Tasks finished so far.
        progress: Called with (finished, total, row) after each task.
    """

    runner: Runner
    cfg: AgentConfig
    models: AgentModels
    standard: dict[str, dict]
    rows_path: Path
    candidates_path: Path
    messages_path: Path
    config_hash: str
    seed: int
    total: int
    finished: int
    progress: TaskProgress | None = None
    _semaphore: asyncio.Semaphore = field(init=False)
    _servers: McpServers = field(init=False)

    async def run(self, tasks: list[Instance], concurrency: int) -> None:
        """Run ``tasks``, at most ``concurrency`` at a time."""
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._servers = McpServers(self.runner, Counter(task.db for task in tasks))
        try:
            await asyncio.gather(*(self._task(task) for task in tasks))
        finally:
            await self._servers.aclose()
            self.runner.close()

    async def _task(self, task: Instance) -> None:
        """Answer and score one task; a task-level failure is an error row, not a crash."""
        async with self._semaphore:
            started = time.perf_counter()
            try:
                result = await self._answer(task)
            except Exception as error:
                row = {
                    "instance_id": task.instance_id,
                    "db": task.db,
                    "ex": 0,
                    "error": f"{type(error).__name__}: {error}"[:TASK_ERROR_CHARS],
                    "ms": round((time.perf_counter() - started) * 1000),
                    "config_hash": self.config_hash,
                }
            else:
                row = await asyncio.to_thread(
                    _score_task, self.runner, task, result, self.standard
                )
                row["ms"] = round((time.perf_counter() - started) * 1000)
                row["config_hash"] = self.config_hash
                for candidate in result.candidates:
                    match = row["candidate_ex"].get(candidate.id)
                    append_record(self.candidates_path, _candidate_record(task, candidate, match))
                for transcript in result.transcripts:
                    record = {"instance_id": task.instance_id, **transcript.model_dump(mode="json")}
                    append_record(self.messages_path, record)
            append_record(self.rows_path, row)
            self.finished += 1
            if self.progress:
                self.progress(self.finished, self.total, row)

    async def _answer(self, task: Instance) -> AnswerResult:
        """Answer one task over its database's MCP server."""
        task_seed = self.seed + zlib.crc32(task.instance_id.encode()) % TASK_SEED_SPREAD
        try:
            url = await self._servers.url(task.db)
            executor = await asyncio.to_thread(self.runner.executor, task.db)
            answerer = new_answerer(url, executor, replace(self.cfg, seed=task_seed), self.models)
            return await answerer.answer(task.question, evidence=self.runner.evidence(task))
        finally:
            await self._servers.task_done(task.db)


def _candidate_record(task: Instance, candidate: Candidate, match: int | None) -> dict:
    """Return one candidate as the candidates file stores it."""
    judgement = candidate.judgement
    result = candidate.exec
    return {
        "instance_id": task.instance_id,
        "id": candidate.id,
        "parent_id": candidate.parent_id,
        "depth": candidate.depth,
        "action": candidate.action,
        "sql": candidate.sql,
        "rationale": candidate.rationale,
        "advice": candidate.advice,
        "feedback": candidate.feedback,
        "score": candidate.score,
        "score_parts": candidate.score_parts,
        "rubric": judgement.fields if judgement else None,
        "missing": judgement.missing if judgement else None,
        "findings": [finding.code for finding in candidate.checks.findings]
        if candidate.checks
        else [],
        "ok": bool(result and result.ok),
        "row_count": result.row_count if result else None,
        "error": candidate.error,
        "ex": match,
    }


def all_failed(candidates: list[Candidate]) -> str:
    """Return the error message of a task whose every node failed."""
    first = candidates[0].error if candidates else "none ran"
    return f"all {len(candidates)} nodes failed: {first}"


def _score_task(runner: Runner, task: Instance, result: AnswerResult, standard: dict) -> dict:
    """Return a task's row: EX of the pick, of the top-score candidate and of any candidate."""
    ex, ex_error = runner.execution_match(task, result.sql, standard)
    candidate_ex = {
        candidate.id: runner.execution_match(task, candidate.sql, standard)[0]
        for candidate in result.candidates
        if candidate.sql
    }
    by_score = max(result.candidates, key=lambda candidate: candidate.score, default=None)
    final = next((c for c in result.candidates if c.id == result.chosen_id), None)
    gold = {canon(table, {}) for table in task.gold_raw}
    read = final.checks.tables if final and final.checks else []
    used = {canon(table, {}) for table in read}
    usage = {
        role: summary.model_dump(include=_ROW_USAGE_FIELDS)
        for role, summary in result.usage.by_role.items()
    }
    # every node failed (provider down, missing key, ...): an error row, retried on resume
    any_ran = any(not candidate.error for candidate in result.candidates)
    return {
        "instance_id": task.instance_id,
        "db": task.db,
        "ex": ex,
        "ex_error": ex_error,
        "ex_by_score": candidate_ex.get(by_score.id, 0) if by_score else 0,
        "oracle": int(any(candidate_ex.values())),
        "chosen_by": result.chosen_by,
        "chosen_id": result.chosen_id,
        "score": round(result.score, 4),
        "nodes": result.nodes,
        "refinements": sum(1 for candidate in result.candidates if candidate.parent_id),
        "stopped_early": result.stopped_early,
        "final_ok": bool(result.result and result.result.ok),
        "table_recall": round(len(gold & used) / len(gold), 4) if gold else None,
        "tokens_in": result.usage.total.input_tokens,
        "tokens_out": result.usage.total.output_tokens,
        "tokens_reasoning": result.usage.total.reasoning_tokens,
        "cost_usd": round(result.usage.total.cost_usd, COST_DECIMALS),
        "unpriced": result.usage.total.unpriced,
        "usage": usage,
        "candidate_ex": candidate_ex,
        "sql": result.sql,
        "error": None if any_ran else all_failed(result.candidates),
    }


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(row.get(key) or 0 for row in rows)


def _percent(rows: list[dict], key: str) -> float:
    return round(100 * sum(row.get(key) or 0 for row in rows) / len(rows), 2)


def _per_task_by_role(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Return the mean usage per task, per role."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for role, usage in (row.get("usage") or {}).items():
            for name, value in usage.items():
                totals[role][name] += value
    return {
        role: {
            name: round(value / len(rows), COST_DECIMALS if name == "cost_usd" else 1)
            for name, value in usage.items()
        }
        for role, usage in totals.items()
    }


def _quantile(values: list[float], q: float) -> float:
    """Return the nearest-rank ``q`` quantile of ``values`` (0 for none)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _costs(rows: list[dict]) -> dict:
    """Return the cost per task (mean, p50, p90, max), the total and the unpriced responses.

    Rows written before costs were recorded count as zero; ``unpriced`` says how many model
    responses carried neither a reported cost nor a fallback price.
    """
    costs = [row.get("cost_usd") or 0.0 for row in rows]
    return {
        "avg_cost_usd": round(statistics.mean(costs), COST_DECIMALS),
        "p50_cost_usd": round(_quantile(costs, 0.5), COST_DECIMALS),
        "p90_cost_usd": round(_quantile(costs, 0.9), COST_DECIMALS),
        "max_cost_usd": round(max(costs), COST_DECIMALS),
        "total_cost_usd": round(sum(costs), COST_DECIMALS),
        "avg_tokens_reasoning": int(_mean(rows, "tokens_reasoning")),
        "unpriced": sum(row.get("unpriced") or 0 for row in rows),
    }


def _aggregate(rows: list[dict]) -> dict:
    """Summarise a group of rows (empty for no rows)."""
    if not rows:
        return {}
    recalls = [row["table_recall"] for row in rows if row.get("table_recall") is not None]
    summary = {
        "n": len(rows),
        "ex": _percent(rows, "ex"),
        "ex_by_score": _percent(rows, "ex_by_score"),
        "oracle": _percent(rows, "oracle"),
        "errors": sum(1 for row in rows if row.get("error")),
        "avg_nodes": round(_mean(rows, "nodes"), 2),
        "avg_refinements": round(_mean(rows, "refinements"), 2),
        "early_stop": _percent(rows, "stopped_early"),
        "table_recall": round(100 * statistics.mean(recalls), 2) if recalls else None,
        "avg_tokens_in": int(_mean(rows, "tokens_in")),
        "avg_tokens_out": int(_mean(rows, "tokens_out")),
        "p50_s": round(statistics.median(row.get("ms") or 0 for row in rows) / 1000, 1),
        **_costs(rows),
    }
    summary["per_task_by_role"] = _per_task_by_role(rows)
    return summary


def summarize(rows: list[dict]) -> dict:
    """Return the overall summary and the n / EX / oracle per database."""
    by_db: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_db[row["db"]].append(row)
    per_db = {
        db: {key: value for key, value in _aggregate(db_rows).items() if key in _DB_SUMMARY_KEYS}
        for db, db_rows in sorted(by_db.items())
    }
    return {"overall": _aggregate(rows), "by_db": per_db}


def format_table(summary: dict) -> str:
    """Return the overall summary as a one-row markdown table."""
    overall = summary["overall"]
    lines = [
        "| " + " | ".join(_SUMMARY_COLUMNS) + " |",
        "|" + "|".join("---" for _ in _SUMMARY_COLUMNS) + "|",
        "| " + " | ".join(str(overall.get(column, "")) for column in _SUMMARY_COLUMNS) + " |",
    ]
    return "\n".join(lines)
