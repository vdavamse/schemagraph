"""The judge study on the Spider 2.0-Lite local SQLite tasks: does the judge predict EX?

One candidate pool (best-of-N, no judge, no selector, no early stop), each candidate labelled with
execution match, is scored by each judge model; the report gives the AUROC of each judge's mean,
of each rubric field, of the deterministic score and of the combined score against the label,
plus pick accuracy, tokens and latency. Shares the task loading, databases and per-database MCP
servers of :mod:`schemagraph.bench.spider2_exec`.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from schemagraph.bench.spider2_exec import (
    TASK_ERROR_CHARS,
    McpServers,
    Runner,
    all_failed,
    append_record,
    auroc,
    done_ids,
    load_tasks,
    new_answerer,
    read_rows,
)
from schemagraph.bench.spider2_lite import Instance

if TYPE_CHECKING:
    from schemagraph.agent.answer import Answerer
    from schemagraph.agent.models import AgentModels
    from schemagraph.agent.results import AgentConfig, Judgement

StudyProgress = Callable[[str, int, int], None]
# The pool's early stop: above any score, so every task gets its full pool.
_NO_EARLY_STOP = 1.01


def judge_only(
    spider2_root: str | Path,
    *,
    judges: list[str],
    cfg: AgentConfig | None = None,
    gen_models: AgentModels | None = None,
    pool_size: int = 8,
    pool: str | Path | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
    tag: str = "judge",
    out_dir: str | Path = "bench_results",
    use_docs: bool = True,
    progress: StudyProgress | None = None,
    judge_models: dict[str, Any] | None = None,
) -> dict:
    """Measure how well each judge's score predicts execution match.

    1. A pool of ``pool_size`` best-of-N candidates per task (no judge, no selector, no early
       stop), each labelled with execution match; resumable per task.
    2. Every executed candidate scored by each judge with the same rubric and material.
    3. AUROC of each judge's mean, of each rubric field, of the deterministic score and of the
       combined score against the label, plus pick accuracy, tokens and latency.

    Args:
        spider2_root: The Spider2 clone.
        judges: Judge model names.
        cfg: The base agent settings.
        gen_models: Ready generator models (tests); None resolves them.
        pool_size: Candidates generated per task.
        pool: A candidate pool (jsonl) to reuse or extend; default under ``out_dir``.
        limit: Only the first this many tasks.
        only: Only these instance ids.
        tag: Output file tag.
        out_dir: Where the pool and the report go.
        use_docs: Pass each task's external-knowledge document.
        progress: Called with (phase, done, total): the phase is ``pool`` or a judge name.
        judge_models: Judge name -> ready model (tests); other names are resolved.

    Returns:
        The report, also written to ``spider2_judge_<tag>.json``.
    """
    from schemagraph.agent.results import AgentConfig

    root, out = Path(spider2_root), Path(out_dir)
    base = cfg or AgentConfig()
    out.mkdir(parents=True, exist_ok=True)
    pool_path = Path(pool) if pool else out / f"spider2_judge_{tag}_pool.jsonl"
    tasks, standard = load_tasks(root, limit=limit, only=only)
    study = _JudgeStudy(
        runner=Runner(root, use_docs=use_docs),
        standard=standard,
        by_id={task.instance_id: task for task in tasks},
        progress=progress,
    )
    report = asyncio.run(
        study.run(tasks, pool_path, base, gen_models, judges, judge_models or {}, pool_size)
    )
    (out / f"spider2_judge_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@dataclass
class _JudgeScores:
    """What one judge scored across the pool.

    Attributes:
        per_field: Rubric field -> the judge's value per judged candidate.
        means: The judge's mean per judged candidate.
        combined: The combined score per judged candidate.
        labels: Execution match per judged candidate.
        all_scores: The combined score of every candidate; unexecuted ones score without a judge.
        all_labels: Execution match of every candidate.
        ms: Judge latency per judged candidate.
        tokens_in: Judge input tokens.
        tokens_out: Judge output tokens.
        failures: Judge calls that failed.
        picks_ok: Tasks whose top-scored candidate matches.
        picks: Tasks with at least one matching judged candidate.
    """

    per_field: dict[str, list[float]]
    means: list[float] = field(default_factory=list)
    combined: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    all_scores: list[float] = field(default_factory=list)
    all_labels: list[int] = field(default_factory=list)
    ms: list[float] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    failures: int = 0
    picks_ok: int = 0
    picks: int = 0

    def add(self, judgement: Judgement, score: float, label: int) -> None:
        """Record one judged candidate."""
        for name in self.per_field:
            self.per_field[name].append(judgement.fields.get(name, 0.0))
        self.means.append(judgement.mean)
        self.combined.append(score)
        self.labels.append(label)
        self.all_scores.append(score)
        self.all_labels.append(label)
        self.ms.append(judgement.ms)

    def report(self) -> dict[str, Any]:
        """Return the judge's AUROCs, pick accuracy, tokens and latency."""
        ms = self.ms
        return {
            "n": len(self.labels),
            "judge_failures": self.failures,
            "auroc_mean": auroc(self.means, self.labels),
            "auroc_combined": auroc(self.combined, self.labels),
            "auroc_combined_all": auroc(self.all_scores, self.all_labels),
            "auroc_fields": {name: auroc(v, self.labels) for name, v in self.per_field.items()},
            "pick_accuracy": round(100 * self.picks_ok / self.picks, 2) if self.picks else None,
            "pick_tasks": self.picks,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "p50_ms": round(statistics.median(ms), 1) if ms else None,
            "p95_ms": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 1) if ms else None,
        }


@dataclass
class _JudgeStudy:
    """The judge study's shared state.

    Attributes:
        runner: Linkers, executors and the EX cache.
        standard: The evaluation standard per instance id.
        by_id: The study's tasks by instance id.
        progress: Called with (phase, done, total).
    """

    runner: Runner
    standard: dict[str, dict]
    by_id: dict[str, Instance]
    progress: StudyProgress | None = None

    async def run(
        self,
        tasks: list[Instance],
        pool_path: Path,
        base: AgentConfig,
        gen_models: AgentModels | None,
        judges: list[str],
        judge_models: dict[str, Any],
        pool_size: int,
    ) -> dict[str, Any]:
        """Build the pool, then score it with every judge; return the report."""
        try:
            await self._build_pool(tasks, pool_path, base, gen_models, pool_size)
            pool_rows = [row for row in read_rows(pool_path) if row["instance_id"] in self.by_id]
            entries = [row for row in pool_rows if not row.get("error")]  # failures are no data
            report = _pool_report(pool_path, pool_rows, entries)
            for name in judges:
                model = judge_models.get(name)
                scores = await self._score_with(name, model, entries, base)
                report["judges"][name] = scores.report()
            return report
        finally:
            self.runner.close()

    async def _build_pool(
        self,
        tasks: list[Instance],
        pool_path: Path,
        base: AgentConfig,
        gen_models: AgentModels | None,
        pool_size: int,
    ) -> None:
        """Generate the candidates of every task not yet in the pool file."""
        from schemagraph.agent.models import AgentModels

        pool_cfg = replace(
            base,
            strategy="best_of_n",
            budget=pool_size,
            judge=False,
            selector=False,
            early_stop=_NO_EARLY_STOP,
        )
        have = done_ids(pool_path)
        missing = [task for task in tasks if task.instance_id not in have]
        if not missing:
            return  # a complete pool needs no generator key
        models = gen_models or AgentModels.resolve(pool_cfg)
        servers = McpServers(self.runner, Counter(task.db for task in missing))
        try:
            for position, task in enumerate(tasks):
                if task.instance_id in have:
                    continue
                entry = await self._pool_entry(task, servers, pool_cfg, models)
                append_record(pool_path, entry)
                if self.progress:
                    self.progress("pool", position + 1, len(tasks))
        finally:
            await servers.aclose()

    async def _pool_entry(
        self,
        task: Instance,
        servers: McpServers,
        pool_cfg: AgentConfig,
        models: AgentModels,
    ) -> dict[str, Any]:
        """Generate one task's candidates, each labelled with execution match."""
        error = None
        try:
            url = await servers.url(task.db)
            answerer = new_answerer(url, self.runner.executor(task.db), pool_cfg, models)
            result = await answerer.answer(task.question, evidence=self.runner.evidence(task))
            candidates = [
                {
                    **candidate.model_dump(mode="json"),
                    "ex": self.runner.execution_match(task, candidate.sql, self.standard)[0],
                }
                for candidate in result.candidates
            ]
            if not any(candidate.sql for candidate in result.candidates):  # retried on resume
                error = all_failed(result.candidates)
        except Exception as failure:  # recorded; the task contributes no candidates
            candidates, error = [], f"{type(failure).__name__}: {failure}"[:TASK_ERROR_CHARS]
        finally:
            await servers.task_done(task.db)
        return {"instance_id": task.instance_id, "candidates": candidates, "error": error}

    async def _score_with(
        self,
        name: str,
        model: Any,
        entries: list[dict],
        base: AgentConfig,
    ) -> _JudgeScores:
        """Judge every executed pool candidate with one judge model."""
        from schemagraph.agent.models import AgentModels, resolve_model
        from schemagraph.agent.results import RUBRIC_FIELDS

        judge_cfg = replace(base, judge_model=name)
        models = AgentModels(
            gen=None,
            judge=model or resolve_model(name),
            selector=None,
            critic=None,
            names={"generator": "", "judge": name, "selector": name, "critic": ""},
        )
        scores = _JudgeScores(per_field={field_name: [] for field_name in RUBRIC_FIELDS})
        servers = McpServers(
            self.runner, Counter(self.by_id[entry["instance_id"]].db for entry in entries)
        )
        try:
            for position, entry in enumerate(entries):
                task = self.by_id[entry["instance_id"]]
                url = await servers.url(task.db)
                answerer = new_answerer(url, self.runner.executor(task.db), judge_cfg, models)
                async with answerer.schema:
                    await self._judge_task(answerer, task, entry, scores)
                await servers.task_done(task.db)
                if self.progress:
                    self.progress(name, position + 1, len(entries))
        finally:
            await servers.aclose()
        return scores

    async def _judge_task(
        self,
        answerer: Answerer,
        task: Instance,
        entry: dict,
        scores: _JudgeScores,
    ) -> None:
        """Judge one task's executed candidates into ``scores``."""
        from schemagraph.agent.results import Candidate
        from schemagraph.agent.score import combine

        weights = answerer.cfg.weights
        await answerer.prepare(task.question, evidence=self.runner.evidence(task))
        task_scores: list[tuple[float, int]] = []
        for raw in entry["candidates"]:
            label = raw["ex"]
            candidate = Candidate.model_validate({k: v for k, v in raw.items() if k != "ex"})
            if not (candidate.exec and candidate.exec.ok):
                unjudged, _ = combine(candidate.checks, candidate.exec, None, weights)
                scores.all_scores.append(unjudged)
                scores.all_labels.append(label)
                continue
            records_before = len(answerer.records)
            judgement = await answerer.judge(candidate)
            for record in answerer.records[records_before:]:
                scores.tokens_in += record.input_tokens
                scores.tokens_out += record.output_tokens
            if judgement is None:
                scores.failures += 1
                continue
            score, _ = combine(candidate.checks, candidate.exec, judgement, weights)
            scores.add(judgement, score, label)
            task_scores.append((score, label))
        if task_scores and any(label for _, label in task_scores):
            scores.picks += 1
            scores.picks_ok += max(task_scores, key=lambda pair: pair[0])[1]


def _pool_report(pool_path: Path, pool_rows: list[dict], entries: list[dict]) -> dict[str, Any]:
    """Return the report's pool part: counts and the deterministic score's AUROC."""
    report: dict[str, Any] = {
        "pool": str(pool_path),
        "tasks": len(entries),
        "pool_errors": len(pool_rows) - len(entries),
        "candidates": sum(len(entry["candidates"]) for entry in entries),
        "positives": sum(raw["ex"] for entry in entries for raw in entry["candidates"]),
        "judges": {},
    }
    det_scores, det_labels = [], []
    for entry in entries:
        for raw in entry["candidates"]:
            if raw.get("exec") and raw["exec"]["ok"]:
                det_scores.append(raw["checks"]["det"] if raw.get("checks") else 0.0)
                det_labels.append(raw["ex"])
    report["det_auroc"] = auroc(det_scores, det_labels)
    report["executed"] = len(det_labels)
    return report
