"""Replay the final pairwise pick on saved candidates: the bare selector prompt vs judge context.

Usage: ``uv run python selector_eval.py <candidates.jsonl> <out.json>`` with
``SCHEMAGRAPH_JUDGE_MODEL`` set (the selector is the judge model) and the Spider2 clone in
``~/data/Spider2``.

Every saved query is executed and checked again (no judge) and keeps the run's own score; then
the selector picks among the top candidates once per variant of :data:`VARIANTS`. One row per
task records the execution match of the top score, of each variant's pick and of the best
candidate (the oracle), plus the selector's cost.
"""

import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from schemagraph.agent.answer import Answerer
from schemagraph.agent.models import AgentModels, resolve_model
from schemagraph.agent.results import AgentConfig, Candidate
from schemagraph.agent.search import select_final
from schemagraph.bench.spider2_exec import McpServers, Runner, load_tasks, new_answerer
from schemagraph.bench.spider2_lite import Instance

SPIDER2_ROOT = Path.home() / "data/Spider2"
VARIANTS = {
    "bare": AgentConfig(judge=False, selector_context=False),
    "context": AgentConfig(judge=False, selector_context=True),
}


def load_candidates(path: Path) -> dict[str, list[dict]]:
    """Group the records of a bench-spider2-exec candidates file by task."""
    saved: dict[str, list[dict]] = defaultdict(list)
    with open(path) as lines:
        for line in lines:
            record = json.loads(line)
            saved[record["instance_id"]].append(record)
    return saved


def selector_models() -> AgentModels:
    """Resolve the selector alone, as the judge model named by ``SCHEMAGRAPH_JUDGE_MODEL``."""
    judge = os.environ["SCHEMAGRAPH_JUDGE_MODEL"]
    names = {"generator": "", "judge": judge, "selector": judge, "critic": ""}
    return AgentModels(None, None, resolve_model(judge), None, names)


async def rebuild_pool(answerer: Answerer, records: list[dict]) -> list[tuple[Candidate, int]]:
    """Execute and check each saved query again; keep the run's score and execution match."""
    pool = []
    for record in records:
        if not record.get("sql"):
            continue
        candidate = Candidate(
            id=record["id"],
            parent_id=record["parent_id"],
            depth=record["depth"],
            action=record["action"],
            sql=record["sql"],
        )
        await answerer.score(candidate)  # execute and check again; the judge is off
        candidate.score = record["score"]  # the run's own combined score
        pool.append((candidate, record["ex"] or 0))
    return pool


async def replay_task(
    task: Instance,
    records: list[dict],
    servers: McpServers,
    runner: Runner,
    models: AgentModels,
) -> dict:
    """Replay one task's final pick under every variant; return its row.

    Args:
        task: The Spider2 task.
        records: Its saved candidates, as the candidates file stores them.
        servers: The MCP servers, one per database.
        runner: Executors and evidence for the tasks.
        models: The resolved models; only the selector runs.

    Returns:
        The task's row: the execution match of the top score, of each variant's pick and of
        the oracle, how each variant chose, and the selector's cost.
    """
    url = await servers.url(task.db)
    answerer = new_answerer(url, runner.executor(task.db), VARIANTS["bare"], models)
    row: dict = {"instance_id": task.instance_id}
    async with answerer.schema:
        await answerer.prepare(task.question, evidence=runner.evidence(task))
        pool = await rebuild_pool(answerer, records)
        execution_match = {candidate.id: ex for candidate, ex in pool}
        candidates = [candidate for candidate, _ in pool]
        top = max(candidates, key=lambda candidate: candidate.score, default=None)
        row["top_score_ex"] = execution_match[top.id] if top is not None else 0
        for name, cfg in VARIANTS.items():
            answerer.cfg = cfg  # the selector's prompt follows cfg.selector_context
            select_cfg = replace(cfg, selector=True)
            chosen, chosen_by, _ = await select_final(candidates, select_cfg, answerer._pick)
            row[name] = execution_match.get(chosen.id, 0) if chosen is not None else 0
            row[f"{name}_by"] = chosen_by
        row["oracle"] = int(any(ex for _, ex in pool))
        row["cost"] = sum(record.cost_usd for record in answerer.records)
    await servers.task_done(task.db)
    return row


async def main(candidates_path: Path, out_path: Path) -> None:
    """Replay every task of the candidates file, then print the totals."""
    saved = load_candidates(candidates_path)
    tasks, _ = load_tasks(SPIDER2_ROOT, only=set(saved))
    # the candidates file drives the loop: a saved id with no task fails loudly (KeyError)
    by_id = {task.instance_id: task for task in tasks}
    runner = Runner(SPIDER2_ROOT)
    models = selector_models()
    servers = McpServers(runner, Counter(by_id[instance_id].db for instance_id in saved))
    results = []
    try:
        for instance_id in sorted(saved):
            task = by_id[instance_id]
            row = await replay_task(task, saved[instance_id], servers, runner, models)
            results.append(row)
            print(row, flush=True)
    finally:
        await servers.aclose()
        runner.close()
    out_path.write_text(json.dumps(results, indent=1))
    for key in ("top_score_ex", "bare", "context", "oracle"):
        print(key, sum(row[key] for row in results), "/", len(results))
    print(f"selector cost ${sum(row['cost'] for row in results):.4f}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
