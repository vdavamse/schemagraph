"""Rebuild a judge-study pool from a bench-spider2-exec candidates file: re-execute and re-check
every saved SQL (no model calls) and keep its execution-match label."""
import asyncio, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from schemagraph.agent.models import AgentModels
from schemagraph.agent.results import AgentConfig, Candidate
from schemagraph.bench.spider2_exec import McpServers, Runner, load_tasks, new_answerer

root = Path.home() / "data/Spider2"
cands_path, pool_path = Path(sys.argv[1]), Path(sys.argv[2])
saved = defaultdict(list)
for line in open(cands_path):
    c = json.loads(line); saved[c["instance_id"]].append(c)
tasks, _ = load_tasks(root, only=set(saved))
runner = Runner(root)
cfg = AgentConfig(judge=False, selector=False)
models = AgentModels(None, None, None, None, {"generator": "", "judge": "", "selector": "", "critic": ""})

async def main():
    servers = McpServers(runner, Counter(t.db for t in tasks))
    try:
        with open(pool_path, "w") as out:
            for task in tasks:
                url = await servers.url(task.db)
                answerer = new_answerer(url, runner.executor(task.db), cfg, models)
                async with answerer.schema:
                    await answerer.prepare(task.question, evidence=runner.evidence(task))
                    pool = []
                    for raw in saved[task.instance_id]:
                        if not raw.get("sql"):
                            continue
                        c = Candidate(id=raw["id"], parent_id=raw["parent_id"], depth=raw["depth"],
                                      action=raw["action"], sql=raw["sql"])
                        await answerer.score(c)
                        pool.append({**c.model_dump(mode="json"), "ex": raw["ex"] or 0})
                await servers.task_done(task.db)
                out.write(json.dumps({"instance_id": task.instance_id, "candidates": pool, "error": None}) + "\n")
                print(task.instance_id, len(pool), sum(p["ex"] for p in pool), flush=True)
    finally:
        await servers.aclose(); runner.close()

asyncio.run(main())
