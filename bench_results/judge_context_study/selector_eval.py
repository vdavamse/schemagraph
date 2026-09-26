"""Replay the final pairwise pick on saved candidates: bare selector material vs the judge's context."""
import asyncio, json, os, sys, collections
from dataclasses import replace
from pathlib import Path
from schemagraph.agent.models import AgentModels, resolve_model
from schemagraph.agent.results import AgentConfig, Candidate
from schemagraph.agent.search import select_final
from schemagraph.bench.spider2_exec import McpServers, Runner, load_tasks, new_answerer

root = Path.home() / "data/Spider2"
cands_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
saved = collections.defaultdict(list)
for l in open(cands_path):
    c = json.loads(l); saved[c["instance_id"]].append(c)
tasks, _ = load_tasks(root, only=set(saved)); by_id = {t.instance_id: t for t in tasks}
runner = Runner(root)
jev = os.environ["SCHEMAGRAPH_JUDGE_MODEL"]
models = AgentModels(None, None, resolve_model(jev), None, {"generator": "", "judge": jev, "selector": jev, "critic": ""})
VARIANTS = {"bare": AgentConfig(judge=False, selector_context=False), "context": AgentConfig(judge=False, selector_context=True)}

async def main():
    per_db = collections.Counter(by_id[i].db for i in saved)
    servers = McpServers(runner, collections.Counter({db: n for db, n in per_db.items()}))
    results = []
    try:
        for iid in sorted(saved):
            task = by_id[iid]; url = await servers.url(task.db)
            base = new_answerer(url, runner.executor(task.db), VARIANTS["bare"], models)
            row = {"instance_id": iid}
            async with base.schema:
                await base.prepare(task.question, evidence=runner.evidence(task))
                pool = []
                for raw in saved[iid]:
                    if not raw.get("sql"):
                        continue
                    c = Candidate(id=raw["id"], parent_id=raw["parent_id"], depth=raw["depth"], action=raw["action"], sql=raw["sql"])
                    await base.score(c)          # execute and check again, no judge
                    c.score = raw["score"]        # the run's own combined score
                    pool.append((c, raw["ex"] or 0))
                ex_of = {c.id: ex for c, ex in pool}
                row["top_score_ex"] = ex_of[max(pool, key=lambda p: p[0].score)[0].id] if pool else 0
                for name, cfg in VARIANTS.items():
                    base.cfg = cfg
                    chosen, how, _ = await select_final([c for c, _ in pool], replace(cfg, selector=True), base._pick)
                    row[name] = ex_of.get(chosen.id, 0) if chosen else 0
                    row[name + "_by"] = how
                row["oracle"] = int(any(ex for _, ex in pool))
                row["cost"] = sum(r.cost_usd for r in base.records)
            await servers.task_done(task.db)
            results.append(row); print(row, flush=True)
    finally:
        await servers.aclose(); runner.close()
    json.dump(results, open(out_path, "w"), indent=1)
    for k in ("top_score_ex", "bare", "context", "oracle"):
        print(k, sum(r[k] for r in results), "/", len(results))
    print("selector cost $%.4f" % sum(r["cost"] for r in results))
asyncio.run(main())
