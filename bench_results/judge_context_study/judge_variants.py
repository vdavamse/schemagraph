"""Re-judge a rebuilt pool with Jev under several context variants; keep every score."""
import asyncio, json, os, sys, time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from schemagraph.agent.models import AgentModels, resolve_model
from schemagraph.agent.results import AgentConfig, Candidate
from schemagraph.agent.score import combine
from schemagraph.bench.spider2_exec import McpServers, Runner, load_tasks, new_answerer

root = Path.home() / "data/Spider2"
pool_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
JEV = os.environ["SCHEMAGRAPH_JUDGE_MODEL"]
base = AgentConfig()
VARIANTS = {
    "A_baseline": base,
    "B_schema": replace(base, judge_schema=True),
    "C_schema_notes_checks": replace(base, judge_schema=True, judge_evidence_chars=4000, judge_findings=True),
    "D_C_plus_stats": replace(base, judge_schema=True, judge_evidence_chars=4000, judge_findings=True,
                              judge_stats=True, preview_rows=20),
    "E_D_plus_readings": replace(base, judge_schema=True, judge_evidence_chars=4000, judge_findings=True,
                                 judge_stats=True, preview_rows=20, judge_ambiguity=True),
}
pool = [json.loads(l) for l in open(pool_path)]
if os.environ.get("VARIANTS"):
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in os.environ["VARIANTS"].split(",")}
if only: pool = [p for p in pool if p["instance_id"] in only]
tasks, _ = load_tasks(root, only={p["instance_id"] for p in pool})
by_id = {t.instance_id: t for t in tasks}
runner = Runner(root)
jev = resolve_model(JEV)
done = set()
if out_path.exists():
    for l in open(out_path):
        r = json.loads(l); done.add((r["variant"], r["instance_id"]))

async def main():
    per_db = Counter(by_id[p["instance_id"]].db for p in pool)
    servers = McpServers(runner, Counter({db: n * len(VARIANTS) for db, n in per_db.items()}))
    try:
        with open(out_path, "a") as out:
            for name, cfg in VARIANTS.items():
                models = AgentModels(None, jev, None, None, {"generator": "", "judge": JEV, "selector": JEV, "critic": ""})
                for entry in pool:
                    iid = entry["instance_id"]; task = by_id[iid]
                    url = await servers.url(task.db)
                    if (name, iid) in done:
                        await servers.task_done(task.db); continue
                    answerer = new_answerer(url, runner.executor(task.db), cfg, models)
                    scored = []
                    async with answerer.schema:
                        await answerer.prepare(task.question, evidence=runner.evidence(task))
                        for raw in entry["candidates"]:
                            c = Candidate.model_validate({k: v for k, v in raw.items() if k != "ex"})
                            judgement = None
                            if c.exec and c.exec.ok:
                                judgement = await answerer.judge(c)
                            score, parts = combine(c.checks, c.exec, judgement, cfg.weights)
                            scored.append({"id": c.id, "ex": raw["ex"], "score": score,
                                           "judge": parts["judge"], "fields": judgement.fields if judgement else None,
                                           "rows": c.exec.row_count if c.exec and c.exec.ok else None})
                    await servers.task_done(task.db)
                    tokens = sum(r.input_tokens for r in answerer.records)
                    cost = sum(r.cost_usd for r in answerer.records)
                    out.write(json.dumps({"variant": name, "instance_id": iid, "tokens_in": tokens,
                                          "cost_usd": cost, "failures": sum(1 for r in answerer.records if not r.ok),
                                          "candidates": scored}) + "\n"); out.flush()
                    print(name, iid, f"{tokens/ max(len(answerer.records),1):.0f} tok/call", flush=True)
    finally:
        await servers.aclose(); runner.close()

asyncio.run(main())
