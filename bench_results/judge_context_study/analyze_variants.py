import json, sys, collections
from schemagraph.bench.spider2_exec import auroc
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in rows: by[r["variant"]][r["instance_id"]] = r
print(f"{'variant':24} {'AUROC':>6} {'in-task':>7} {'pick':>6} {'mixed':>6} {'tok/call':>8} {'cost':>7} fail")
for v, tasks in by.items():
    s, l, within, picks, mixed_ok, mixed_n, calls, tok, cost, fail = [], [], [], 0, 0, 0, 0, 0, 0.0, 0
    per_task = {}
    for iid, r in sorted(tasks.items()):
        cs = [c for c in r["candidates"] if c["judge"] is not None]
        calls += len(cs); tok += r["tokens_in"]; cost += r["cost_usd"]; fail += r["failures"]
        s += [c["score"] for c in cs]; l += [c["ex"] for c in cs]
        allc = r["candidates"]
        if any(c["ex"] for c in allc):
            top = max(allc, key=lambda c: c["score"])
            picks += top["ex"]; per_task[iid] = top["ex"]
            if not all(c["ex"] for c in allc):
                mixed_n += 1; mixed_ok += top["ex"]
                a = auroc([c["score"] for c in allc], [c["ex"] for c in allc])
                if a is not None: within.append(a)
    wa = sum(within) / len(within) if within else float("nan")
    print(f"{v:24} {auroc(s, l):6.3f} {wa:7.3f} {picks:3}/17 {mixed_ok:3}/{mixed_n:<2} {tok/max(calls,1):8.0f} ${cost:6.3f} {fail}")
    by[v]["_picks"] = per_task
base = by["A_baseline"]["_picks"]
for v in by:
    if v == "A_baseline": continue
    diff = {i: (base[i], p) for i, p in by[v]["_picks"].items() if p != base[i]}
    print(v, "changed picks vs baseline:", diff)
