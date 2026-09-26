"""Replay selection rules on judged pools: top score vs agreement tie-break; early-stop agreement."""
import json, sqlite3, sys, collections, pickle, os
from pathlib import Path
DBS = Path.home() / "data/Spider2/spider2-lite/resource/databases/spider2-localdb"
pool = {json.loads(l)["instance_id"]: json.loads(l) for l in open(sys.argv[1])}
runs = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
def db_path(db):
    return next(p for p in DBS.glob("*.sqlite") if p.stem.lower().replace("-", "_") == db.lower().replace("-", "_"))
dbs = {}
for line in open(Path.home() / "data/Spider2/spider2-lite/spider2-lite.jsonl"):
    t = json.loads(line); dbs[t["instance_id"]] = t["db"]
def norm(v):
    return round(v, 2) if isinstance(v, float) else (float(v) if isinstance(v, int) else v)
CACHE = Path(sys.argv[1]).with_suffix('.sigs.pkl')
sig_cache = pickle.load(open(CACHE, 'rb')) if CACHE.exists() else {}
def sig(iid, cid):
    if (iid, cid) not in sig_cache:
        sql = next(c["sql"] for c in pool[iid]["candidates"] if c["id"] == cid)
        con = sqlite3.connect(f"file:{db_path(dbs[iid])}?mode=ro", uri=True)
        import time
        deadline = time.monotonic() + 30  # the agent's execution timeout
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            data = con.execute(sql).fetchmany(100_000)
            sig_cache[(iid, cid)] = (len(data), frozenset(collections.Counter(tuple(sorted((norm(v) for v in col), key=repr)) for col in zip(*data)).items()) if data else frozenset())
        except Exception:
            sig_cache[(iid, cid)] = None
    return sig_cache[(iid, cid)]
def agree(a, b):
    if not a or not b or a[0] != b[0] or not a[1]: return 0.0
    return len(a[1] & b[1]) / max(len(a[1]), len(b[1]))
EPS = [0.0, 0.01, 0.02, 0.03, 0.05]
for variant in sorted({r["variant"] for r in runs}):
    for eps in EPS:
        ok = total = 0
        for r in runs:
            if r["variant"] != variant: continue
            cs = r["candidates"]
            if not any(c["ex"] for c in cs): continue
            total += 1
            best = max(c["score"] for c in cs)
            near = [c for c in cs if c["score"] >= best - eps]
            if len(near) > 1:
                support = {c["id"]: sum(agree(sig(r["instance_id"], c["id"]), sig(r["instance_id"], o["id"])) for o in cs if o is not c) for c in near}
                pick = max(near, key=lambda c: (support[c["id"]], c["score"]))
            else:
                pick = near[0]
            ok += pick["ex"]
        print(f"{variant:24} tie-break eps={eps:<5} pick {ok}/{total}")

pickle.dump(sig_cache, open(CACHE, 'wb'))
