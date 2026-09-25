# What SPRIG has to say about the linker — recommendations

Status: **measured and rejected (2026-09-23)**; see *Results* at the end. `LinkOptions.seed_bm25` /
`seed_k` / `seed_w` implement the proposal and stay off for ablation.

Source: **SPRIG** — *Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA*
(arXiv:2602.23372v1, Qizhi Wang, PingCAP). Read in full; paper and analysis are vendored in the
CTRA repo at `docs/2602.23372v1.pdf` / `docs/2602.23372v1.md`, along with the six source papers
behind its seeding variants (BM25, RRF, BGE, HippoRAG, LightPAL, MixPR).

SPRIG builds an entity–document co-occurrence graph with lightweight NER and retrieves by PPR —
the same shape as this linker, but over 5M-passage text corpora instead of a schema. It is the
only paper that systematically ablates **where the PPR seeds come from**, which is exactly the
question `LinkOptions.ranker` is answering.

## The one finding that matters

SPRIG's variants are one algorithm with three seed sources: `GraphHybrid` (BM25 seeds),
`GraphDense` (dense seeds), `GraphRRF` (seeds from the RRF-fused list). Plus two fusion
baselines. Table 5, Recall@10:

| Dataset | RRF alone | **GraphRRF** (seed-side) | **RRF+PPR** (score-side) |
|---|---|---|---|
| HotpotQA | 0.851 | **0.867** | **0.782** |
| 2WikiMultiHopQA | 0.697 | **0.794** | **0.602** |

**Seeding PPR from the fused list beats fusing PPR's output with the lexical ranking afterwards.**
Score-side fusion is worse than not fusing at all — dramatically so on 2Wiki.

`ranker="rrf"` (`linking/linker.py:129-140`) is structurally SPRIG's **`RRF+PPR (fusion)`**:

```python
ppr_rank    = [f for f, s in sorted(tscores.items(), ...) if s > 0]
sparse_rank = [f for f, s in sorted(sparse.items(), ...)]
tscores     = reciprocal_rank_fusion(ppr_rank, sparse_rank, k=opts.rrf_k)
```

**This is not a verdict on the current ranker.** SPRIG's score fusion failed because its PPR input
was junk — query-entity-only `Graph` scores **0.464 / 0.357**, *below plain BM25*. Fusing a bad
ranking into a good one drags it down. Here PPR-alone is **95.66** against BM25F's **94.91**, so
the failure mechanism does not apply, which is why `rrf` gains (**95.85**, the 2026-09-17 default
row; 96.04 is `rrf_nogate`, rrf with the evidence gates off, not the default).

What transfers is the positive result: **seed-side fusion is the untested cell**.

## `ranker="seedrrf"` — the proposal

BM25F becomes a *seed source* rather than a second ranking. Insertion point is **after** the
embedding block (`linker.py:107-114`, which also bumps seeds) and immediately before
`personalized_pagerank` (`linker.py:115`), so every seed source is in place when PPR runs;
`tnode` is already imported at `linker.py:25`.

The BM25 seeds must go into the PPR personalization **only**. If they are added with
`act.bump(tnode(fqn), ...)`, the direct-lexical bonus loop at `linker.py:118-124` (which adds
`0.35 * scale * min(w, 2.0)` for every table seed) picks them up too, and "seed-side only" would
silently carry a BM25 score-side term — contaminating exactly the seed-side vs score-side
comparison this experiment exists to make. Keep them in a separate dict merged into the seeds
passed to PPR:

```python
seeds = act.seeds
if opts.ranker in {"seedrrf", "seedrrf_fuse"}:
    if self._bm25 is None:
        self._bm25 = build_bm25(schema_graph)
    sparse = bm25_scores(self._bm25, question)
    ranked = sorted(sparse.items(), key=lambda x: (-x[1], x[0]))[: opts.seed_k]
    seeds = dict(act.seeds)
    for rank, (fqn, _) in enumerate(ranked):
        n = tnode(fqn)
        seeds[n] = seeds.get(n, 0.0) + opts.seed_w / (rank + 1)
node_scores = personalized_pagerank(schema_graph, seeds, ...)
# the lexical bonus loop below keeps iterating act.seeds, not seeds
```

**Seed weight scale.** `seed_w / (rrf_k + rank + 1)` with `rrf_k=60` gives ≈0.016 per BM25 seed,
against 1.0 for a name hit and up to 0.8 for an embedding hit. PPR normalizes the personalization
vector, so BM25 seeds would hold ~1–2 % of teleport mass and the experiment would measure nothing.
Use `seed_w / (rank + 1)` instead (rank-based, but the top seed lands on the name-hit scale at
`seed_w=1`), and put `seed_w` in the grid. Note that `specificity_weights` also multiplies table
seeds (`1/log(1+cnt)`), scaling them down further on schemas with repeated table names.

Three details from SPRIG's own setup, all of which matter:

* **Rank-based seed weights, not score-based** — `1/(rank + 1)` (RRF's shape without the
  `k=60` offset, which would flatten the weights to noise; see above). BM25F
  scores and the activator's lexical weights (name hit 1.0, desc hit 0.35, embed 0.8·cosine) are
  on unrelated scales with nothing to calibrate them against. Ranks sidestep that, which is the
  whole reason Cormack et al. chose reciprocal ranks in the first place.
* **Small seed counts** — SPRIG uses **k=5–10** (k=10 HotpotQA / k=5 2Wiki for BM25 seeds,
  k=5 / k=3 for dense). Not top-50.
* **Downweight the query-side seeds** — SPRIG scales entity seeds by `df(e)^−q` so externally
  seeded mass isn't swamped by activation mass. `specificity_weights` is the existing analogue;
  see the exponent item below.

Then either take PPR's output directly, or **also** score-fuse on top. SPRIG never tested
seed-side *and* score-side together; with two strong rankers here that is plausibly the best cell
in the grid and costs one extra config.

## Check this first — it may say don't bother

SPRIG A.9: *"GraphHybrid is more robust because BM25 seeding compensates for NER errors."*

The analogue here is a table whose **name** doesn't token-match the question but whose
**description** does: `activate` yields few or no seeds and PPR has nothing to spread. That is the
only mechanism by which seed-side fusion can help.

So before writing any code, segment the Spider2-Lite misses by `len(act.seeds)`. **If the failures
are not concentrated in the low-seed bucket, there is no headroom and this whole plan is dead.**
One pass over the existing harness with `debug=True`.

## Smaller items, worth a sweep

* **Seed specificity exponent.** `graph/ppr.py:specificity_weights` hardcodes `1/log(1+cnt)`.
  SPRIG uses `df(e)^−q` and tuned **q=0.5 on HotpotQA, q=1.0 on 2Wiki** — dataset-dependent
  enough that the curve shape is worth exposing as an option rather than fixed.
* **Hub penalty.** SPRIG applies `p=0.5` in graph normalization, *separate from* seed specificity.
  `PPRMatrix.__init__` does plain degree normalization (`inv[nz] = 1.0 / deg[nz]`); a `p` exponent
  interpolates between that and none. The hubs here are shared key columns and the `contains` star
  out of wide tables — `graph/infer.py`'s `max_fanout=12` covers inferred edges, not this.
* **SPRIG-MIX.** Explicit entity–document seed mixing. Partly present already: seeds span tokens,
  glossary terms, columns and tables, and the direct lexical bonus at `linker.py:118-124` adds
  table-level evidence after PPR. Worth reading their formulation before changing that bonus.

## What SPRIG confirms is already right

* **`rrf_k=60`.** Cormack et al. 2009 fixed k=60 in a pilot and never re-tuned it; their sweep
  shows MAP moves 0.2134 → 0.2147 across k=20…100, peaking at k=80. The comment in `LinkOptions`
  about smaller k weighting the head is correct but the effect is below noise. **Don't sweep it.**
* **`ppr_alpha=0.85`.** Teleport 0.15 — *exactly* SPRIG's α. Two independently tuned systems
  landing on the same value, against HippoRAG's 0.5 and MixPR's 0.6 teleport. Leave it.
* **`embed_top_k=5` with exact search.** SPRIG found tuned HNSW produced *identical* full-validation
  numbers to defaults on both datasets (ANN sensitivity shows up on 2k-query subsets and washes out
  at scale). No reason to add an ANN index.

## What does not transfer

* **SPRIG-PRUNE** (hub pruning / edge caps; 16–28% query-time reduction) targets a bottleneck
  already eliminated — the 300–400 ms was NetworkX subgraph conversion, now cached in `PPRMatrix`,
  and the iteration itself is a few ms.
* **Push-based PPR** pays off on 335k-node graphs, not schema graphs.
* **`max_iter=5`.** SPRIG truncates hard, which keeps mass local. `ppr.py` does the opposite with
  `tol=1e-12`, and the comment there says why: at `1e-6` a 7,000-node schema stopped with L1 error
  near 1e-2, enough to reorder near-tied tables at rank 1. That reasoning holds, and a schema
  graph's diameter is small enough that 5 iterations is near-converged anyway. Low value either way.

## Suggested order

1. **Seed-count diagnostic** on Spider2-Lite misses (above). Gate on this.
2. `ranker="seedrrf"` with `seed_k=5`, rank-based weights. Baseline to beat: **rerun the current
   default first** — 95.85 was the `rrf` default on 2026-09-17, and the embedding activator
   (+0.38, 96.23 with `embed=true`) has landed since, so neither 95.85 nor 96.04 is current.
   Quote strict and precise strict@7.
3. Grid: `seed_k ∈ {3, 5, 10}` × `seed_w ∈ {0.5, 1, 2}` × `{seedrrf, seedrrf + score-fusion}`.
4. Only if 2–3 move the number: specificity exponent `q`, then hub penalty `p`.

Every item is measurable today against `bench_results/` — unlike the CTRA side of this research,
where the equivalent retrieval harness doesn't exist yet. That asymmetry is the reason to run these
experiments here first.

## Results (2026-09-23)

Spider2-Lite, n = 530, precise sample n = 323 (gold < 5 tables, db > 7 tables), linked n = 152 (db > 20
tables). One process builds each graph once and runs every config (scratch driver, not committed); it
reproduces the published baselines exactly (rrf 95.85 / @7 70.6, ppr 95.66 / 66.9).

**1. Diagnostic.** Under `rrf`, 95 precise tasks miss at @7. Only **18** have a gold table with no
direct table/column seed (miss rate 69 % in that bucket, n = 26); **77** have every gold table
seeded, so they fail on ranking rather than on seeding. The mechanism SPRIG describes exists, but
it covers at most 18 tasks (5.6 points @7).

**2–3. Grid** (`seed_k ∈ {3,5,10}`, `seed_w ∈ {0.5,1,2,4,8}`, seed-only on `ppr` vs seed + score
fusion on `rrf`). Selected rows:

| config | strict | anchor_hit | precise @3 | @5 | @7 | @10 | linked @7 |
|---|---|---|---|---|---|---|---|
| ppr | 95.66 | 67.17 | 38.1 | 56.7 | 66.9 | 78.9 | 45.4 |
| ppr + seeds k5 w2 (SPRIG's GraphRRF cell) | 95.66 | 67.55 | 40.9 | 57.9 | 69.0 | 79.3 | 47.4 |
| **rrf (default)** | **95.85** | 70.38 | 44.3 | 58.8 | **70.6** | 79.6 | 49.3 |
| rrf + seeds k5 w2 | 95.85 | 70.19 | 44.9 | 59.4 | 71.5 | 79.3 | 49.3 |
| rrf + seeds k5 w4 | 96.04 | 70.19 | 45.5 | 59.8 | 71.2 | 79.6 | 49.3 |
| **rrf + embed** (Engine default) | **96.23** | 72.64 | 47.1 | 62.5 | **73.7** | 82.4 | 52.6 |
| rrf + embed + seeds k5 w2 | 96.42 | 72.08 | 46.7 | 61.6 | 73.4 | 82.7 | 52.6 |
| rrf + embed + seeds k5 w4 | 96.60 | 71.70 | 46.4 | 61.6 | 73.1 | 82.4 | 52.6 |

* Seed-side alone (ppr + seeds) **does not** beat score-side fusion (69.0 vs 70.6): the reverse
  of SPRIG Table 5, as predicted above, because PPR here is a strong ranker.
* Seeds + score fusion: +0.9 @7 at best, which is **3 tasks**, all on one 9-table database
  (`sf_bq182`, `sf_bq217`, `sf_bq192`), all in the unseeded-gold bucket. Weights 0.5–1 do nothing,
  so the realistic range is w ≥ 2.
* With embeddings on (what users actually get), seeding trades +1–2 strict tasks for −0.3 to −0.6
  @7 and −0.5 to −0.9 anchor_hit. The embedding activator already covers the paraphrase and
  description cases BM25 seeds would reach.

**Verdict:** not adopted. Per step 4, the `q` (specificity exponent) and `p` (hub penalty) sweeps
are gated on 2–3 moving the number, and they didn't, so they were not run. The larger bucket, 77
misses with every gold table seeded, is a ranking problem among seeded tables and is where the
next lever is.
