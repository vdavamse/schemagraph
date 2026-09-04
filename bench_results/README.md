# Spider 2.0-Lite schema-linking benchmark

Gold-table recall of `link_schema` over the Spider 2.0-Lite tasks, using the schema
files shipped in the `xlang-ai/Spider2` repo and the gold table lists in
`methods/gold-tables`. No SQL execution, no cloud credentials, **no LLM** — every
number below is the deterministic lexical → PPR → path-union → prune pipeline.

```bash
bench_results/run_sweep.sh /path/to/Spider2            # all configs, resumable
uv run schemagraph bench-spider2-lite /path/to/Spider2  # one config; --opt key=value for LinkOptions
uv run schemagraph bench-spider2-lite /path/to/Spider2 --min-db-tables 100   # large-schema subset
```

530 of 547 tasks scored (12 SQLite tasks skip because the clone ships no schema files
for `sqlite-sakila` / `Db-IMDB`; 5 have empty gold lists). Last run: 2026-09-04. The same
command with `--suite snow` scores the 547 Spider 2.0-Snow tasks (see below).

## Metrics

Table names are canonicalised on both sides so a partition family counts as one logical
table: date/year shards (`events_20210101…`, `gsod1929…gsod2023`) and, more generally,
tables in the same schema whose names differ only in digit runs *and* share ≥ 80 % of
their columns (`2000_q1…2018_q4`, `county_2018_1yr`, `zip_codes_2017_5yr`). The linker
returns the family with its member list; picking the shard is a value-level decision
for the SQL writer (the same partition-family dedup ReFoRCE and SignalPilot apply).

| metric | definition |
|---|---|
| recall | mean over tasks of \|gold ∩ pred\| / \|gold\| |
| precision | mean over tasks of \|gold ∩ pred\| / \|pred\| |
| **strict_recall** | % of tasks where **every** gold table is returned (EviLink's SRR; the number that tracks execution accuracy) |
| anchor_hit | % of tasks where the PPR anchors alone already cover gold (before path union and fill) |
| gold_in_top10 / top20 | % of tasks where every gold table is within the top 10 / 20 of the candidate ranking (ranking quality, independent of budget) |
| avg_pred_tables | tables returned per task |

## Headline (defaults: max 20 tables, 6 anchors, docs on, inferred edges on)

| split | n | recall | precision | strict_recall | anchor_hit | gold_in_top20 | avg_pred_tables | avg_db_tables | p50_ms |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 530 | 98.12 | 30.68 | **95.66** | 66.79 | 95.47 | 11.76 | 18.0 | 23 |
| bigquery | 205 | 96.64 | 39.14 | 92.68 | 69.27 | 92.68 | 10.23 | 17.4 | 32 |
| snowflake | 202 | 98.47 | 25.21 | 96.04 | 66.34 | 96.04 | 11.98 | 19.9 | 30 |
| sqlite | 123 | 100.0 | 25.58 | 100.0 | 63.41 | 99.19 | 13.93 | 16.0 | 15 |

Large-schema subset (databases with ≥ 100 raw tables, n = 83): **90.36 strict**, 95.10 recall, 13.4 tables returned.

(p50 latency on 2026-09-04 was measured with three benchmark processes running concurrently; the 2026-09-03 single-process numbers were 17 / 24 / 22 / 11 ms.)

## Budget

| config | recall | precision | strict_recall | anchor_hit | avg_pred_tables |
|---|---|---|---|---|---|
| max 6, anchors 3 | 89.60 | 38.18 | 77.17 | 53.77 | 7.5 |
| max 12, anchors 4 | 95.79 | 33.03 | 90.00 | 60.00 | 9.6 |
| **max 20, anchors 6 (default)** | 98.12 | 30.68 | **95.66** | 66.79 | 11.8 |

`gold_in_top20` is the same in every row (95.3 % on 2026-09-03, 95.5 % now): the ranking is the same, only the cut changes. At 20 tables the returned set equals the ranking's top-20 coverage, i.e. the budget no longer loses anything the ranking found.

## Ablations (full set, defaults otherwise)

| variant | strict_recall | Δ | what it isolates |
|---|---|---|---|
| **default** | **95.66** | | (95.28 on 2026-09-03, when the ablation rows below were run) |
| no partition-family collapse | 86.79 | −8.5 | year/quarter families (`gsod2019`, `2000_q1`, `zip_codes_2017_5yr`) as separate tables |
| no adaptive budget | 86.23 | −9.1 | measured at max 12: schemas > 30 tables not widened to 20 |
| no bypass-if-fits | 87.74 | −7.5 | measured at max 12: schemas that fit the budget still filtered |
| no IDF token weighting | 88.30 | −7.0 | `average`, `population`, `data` weighted like rare tokens |
| schema routing on (0.5) | 88.87 | −6.4 | DBCopilot-style dataset boost — *hurts*; gold tables often span datasets |
| sum column aggregation | 89.81 | −0.2 | top-3 vs sum (−2.4 on the large subset) |
| no inferred edges | 89.25 | −0.75 | name-based `x_id → x.id` joins (−3.3 on sqlite) |
| no external-knowledge docs | 89.62 | −0.4 | |

(The ablation rows that say "measured at max 12" were run before the default budget moved to 20; their deltas are relative to the 90.00 max-12 baseline.)

## The iteration log (large-schema problem)

Starting point on 2026-09-02: 75.7 % strict overall, 18.9 % on schemas > 100 tables.
Instrumenting the ranking (where do gold tables land?) showed the failures were
*ranking* failures, not budget failures: in 42 % of large-schema tasks a gold table was
outside the top 20. Dissecting activations gave four mechanisms, each fixed and measured:

| step | change | strict overall | strict large (≥100) |
|---|---|---|---|
| 0 | baseline (date/year shard collapse only) | 75.66 | 51.81 |
| 1 | IDF weighting of token seeds; top-3 column aggregation; question-scaffolding stopwords; ignore digit tokens < 4 chars | | |
| 2 | generalised family collapse (digit-run signature + column Jaccard ≥ 0.8); strip table names (NHTSA `" accident_2015"`) | 80.38 | 67.47 |
| 3 | adaptive budget: schemas > 30 tables get 20 tables / 6 anchors | 84.15 | 79.52 |
| 4 | bypass-if-fits: schema ≤ budget → return all tables, keep anchors and paths in the DDL | 90.00 | 80.72 |
| 5 | default budget 12 → 20 (same ranking; cut moves to where the ranking already is) | 95.28 | 87.95 |
| 6 | 2026-09-04 fixes: whole-schema bypass also fires when the question activates nothing; family collapse no longer drops a second cluster with the same digit-run signature (`imaging_level2_metadata_r*` vs `imaging_level4_metadata_r*`); score ties broken by name so results no longer depend on `PYTHONHASHSEED` (one large-schema task flipped between runs) | **95.66** | **90.36** |

Tried and rejected: **schema routing** (blend each table's score with its dataset's
activation mass, DBCopilot-style). −6 to −7 strict points everywhere: Spider 2 gold sets
routinely join across datasets (`zip_codes` from one, `gsod` from another), and the boost
suppresses the second dataset.

## What still misses (23 of 530 at defaults)

| class | n | example | direction |
|---|---|---|---|
| gold at rank 17–44 in 24–170-table schemas (rank 17 misses because the fill floor cuts before the budget) | 18 | META_KAGGLE `users` (26), OPENAQ `global_air_quality` (33), PATENTS `match_app` (44) | second-tier evidence: column descriptions carry the hook but at 0.35 weight; a query-rewrite round (LinkAlign) or the optional Claude anchor pass |
| gold at rank 78 under a 4,000-char external document | 1 | census_bureau_acs_1 (2,900 seeds activated) | the document activates nearly the whole graph; cap or IDF-weight document tokens separately |
| gold table absent from the shipped schema files | 4 | `persistent_udfs.*` (UDFs), `census_bureau_acs.zip_codes_2017_5yr` referenced from the `fda` task, `idc.columns` / `idc.rows`, mitelman `copy_number_segment_*` | unwinnable on Lite; the Snow files include most of these, which is why Snow scores higher |

The benchmark keeps the full ranking (`ranking_limit=0`), so an empty worst-rank now means the
table is not in the graph, never "beyond the top 60". Per-task rows with the missed tables and
each gold table's worst rank are in `spider2_lite_<config>.csv`.

## Spider 2.0-Snow (`--suite snow`)

All 547 tasks on Snowflake; 530 of them are the Lite tasks ported to Snowflake on the same
databases, so after partition-family collapse the graphs are the same size (18.0 tables). This
is the gold-table file the EviLink / APEX-SQL / RSL-SQL / LinkAlign comparisons use, so it removes
the Lite-versus-Snow objection to comparing with them; it does not remove the table-versus-field
one (they score columns). Output files: `spider2_snow_<config>.*`.

| split | n | recall | precision | strict_recall | anchor_hit | gold_in_top20 | avg_pred_tables | avg_db_tables | p50_ms |
|---|---|---|---|---|---|---|---|---|---|
| **snow** | 547 | 98.35 | 30.34 | **96.53** | 66.73 | 95.98 | 11.82 | 18.0 | 21 |

Paired with Lite by task: 504 hit on both, 4 hit only on Snow (Lite gold referenced tables
outside the task's database folder; the Snow files include them), 1 hit only on Lite before the
2026-09-04 bypass fix, and the 17 Snow-only tasks all hit. 19 misses, none absent from the schema
files: 18 at rank 24–44 and one at rank 78 under a long external document — the same tasks as
the Lite miss list.

Field-level strict recall reported on Spider2-Snow by LLM-driven linkers (EviLink, 2605.29670):
EviLink 90.15, RSL-SQL 83.20, APEX-SQL 81.85, AutoLink 73.36, LinkAlign 64.67, ReFoRCE 42.28 at
79k–575k tokens per question. schemagraph's 96.53 is table-level and costs zero tokens; a
field-level number needs gold columns parsed from the 120 public Snow gold SQL files.

## Reading the numbers against the literature

The deep-dive notes record (field-level SRR on Spider2-Snow) ReFoRCE 42.3, LinkAlign 64.7,
AutoLink 73.4, APEX-SQL 81.9, EviLink 90.2 — all LLM-driven at 79k–574k tokens per
question. schemagraph's 96.5 table-level strict recall on Spider 2.0-Snow costs zero
tokens and ~20 ms. The metrics are not identical (table vs field level, Lite vs Snow
gold, partition families collapsed here), so this is a sanity check of the approach, not
a leaderboard entry. Precision is deliberately low (30 %): recall-first, because "LLMs
can ignore noise but cannot guess missing joins" (SchemaGraphSQL), and 12 tables of
annotated DDL is still a small prompt.
