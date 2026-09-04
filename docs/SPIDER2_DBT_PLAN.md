# Spider 2.0-DBT end-to-end benchmark — how to proceed

Status: research and plan (2026-09-03). Nothing here is built yet.

## What the benchmark actually is

* **67 tasks** (`spider2-dbt/examples/spider2-dbt.jsonl`), each a real dbt project folder plus a
  one-paragraph instruction ("complete the project so that…"). The agent edits/creates dbt
  models (SQL + Jinja + YAML), runs `dbt run`, and the **task's DuckDB file** is the answer.
* **Evaluation** (`evaluation_suite/eval_utils.py::duckdb_match`): for each task, specific
  tables (`condition_tabs`) and column indexes in the agent's DuckDB are compared against the
  gold DuckDB, with numeric tolerance 1e-2 and optional order-insensitivity. Submission format
  is `results_metadata.jsonl` + one folder per task holding the result `.duckdb`.
* **Data not in the clone**: two Google Drive zips (`DBT_start_db.zip` → starting DuckDB per
  task under `examples/<task>/`, `dbt_gold.zip` → gold DuckDB under `evaluation_suite/gold/<task>/`),
  installed by `spider2-dbt/setup.py`. Google Drive is reachable from this machine; `gdown` is not installed.
* **Local prerequisites** (checked 2026-09-03): Docker 28 ✅ · Claude Code CLI 2.1.259 with OAuth
  credentials ✅ (the Claude Agent SDK reuses them) · dbt ❌ (not installed) · gdown ❌ · 433 GB free.

## The three harness options

### A. Reference `methods/spider-agent-dbt` — not worth extending
Docker-based ReAct loop (Bash / CreateFile / EditFile / LOCAL_DB_SQL / Terminate, 30 steps),
OpenAI/Azure/Gemini clients, `openai==1.31` pin and a heavy requirements file. It is the
harness that scores 3–13 % with frontier models on the leaderboard. Useful only as the
published baseline number; no MCP support, no Anthropic client.

### B. Fork SignalPilot's `benchmark/` — the fairest comparison, highest coupling
The #1 entry (65.6). Claude Agent SDK with `permission_mode=bypassPermissions`, Docker image
with `dbt-duckdb` + **libfaketime** (per-task `FAKETIME` derived from the gold DB so
`current_date` models are deterministic), a workdir with `.mcp.json` + `CLAUDE.md` + five dbt
skills, a **verifier subagent** that rebuilds missing models, and a comparator that replicates
Spider2's `compare_pandas_table`.

Coupling to remove for a schemagraph arm: the gateway (Postgres store, `docker compose up gateway`,
`register_local_connection`), the `mcp__signalpilot__*` tool names hard-coded in `core/workdir.py`
and the skills, and the `signalpilot-dbt` Claude plugin. Everything else (faketime wrapper,
`derive_gold_dates.py`, comparator, SDK runner, verifier prompt) is reusable — Apache-2.0.

### C. Own minimal harness in schemagraph — recommended first
`bench/spider2_dbt.py`, ~400 lines, borrowing B's reusable parts:

1. **prepare(task)**: copy `examples/<task>` to a fresh workdir; copy the starting `.duckdb`;
   build a schemagraph home there with two connections — `dbt` (project dir, parsed without
   dbt; declared-but-missing models such as `dim_listings_hosts` show up as tables with their
   YAML columns, which tells the agent what it must build) and `duckdb` (the task DB: tables,
   FKs, samples); write `.mcp.json` pointing at `uv run schemagraph mcp --home <workdir>/.schemagraph`.
2. **run(task, arm, model)**: Claude Agent SDK (`claude_agent_sdk.query`) in the workdir,
   `bypassPermissions`, `max_turns` ≈ 60, timeout ≈ 30 min, system prompt adapted from
   SignalPilot's `prompts/dbt_local_system.md` (public) plus Spider-Agent's task framing.
   Arm **baseline**: no MCP servers. Arm **schemagraph**: `.mcp.json` present and a
   CLAUDE.md line telling the agent to call `link_schema` before writing models and
   `find_join_path` before writing joins. Log every MCP call.
3. **collect + evaluate**: copy the workdir `.duckdb` into a submission folder, write
   `results_metadata.jsonl`, call Spider2's `evaluate.py` (or vendor `duckdb_match`).
4. **report**: pass rate per arm, turns and tokens per task, MCP call counts, and for the
   schemagraph arm whether `link_schema`'s tables covered the tables the gold models touch.

Determinism: reuse SignalPilot's `derive_gold_dates.py` output and the libfaketime `dbt` wrapper
(Docker) from the start; without it the `current_date` tasks are non-reproducible.

Isolation: `bypassPermissions` on a workdir is acceptable natively in WSL for a pilot, but the
full run should use a container (start from `Dockerfile.dbt-agent`, drop the gateway install).

## What this measures — and what it doesn't

It benchmarks **an agent with and without schemagraph context**, holding model, prompt and
budget fixed. The interesting numbers are the delta in pass rate and, just as much, the delta
in turns/tokens (SignalPilot's thesis: scaffolding, not model strength, wins DBT). It does
not measure schemagraph in isolation; the Lite linking benchmark does that.

A cheap intermediate exists and is worth doing first: once the gold zips are downloaded,
check whether `dbt_gold.zip` contains the gold dbt projects (setup.py only moves `.duckdb`
files, so the zip may hold more). If it does, "tables referenced by the gold models" is a
gold set for a **no-execution DBT linking benchmark**, exactly like Lite, run in seconds.

## Cost and time

* Per task: 20–60 agent turns, roughly 0.2–1 M tokens → **$1–5 per task per arm** at
  Sonnet-class pricing, 5–15 min wall-clock. Two arms × 67 tasks ≈ **$150–700** and
  ~4–6 h per arm at 3 concurrent containers. With the Claude Code subscription (OAuth) it
  draws on plan usage instead.
* Build effort: data + env half a day; harness + 5-task pilot one day; full runs and write-up one day.

## Decisions needed before starting

1. **Model** for both arms (Sonnet-class for cost, or Opus 5 to match the top of the board).
2. **Credentials**: Claude Code subscription (OAuth, already logged in here) vs an API key.
3. **Isolation**: native WSL pilot first, or Docker from day one.
4. **Scope**: all 67 tasks, or a stratified 20-task subset for the first comparison.

## Step-by-step

```bash
# 1. data
uv pip install gdown && cd Spider2/spider2-dbt \
  && gdown 'https://drive.google.com/uc?id=1N3f7BSWC4foj-V-1C9n8M2XmgV7FOcqL' \
  && gdown 'https://drive.google.com/uc?id=1s0USV_iQLo4oe05QqAMnhGGp5jeejCzp' \
  && python setup.py
# 2. environment (in schemagraph)
uv add --group bench dbt-duckdb claude-agent-sdk gdown
cd Spider2/spider2-dbt/examples/airbnb001 && dbt deps && dbt run      # sanity: starting project builds
# 3. harness: src/schemagraph/bench/spider2_dbt.py  (prepare / run / collect / evaluate / report)
# 4. pilot: 5 tasks × 2 arms; inspect traces; fix prompt + tool instructions
# 5. full run in Docker, 3 concurrent; 6. write bench_results/spider2_dbt/README.md
```
