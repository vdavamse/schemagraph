# SignalPilot's Spider 2.0-DBT harness — how it is built

Investigation of `/mnt/c/Users/Jorge/projects/semantic-layer/SignalPilot` (main, `b1faa787`, 2026-09-01),
done 2026-09-03 for the schemagraph DBT benchmark plan. File:line references are relative to that
repo. Where the code contradicts the README, the code is reported.

## 1. Runtime: Claude Agent SDK driving the Claude Code CLI — no LangGraph

There is no LangGraph, LangChain, or custom ReAct loop. `benchmark/requirements.txt` lists only
`claude-agent-sdk`, `python-dotenv`, `httpx`. `benchmark/agent/sdk_runner.py:39-300` calls
`claude_agent_sdk.query()` with `ClaudeAgentOptions`:

| field | value |
|---|---|
| `model` | `--model` arg, default `claude-sonnet-4-6` (`runners/direct.py:664`) |
| `max_turns` | default 200 |
| `permission_mode` | `bypassPermissions` |
| `cwd` | the task workdir |
| `mcp_servers` | from `mcp_config.json` / baked container config (§7) |
| `thinking` | `enabled`, `budget_tokens=20000` |
| `system_prompt` | `prompts/dbt_local_system.md` (23 lines), **replaces** Claude Code's default prompt |

Not set: `allowed_tools`, `disallowed_tools`, `agents`, `hooks`, `setting_sources`. So the agent
has every built-in Claude Code tool plus every MCP tool, and the sub-agents and skills come from a
Claude Code **plugin** installed at user scope in the container (`signalpilot-dbt`), not from SDK
options. Pinned versions: `claude-agent-sdk==0.1.59`, `@anthropic-ai/claude-code@2.1.156`
(`Dockerfile.dbt-agent:14,28`); a later "opus5 state-audit" image bumps to CC 2.1.228 / SDK 0.2.136.

Retry: 3 attempts, only on `529`/`overloaded`, backoff `30·attempt` s. Turns are counted per
`AssistantMessage`. The `timeout` argument (1800 s) is **only logged** — there is no `asyncio.wait_for`,
so no wall-clock limit is enforced. `total_cost_usd`/`usage` are captured from the `ResultMessage`
but dropped before the audit record is written (`direct.py:91-97` vs `audit.py:499-533`).

## 2. Per-task lifecycle (`runners/direct.py:659-805`)

1. Delete any previous gateway connection named `<instance_id>`; MCP stdio sanity check (`list_tools`).
2. Load the task from `spider2-dbt.jsonl` and its eval entry (`condition_tabs`) — the latter is loaded
   but **not** passed to the agent (`agent/prompts.py:8-20` ignores it).
3. `prepare_workdir` (`core/workdir.py:29-76`): `rmtree` + copy `examples/<id>` → `_dbt_workdir/<id>`,
   delete `target/partial_parse.msgpack`, write `.mcp.json`, run `dbt deps` (120 s) then **delete
   `packages.yml`**, `git init`. No skills are copied for DBT (they come from the plugin); no
   reference snapshot is taken (the `reference_snapshot.md` the legacy verifier prompt reads is never written).
4. `write_claude_md` (`workdir.py:261-303`): task instruction, connection name, DuckDB path, a list of
   19 `mcp__signalpilot__*` tools, two key rules.
5. `register_local_connection` (`core/mcp.py:51-103`): writes a DuckDB connection into the **gateway's
   Postgres store** (falls back to `POST /api/connections`). This is what lets `query_database` reach the file.
6. `run_agent`: user prompt is literally `DBT TASK: {instruction}`; system prompt from the file above.
7. `CHECKPOINT` the DuckDB, delete the connection.
8. `evaluate` (§6). Audit record under `$BENCHMARK_AUDIT_DIR/runs/single-<id>-<ts>/{run_metadata.json,
   tasks/<id>.json, traces/<id>.json, queries/<id>.jsonl, projects/<id>/}`.

**Dead code.** `_post_agent_dbt_run` (post-agent `dbt run --select <condition_tabs>` + a "quick-fix"
agent whose prompt embeds the eval-critical models' YML columns), `_run_name_fix_stage` (renames
missing `condition_tabs` tables), their async twins, `execute_dbt_task` (in-process parallel runner),
and `run_value_verify_agent` (hard-coded `claude-opus-4-6`) are all defined and never called.
`prompts/dbt_verify_subagent.md` has no consumer. `main()` goes straight from the agent to evaluation.
Had those stages been live they would have leaked `condition_tabs` into the agent's context; as shipped,
the agent sees only the instruction.

**Not in the repo.** `run4.sh` (the 3-container orchestrator the README describes), `gold_build_dates.json`
(output of `derive_gold_dates.py`, gitignored), `prompts/dbt_local_user.md`. Concurrency and per-task
FAKETIME injection are documented, not present.

**FAKETIME.** `Dockerfile.dbt-agent:71-75` sets `FAKETIME="2024-09-08 12:00:00"` and replaces the `dbt`
binary with a wrapper: `run|compile|build|run-operation|seed|snapshot|test` run under
`LD_PRELOAD=libfaketime` with `FAKETIME_DONT_FAKE_MONOTONIC=1 FAKETIME_NO_CACHE=1 DO_NOT_TRACK=1
DBT_NO_VERSION_CHECK=1`; `deps`/`parse` use the real clock. `derive_gold_dates.py` reads each **gold**
DuckDB, looks for calendar/spine tables, and sets build date = `max(date_day)+1` (fallback: today).
Per-task values are injected only via `docker run -e FAKETIME`.

## 3. Prompts

* `dbt_local_system.md` (the live system prompt, 23 lines): skills/MCP instructions override "be
  efficient"; never replace a prescribed tool or sub-agent with a script; derive SQL from data, not
  description words; trust the YML contract (model names, columns, materializations; `name:` =
  filename; never create models absent from YML); minimal edits to complete models; **"Always load the
  dbt-workflow skill before any other action."**
* User prompt: `DBT TASK: {instruction}`.
* `dbt_verify_subagent.md` (129 lines, unused), `system_general.md` (SQL suites; still lists two tools
  that no longer exist, `dbt_project_map`/`dbt_project_validate`), `post_grade_review.md` (ADE-bench,
  explicitly gold-revealing, post-grade only), `kb_generation_system.md` (knowledge-base generation agent).

## 4. Skills and sub-agents (the plugin, `signalpilot-plugin/`)

Installed as `signalpilot-dbt@signalpilot`, invoked as `/signalpilot-dbt:<skill>`. The older
`benchmark/skills/` tree is used only for the SQL suites.

**dbt-workflow (363 lines) — the 8-step workflow the agent must follow:**
1. Map: run `scan_project.py` and the MCP tool `analyze_project_db` in parallel; write `prebuild_state.md`
   (per model: CREATE / REWRITE STUB / MODIFY / VERIFY, materialization, tables, sampled columns, every
   `ref()/source()/var()` binding) before any materializing dbt command.
2. Load `dbt-write`, the DB SQL skill (`duckdb-sql`), one of seven `domain-*` skills chosen by keyword,
   and conditionally `dbt-testing`/`dbt-snapshots`/`dbt-versioning`.
3. `validate_project.py` (`dbt parse` → structured errors, orphan YML patches); rebuild pre-existing
   models flagged for `current_date`.
4. Read macro definitions (a macro defines an output column only if its inputs exist upstream).
5. Research: **driving-table rule** (COUNT/SUM metrics → `FROM parent LEFT JOIN child`; ratio/avg →
   aggregate child then `INNER JOIN`), cardinalities, YML contract, `map_columns`, sibling patterns,
   `SELECT DISTINCT` on categoricals.
6. `/signalpilot-dbt:knowledge-base` → write `technical_spec.md` (seven mandatory fields per model).
7. Write SQL; `inspect_model_state.py`; `dbt run --select <models>` (never bare `dbt run`, never `+`).
8. Dispatch the `verifier` and `value-verifier` sub-agents in parallel; reproduce FAILs; loop to 6–7
   until all PASS.

Definitions the skill fixes: **stub** = SQL under 5 chars, `select * from`, trailing comma/paren,
unbalanced parens, or `TODO/FIXME/PLACEHOLDER` (`scan_project.py:202-210`). **Output shape** = parse the
YML `description:` for entity, qualifier, temporal scope (MoM/WoW → latest date only), date boundary,
period-over-period (`CAST(NULL AS DOUBLE)` on first build). **Grain inference** priority: unique-key
composition → column list → upstream grain → source cardinality → sibling row counts. **What to trust in
YML**: names, descriptions, refs — not `unique`/`not_null` tests for grain.

**dbt-write (304 lines)**: exact YML aliases, COALESCE ordering by NULL profile, types and MD5
surrogate keys checked against the "pre-existing reference table", sibling copy rules, JOIN ladder
(sibling → producer → measure → tests), filters only when explicit, `materialized='table'`.
**dbt-debugging**, **duckdb-sql**, **knowledge-base**, **dbt-knowledgebase**: cheat sheets and KB rules.

**Sub-agents** (`agents/verifier.md` 100 lines, `agents/value-verifier.md` 69 lines): both read-only
("Fix nothing"); the verifier checks required models exist, column schema (`check_model_schema`),
sources (`audit_model_sources`), row counts; the value-verifier calls `verify_model_values` and
`analyze_project_db`. Fixes are applied by the main agent after re-entering step 6.

Scripts that are portable (filesystem + `dbt parse` only): `scan_project.py` (610 lines),
`validate_project.py` (235), `inspect_model_state.py` (260).

## 5. MCP tool surface (gateway `mcp/tools/`, ~62 tools registered, all exposed)

| group | tools |
|---|---|
| connections | `list_database_connections`, `connection_health`, `connector_capabilities` |
| query | `plan_query`, `query_database` (read-only, LIMIT injection), `check_budget`, `explain_query`, `validate_sql`, `query_history`, `estimate_query_cost`, `debug_cte_query` |
| schema | `describe_table`, `list_tables`, `get_date_boundaries`, `find_join_path`, `get_relationships`, `explore_table`, `explore_column(s)`, `schema_overview`, `schema_ddl`, `schema_link`, `schema_statistics`, `schema_diff*`, `get_dbt_profile`, Xata branch tools |
| **dbt-specific** | `dbt_error_parser`, `generate_sql_skeleton`; `analyze_project_db`, `map_columns`, `find_column_producers` (`model_map.py`, 1163 lines); `check_model_schema`, `analyze_grain`, `validate_model_output`, `audit_model_sources`, `compare_join_types`, `verify_model_values` (`model_verify.py`, 1067 lines); `dbt_execute` (gated off in the benchmark) |
| knowledge | `get_knowledge`, `search_knowledge`, `read_knowledge`, `propose_knowledge`, `archive_knowledge` |
| other | semantic-layer metric tools (Snowflake/Databricks views only), notebooks, Notion, sandbox exec, workspace projects |

How the dbt tools work (they are self-contained in the two files above and read the DuckDB catalog,
the model SQL, and the YML — never gold):

* `analyze_project_db`: lookup-join detection (`*_id` column ↔ table `{p}s|stg_{p}|dim_{p}s` with `id`
  and `name`), staging-vs-raw row gaps, and **driving-table hints** by pigeonhole on
  `COUNT(DISTINCT parent.id)` vs `COUNT(DISTINCT child.fk)`; cap 10 hints.
* `map_columns`: reads the model's `ref()/source()` upstreams and YML columns; reports per upstream
  column YML-MATCH / SOURCE-ONLY / positional alignment.
* `find_column_producers`: static scan of every `*.sql` (including packages) for the final SELECT
  projections; lists which existing models already produce each requested column.
* `check_model_schema`: `PRAGMA table_info` vs the caller-supplied YML columns.
* `audit_model_sources`: model vs source row-count ratios (flag <0.5 or >2.0), NULL fraction and
  distinct count per column.
* `verify_model_values`: pick the top row by the first numeric column, slice on the first dimension,
  compare each metric to `COUNT(*)` and `COUNT(DISTINCT *_id)` of the three largest upstream tables;
  MATCH within ±10 %. Heuristic; reads no YML/SQL.
* `map_columns`/`find_column_producers` require `SP_WORKSPACE_ROOT` (or the MCP process cwd) to contain
  the project dir, otherwise they refuse.

**Knowledge base**: Postgres-backed docs scoped org/project/connection; `get_knowledge(task)` returns all
`context` docs plus up to five task-matched `decisions|rules|troubleshooting` docs via hybrid search.
Populated by `runners/kb_generator.py` (an exploration-only agent on the task's own project + DuckDB) and
by agents calling `propose_knowledge`. Contamination notes: the KB is **global and persistent across
runs** and is never reset by the harness; an ADE post-grade session that has gold in context has the
same `propose_knowledge` tool. Nothing in the benchmark code reads gold to build KB entries, but nothing
prevents it either.

## 6. Evaluation (`benchmark/evaluation/comparator.py`, 264 lines)

Replicates Spider 2's `compare_pandas_table`: per gold column (positional `condition_cols`) any
predicted column must match as a vector, `abs_tol = 1e-2`, row counts must match, NULL-first numeric-aware
sort when `ignore_order`. Task passes only if every `condition_tabs` table passes.

More lenient than the official matcher in four places: (a) date/datetime vs string values are
normalised by stripping ` 00:00:00`, `.0`, `T00:00:00`; (b) table names resolve `fct_`↔`fact_` and
case-insensitively on both sides; (c) nullable-dtype normalisation; (d) if the expected `.duckdb`
filename is absent, the **largest** `.duckdb` in the workdir is used. The README claims parity.

## 7. Container (`benchmark/Dockerfile.dbt-agent`, 77 lines)

`python:3.12-slim`, Node 22, Claude Code 2.1.156, the gateway package installed from source,
`dbt-duckdb duckdb pandas pyarrow` **unpinned**, `claude-agent-sdk==0.1.59`, non-root `agentuser`,
`libfaketime` + the `dbt` wrapper, plugin marketplace add + `claude plugin install signalpilot-dbt
--scope user`, baked MCP config launching `gateway.mcp.server` over stdio with `SP_DISABLE_SANDBOX=1`.
Runtime env: `CLAUDE_CODE_OAUTH_TOKEN`, `SP_GATEWAY_URL` (default `http://localhost:3300`),
`DATABASE_URL` (shared Postgres), `SP_ORG_ID`, `SPIDER2_DBT_DIR`, `BENCHMARK_WORK_DIR`,
`BENCHMARK_AUDIT_DIR`, `FAKETIME`, `SP_WORKSPACE_ROOT`.

## 8. What is coupled to the gateway (what a schemagraph arm must replace)

* **Server wiring**: `mcp_config.json`, the baked config in the Dockerfile, `core/paths.py:40-42`,
  `core/mcp.py:14-44` (env injection, `cwd` rewrite), `_mcp_sanity_check` requiring the key `signalpilot`,
  `workdir.py:48-55` (`.mcp.json`). The key name determines the `mcp__signalpilot__` prefix.
* **Connection registration** into the gateway Postgres store (`core/mcp.py:51-141`, callers in
  `direct.py`): a replacement server must bind `connection_name == instance_id` to the workdir DuckDB
  some other way (e.g. take the path as an argument, or build a schemagraph home per workdir).
* **Hard-coded tool names**: `workdir.py:276-296` (CLAUDE.md), SQL prompt builders, `system_general.md`,
  the (dead) fix prompts, validation tests.
* **The plugin**: `dbt-workflow` steps 1, 5, 8 and both sub-agents call `analyze_project_db`,
  `map_columns`, `find_column_producers`, `check_model_schema`, `audit_model_sources`,
  `compare_join_types`, `verify_model_values`, `get_knowledge`, `query_database`, `list_tables`,
  `explore_columns`, `get_date_boundaries` by name. `dbt-write` and `dbt-debugging` still cite the
  removed `dbt_project_map`/`dbt_project_validate`.
* **Portable as-is**: `scan_project.py`, `validate_project.py`, `inspect_model_state.py`, the FAKETIME
  wrapper, `derive_gold_dates.py`, `evaluation/*`.

Mapping burden for schemagraph: `list_tables`/`describe_table`/`explore_*` → `get_table`;
`schema_link` → `link_schema`; `find_join_path` → same. `query_database` is used ~15 times across skills
and sub-agents and has no counterpart (schemagraph deliberately does not execute SQL) — the harness would
need a separate read-only DuckDB query tool. The dbt verification tools (`check_model_schema`,
`audit_model_sources`, `verify_model_values`, `analyze_project_db`) have no counterpart; steps 1, 5 and 8
of the workflow and both verifier agents would have to be rewritten around `dbt_project_status` plus a
query tool, or those tools reimplemented.

## Architecture in one diagram

```
host (run4.sh — documented, not in repo)              gateway stack (docker compose)
  per task: fresh volume, -e FAKETIME=<gold date>       Postgres: connections, knowledge docs, audit
        │                                                       ▲
        ▼                                                       │ store / HTTP :3300
┌─ container (Dockerfile.dbt-agent) ─────────────────────────────┼─────────────────────────────┐
│ python -m benchmark.run_direct <id>                            │                             │
│  1 prepare_workdir  copy examples/<id>; .mcp.json; dbt deps; rm packages.yml; git init       │
│  2 write CLAUDE.md  instruction + 19 mcp__signalpilot__* tool names                          │
│  3 register DuckDB connection "<id>" in the gateway store ─────┘                             │
│  4 claude_agent_sdk.query → claude CLI 2.1.156                                               │
│       system = dbt_local_system.md   user = "DBT TASK: <instruction>"                        │
│       max_turns 200 · bypassPermissions · thinking 20k · no wall-clock timeout · retry×3/529 │
│       ┌── Claude Code ──────────────────────────────────────────────────────┐                │
│       │ plugin signalpilot-dbt: dbt-workflow (8 steps), dbt-write, domain-*,│                │
│       │   duckdb-sql, knowledge-base; agents verifier + value-verifier (RO) │                │
│       │ scripts: scan_project / validate_project / inspect_model_state      │                │
│       │ Bash dbt → libfaketime wrapper for run/compile/build                │                │
│       │ MCP stdio → gateway.mcp.server (~62 tools): query_database,         │──► workdir     │
│       │   list_tables, analyze_project_db, map_columns, check_model_schema, │    .duckdb     │
│       │   audit_model_sources, verify_model_values, get_knowledge …         │                │
│       └─────────────────────────────────────────────────────────────────────┘                │
│  (dead: post-agent dbt run, quick-fix agent, name-fix agent, value-verify agent)             │
│  5 CHECKPOINT; evaluate vs gold/<id>/*.duckdb (condition_tabs/cols, abs_tol 1e-2, fct_↔fact_)│
│  audit → runs/single-<id>-<ts>/{run_metadata, tasks/, traces/, queries/, projects/}          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Consequences for the schemagraph DBT plan

1. The "agent" is Claude Code itself; the harness's value is the **plugin** (workflow skill, two
   verifier agents, three scripts) and the **dbt verification MCP tools**. That is what to reuse or
   re-implement, not the runner.
2. A two-arm comparison inside this harness is not just swapping `.mcp.json`: the workflow skill and
   verifier agents are written against SignalPilot's tool names and semantics. A fair schemagraph arm
   needs a fork of the plugin with steps 1/5/8 rewritten, plus a read-only DuckDB query tool.
3. Reuse verbatim: the FAKETIME wrapper, `derive_gold_dates.py`, the three project scripts, and the
   comparator (noting its leniency, or use Spider 2's official `evaluate.py` for the headline number).
4. Fix in our copy: enforce a wall-clock timeout around `query()`, keep `total_cost_usd`/`usage` in the
   audit record, and reset any shared knowledge state between tasks.
