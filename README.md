# schemagraph

Graph-native schema context engine for text-to-SQL.

Catalog metadata in — pasted DDL, DuckDB, dbt projects, **Unity Catalog**, **AWS Glue**, **Collibra** — one provenance-tagged schema graph in the middle, and a linked, join-complete sub-schema out, rendered as annotated DDL an LLM can write SQL against. Served over HTTP and MCP, fed and curated through a small web UI.

The linking core is graph search, not embedding search:

* **LinearRAG**-style entity activation from the question (tokens, n-grams, abbreviations, glossary phrases, sample values) — no LLM, no vectors.
* **HippoRAG** personalized PageRank over tables, columns, terms and tokens to rank candidate tables.
* **SchemaGraphSQL** union of shortest paths between anchor tables, so bridge tables are included by construction.
* **PathRAG** flow-based pruning of redundant join paths, serialized most-reliable-last.
* Optional one-call **Claude** pass to pick source/destination tables when you want it.

Design notes: [`docs/DESIGN.md`](docs/DESIGN.md).

## Quick start

```bash
uv sync --all-extras                # Python 3.11+
uv run schemagraph add-ddl examples/store.sql --dialect postgres --schema public
uv run schemagraph link "total revenue by product category for customers in california"
uv run schemagraph serve            # http://127.0.0.1:8765  (UI + API)
```

Build the UI once (served by `serve` from `web/dist`):

```bash
cd web && npm install && npm run build
```

Dev loop for the UI: `npm run dev` (proxies `/api` to port 8765).

## Connect a catalog

From the UI (**Connections**), or the CLI:

```bash
# Unity Catalog (Databricks or OSS server)
uv run schemagraph add unity_catalog prod -c '{"host":"https://adb-123.azuredatabricks.net","token":"${DATABRICKS_TOKEN}","catalogs":["sales"],"lineage":true}'

# AWS Glue
uv run schemagraph add aws_glue lake -c '{"region":"eu-west-1","databases":["lake"],"profile":"prod"}'

# Collibra (asset-type / relation-role names are configurable per operating model)
uv run schemagraph add collibra gov -c '{"host":"https://acme.collibra.com","username":"svc","password":"${COLLIBRA_PASSWORD}"}'

# dbt project, parsed without dbt installed
uv run schemagraph add-dbt --project-dir ~/analytics --name dbt
```

Every table, edge and glossary term remembers which source it came from, and sources merge: Glue gives you columns, dbt gives you lineage and `relationships` tests, Collibra gives you curated relations, descriptions, classifications and business terms, and you can add join hints and glossary entries by hand.

## Use it from an agent

```bash
uv run schemagraph mcp                    # stdio MCP server: link_schema, get_table, find_join_path, ...
uv run schemagraph mcp --transport http   # the same tools over streamable HTTP at http://127.0.0.1:8766/mcp
```

`schemagraph serve` also serves the MCP server at `http://127.0.0.1:8765/mcp`, over the same graph as the API and UI. The tools are read-only and never execute SQL. Or `POST /api/link {"question": "..."}` and paste the returned `ddl` into your prompt.

## Answer questions (optional)

```bash
uv sync --extra agent
export DASHSCOPE_API_KEY=...        # Qwen generator + critic (alibaba:qwen3.8-max)
export TYPESAFE_API_KEY=...         # Jev judge + selector (typesafe:jev-1.13.0)
uv run schemagraph add-duckdb shop.duckdb -n shop
uv run schemagraph ask "revenue by product category for customers in california" -c shop --strategy abmcts --budget 16
```

Or both models through one OpenRouter key: copy `.env.example` to `.env`, fill in the key, and run with `uv run --env-file .env schemagraph ask ...`. `openrouter:` models reason at `SCHEMAGRAPH_REASONING` effort (`off | low | medium | high`, default `medium`; forced tool choice is off while they reason), and Jev reaches OpenRouter's TypeSafe-compatible endpoint through `TYPESAFE_BASE_URL=https://openrouter.ai/api`.

`ask` reads the schema through schemagraph's own MCP server, writes SQL with Qwen, runs it **read-only** against the DuckDB file (guard + engine statement check + hardened connection, timeout and row caps), scores each candidate with deterministic checks and a Jev rubric, and searches with TreeQuest AB-MCTS (`--strategy best_of_n | refine | single` for the baselines). Any pydantic-ai model string works per role via `SCHEMAGRAPH_GEN_MODEL`, `SCHEMAGRAPH_JUDGE_MODEL`, `SCHEMAGRAPH_CRITIC_MODEL`. Cost scales with `--budget` (generator nodes). Every usage record carries the billed cost in USD (OpenRouter's reported cost per response, failed and retried requests included; Jev at its published $0.042 per million input tokens when the response has no cost), reasoning and cache tokens; `ask` prints it per role and `bench-spider2-exec` summarises it per task (mean, p50, p90, max, total).

By default `ask` serves this home's schema, scoped to the connection, on an ephemeral localhost port for the length of the call. `--mcp-url http://127.0.0.1:8765/mcp` points the agents at a running server instead (`serve`'s `/mcp`, or `mcp --transport http`); that server is not scoped, so `-c` then only picks the database the SQL runs on. The generator gets the server's `link_schema`, `search_tables`, `get_table`, `find_join_path` and `list_glossary` tools; `sample_values` and `run_query`, the two that execute, stay local to the agent.

## Layout

```
src/schemagraph/
  model.py          Table / Column / Edge / BusinessTerm / SchemaSnapshot / LinkResult
  connectors/       ddl, duckdb, dbt, unity, glue, collibra
  graph/            build (merge), pathfinding (SchemaGraphSQL), ppr (HippoRAG), pruning (PathRAG)
  linking/          lexical (LinearRAG activation), linker (pipeline), render (DDL)
  llm/anchors.py    optional Claude anchor-table pass
  store.py          DuckDB persistence   engine.py  the one object everything drives
  mcp/              MCP server over a SchemaSource (Engine, LinkerSource, ScopedSource); stdio or streamable HTTP
  agent/            optional: read-only execution, checks, Qwen/Jev agents, TreeQuest search (`ask`)
  api/  cli.py      FastAPI (with MCP at /mcp) and Typer
  bench/            Spider 2.0 / Spider benchmarks, execution accuracy and the judge study
web/                Vite + React UI
tests/              no network (catalog connectors run against canned payloads, agents against scripted models)
```

## Benchmark

```bash
uv run schemagraph bench-spider2-lite /path/to/Spider2   # gold-table recall on 547 Spider 2.0-Lite tasks, no LLM
```

2026-09-03, defaults: **95.3% strict table recall, 97.8% recall, 11.8 tables returned, ~17 ms/question**; 88.0% strict on schemas with 100+ tables. Details, ablations and the iteration log in [`bench_results/README.md`](bench_results/README.md).

## Status

v1. No writes and no governance by design — the agent calling `link_schema` owns that boundary; SQL runs only in the optional `ask` loop, read-only. Catalog connectors are written against the documented Unity Catalog 2.1, Glue and Collibra 2.0 APIs and tested with recorded payloads; run **check** on a real connection before trusting a build.
