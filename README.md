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
uv run schemagraph mcp        # stdio MCP server: link_schema, get_table, find_join_path, ...
```

Or `POST /api/link {"question": "..."}` and paste the returned `ddl` into your prompt.

## Layout

```
src/schemagraph/
  model.py          Table / Column / Edge / BusinessTerm / SchemaSnapshot / LinkResult
  connectors/       ddl, duckdb, dbt, unity, glue, collibra
  graph/            build (merge), pathfinding (SchemaGraphSQL), ppr (HippoRAG), pruning (PathRAG)
  linking/          lexical (LinearRAG activation), linker (pipeline), render (DDL)
  llm/anchors.py    optional Claude anchor-table pass
  store.py          DuckDB persistence   engine.py  the one object everything drives
  api/  mcp/  cli.py
web/                Vite + React UI
tests/              33 tests, no network (catalog connectors run against canned payloads)
```

## Benchmark

```bash
uv run schemagraph bench-spider2-lite /path/to/Spider2   # gold-table recall on 547 Spider 2.0-Lite tasks, no LLM
```

2026-09-03, defaults: **95.3% strict table recall, 97.8% recall, 11.8 tables returned, ~17 ms/question**; 88.0% strict on schemas with 100+ tables. Details, ablations and the iteration log in [`bench_results/README.md`](bench_results/README.md).

## Status

v1. No SQL execution and no governance by design — the agent calling `link_schema` owns that boundary. Catalog connectors are written against the documented Unity Catalog 2.1, Glue and Collibra 2.0 APIs and tested with recorded payloads; run **check** on a real connection before trusting a build.
