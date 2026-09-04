# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

schemagraph is a schema context engine for text-to-SQL: catalog metadata in (pasted DDL, DuckDB, dbt, Unity Catalog, AWS Glue, Collibra), one provenance-tagged NetworkX graph in the middle, and a linked, join-complete sub-schema out as annotated DDL. Served over HTTP, MCP, a CLI and a small React UI. The linking core is deterministic graph search (no embeddings, no LLM by default); a single optional Claude call can pick anchor tables. **By design there is no SQL execution and no governance** — the agent that calls `link_schema` owns that boundary. Don't add either without asking.

`docs/DESIGN.md` explains the research behind each stage (SchemaGraphSQL, HippoRAG, PathRAG, LinearRAG, SignalPilot); `docs/ENTITIES.md` traces how schema objects become lexical entities and query seeds (there is no NER or embedding step); `bench_results/README.md` holds the benchmark numbers and the iteration log. Read those before changing the linker.

## Commands

```bash
uv sync --all-extras                       # Python 3.11+, installs anthropic + boto3 extras and dev deps
uv run pytest -q                           # whole suite, ~6 s, no network
uv run pytest -q tests/test_linking.py     # one file
uv run pytest -q tests/test_linking.py -k bypass   # one test by keyword
uv run ruff check --fix src tests          # lint (ruff is the only linter; line length 100, E501 ignored)

uv run schemagraph add-ddl examples/store.sql --dialect postgres --schema public
uv run schemagraph link "revenue by product category for customers in california"
uv run schemagraph serve                   # API + UI on http://127.0.0.1:8765 (UI only if web/dist exists)
uv run schemagraph mcp                     # stdio MCP server
uv run schemagraph explain "<question>"    # activation seeds + PPR scores, for debugging the linker

cd web && npm install && npm run build     # builds web/dist; `npm run build` also runs tsc --noEmit
cd web && npm run dev                      # Vite dev server on :5173 proxying /api to :8765
```

Benchmark (needs a local clone of `xlang-ai/Spider2`; no credentials, no LLM):

```bash
uv run schemagraph bench-spider2-lite /path/to/Spider2                   # all 547 Lite tasks, ~80 s
uv run schemagraph bench-spider2-lite /path/to/Spider2 --min-db-tables 100 --tag large   # large-schema subset
uv run schemagraph bench-spider2-lite /path/to/Spider2 --opt idf=false   # any LinkOptions field via --opt key=value
bench_results/run_sweep.sh /path/to/Spider2   # full sweep + ablations; resumable (skips tags with existing results)
```

State lives in `.schemagraph/schemagraph.duckdb` (override with `--home` or `SCHEMAGRAPH_HOME`). Delete the directory to reset.

Environment notes: this repo usually lives on a Windows mount under WSL, so Python imports take ~10 s and the first `link()` call pays a ~1 s SciPy import; a warm link is milliseconds. `zsh` is the shell — avoid `|` inside `${var%%|*}` parameter expansions and don't `pkill -f` a pattern that appears in your own command line.

## Architecture

### Data flow

```
connectors/*  ──SchemaSnapshot──►  store.py (DuckDB)  ──►  graph/build.py (NetworkX)  ──►  linking/linker.py  ──►  DDL
                                        ▲                                                        ▲
                             engine.py owns all three; api/, mcp/, cli.py only ever talk to Engine
```

* **`model.py`** is the one vocabulary: `Table`/`Column`/`Edge`/`BusinessTerm` bundled per source in a `SchemaSnapshot`; `LinkResult` is what the linker returns. Every object carries `source` so provenance survives merging. `Table.schema_name` is aliased to `schema` for JSON (`by_alias=True` when dumping).
* **Connectors** (`connectors/`) implement `introspect() -> SchemaSnapshot` and `check() -> str`, have a pydantic `Config`, and register with `@register`. The registry drives `make_connector`, the API's `/api/connector-types`, and the UI form (rendered from the config's JSON schema). `connectors/__init__.py` imports every connector to register it — a new connector must be added there. `connectors/spider2.py` is benchmark-only and deliberately *not* registered.
* **`store.py`** persists connections, raw snapshot JSON, a user glossary and join hints in one DuckDB file. DuckDB is the durable form only; `Engine.reload()` rebuilds the in-memory graph from all snapshots plus a synthetic `user` snapshot merged **last**, so human curation overrides catalogs. `${ENV_VAR}` references in configs are substituted at connector instantiation and never returned by the API.
* **`graph/build.py`** merges snapshots field-by-field (first source to fill a field wins; columns/tags/samples union). Node ids are prefixed: `t:` table, `c:` column, `k:` glossary term, `w:` lexical token. Two tables share one `relation` edge carrying *all* evidence (`relations: list[Edge]`) with `weight = min` over kinds (`RELATION_WEIGHT`: FK 1.0 < catalog relation 1.3 < lineage 1.6 < inferred 2.5). `table_graph()` projects to tables only for path-finding. Tables referenced by an edge but never introspected become stub nodes.
* **`linking/linker.py`** is the pipeline; every stage is a `LinkOptions` field so it can be ablated from the benchmark CLI:
  1. `lexical.activate` — tokens, n-grams, abbreviations, lemmas, glossary phrases, sample values → seed weights with reason strings. Single-token evidence is IDF-scaled; n-gram/value/glossary hits are not.
  2. `graph/ppr.personalized_pagerank` — HippoRAG-style PPR with node specificity; `table_scores(agg="top3")` folds columns into tables.
  3. anchors — top-k tables, or `llm/anchors.py` (one Claude call) when `use_llm` and a key is set.
  4. `graph/pathfinding.union_of_shortest_paths` — all weighted-shortest simple paths between anchors; this is what pulls in bridge tables.
  5. `graph/pruning.prune_paths` — PathRAG flow pruning, sub-path dedup.
  6. budget: `adaptive_budget` widens to 20 tables / 6 anchors above `large_threshold`; `bypass_if_fits` returns the whole schema when it fits `max_tables` (anchors and paths are still computed for the DDL).
  7. column selection (anchors: all columns; bridge tables: keys + activated) and `render.render_ddl`.
* **`graph/infer.py`** adds `inferred` edges from naming conventions for catalogs with no FKs. Opt-in per snapshot (`with_inferred_edges`); the benchmark turns it on, the engine does not yet.
* **Surfaces**: `api/app.py` (FastAPI, serves `web/dist` at `/`), `mcp/server.py` (FastMCP, read-only tools), `cli.py` (Typer). All construct an `Engine` and nothing else.

### Things that are easy to get wrong

* `Linker(sg)` builds the lexical index and adds `w:` token nodes to the graph in place; build the graph, then the linker, never reuse a graph across differently-configured indexes.
* `Engine` holds an `RLock`; `link()` runs under it because `reload()` swaps the graph. Long-running work inside connectors should not hold it.
* `LinkResult.ranking` is populated only with `LinkOptions(debug=True)`; the benchmark relies on it for gold-rank metrics.
* Partition families: `connectors/spider2.py` collapses tables whose names differ only in digit runs and share ≥ 80 % of columns into one table with `properties["members"]`. The benchmark canonicalises gold names through that member map (`bench/spider2_lite.canon`). The same treatment is *not* yet in the Unity/Glue connectors.
* Test fixture `tests/fixtures/dbt_airbnb` is a Spider 2.0-DBT project copied without `dbt_packages`; `dim_listings_hosts` is declared in YAML but has no SQL on purpose (that's the benchmark task).
* `web/dist` is gitignored; `schemagraph serve` silently runs API-only until you build the UI.

### Benchmark defaults are evidence, not taste

`LinkOptions` defaults (`max_tables=20`, `anchor_k=6`, `idf=True`, `agg="top3"`, `adaptive_budget`, `bypass_if_fits`, `schema_routing=0.0`) were each set by an ablation on Spider 2.0-Lite (see `bench_results/README.md`). Schema routing was measured and rejected (−6 strict points); don't re-enable it without new evidence. When you change anything in `linking/` or `graph/`, rerun `bench-spider2-lite` before and after and quote strict recall in the change description.
