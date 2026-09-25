# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

schemagraph is a schema context engine for text-to-SQL: catalog metadata in (pasted DDL, DuckDB, dbt, Unity Catalog, AWS Glue, Collibra), one provenance-tagged NetworkX graph in the middle, and a linked, join-complete sub-schema out as annotated DDL. Served over HTTP, MCP, a CLI and a small React UI. The linking core is deterministic graph search (no LLM by default, no embeddings in the benchmark defaults); the optional `embed` extra adds paraphrase seeds from a small static embedding model (the Engine turns it on when installed), and a single optional Claude call can pick anchor tables. **Execution exists only in the optional `agent` extra** (`src/schemagraph/agent/`, issue #6): `schemagraph ask` / `Engine.answer` write SQL with a Qwen generator, run it **read-only** (guard + engine statement check + hardened connection, bounded by timeout and row caps), score it with deterministic checks and a TypeSafe Jev judge, and search with TreeQuest AB-MCTS. The agents read the schema through schemagraph's own MCP server over streamable HTTP, like any other client. The linker, `LinkOptions` and the MCP/HTTP tools do not execute; only the agent-local `sample_values` / `run_query` tools and the orchestrator do. **No writes and no governance** (row-level security, PII, budgets) — the caller owns that; don't widen execution beyond single read-only queries without asking.

`docs/DESIGN.md` explains the research behind each stage (SchemaGraphSQL, HippoRAG, PathRAG, LinearRAG, SignalPilot); `docs/ENTITIES.md` traces how schema objects become lexical entities and query seeds (there is no NER or embedding step); `bench_results/README.md` holds the benchmark numbers and the iteration log. Read those before changing the linker.

## Commands

```bash
uv sync --all-extras                       # Python 3.11+, installs anthropic + boto3 + model2vec (embed) + bm25s + agent (pydantic-ai, treequest) extras and dev deps
uv run pytest -q                           # whole suite, ~40 s on the Windows mount (agent tests ~20 s of it), no network
uv run pytest -q tests/test_linking.py     # one file
uv run pytest -q tests/test_linking.py -k bypass   # one test by keyword
uv run pytest -q tests/test_golden.py      # byte-exact regression oracle (see "Golden files" below)
uv run ruff check --fix src tests          # lint (ruff is the only linter; docstrings, complexity and line length 100 enforced in src)

uv run schemagraph add-ddl examples/store.sql --dialect postgres --schema public
uv run schemagraph link "revenue by product category for customers in california"
uv run schemagraph serve                   # API + UI on http://127.0.0.1:8765 (UI only if web/dist exists), MCP at /mcp
uv run schemagraph mcp                     # stdio MCP server
uv run schemagraph mcp --transport http    # streamable-HTTP MCP server on http://127.0.0.1:8766/mcp (--host, --port)
uv run schemagraph explain "<question>"    # activation seeds + PPR scores, for debugging the linker
uv run schemagraph ask "<question>" -c <duckdb connection> --strategy abmcts --budget 16   # agent extra + DASHSCOPE_API_KEY + TYPESAFE_API_KEY; serves its own MCP for the call
uv run schemagraph ask "<question>" -c <connection> --mcp-url http://127.0.0.1:8765/mcp     # read the schema from a running server instead (not scoped to -c)

cd web && npm install && npm run build     # builds web/dist; `npm run build` also runs tsc --noEmit
cd web && npm run dev                      # Vite dev server on :5173 proxying /api to :8765
```

Benchmark (needs a local clone of `xlang-ai/Spider2`; no credentials, no LLM):

```bash
uv run schemagraph bench-spider2-lite /path/to/Spider2                   # all 547 Lite tasks, ~9 min, ~3.4 GB peak; run configs one at a time
uv run schemagraph bench-spider2-lite /path/to/Spider2 --suite snow      # the 547 Snow tasks (same databases on Snowflake; output spider2_snow_*)
uv run schemagraph bench-spider2-lite /path/to/Spider2 --min-db-tables 100 --tag large   # large-schema subset
uv run schemagraph bench-spider2-lite /path/to/Spider2 --opt idf=false   # any LinkOptions field via --opt key=value
uv run schemagraph bench-spider1 <tables.json> <questions.json> --max-tables 6 --anchor-k 3   # Spider-format FK graphs (LinkAlign's Spider dev copy); measures paths/pruning
bench_results/run_sweep.sh /path/to/Spider2   # full sweep + ablations; resumable (skips tags with existing results)
uv run schemagraph bench-spider2-exec /path/to/Spider2 --judge-only        # judge study first: AUROC of each judge vs execution match on one candidate pool
uv run schemagraph bench-spider2-exec /path/to/Spider2 --strategy abmcts --budget 16   # EX on the 135 local SQLite tasks (needs local_sqlite.zip unpacked into spider2-lite/resource/databases/spider2-localdb/); resumable
```

State lives in `.schemagraph/schemagraph.duckdb` (override with `--home` or `SCHEMAGRAPH_HOME`). Delete the directory to reset.

Environment notes: this repo usually lives on a Windows mount under WSL, so Python imports take ~10 s and the first `link()` call pays a ~1 s SciPy import; a warm link is milliseconds. `zsh` is the shell — avoid `|` inside `${var%%|*}` parameter expansions and don't `pkill -f` a pattern that appears in your own command line.

## Architecture

### Data flow

```
connectors/*  ──SchemaSnapshot──►  store.py (DuckDB)  ──►  graph/build.py (NetworkX)  ──►  linking/linker.py  ──►  DDL
                                        ▲                                                        ▲
                             engine.py owns all three; api/ and cli.py talk to Engine, mcp/ to a SchemaSource (Engine by default)

agent/ (optional extra), per Engine.answer call:
  serve_http_async(create_server(engine, connection=...)) ─► http://127.0.0.1:<ephemeral>/mcp   (or AgentConfig.mcp_url / ask --mcp-url)
  Answerer ─ SchemaClient (fastmcp) ─────────────┐
  generator ─ MCPToolset (5 read-only tools) ────┴─► MCP over streamable HTTP ─► link / get_table / find_join_path
  generator ─ local sample_values, run_query; Answerer ─ candidates ─► guard.py + execute.py (read-only DuckDB/SQLite)
```

* **`model.py`** is the one vocabulary: `Table`/`Column`/`Edge`/`BusinessTerm` bundled per source in a `SchemaSnapshot`; `LinkResult` is what the linker returns. Every object carries `source` so provenance survives merging. `Table.schema_name` is aliased to `schema` for JSON (`by_alias=True` when dumping).
* **Connectors** (`connectors/`) implement `introspect() -> SchemaSnapshot` and `check() -> str`, have a pydantic `Config`, and register with `@register`. The registry drives `make_connector`, the API's `/api/connector-types`, and the UI form (rendered from the config's JSON schema). `connectors/__init__.py` imports every connector to register it — a new connector must be added there. `connectors/spider2.py` is benchmark-only and deliberately *not* registered.
* **`store.py`** persists connections, raw snapshot JSON, a user glossary and join hints in one DuckDB file. DuckDB is the durable form only; `Engine.reload()` rebuilds the in-memory graph from all snapshots plus a synthetic `user` snapshot at priority 0, merged **first** so human curation wins conflicting fields. Snapshots merge in `SOURCE_PRIORITY` order (user > collibra > dbt > unity_catalog > duckdb > ddl > aws_glue) unless the connection sets `priority`; relation edges and glossary targets are resolved after every snapshot has loaded its tables, so nothing depends on connection names. `${ENV_VAR}` references in configs are substituted at connector instantiation and never returned by the API.
* **`graph/build.py`** holds `SchemaGraph`, whose `graph` attribute is the NetworkX graph (from the Engine: `engine.graph` is the `SchemaGraph`, `engine.graph.graph` the NetworkX graph). It merges snapshots field-by-field (the highest-priority source to fill a field wins; columns/tags/samples union; a stub for a referenced-only table is replaced when the real table arrives). Node ids are prefixed: `t:` table, `c:` column, `k:` glossary term, `w:` lexical token. Two tables share one `relation` edge carrying *all* evidence (`relations: list[Edge]`) with `weight = min` over kinds (`RELATION_WEIGHT`: FK 1.0 < catalog relation 1.3 < lineage 1.6 < inferred 2.5). `table_graph()` projects to tables and join-capable relation kinds only for path-finding (`JOIN_KINDS`; lineage is provenance, rendered as `built from` / `feeds` context, never a join path). Tables referenced by an edge but never introspected become stub nodes.
* **`linking/linker.py`** is the pipeline; every stage is a `LinkOptions` field so it can be ablated from the benchmark CLI:
  1. `lexical.activate` — tokens, n-grams (with and without stopwords, `ngram_stop`), abbreviations, lemmas, glossary phrases, sample values (word n-gram lookup, not a regex per value) → seed weights with reason strings. Single-token evidence is IDF-scaled; n-gram/value/glossary hits are not. With `embed=True` (`linking/embed.py`, extra `embed`) the question's phrases also seed the closest objects by static-embedding cosine, which is what recovers paraphrases.
  2. `graph/ppr.personalized_pagerank` — HippoRAG-style PPR with node specificity, run as a power iteration on a cached sparse matrix (`PPRMatrix`, tol 1e-12); `table_scores(agg="top3")` folds columns into tables. `ppr_edge_attr` picks the edge attribute read as transition mass (`weight` = join cost, the measured default; `affinity` = uniform per kind).
  3. ranking — `ranker="rrf"` fuses the PPR table ranking with BM25F over one document per table (`linking/bm25.py`) by reciprocal rank; `ppr` and `bm25` alone are the ablations. `bm25_backend="bm25s"` (extra `bm25s`, `linking/bm25s_backend.py`) swaps BM25F for the bm25s library over one field-repeated document per table; an ablation, BM25F stays the default. Anchor and fill gates read per-ranker relative *evidence*, not the fused score.
  4. anchors — top-k tables, or `llm/anchors.py` (one Claude call) when `use_llm` and a key is set.
  5. `graph/pathfinding.union_of_shortest_paths` — all weighted-shortest simple paths between anchors via Yen's `shortest_simple_paths` (capped at 64 per pair); this is what pulls in bridge tables.
  6. `graph/pruning.prune_paths` — PathRAG flow pruning, sub-path dedup (inert on Spider2-Lite, which has almost no relation edges).
  7. budget: `adaptive_budget` widens to 20 tables / 6 anchors above `large_threshold`; `bypass_if_fits` returns the whole schema when it fits `max_tables` (anchors and paths are still computed for the DDL).
  8. column selection (rank-1 table: every column; others: keys + activated + best-scored up to `max_columns_per_table=40`) and `render.render_ddl`.
* **`graph/infer.py`** adds `inferred` edges from naming conventions for catalogs with no FKs. Opt-in per snapshot (`with_inferred_edges`); the benchmark turns it on, the engine does not yet.
* **`mcp/`**: `create_server(source, *, connection=None, host=)` builds the FastMCP server over a `SchemaSource` (`mcp/source.py`: `graph`, `link`, `tables`, `table`, `edges`, `terms`, `join_path`, `stats`). `Engine` satisfies it as is; `LinkerSource(linker, dialect=)` adapts a bare `Linker` (the benchmark has no Store); `ScopedSource(source, connection)` (what `connection=` wraps) keeps only tables whose comma-joined `Table.source` names that connection, and its `link` over-fetches 2× without bypass, drops other connections' tables, keeps join-path bridges, fills the budget in rank order and re-renders; its `terms` keeps a glossary term with a target in scope, listing only those targets. Both sources share `engine.find_join_paths`. Transports: stdio (`server.run()`), SSE (`mcp --transport sse`, as on main), `mcp/http.run_http` (`mcp --transport http`, alias `streamable-http`), the `/mcp` routes on `serve`'s FastAPI app, and `mcp/http.serve_http(server)`, a context manager that runs uvicorn in a daemon thread on `127.0.0.1:0` and yields the `/mcp` URL (`serve_http_async` is its `async with` twin, starting and stopping in a worker thread so the caller's loop never stalls; `Engine.answer_async` and the exec bench use it).
* **`agent/`** (optional `agent` extra; issue #6) answers questions: `Engine.answer` -> `answer.Answerer(SchemaClient(mcp_url), executor, cfg)` -> `search.run_search` (TreeQuest `ABMCTSA`, or the `single` / `best_of_n` / `refine` baselines on the same `generate(node_id, parent, action)`) -> per node: link through `schema_client.SchemaClient` (memoised `link_schema_json` with `include_ddl`; action `tight` = 7 tables with `adaptive_budget=False`, `wide` = 20) -> Qwen generator (`agents.py`: `MCPToolset` filtered to five schema tools, results capped at `TOOL_RESULT_CHARS`, plus local `sample_values` / `run_query`) -> `guard.py` + `execute.py` (read-only DuckDB/SQLite, one `Executor` protocol) -> `checks.py` (pure and sync; joins read pre-fetched lookups) -> Jev rubric judge -> `score.combine` -> both-order Jev selector over the top-k. `guard`, `execute`, `checks`, `score`, `results` import only core deps; `agents`/`answer`/`search`/`schema_client` need the extra. `bench/spider2_exec.py` (one MCP server per database, via `LinkerSource` + `serve_http_async`), `bench/spider2_judge.py` (the `--judge-only` study) and `bench/spider2_eval.py` (plain-Python port of the official comparator) measure EX.
* **Surfaces**: `api/app.py` (FastAPI, serves `web/dist` at `/` and the MCP server at `/mcp`), `mcp/server.py` (FastMCP, read-only tools), `cli.py` (Typer). The surfaces construct an `Engine` and nothing else; only the benchmark builds an MCP server over a `LinkerSource`.

### Things that are easy to get wrong

* `Linker(schema_graph)` builds the lexical index and adds `w:` token nodes to the graph in place, then caches the PPR matrix on first use; build the graph, then the linker, never reuse a graph across differently-configured indexes and never mutate the graph after the linker exists.
* Edge attribute `weight` is a *join cost* on `relation` edges (FK 1.0 < inferred 2.5) and a transition affinity everywhere else; PPR reads it as affinity on purpose (measured), the separate `affinity` attribute is the semantically clean alternative.
* Fused (`rrf`) table scores are rank-based and flat; anything that needs a magnitude (anchor ratio, fill floor) must use the `evidence` that `_rank` computes (`_Ranking.evidence`), not the score.
* `Engine` holds an `RLock`; `link()` runs under it because `reload()` swaps the graph. Long-running work inside connectors should not hold it.
* `LinkResult.ranking` is populated only with `LinkOptions(debug=True)`; the benchmark relies on it for gold-rank metrics.
* Partition families: `connectors/spider2.py` collapses tables whose names differ only in digit runs and share ≥ 80 % of columns into one table with `properties["members"]`. The benchmark canonicalises gold names through that member map (`bench/spider2_lite.canon`). The same treatment is *not* yet in the Unity/Glue connectors.
* Test fixture `tests/fixtures/dbt_airbnb` is a Spider 2.0-DBT project copied without `dbt_packages`; `dim_listings_hosts` is declared in YAML but has no SQL on purpose (that's the benchmark task).
* `web/dist` is gitignored; `schemagraph serve` silently runs API-only until you build the UI.
* Agent layer: the Engine lock is taken only inside `Engine.link`, which the async MCP tools call in a worker thread (`anyio.to_thread`) so a waiting link never blocks the event loop `serve` shares with the API; never hold it across an `await`. Executors are synchronous and run via `asyncio.to_thread`.
* A FastMCP server can be served only once: its streamable-HTTP session manager's `run()` refuses a second start. Build a new `create_server(...)` per `serve_http` block (one per answer, one per benchmark database). `serve` mounts the routes of `streamable_http_app()` on FastAPI, and a mounted app's lifespan never runs, so `create_app`'s lifespan starts `session_manager.run()` itself; the routes are plain Starlette routes, so the OpenAPI golden does not see them.
* DNS-rebinding protection: on a loopback `host` FastMCP accepts only loopback `Host` / `Origin` headers (421 / 403 otherwise), and on any other host it is off. Pass the host you actually bind to `create_server` / `create_app` (the CLI does); a proxy that rewrites `Host` needs its own `transport_security`.
* MCP never executes SQL. `sample_values` and `run_query` are agent-local tools on the executor; don't move them into `mcp/`. An external `--mcp-url` server (e.g. `serve`'s `/mcp`) is unscoped: `--connection` then only picks the database to execute on, not the tables the agents see.
* Static checks see executor catalog keys (bare `orders` on SQLite, `schema.table` on DuckDB), not graph FQNs. `Answerer._join_lookups` resolves each key through `SchemaClient.resolve` before pre-fetching relations and join paths, and the lookup dicts are keyed by catalog key; skip the resolution and the join finding silently never fires.
* `AgentConfig.mcp_url` is excluded from the exec bench's `config_hash` (`_run_config`; `concurrency` too, in `config_hash`): the bench always serves each database itself, and rows written before the field existed must still resume. Keep new transport-only fields out of the hash the same way.
* Read-only needs **all three** layers: DuckDB `read_only` alone still reads files (`read_csv('/etc/passwd')`), SQLite `mode=ro` alone still lets `ATTACH` create a file. The SQLite authorizer also blocks `pragma_table_info`, so catalog introspection uses a separate trusted connection with fixed SQL.
* Jev cannot produce text: rubric and pick fields are `float = Field(ge=0, le=1)` (the value *is* the probability; `bool` fields expose only a threshold margin), refinement feedback comes from the lazy Qwen critic, and the judge has no output validators (Jev cannot revise). Qwen on DashScope needs `openai_supports_tool_choice_required=False` (`agent/models.py`).
* Tests never call a model: `ALLOW_MODEL_REQUESTS=False` in conftest; use `FunctionModel`/`TestModel` and pass `AgentModels` (or monkeypatch `agent.models.resolve_model`). Agent tests `importorskip` pydantic_ai/treequest so the suite passes without the extra.
* The exec bench keeps families off (the generator must see real table names) and resolves `Db-IMDB` -> `DB_IMDB` schema folders; the Lite bench keeps its exact folder lookup so its numbers cannot move. The official comparator's quirks (pandas CSV round trip, frame upcast, str-sorted `ignore_order`, flat `condition_cols` with one `_a` gold) are reproduced on purpose; 8 of the 24 public local gold SQL files score 0 against their own gold CSVs under the official evaluator too.

### Golden files

`tests/test_golden.py` pins, byte for byte, every connector's snapshot, the merged graphs, the lexical index, `LinkResult` + DDL for a set of questions under 16 `LinkOptions` variants, the LLM request, and the user/agent-visible surfaces (OpenAPI, MCP tool descriptions, CLI options, connector config schemas) in `tests/golden/*.json`. A refactor must leave them unchanged. Regenerate (`SCHEMAGRAPH_UPDATE_GOLDEN=1 uv run pytest -q tests/test_golden.py`) only for an intentional, benchmarked behaviour change, and review the golden diff as part of the change.

### Benchmark defaults are evidence, not taste

`LinkOptions` defaults (`max_tables=20`, `anchor_k=6`, `idf=True`, `agg="top3"`, `adaptive_budget`, `bypass_if_fits`, `schema_routing=0.0`, `ranker="rrf"`, `rrf_k=60`, `ngram_stop=True`, `ppr_edge_attr="weight"`, `columns_top_uncapped=1`, `max_columns_per_table=40`) were each set by an ablation on Spider 2.0-Lite (see `bench_results/README.md`). Schema routing was measured and rejected (−6 strict points); uniform PPR affinity and BM25-only ranking were measured and rejected too; don't re-enable them without new evidence. When you change anything in `linking/` or `graph/`, rerun `bench-spider2-lite` before and after and quote strict recall **and** strict@7 on the precise sample (ranking quality; strict at 20 tables saturates) in the change description. A full Lite run takes ~2.6 min from a copy of the Spider2 clone on the Linux filesystem and ~9 min from the Windows mount.

## Code style

Ruff enforces Google-style docstrings (`D`, except `D105`/`D107`), complexity (`C901` at 12) and line length 100 on `src/`. `graph/build.py` is the reference module.

* **Docstrings**: every module, public class and public function; private helpers too when the name doesn't say everything, when they mutate an argument, or when they are longer than ~10 lines. One-line imperative summary. Add `Args:`/`Returns:`/`Raises:` when there are ≥ 3 parameters, a parameter with units or a tuning meaning, a non-obvious return shape, a mutation, or an intentional raise. Dataclasses document fields in `Attributes:` rather than trailing comments.
* **User- and agent-visible text is not free**: a pydantic model's docstring becomes its JSON-schema/OpenAPI description, a FastAPI handler's docstring its operation description, an MCP tool's docstring what agents read, a Typer command's docstring its `--help`. Don't add `Field(description=...)` to connector configs (the UI masks secrets by description).
* **Names**: descriptive names (`table`, `column`, `edge`, `relation`, `node`, `schema_graph`, `activation`). Accepted abbreviations: `fqn`, `snap`, `cfg`, `opts`, `db`, `pk`, `fk`, `ddl`, `sql`, `uid`, `idf`, `ppr`, `rrf`, and conventional math names (`u`/`v`, `k1`, `b`, `tf`, `df`, `alpha`). Single letters only inside a one-line comprehension.
* **Structure**: split long functions into named helpers; tuned numbers become module-level `UPPER_SNAKE` constants with a comment. Tuned numbers and policy constants are public; implementation details (SQL text, regexes, markers and lookup sets used only inside the module, CSV headers) are `_PRIVATE`. Don't give two constants the same name with different meanings; import a shared value instead of copying it. When moving linker or graph code, keep insertion order, float operand grouping and sort tie-breaks exactly as they were, since they feed PPR and the rankings.
