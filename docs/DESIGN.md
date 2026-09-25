# schemagraph — design

**One sentence.** Catalog metadata in (DDL, DuckDB, dbt, Unity Catalog, AWS Glue, Collibra), a single provenance-tagged schema graph in the middle, and a linked, join-complete sub-schema out — as annotated DDL for an LLM, over HTTP and MCP, with a UI to feed and curate it.

It sits in the same slot as JetBrains' Databao Context Engine and SignalPilot's semantic layer, minus governance (that belongs to the agent that calls this); read-only SQL execution exists only in the optional agent layer (see *Answer loop*), and with the schema-linking core rebuilt around **graph traversal** instead of embedding search.

## Why graph traversal

The research behind this (see the Obsidian hub note *Semantic Layer for Text-to-SQL*) converges on a few facts:

* Schema-linking errors are the dominant failure at enterprise scale (LinkAlign: >60% of failures), and the specific failure pure retrieval cannot fix is the **missing bridge table** — semantically irrelevant, structurally mandatory.
* **SchemaGraphSQL** shows that one cheap anchor-table pick plus a deterministic *union of shortest paths* over the FK graph lands within ~1.5% of oracle linking on BIRD. "LLMs can ignore noise but cannot guess missing joins."
* On repository-level data work (Spider 2.0-DBT) scaffolding beats model strength by 20×; SignalPilot leads that board with **no embeddings at all**.
* Real warehouses rarely declare FKs, so the graph has to come from more than `information_schema`: dbt lineage, `relationships` tests, Unity constraints and lineage, Collibra curated relations, human join hints.

And from the GraphRAG-alternatives research:

* **HippoRAG**: personalized PageRank over a knowledge graph gives the best multi-hop retrieval at ~1k tokens/query — basically free. Node specificity (down-weight ubiquitous nodes) matters.
* **PathRAG**: graph retrieval fails from *redundancy*, not scarcity; flow-based pruning with a reliability score per path fixes it, and paths should be serialized most-reliable-last.
* **LinearRAG**: aligned *entities* are the anchors; relations can stay in the source and be read at inference time. Zero LLM tokens for indexing.
* **Hybrid retrieval**: sparse and graph rankers fail differently, and reciprocal-rank fusion keeps the head of one with the tail of the other. On Spider 2.0-Lite, BM25F over one document per table beats the PPR ranking by 8 points at strict@3 and loses 7 at gold-in-top-20; fusing the two beats both (`bench_results/README.md`, 2026-09-17).

schemagraph maps those one-to-one onto a schema:

| Concept | In a document corpus | In schemagraph |
|---|---|---|
| Entity | noun phrase | table, column, glossary term, sample value |
| Sentence/passage | text chunk | object name, description, tags |
| Mention matrix | entity ↔ sentence | `w:<token>` ↔ object (`mention` edges) |
| Relation | extracted triple | FK / lineage / test / catalog relation / hint, read verbatim at query time |
| PPR seed | query entities | activated objects, weighted by lexical evidence × specificity |
| Path | reasoning chain | join path between anchor tables |

## Pipeline

```
question
  │  lexical.activate        tokens, n-grams, abbreviations, lemmas, glossary phrases, sample values → seeds
  ▼
ppr.personalized_pagerank   HippoRAG PPR over {tables, columns, terms, tokens}; specificity-weighted seeds;
  │                         a cached sparse power iteration converged to 1e-12 (graph/ppr.PPRMatrix)
  │  ppr.table_scores        own + best + 0.5·second + 0.25·third column (IDF-weighted seeds), plus direct lexical bonus
  ▼
ranking                     reciprocal-rank fusion of the PPR table ranking with BM25F over one document per
  │                         table (linking/bm25.py): PPR carries the tail (join-implied tables with no shared
  │                         token), BM25F the head. ranker = rrf | ppr | bm25
  ▼
anchors                     top-k by fused rank (k = 6; evidence ≥ 15% of the best under either ranker), or an
  │                         optional Claude call that picks source/destination tables among the candidates
  ▼
pathfinding.union_of_shortest_paths
  │                         all weighted-shortest simple paths between anchor pairs (Yen's k-shortest paths);
  │                         FK = 1.0, catalog relation = 1.3, inferred = 2.5 (lineage is context, not a join path)
  ▼
pruning.prune_paths         PathRAG flow propagation (α = 0.8, θ = 0.05), reliability = mean resource,
  │                         drop sub-paths of kept paths, keep top-k
  ▼
column selection            top-ranked table: every column; other tables: PK + join keys + activated
  │                         columns + best-scored rest up to 40 (table rank, not column evidence,
  │                         predicts gold columns: bench_results/README.md, column level)
  ▼
render.render_ddl           CREATE TABLE … with descriptions, glossary header, join paths (most reliable
                            last), sample values footer
```

Warm-path latency on a 7-table schema is ~3 ms; the first call pays a one-off SciPy import. How the entities behind the first two stages are formed, field by field and source by source, is in [`ENTITIES.md`](ENTITIES.md).

## Graph model (`graph/build.py`)

Undirected NetworkX graph. Node ids are prefixed: `t:` table, `c:` column, `k:` glossary term, `w:` token. Edge attribute `etype` ∈ {`contains`, `relation`, `fk_col`, `glossary`, `mention`}. A `relation` edge between two tables carries `relations: list[Edge]` — every piece of evidence from every source, deduplicated by (kind, tables, columns) — and `weight = min` over the kinds. `table_graph()` projects to tables + `relation` edges of join-capable kinds (`JOIN_KINDS`: FK, relationship test, join hint, catalog relation, inferred) for path-finding; `lineage` edges stay in the full graph for PPR and are rendered as context (`built from`, `feeds`), because two models fed by the same source are related but not joinable through it.

Merging is field-level and in an explicit order: snapshots are sorted by priority (`SOURCE_PRIORITY` by source type: user, collibra, dbt, unity_catalog, duckdb, ddl, aws_glue; or a per-connection `priority`), the first source to fill a field wins, later sources fill blanks and union columns/tags/samples. The **user snapshot** (glossary + join hints from the UI) has priority 0, so human curation wins where it conflicts. Relation edges and glossary targets are applied after every snapshot has contributed its tables, so a foreign key or term pointing at a table another source introspects resolves whatever the load order (until 2026-09-17 snapshots loaded alphabetically by connection name and unresolved targets were dropped).

Referenced-but-unknown tables become stub nodes (`properties.stub = "true"`) so paths through them still exist.

## Connectors (`connectors/`)

| type | edges produced | notes |
|---|---|---|
| `ddl` | `foreign_key` | sqlglot; inline `REFERENCES`, table-level FK, `ALTER TABLE ADD FOREIGN KEY`, `COMMENT` / `COMMENT ON` |
| `duckdb` | `foreign_key` | `duckdb_tables/columns/constraints`, row counts, sample values |
| `dbt` | `lineage`, `relationship_test`, `foreign_key` (model constraints) | manifest.json, or project dir parsed without dbt (regex over `ref()`/`source()`); `unique`+`not_null` → PK, `accepted_values` → samples, semantic models → glossary. Lineage feeds PPR and the DDL notes (`built from` / `feeds`) but is not a join path; joins come from `relationships` tests, constraints and hints |
| `unity_catalog` | `foreign_key`, `lineage` | REST 2.1 catalogs/schemas/tables + `table_constraints`; optional lineage-tracking API |
| `aws_glue` | none | boto3; columns, partition keys, parameters, optional LF-tags. Relations come from other sources or hints |
| `collibra` | `foreign_key`, `catalog_relation`, glossary | REST 2.0 assets/relations/attributes; asset types, relation roles and attribute names are all configurable |

Connector configs are pydantic models; the UI renders a form from their JSON schema. Secrets may be `${ENV_VAR}` references, substituted at run time and never returned by the API.

## Persistence (`store.py`)

One DuckDB file: `connections`, `snapshots` (full JSON per source), `glossary`, `join_hints`. The graph is rebuilt in memory on load — DuckDB is the durable form, NetworkX the working one.

## Surfaces

* **HTTP** (`api/app.py`): `/api/connections`, `/api/ddl`, `/api/build`, `/api/graph/*`, `/api/link`, `/api/explain`, `/api/glossary`, `/api/join-hints`; serves the MCP server at `/mcp` (streamable HTTP, same Engine) and `web/dist` at `/`.
* **MCP** (`mcp/server.py`): `link_schema`, `link_schema_json`, `search_tables`, `get_table`, `find_join_path`, `list_glossary`, `graph_stats`. Read-only; no tool executes SQL. `link_schema` and `link_schema_json` take `adaptive_budget` (false keeps a small `max_tables` on a large schema), and `link_schema_json` takes `include_ddl` so one call returns the tables and the DDL. The server reads a `SchemaSource` (`mcp/source.py`): the Engine, a bare `Linker` through `LinkerSource` (the benchmark, which has no Store), or either restricted to one connection through `ScopedSource`. It runs over stdio or streamable HTTP (`mcp --transport http`, the `/mcp` mount on `serve`, or `mcp/http.serve_http`, which serves it on an ephemeral localhost port for the length of a `with` block).
* **CLI** (`cli.py`): `add-ddl`, `add-dbt`, `add-duckdb`, `add <type> <name> -c '{json}'`, `build`, `link`, `explain`, `path`, `serve`, `mcp [--transport stdio|http|sse]`, and with the `agent` extra `ask [--mcp-url URL]` and `bench-spider2-exec`.
* **Web** (`web/`): Connections, Paste DDL, Link playground, Graph (cytoscape), Glossary & hints.

## LLM use

Optional and small: one call per question (`llm/anchors.py`) that returns source/destination tables via a JSON-schema output format, using `claude-opus-5` by default with `effort: low` and server-side refusal fallbacks enabled. Enabled only when `ANTHROPIC_API_KEY` is set and the caller passes `use_llm=true`. Everything else is deterministic and free.

## Answer loop (optional `agent` extra, issue #6)

`schemagraph ask` / `Engine.answer` go past context: write SQL, run it read-only, score it, search.

* **The schema comes over MCP.** The agents are MCP clients of schemagraph itself, so they see exactly what any other agent sees. The generator's schema tools are the server's `link_schema`, `search_tables`, `get_table`, `find_join_path` and `list_glossary` (a pydantic-ai `MCPToolset` filtered to those five, each result capped at 12,000 characters); the orchestrator reads links, tables, relations, neighbours and join paths through `SchemaClient`, one memoised fastmcp session (linking is deterministic). `Engine.answer` serves the engine, scoped to the connection, on an ephemeral `127.0.0.1` port for the call, unless `AgentConfig.mcp_url` (`ask --mcp-url`) names a running server; an external server such as `serve`'s `/mcp` is not scoped. Execution is the one thing MCP does not provide: `sample_values` and the bounded `run_query` probe are local tools over the read-only executor, so the server stays execution-free.

* **Why AB-MCTS.** TreeQuest's `ABMCTSA` (Sakana, arXiv:2503.04412) uses Thompson sampling per node to choose between a new child ("go wider": a fresh draft at the root) and expanding an existing one ("go deeper": a refinement fed with the parent's feedback), and between actions. With `generate(parent)`, `parent=None` is a best-of-N draft and `parent=<candidate>` a Refine step, so BestOfN (breadth only: independent drafts, alternating the tight and wide context actions) and a DSPy-Refine-style chain (depth only) are special cases; `search.py` implements all three on the same generate/score functions and the benchmark compares them at equal budget. DSPy itself is not a dependency.
* **Actions are context widths.** `tight` links 7 tables with the adaptive budget off; `wide` links 20. The tree searches over schema context as well as SQL; the generator can widen or narrow it further with its MCP tools and probe the data with `sample_values` and `run_query`.
* **Every published text-to-SQL tree search scores with execution** (Alpha-SQL, CHASE-SQL, ReFoRCE); an LLM-only judge gets optimised for its own biases. So execution is the signal, read-only in three layers: a sqlglot guard (one query, no DDL/DML/ATTACH/PRAGMA/SET/COPY/INSTALL/LOAD/INTO, no file table functions or file-like names), the engine's own statement check, and a hardened connection (DuckDB `read_only` + `enable_external_access=false` + `lock_configuration`; SQLite `mode=ro` + an allow-list authorizer + no ATTACH), each bounded by a timeout and row caps.
* **Score** in [0, 1]: 0 for a guard reject, 0.05 for an execution error, else `0.15 + 0.85 * (0.4 * det + 0.6 * judge)`, where `det` = 1 − penalties of deterministic findings (unknown table/column with closest-name suggestions, cartesian product, ungrouped column, empty result, all-NULL columns, row explosion; joins off the relation graph are reported, not penalised, until measured) and `judge` the mean of six rubric probabilities. Findings and judge doubts become the refine prompt's feedback.
* **Judge = TypeSafe Jev**, a classifier that answers typed questions with a probability each (not an LLM): one plain question per `float` field in [0, 1], the prompt is only the material (question, SQL, result columns and ≤ 10 preview rows, no DDL), and `missing` is a `list[Literal[...]]` built from the linked and neighbouring tables the query skips. Jev is weak at arithmetic, dates and multi-hop questions, so those stay in the deterministic checks; its confidences are not calibrated out of the box (per-field logistic calibration against execution match is the follow-up). Jev cannot write text, so refinement advice comes from a lazy Qwen critic, called once per expanded parent. The final pick is a round-robin pairwise Jev selector over the top-k results (deduplicated by result), asked in both orders to cancel position bias (CHASE-SQL).
* **What the model sees.** Tool results and previews come from the database, so a query like `duckdb_databases()` or `pragma_database_list` shows the database's own path and settings to the generator (a remote API). Nothing outside the database is readable. Queries run against the whole database file: a DuckDB connection's `schemas` setting limits what the agent is shown (catalog, suggestions, checks), not what a query can read. Data in preview rows can also carry prompt injection into the judge; previews are cut to 10 rows of 60 characters and the judge has no tools.
* **Judge first.** `bench-spider2-exec --judge-only` scores one fixed candidate pool with each judge (Jev and the same rubric on Qwen) and reports AUROC against execution match before any search run.

## Non-goals (v1)

* No writes and no governance (row-level security, PII redaction, budgets): the calling agent owns that boundary. SQL execution is confined to the optional agent layer and is read-only and bounded; the linker, MCP and HTTP tools never execute.
* No embeddings in the core. The lexical + PPR path is the LinearRAG bet; the optional `embed` extra (`linking/embed.py`, a 30 MB static model from the Hugging Face Hub, no torch; set `HF_HUB_OFFLINE=1` on air-gapped hosts once it is cached) adds paraphrase seeds behind the same `Activation` interface and is measured at +0.4 strict / +3 strict@7 on Spider 2.0-Lite.
* No BI metrics layer. Glossary terms map words to columns; they are not governed calculations.

## Benchmark

`schemagraph bench-spider2-lite <Spider2 clone>` scores gold-table recall of `link_schema` on the 547 Spider 2.0-Lite tasks with no execution, credentials, or LLM. Results, ablations, the iteration log and error analysis: [`bench_results/README.md`](../bench_results/README.md). Headline (2026-09-17): **95.9% strict table recall, 98.3% recall, 11.9 tables returned, ~12 ms per question** on Lite and **96.7% strict on Spider 2.0-Snow** (`--suite snow`); 91.6% strict on schemas with ≥ 100 tables (was 18.9% before the large-schema loop). Strict recall at 20 tables saturates, so ranking changes are judged on strict@7 over the precise sample (70.6, from 65.9 with PPR alone).

What the benchmark forced into the core, each ablated:

* `linking/lexical.py` — **IDF weighting** of token evidence (a token shared by half the columns is nearly worthless as a seed), a stopword list of question scaffolding that also appears in descriptions, and no seeds from digit tokens shorter than 4 characters.
* `graph/ppr.py` — **top-3 column aggregation** into table scores instead of a sum, so a wide table with forty weak matches no longer swamps a table with one strong name match. Since 2026-09-17 the walk is a cached sparse power iteration converged to 1e-12; `nx.pagerank` on a subgraph view stopped at `N × 1e-6`, which on 7,000-node schemas left an L1 error near 1e-2 and reordered near-tied tables.
* `linking/linker.py` — **rank-tiered column cap**: the best-ranked table keeps every column, the rest 40. Column strict recall 81 → 91 on Lite at the same column count; a simulation showed column-level evidence (seeds, PPR score) does not predict gold columns while table rank does.
* `bench/spider1.py` — **FK-graph benchmark** (Spider-format `tables.json` + questions with gold SQL; the LinkAlign copy of Spider dev, 517 multi-join questions). Path union recovers bridge tables (+0.2 to +3.5 strict by budget); PathRAG pruning is inert there as on Lite, so it is kept only to order the DDL's path list.
* `linking/bm25.py` — **BM25F per table fused with PPR by reciprocal rank** (`ranker="rrf"`): +3.7 strict@7 on the precise sample and +3.6 anchor hit over PPR alone, with BM25F alone 0.75 strict below PPR because three gold tables share no token with their question. Anchor and fill gates read per-ranker relative evidence, because fused scores are flat.
* `linking/bm25s_backend.py` — **the bm25s library as an optional sparse backend** (`bm25_backend="bm25s"`, extra `bm25s`; issue #10, 2026-09-25), measured because it would mean less custom code. bm25s scores one flat document with plain BM25 variants. It has no fields, field weights or weighted query terms, so the adapter repeats name, column and business tokens 3x and scores query expansions in a second weighted call. Under `rrf`, `lucene` ties the default on strict recall (95.85) and is one task behind at strict@7 (70.3 vs 70.6). On its own (`ranker=bm25`) it is 1.9 @7 below BM25F, because one length normalisation over the whole document dilutes a name hit on a wide table. BM25F stays the default and the library stays an ablation.
* `graph/infer.py` — name-based `inferred` edges (`x_id` ↔ `x.id`, shared key columns) for catalogs with no declared FKs.
* `connectors/spider2.py` — **partition-family collapse**: tables in one schema whose names differ only in digit runs and share ≥ 80% of columns become one logical table with a member list. The same treatment belongs in the Unity/Glue connectors for partitioned datasets.
* `linking/linker.py` — **adaptive budget** (schemas over 30 tables get 20 tables and 6 anchors) and **bypass-if-fits** (a schema that fits the budget is returned whole, with anchors and join paths still computed) — the "Death of Schema Linking" result, operationalised. Default budget is 20 tables.
* Rejected: DBCopilot-style **schema routing** (boosting the dataset with the most activation mass) — Spider 2 gold sets routinely join across datasets and it cost 6–7 strict points. Kept as an option (`schema_routing`), off.

`schemagraph bench-spider2-exec <Spider2 clone>` (agent extra, model keys, the local SQLite databases) measures execution accuracy of the answer loop on the 135 `local*` tasks, serving each database's linker on its own in-process MCP server (`LinkerSource` + `serve_http`) while that database's tasks run, with a plain-Python port of the official comparison (`bench/spider2_eval.py`, parity-tested against the original on real gold SQL and random tables): EX of the final pick, of the top-score candidate and of any candidate (oracle), per strategy at equal budget.

## Next steps

1. The last 4%: gold tables at rank 21–80 in mid-size schemas, vocabulary gaps and join-implied tables. The seed-side embedding pass (`embed` extra) recovers two of them and lifts strict@7 by 3; an LLM entity pass before PPR (HippoRAG's placement) is the next step for the rest, and the Spider-dev FK benchmark shows join-implied bridge tables still rank 6th–8th of 10 even with declared keys.
2. Shard-family collapse in the Unity Catalog and Glue connectors; value grounding on `_TABLE_SUFFIX`-style shard keys.
3. A larger embedding model (Ollama / sentence-transformers) behind the same activator as `linking/embed.py`.
4. GATE-style grounding memory: persist resolved value/format groundings per column so the agent stops re-discovering them.
