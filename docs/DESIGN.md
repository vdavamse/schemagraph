# schemagraph — design

**One sentence.** Catalog metadata in (DDL, DuckDB, dbt, Unity Catalog, AWS Glue, Collibra), a single provenance-tagged schema graph in the middle, and a linked, join-complete sub-schema out — as annotated DDL for an LLM, over HTTP and MCP, with a UI to feed and curate it.

It sits in the same slot as JetBrains' Databao Context Engine and SignalPilot's semantic layer, minus governance and SQL execution (those belong to the agent that calls this), and with the schema-linking core rebuilt around **graph traversal** instead of embedding search.

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
ppr.personalized_pagerank   HippoRAG PPR over {tables, columns, terms, tokens}; specificity-weighted seeds
  │  ppr.table_scores        own + best + 0.5·second + 0.25·third column (IDF-weighted seeds), plus direct lexical bonus
  ▼
anchors                     top-k by score (k = 6; ≥ 15% of best), or an optional Claude call that picks
  │                         source/destination tables among the PPR candidates (SchemaGraphSQL step 1)
  ▼
pathfinding.union_of_shortest_paths
  │                         all weighted-shortest simple paths between anchor pairs; FK = 1.0,
  │                         catalog relation = 1.3, lineage = 1.6, inferred = 2.5
  ▼
pruning.prune_paths         PathRAG flow propagation (α = 0.8, θ = 0.05), reliability = mean resource,
  │                         drop sub-paths of kept paths, keep top-k
  ▼
column selection            anchors: all columns; bridge tables: PK + join keys + activated columns
  ▼
render.render_ddl           CREATE TABLE … with descriptions, glossary header, join paths (most reliable
                            last), sample values footer
```

Warm-path latency on a 7-table schema is ~3 ms; the first call pays a one-off SciPy import. How the entities behind the first two stages are formed, field by field and source by source, is in [`ENTITIES.md`](ENTITIES.md).

## Graph model (`graph/build.py`)

Undirected NetworkX graph. Node ids are prefixed: `t:` table, `c:` column, `k:` glossary term, `w:` token. Edge attribute `etype` ∈ {`contains`, `relation`, `fk_col`, `glossary`, `mention`}. A `relation` edge between two tables carries `relations: list[Edge]` — every piece of evidence from every source, deduplicated by (kind, tables, columns) — and `weight = min` over the kinds. `table_graph()` projects to tables + `relation` edges for path-finding.

Merging is field-level: the first source to fill a field wins, later sources fill blanks and union columns/tags/samples, and the **user snapshot** (glossary + join hints from the UI) is merged last so human curation overrides catalogs where it conflicts.

Referenced-but-unknown tables become stub nodes (`properties.stub = "true"`) so paths through them still exist.

## Connectors (`connectors/`)

| type | edges produced | notes |
|---|---|---|
| `ddl` | `foreign_key` | sqlglot; inline `REFERENCES`, table-level FK, `ALTER TABLE ADD FOREIGN KEY`, `COMMENT` / `COMMENT ON` |
| `duckdb` | `foreign_key` | `duckdb_tables/columns/constraints`, row counts, sample values |
| `dbt` | `lineage`, `relationship_test`, `foreign_key` (model constraints) | manifest.json, or project dir parsed without dbt (regex over `ref()`/`source()`); `unique`+`not_null` → PK, `accepted_values` → samples, semantic models → glossary |
| `unity_catalog` | `foreign_key`, `lineage` | REST 2.1 catalogs/schemas/tables + `table_constraints`; optional lineage-tracking API |
| `aws_glue` | none | boto3; columns, partition keys, parameters, optional LF-tags. Relations come from other sources or hints |
| `collibra` | `foreign_key`, `catalog_relation`, glossary | REST 2.0 assets/relations/attributes; asset types, relation roles and attribute names are all configurable |

Connector configs are pydantic models; the UI renders a form from their JSON schema. Secrets may be `${ENV_VAR}` references, substituted at run time and never returned by the API.

## Persistence (`store.py`)

One DuckDB file: `connections`, `snapshots` (full JSON per source), `glossary`, `join_hints`. The graph is rebuilt in memory on load — DuckDB is the durable form, NetworkX the working one.

## Surfaces

* **HTTP** (`api/app.py`): `/api/connections`, `/api/ddl`, `/api/build`, `/api/graph/*`, `/api/link`, `/api/explain`, `/api/glossary`, `/api/join-hints`; serves `web/dist` at `/`.
* **MCP** (`mcp/server.py`): `link_schema`, `link_schema_json`, `search_tables`, `get_table`, `find_join_path`, `list_glossary`, `graph_stats`. Read-only.
* **CLI** (`cli.py`): `add-ddl`, `add-dbt`, `add-duckdb`, `add <type> <name> -c '{json}'`, `build`, `link`, `explain`, `path`, `serve`, `mcp`.
* **Web** (`web/`): Connections, Paste DDL, Link playground, Graph (cytoscape), Glossary & hints.

## LLM use

Optional and small: one call per question (`llm/anchors.py`) that returns source/destination tables via a JSON-schema output format, using `claude-opus-5` by default with `effort: low` and server-side refusal fallbacks enabled. Enabled only when `ANTHROPIC_API_KEY` is set and the caller passes `use_llm=true`. Everything else is deterministic and free.

## Non-goals (v1)

* No SQL execution, no governance (validation, LIMIT injection, PII redaction, budgets). The calling agent owns that boundary.
* No embeddings. The lexical + PPR path is the LinearRAG bet; an embedding-based activation stage can be added behind the same `Activation` interface if recall on undocumented schemas proves insufficient.
* No BI metrics layer. Glossary terms map words to columns; they are not governed calculations.

## Benchmark

`schemagraph bench-spider2-lite <Spider2 clone>` scores gold-table recall of `link_schema` on the 547 Spider 2.0-Lite tasks with no execution, credentials, or LLM. Results, ablations, the iteration log and error analysis: [`bench_results/README.md`](../bench_results/README.md). Headline (2026-09-04): **95.7% strict table recall, 98.1% recall, 11.8 tables returned, ~20 ms per question** on Lite and **96.5% strict on Spider 2.0-Snow** (`--suite snow`); 90.4% strict on schemas with ≥ 100 tables (was 18.9% before the large-schema loop).

What the benchmark forced into the core, each ablated:

* `linking/lexical.py` — **IDF weighting** of token evidence (a token shared by half the columns is nearly worthless as a seed), a stopword list of question scaffolding that also appears in descriptions, and no seeds from digit tokens shorter than 4 characters.
* `graph/ppr.py` — **top-3 column aggregation** into table scores instead of a sum, so a wide table with forty weak matches no longer swamps a table with one strong name match.
* `graph/infer.py` — name-based `inferred` edges (`x_id` ↔ `x.id`, shared key columns) for catalogs with no declared FKs.
* `connectors/spider2.py` — **partition-family collapse**: tables in one schema whose names differ only in digit runs and share ≥ 80% of columns become one logical table with a member list. The same treatment belongs in the Unity/Glue connectors for partitioned datasets.
* `linking/linker.py` — **adaptive budget** (schemas over 30 tables get 20 tables and 6 anchors) and **bypass-if-fits** (a schema that fits the budget is returned whole, with anchors and join paths still computed) — the "Death of Schema Linking" result, operationalised. Default budget is 20 tables.
* Rejected: DBCopilot-style **schema routing** (boosting the dataset with the most activation mass) — Spider 2 gold sets routinely join across datasets and it cost 6–7 strict points. Kept as an option (`schema_routing`), off.

## Next steps

1. The last 5%: gold tables at rank 21–40 in mid-size schemas. A LinkAlign-style query-rewrite round or the optional Claude anchor pass (`bench-spider2-lite --llm`) are the candidates; both need an API key to measure.
3. Shard-family collapse in the Unity Catalog and Glue connectors; value grounding on `_TABLE_SUFFIX`-style shard keys.
4. Optional embedding activation (Ollama / sentence-transformers) as a second seed source.
5. GATE-style grounding memory: persist resolved value/format groundings per column so the agent stops re-discovering them.
