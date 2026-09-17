# How entities are formed

Status: description of the code as of 2026-09-04. Read this before touching
`connectors/`, `graph/build.py` or `linking/lexical.py`.

## What an "entity" is here

There is no NLP entity extraction in schemagraph: no named-entity recogniser, no
part-of-speech tagging, no embeddings, no LLM in the indexing path. The LinearRAG
idea is applied literally: the entities **are the schema objects**, and "extraction"
means aligning question words to those objects through their surface forms. Every
entity is a node in the NetworkX graph:

| node id | entity | created by | surface forms indexed |
|---|---|---|---|
| `t:<fqn>` | table / view / dbt model / source / family | `graph/build.py` | name, description, tags, `business_name`, `dbt_name` |
| `c:<fqn>.<col>` | column | `graph/build.py` | name, description, tags, `business_name`, sample values |
| `k:<term>` | business-glossary term | `graph/build.py` | name, synonyms, description |
| `w:<token>` | lexical token (the LinearRAG "mention" anchor) | `linking/lexical.build_index` | itself |

Sample values are not nodes; they live in an exact-match map inside the lexical
index and activate the column that holds them.

The whole path has five stages. Stages 1 to 3 run at build time, 4 and 5 at query time.

```
source metadata ──1 connector──► SchemaSnapshot ──2 merge──► graph nodes ──3 index──► tokens, names, values, phrases
question ──4 activate──► seeds {node: weight, reasons} ──5 PPR + aggregation──► ranked tables ──► anchors
```

## Stage 1: extraction from each source (`connectors/`)

Each connector turns one source into `Table`, `Column`, `Edge` and `BusinessTerm`
objects (`model.py`). This is the only place "extraction" happens, and it is a
field mapping, not text mining. What each source contributes to entity surface forms:

| source | table name / description | column name / description | tags | sample values | business terms |
|---|---|---|---|---|---|
| **DDL** (sqlglot) | `CREATE TABLE` name; `COMMENT 'x'` property or `COMMENT ON TABLE` | column def; inline `COMMENT` or `COMMENT ON COLUMN` | none | none | none |
| **DuckDB** | `duckdb_tables().comment` | `duckdb_columns().comment` | none | `SELECT DISTINCT` on text columns, 5 by default, 80 chars max | none |
| **dbt** (manifest or project dir) | model / source / seed name; YAML `description`; `properties["dbt_name"]` | YAML `columns[].description`; columns guessed from the SELECT when YAML is missing | YAML `tags` | `accepted_values` test values, up to 20 | semantic-model entities and measures (`name` -> `model.expr`) |
| **Unity Catalog** | `comment`, `owner`, non-`delta.` properties | `comment`, `type_text`, `nullable` | none | none | none |
| **AWS Glue** | `Description` or `Parameters.comment`, `classification`, `location` | `Comment`, column `Parameters` | LF-tags as `key=value` on tables and columns; `partition_key` on partition columns | none | none |
| **Collibra** | Table asset `displayName` after the last `>` or `.`; `Description` attribute; `domain`, `status` | Column asset name; `Description`; `Data Type` | classification attributes as `name=value` | none | Business Term assets with `Description`, synonyms via `is synonym of`, targets via `represents` / `is described by` |
| **Spider2** (benchmark only) | JSON `table_description`; families get `business_name = "<schema> <stem>"` | per-column `description` list | `nested` on struct columns | first 5 distinct values of `sample_rows` | none |

Things to know about this stage:

* Names are taken verbatim. Nothing splits `CustomerOrders` here; that happens in stage 3.
* Descriptions are the second-strongest surface form and the only place most
  warehouses carry business vocabulary. Unity, Glue, Collibra and dbt all populate
  them; pasted DDL only does if the author wrote `COMMENT`.
* Sample values come only from DuckDB, dbt `accepted_values` and Spider2. Unity,
  Glue and Collibra never provide them, so value-based linking ("customers in
  California" -> `customer.state`) does not work on those catalogs unless a DuckDB
  or dbt source for the same table is also connected.
* Business terms come only from Collibra, dbt semantic models and the UI glossary.
* Every object is stamped with `source` (`SchemaSnapshot.stamp()`), which is how
  provenance survives the merge and shows up in the rendered DDL (`-- from unity`).

## Stage 2: merge into nodes (`graph/build.py`)

`SchemaGraph.add_snapshot` turns snapshot objects into nodes:

* **Identity of a table** is the lowercase fully-qualified name
  `catalog.schema.table` with empty parts dropped (`Table.fqn`). Snapshots merge in
  priority order (`SOURCE_PRIORITY`: user, collibra, dbt, unity_catalog, duckdb, ddl,
  aws_glue, or the connection's own `priority`); a second snapshot with the same key
  merges into the existing node: scalar fields keep the highest-priority non-empty
  value, columns are unioned by name, tags, sample values (max 20) and primary keys
  are unioned, properties are merged with the higher-priority side winning. A stub
  created for a referenced-only table is replaced when the real table arrives. This
  is gap 1 in the project notes: FQN shapes differ per connector, so the same
  physical table from Unity and Collibra becomes two entities.
* **Identity of a column** is `fqn.column` lowercased. Column nodes get a
  `contains` edge (weight 0.5) to their table.
* **Tables referenced by an edge but never introspected** become stub entities
  (`properties.stub = "true"`) with no columns, so join paths through them exist.
* **Business terms** become `k:` nodes keyed by lowercase name. Their `targets`
  are resolved with `_resolve_target` (exact FQN, `table.column`, or a unique bare
  table name) after every snapshot has loaded its tables, and each hit gets a
  `glossary` edge (weight 0.7). Targets that never resolve are dropped; since
  2026-09-17 resolution no longer depends on snapshot order (the former gap 4).
* Relation edges (`foreign_key`, `lineage`, `relationship_test`,
  `catalog_relation`, `join_hint`, `inferred`) do not create entities; they connect
  table entities and, when they carry columns, add `fk_col` edges between column
  entities.

The node attributes used later are `ntype`, `fqn` and `name` (lowercased bare
name). PPR's specificity weight is computed from how many nodes share a `name`.

## Stage 3: surface forms (`linking/lexical.build_index`)

This is where entities get the vocabulary that the question can hit. For every
table, column and term node the indexer produces four kinds of index entries.

**Tokenisation** (`tokenize`): insert `_` at camelCase boundaries, lowercase, split
on anything that is not `[a-z0-9]`, drop tokens shorter than 2 characters unless
they are digits, and drop stopwords unless `keep_stop=True`. Names are tokenised
with stopwords kept, descriptions without.

**Name expansion** (`_name_tokens`): each name token also emits its abbreviation
expansion from the fixed `ABBREVIATIONS` map (`cust -> customer`, `qty ->
quantity`, `dt -> date`, about 35 entries) and its lemma from a three-rule
stemmer (`ies -> y`, `ses -> se`, trailing `s`). So the column `cust_ids` indexes
the tokens `cust`, `ids`, `customer`, `id`.

**Postings** (`idx.postings[token] -> [(node, weight)]`), one entry per token per
node, keeping the maximum weight when a token appears in several fields:

| field | weight |
|---|---|
| table or column name token, its abbreviation expansion and lemma | 1.0 |
| `business_name` / `dbt_name` property tokens | 0.9 |
| tag tokens (`partition_key`, `Data Classification=PII` -> `data`, `classification`, `pii`) | 0.5 |
| description tokens | 0.35 |
| glossary term name and synonym tokens | 1.0 |
| glossary description tokens | 0.3 |

**Full-name index** (`idx.names`): the normalised whole name
(`"_".join(tokenize(name, keep_stop=True))`) maps to its nodes, so a question
bigram or trigram can hit `product_category` as one unit. Names whose stopword-free
form differs and still has two or more tokens are also indexed under that form
(`idx.names_nostop`: `date_of_birth` -> `date_birth`); in the large Spider 2.0-Lite
schemas one object name in five contains a stopword token (`to`, `in`, `over`, `or`,
`other`, `first`, `last`, `account`, `report`).

**Value index** (`idx.values`): every column sample value, lowercased and stripped,
at least 3 characters and not purely numeric, maps to the column nodes holding it.
For matching, each value is also keyed by its words (`idx.value_grams`, split on
anything non-alphanumeric, up to six words; longer values stay in `idx.long_values`
and are scanned with a regex). A question is matched by looking up its own word
n-grams, so the cost is linear in the question, not in the number of values (one
regex per value was 95 % of activation time on a 2,500-value schema).

**Phrase index** (`idx.phrases`): every glossary term name and synonym, lowercased,
maps to the term node.

**IDF** per token, in `[0.05, 1]`:
`log((N - df + 0.5) / (df + 0.5) + 1) / log(N + 1)` where N is the number of
table, column and term nodes and df the number of nodes the token appears on. A
token unique to one object scores about 1; `id` on half the columns scores near
0.05. Only single-token evidence is scaled by this at query time.

**Token nodes**: for every token with at most 200 postings a `w:<token>` node is
added to the graph with a `mention` edge to each posting node, weight
`0.3 x posting weight`. This is the LinearRAG mention matrix. Tokens on more than
200 objects are kept in the postings but get no node, so they cannot become PPR hubs.
The indexer mutates the graph in place, which is why a `Linker` must be built on a
fresh graph.

## Stage 4: activation from the question (`linking/lexical.activate`)

The question is tokenised the same way (camelCase split, lowercase, stopwords
removed). Five matchers run in order, each adding weight to a node through
`Activation.bump(node, weight, reason)`. Weights on the same node add up, and the
reason strings are what `schemagraph explain` and the rendered column comments show.

| # | matcher | matches | weight | IDF-scaled |
|---|---|---|---|---|
| 1 | glossary phrase | a term name or synonym appears in the question as a whole word, longest phrase first | 1.5 on the term, plus 1.2 on every table or column the term targets | no |
| 2 | sample value | an indexed value appears in the question as a whole-word sequence (word n-gram lookup; punctuation inside the value is a word boundary, so `St. Louis` matches "st louis") | 1.5 on each column holding it, once per value | no |
| 3 | n-gram | a question bigram or trigram, formed both with and without stopwords, equals an object name in either its full or its stopword-free form ("date of birth" -> `date_of_birth`, "first name" -> `first_name`); one hit per name however many forms match (`ngram_stop`, default on) | 1.6 | no |
| 4 | single token | token, its lemma (0.8), its abbreviation expansion (0.8), and the reverse expansion (0.8, so `customer` also hits `cust`) against postings | `token weight x posting weight` | yes |
| 5 | fuzzy | only for tokens of 5+ characters with no posting at all: the three closest vocabulary entries by `difflib` at ratio >= 0.86 | `0.6 x posting weight` | yes |

Digit-only tokens shorter than `min_numeric_len` (4) are skipped as single tokens,
so `5` from "5-year" seeds nothing while `2019` can still hit a shard family.

The output is `Activation(seeds, matched_terms, matched_values, tokens, reasons)`.
That dictionary is the complete entity layer. Everything downstream consumes only
`seeds`, `matched_terms` and `reasons`.

## Stage 5: from seeds to anchor tables (`graph/ppr.py`, `linking/linker.py`)

1. **Specificity**: each seed weight is multiplied by
   `1 / log(1 + n)` where n is the number of table or column nodes sharing the
   same bare name. A column called `id` on forty tables contributes almost nothing.
2. **Personalized PageRank** (`alpha=0.85`) over the whole heterogeneous graph, as
   a power iteration on a row-stochastic sparse matrix built once per graph
   (`graph/ppr.PPRMatrix`) and iterated to an L1 tolerance of 1e-12 per node. The
   walk starts from the teleport vector, so components the seeds do not touch stay
   exactly zero. (Until 2026-09-17 this was `nx.pagerank` on a subgraph view with
   the default tolerance of `N x 1e-6`; on a 7,000-node schema that stopped with an
   L1 error near 1e-2 and reordered near-tied tables as early as rank 1.) Activation
   flows over `contains`, `relation`, `fk_col`, `glossary` and `mention` edges, so a
   table connected to several activated columns, or one hop from an activated table,
   rises. Relation edges carry their join cost as transition mass by default
   (`weight`: inferred 2.5 > FK 1.0), which is backwards as semantics but measured
   better than uniform per-kind affinity on Lite, where inferred edges are the only
   structure (`ppr_edge_attr="affinity"` selects the uniform `PPR_AFFINITY` values).
3. **Table aggregation** (`agg="top3"`): table score = own PPR score + best column
   + 0.5 second + 0.25 third + 0.02 the rest.
4. **Direct lexical bonus**: tables that were seeded directly get
   `0.35 x best table score x min(seed, 2)` added; columns add `0.10 x` the same
   to their table. This counters PPR diluting a strong name match.
5. **Anchors**: candidates are tables scoring at least 15 % of the best, capped at
   `max(3 x anchor_k, 8)`; the top `anchor_k` (6) become anchors. With `use_llm`
   and a key, the Claude anchor picker chooses source and destination tables among
   those candidates instead. It does not see the question's entities, only the
   candidate tables.

Column selection later reuses the same seeds: a column is kept when it is a
primary key, a join key on a returned path, or has a seed, and the seed's reason
becomes the `-- reason` comment in the DDL.

## Worked example

`examples/store.sql`, question *"revenue by product category for customers in
california"* (`schemagraph explain`):

| seed | weight | how it was formed |
|---|---|---|
| `t:public.product_category` | 2.92 | trigram `product_category` equals the table name (1.6) + tokens `product`, `category` on the name (1.0 each, IDF-scaled) |
| `t:public.customer` | 1.05 | `customers` has no posting; fuzzy match to `customer` (0.6) + lemma `customer` (0.8), both IDF-scaled |
| `c:public.orders.customer_id` | 1.05 | same two matchers through the `customer` name token of `customer_id` |
| `t:public.products` | 0.89 | `product` from the name, `category` from the description "items by category and price" (0.35) |
| `c:public.orders.total_amount` | 0.31 | `revenue` appears only in the description "order revenue in USD" (0.35 x IDF) |
| `c:public.customer.state` | 0.31 | `california` appears only in the description "US state, e.g. California"; no sample values exist in a DDL source |

PPR then ranks `product_category`, `products`, `customer`, `order_items`, `orders`.
`order_items` and `orders` were never seeded by the question; they rose through
`relation` edges from seeded neighbours, and the shortest-path union between the
anchors makes them bridge tables. Note that `revenue` reached `total_amount` only
because someone wrote a column comment. Without it the question word has no entity.

## What this design cannot do, and where the seam is

Failure classes that follow directly from the mechanism (all observed in the
Spider 2.0-Lite misses, `bench_results/README.md`):

* **Paraphrase with no shared token**: "turnover" vs `total_amount` with no comment.
  The three-rule stemmer and the 35-entry abbreviation map are the only synonymy.
* **Entities that are values, not names**: a product name, a city, a status code
  in the question hits only if a sample value or `accepted_values` list contains it.
  Unity, Glue and Collibra sources never carry values.
* **Multi-word concepts split across objects**: "gross margin" is two tokens that
  each land on unrelated columns unless a glossary term binds them.
* **Vocabulary and bridge tables with no lexical hook** (`concept_ancestor`,
  `mc3_maf_v5_one_per_tumor_sample`): reachable only through graph structure.
* **Digits and codes**: shard suffixes and numeric identifiers are deliberately
  not seeds.

The seam for a real entity-extraction stage is `Activation`. A new activator only
needs to add `(node, weight, reason)` entries; PPR, anchors, column selection and
the DDL reasons all work unchanged. Three options, cheapest first, and they compose
because seeds add up:

1. **Curated vocabulary**: grow `ABBREVIATIONS` per deployment and lean on the
   glossary. Collibra terms already flow in; the missing piece is gap 4 (targets
   must resolve after all tables are loaded) and gap 1 (targets must resolve to
   the merged entity, not a Collibra twin).
2. **Embedding activation** (implemented 2026-09-17, `linking/embed.py`, extra
   `embed`): each table, column and term's surface forms (name words, business
   name, description) are embedded once with a static model (model2vec
   `potion-base-8M`, numpy only, 5,000 objects in 0.14 s); at query time the
   question's word n-grams are embedded and, per phrase, the closest objects above
   cosine 0.5 are seeded at `0.8 x cosine`. Spider 2.0-Lite: strict 95.85 -> 96.23,
   anchor hit 70.4 -> 72.6, strict@7 on the precise sample 70.6 -> 73.7, p50 latency
   12 -> 51 ms. Off in `LinkOptions` (benchmarks stay dependency-free); the Engine
   turns it on when the extra is installed.
3. **LLM entity pass**: one call that returns question entities as JSON
   (`{measures, dimensions, filters, values, time}`), each then matched by the
   existing matchers, with values routed straight to the value index. Fixes
   multi-word concepts and value grounding at roughly a thousand tokens per
   question; the same call could replace the current anchor picker.

Whichever is chosen, keep the seeds explainable: every entity in the DDL must
still carry a reason string, because that is what the calling agent uses to decide
which columns to trust.
