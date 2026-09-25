"""The schema-linking pipeline: activate -> PPR -> anchors -> paths -> prune -> columns -> DDL.

question
  │  lexical.activate            (LinearRAG entity activation, no LLM)
  ▼
seeds ──► ppr.personalized_pagerank   (HippoRAG PPR over tables/columns/terms/tokens)
  │
  ▼
anchor tables  ◄── optional LLM pass picks source/destination tables (SchemaGraphSQL step 1)
  │
  ▼
pathfinding.union_of_shortest_paths   (SchemaGraphSQL: bridge tables by construction)
  │
  ▼
pruning.prune_paths                   (PathRAG flow pruning of redundant routes)
  │
  ▼
column selection + render.render_ddl  (SignalPilot-style annotated DDL)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from schemagraph.graph.build import SchemaGraph, cnode, tnode
from schemagraph.graph.pathfinding import union_of_shortest_paths
from schemagraph.graph.ppr import (
    PPRMatrix,
    personalized_pagerank,
    specificity_weights,
    table_scores,
)
from schemagraph.graph.pruning import prune_paths
from schemagraph.linking.bm25 import BM25Index, bm25_scores, build_bm25, reciprocal_rank_fusion
from schemagraph.linking.lexical import Activation, LexicalIndex, activate, build_index
from schemagraph.linking.render import render_ddl
from schemagraph.model import Column, JoinPath, LinkedColumn, LinkedTable, LinkResult, Table

if TYPE_CHECKING:
    from schemagraph.linking.embed import EmbeddingActivator

# Direct lexical evidence on a table (or one of its columns) is added on top of its PPR score,
# because PPR dilutes it: bonus = factor * best table score * min(seed weight, cap).
TABLE_SEED_BONUS = 0.35
COLUMN_SEED_BONUS = 0.10
SEED_BONUS_CAP = 2.0
# Column score = PPR score * this + seed weight (PPR scores are tiny next to seed weights).
COLUMN_PPR_SCALE = 1000
# Top ranked tables whose evidence is passed to path pruning as a node prior.
PRIOR_WINDOW = 50
# Anchor candidates: at least this many, or ANCHOR_CANDIDATE_FACTOR * anchor_k.
MIN_ANCHOR_CANDIDATES = 8
ANCHOR_CANDIDATE_FACTOR = 3
# Column cap of a table ranked at or above ``columns_top_uncapped``, and the rank assumed for
# an unranked table (so it is never uncapped).
UNCAPPED = 10**9
# ``explain`` lists this many seeds and PPR nodes, and this many tables.
EXPLAIN_TOP = 40
EXPLAIN_TABLES = 20
# Column reasons that exempt a column from the width cap.
KEY_REASONS = frozenset({"primary key", "join key"})


@dataclass
class LinkOptions:
    """Every knob of the linking pipeline; each stage can be ablated from the benchmark CLI.

    The defaults were each set by an ablation on Spider 2.0-Lite (bench_results/README.md).

    Attributes:
        max_tables: Table budget. 12 -> 20 after the Spider2-Lite loop: +5 strict recall for
            ~2 extra tables.
        anchor_k: Top tables by PPR treated as anchors when there is no LLM pass.
        min_anchor_ratio: Anchors must score >= ratio * best (relative evidence).
        fill_ratio: Non-path tables need >= ratio * best evidence to be included.
        bypass_if_fits: If the whole schema fits ``max_tables``, return all of it ("Death of
            Schema Linking").
        ppr_alpha: PPR damping factor.
        paths: Union of shortest paths between anchors (off = anchors + budget fill only; the
            ablation for bridge tables).
        path_extra: Allow paths this much longer than the shortest.
        path_cutoff: Longest path considered, in hops.
        prune_alpha: PathRAG flow decay.
        prune_theta: PathRAG flow threshold.
        prune_top_k: Paths kept after pruning.
        columns: ``"relevant"`` | ``"all"``.
        max_columns_per_table: Column cap for tables ranked below ``columns_top_uncapped``
            (60 -> 40 on 2026-09-17: same col_strict, fewer tokens).
        columns_top_uncapped: Tables at rank <= this keep every column: Lite col_strict
            81.2 -> 91.2 at -0.5 % tokens (the cap was the whole column-level miss; table rank,
            not column evidence, predicts gold columns; top-3 gives 94.6 at +19 % tokens).
        small_schema_bypass: If > 0 and the graph has <= N tables, return everything.
        use_llm: Let the LLM pick anchor tables (needs ``Linker.llm``).
        render: Render the result as annotated DDL.
        debug: Include the table ranking in the result.
        ranking_limit: With ``debug``, keep this many ranked tables (0 = all; the benchmark
            uses 0 so gold rank is never ambiguous).
        idf: Scale single-token evidence by rarity across the schema.
        agg: Column -> table aggregation: ``"top3"`` | ``"sum"``.
        min_numeric_len: Ignore digit-only tokens shorter than this.
        adaptive_budget: Widen ``max_tables`` / ``anchor_k`` on large schemas.
        large_threshold: A schema with more tables than this is "large".
        max_tables_large: ``max_tables`` on a large schema.
        anchor_k_large: ``anchor_k`` on a large schema.
        schema_routing: 0 = off (hurt on Spider2-Lite: -6 strict on large schemas); blend table
            score with its dataset's activation mass (DBCopilot-style).
        ngram_stop: N-gram name matches survive stopwords inside names ("date of birth" ->
            date_of_birth).
        ppr_edge_attr: Edge attribute PPR reads as transition mass: ``"weight"`` (join cost,
            historical) | ``"affinity"`` (PPR_AFFINITY by kind).
        ranker: ``"rrf"`` (reciprocal-rank fusion of the PPR ranking with BM25F over one
            document per table; Lite 95.85 strict, +3.7 strict@7) | ``"ppr"`` (activation +
            PPR alone, 95.66) | ``"bm25"`` (BM25F alone, 94.91).
        rrf_k: Reciprocal-rank fusion constant: score = sum 1/(k + rank); smaller k weights the
            head of each ranking more.
        seed_bm25: SPRIG seed-side fusion: top ``seed_k`` BM25F tables also seed PPR
            (personalization only). Rejected 2026-09-23: +1-2 strict, -0.3 strict@7 with embed.
        seed_k: BM25F tables used as PPR seeds (SPRIG: 5-10).
        seed_w: BM25F seed weight at rank r (0-based) = seed_w / (r + 1); a name hit is 1.0.
        embed: Add paraphrase seeds from a small static embedding model (linking/embed.py;
            needs the ``embed`` extra).
        embed_model: The model2vec model name.
        embed_threshold: Cosine floor for an embedding seed.
        embed_top_k: Objects considered per question phrase.
        embed_weight: Seed weight = embed_weight * cosine (a description hit is 0.35, a name
            hit 1.0).
    """

    max_tables: int = 20
    anchor_k: int = 6
    min_anchor_ratio: float = 0.15
    fill_ratio: float = 0.06
    bypass_if_fits: bool = True
    ppr_alpha: float = 0.85
    paths: bool = True
    path_extra: float = 0.0
    path_cutoff: int = 6
    prune_alpha: float = 0.8
    prune_theta: float = 0.05
    prune_top_k: int = 8
    columns: str = "relevant"
    max_columns_per_table: int = 40
    columns_top_uncapped: int = 1
    small_schema_bypass: int = 0
    use_llm: bool = False
    render: bool = True
    debug: bool = False
    ranking_limit: int = 60
    idf: bool = True
    agg: str = "top3"
    min_numeric_len: int = 4
    adaptive_budget: bool = True
    large_threshold: int = 30
    max_tables_large: int = 20
    anchor_k_large: int = 6
    schema_routing: float = 0.0
    ngram_stop: bool = True
    ppr_edge_attr: str = "weight"
    ranker: str = "rrf"
    rrf_k: int = 60
    seed_bm25: bool = False
    seed_k: int = 5
    seed_w: float = 1.0
    embed: bool = False
    embed_model: str = "minishlab/potion-base-8M"
    embed_threshold: float = 0.5
    embed_top_k: int = 5
    embed_weight: float = 0.8


@dataclass(frozen=True)
class _Ranking:
    """Everything ``Linker._rank`` computes for one question.

    Attributes:
        activation: The lexical (and embedding) seeds.
        node_scores: PPR score of every node.
        table_scores: The final table scores (fused, BM25F, or PPR plus direct evidence).
        ranked: Tables with a positive score, best first, fqn breaking ties.
        evidence: Each table's score relative to the best table under each ranker (max across
            rankers, in [0, 1]); the anchor and fill gates read it instead of the fused score,
            whose rank-based scale is flat.
    """

    activation: Activation
    node_scores: dict[str, float]
    table_scores: dict[str, float]
    ranked: list[tuple[str, float]]
    evidence: dict[str, float]


def _by_score(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Items sorted by descending score, the key breaking ties so runs are reproducible."""
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _sorted_positive(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Items with a positive score, sorted as :func:`_by_score`."""
    return [(key, score) for key, score in _by_score(scores) if score > 0]


def _relative(scores: dict[str, float]) -> dict[str, float]:
    """Each score divided by the best one (by 1.0 when the best is 0 or there is none)."""
    best = max(scores.values(), default=0.0) or 1.0
    return {key: score / best for key, score in scores.items()}


def _fuse_with_bm25(
    ppr_scores: dict[str, float],
    sparse: dict[str, float],
    evidence: dict[str, float],
    rrf_k: int,
) -> dict[str, float]:
    """Fuse the PPR and BM25F table rankings by reciprocal rank.

    Args:
        ppr_scores: PPR table scores (only positive ones are ranked).
        sparse: BM25F table scores (all ranked).
        evidence: Relative PPR evidence; each BM25F table's relative score is folded in as a
            max. Mutated in place.
        rrf_k: The fusion constant.

    Returns:
        The fused table scores.
    """
    ppr_rank = [fqn for fqn, _ in _sorted_positive(ppr_scores)]
    sparse_rank = [fqn for fqn, _ in _by_score(sparse)]
    fused = reciprocal_rank_fusion(ppr_rank, sparse_rank, k=rrf_k)
    sparse_best = max(sparse.values(), default=0.0) or 1.0
    for fqn, score in sparse.items():
        evidence[fqn] = max(evidence.get(fqn, 0.0), score / sparse_best)
    return fused


def _join_columns(schema_graph: SchemaGraph, join_paths: list[JoinPath]) -> dict[str, set[str]]:
    """Lowercased join-key columns per lowercased table fqn, over every step of every path."""
    join_columns: dict[str, set[str]] = {}
    for join_path in join_paths:
        for step in join_path.steps:
            for relation in schema_graph.relations(step.from_table, step.to_table):
                join_columns.setdefault(relation.from_table.lower(), set()).update(
                    column.lower() for column in relation.from_columns
                )
                join_columns.setdefault(relation.to_table.lower(), set()).update(
                    column.lower() for column in relation.to_columns
                )
    return join_columns


def _keep_reason(
    table: Table,
    column: Column,
    score: float,
    *,
    join_keys: set[str],
    is_anchor: bool,
    activation: Activation,
    node_scores: dict[str, float],
    opts: LinkOptions,
) -> tuple[bool, str | None]:
    """Decide whether a column of a kept table is linked, and why.

    Tried in order: primary key, join key, seeded (its first two reasons), keep-everything
    (``columns="all"``, an anchor table, or no seeds at all: nothing to select by), and finally
    any positive PPR score.

    Args:
        table: The column's table.
        column: The column.
        score: The column's score (``PPR * COLUMN_PPR_SCALE + seed``).
        join_keys: Lowercased join-key columns of the table.
        is_anchor: Whether the table is an anchor.
        activation: The question's seeds and reasons.
        node_scores: PPR score of every node.
        opts: The linking options.

    Returns:
        ``(keep, reason)``; the reason is None unless the column is a key or seeded.
    """
    column_node = cnode(table.fqn, column.name)
    if column.is_primary_key or column.name in table.primary_key:
        return True, "primary key"
    if column.name.lower() in join_keys:
        return True, "join key"
    if column_node in activation.seeds:
        return True, "; ".join(activation.reasons.get(column_node, [])[:2])
    if opts.columns == "all" or is_anchor or not activation.seeds:
        return True, None
    return score > 0 and node_scores.get(column_node, 0.0) > 0, None


def _cap_columns(columns: list[LinkedColumn], cap: int) -> list[LinkedColumn]:
    """Keep every key column plus the best-scored others, up to ``cap`` columns in all."""
    if len(columns) <= cap:
        return columns
    keys = [column for column in columns if column.reason in KEY_REASONS]
    rest = sorted(
        [column for column in columns if column.reason not in KEY_REASONS],
        key=lambda column: -column.score,
    )
    return keys + rest[: max(0, cap - len(keys))]


def _declaration_order(columns: list[LinkedColumn], table: Table) -> list[LinkedColumn]:
    """The linked columns in the order the table declares them (stable for unknown names)."""
    order = {column.name: i for i, column in enumerate(table.columns)}
    return sorted(columns, key=lambda column: order.get(column.name, 0))


def _plain_columns(columns: list[Column]) -> list[LinkedColumn]:
    """Linked columns carrying only name, type and description."""
    return [
        LinkedColumn(name=column.name, data_type=column.data_type, description=column.description)
        for column in columns
    ]


def _elapsed_ms(start: float) -> float:
    """Milliseconds since ``start`` (a ``time.perf_counter()`` value), to one decimal."""
    return round((time.perf_counter() - start) * 1000, 1)


class Linker:
    """Links questions to a sub-schema of one :class:`SchemaGraph`.

    Building a linker indexes the graph (adding ``w:`` token nodes to it in place); the PPR
    matrix is cached per edge attribute on first use, so the graph must not change afterwards.

    Attributes:
        schema_graph: The graph being linked against.
        index: Its lexical index.
        llm: Optional object with ``.anchor_tables(question, candidates) -> (sources,
            destinations)``, used when ``LinkOptions.use_llm``.
    """

    def __init__(self, schema_graph: SchemaGraph, index: LexicalIndex | None = None, llm=None):
        self.schema_graph = schema_graph
        self.index = index or build_index(schema_graph)
        self.llm = llm
        self._spec = specificity_weights(schema_graph)
        self._matrices: dict[str, PPRMatrix] = {}  # per edge attribute
        self._bm25: BM25Index | None = None
        self._embedder = None  # EmbeddingActivator, built on first use with embed=True

    def _matrix(self, edge_attr: str) -> PPRMatrix:
        """The cached PPR transition matrix reading ``edge_attr``, built on first use."""
        matrix = self._matrices.get(edge_attr)
        if matrix is None:
            matrix = self._matrices[edge_attr] = PPRMatrix(self.schema_graph, edge_attr)
        return matrix

    def _bm25_index(self) -> BM25Index:
        """The BM25F index over the graph's tables, built on first use."""
        if self._bm25 is None:
            self._bm25 = build_bm25(self.schema_graph)
        return self._bm25

    def embedder(self, model_name: str) -> EmbeddingActivator:
        """The embedding activator for ``model_name``, built on first use.

        Building it loads the model and encodes every object; a different model replaces it.
        """
        if self._embedder is None or self._embedder.model_name != model_name:
            from schemagraph.linking.embed import EmbeddingActivator

            self._embedder = EmbeddingActivator(self.schema_graph, model_name)
        return self._embedder

    # ---------------------------------------------------------------- ranking
    def _activate(self, question: str, opts: LinkOptions) -> Activation:
        """Lexical activation, plus embedding seeds when ``opts.embed``."""
        activation = activate(
            self.schema_graph,
            self.index,
            question,
            idf=opts.idf,
            min_numeric_len=opts.min_numeric_len,
            ngram_stop=opts.ngram_stop,
        )
        if opts.embed:
            embedding_seeds = self.embedder(opts.embed_model).activate(
                question,
                threshold=opts.embed_threshold,
                top_k=opts.embed_top_k,
                weight=opts.embed_weight,
            )
            for node, weight, why in embedding_seeds:
                activation.bump(node, weight, why)
        return activation

    def _ppr_seeds(
        self,
        activation: Activation,
        question: str,
        opts: LinkOptions,
    ) -> dict[str, float]:
        """The PPR personalization: the activation's seeds, plus top BM25F tables if asked.

        With ``opts.seed_bm25`` the result is a copy; the activation is never changed.
        """
        if not opts.seed_bm25:
            return activation.seeds
        top = _by_score(bm25_scores(self._bm25_index(), question))[: opts.seed_k]
        seeds = dict(activation.seeds)
        for rank, (fqn, _) in enumerate(top):
            seeds[tnode(fqn)] = seeds.get(tnode(fqn), 0.0) + opts.seed_w / (rank + 1)
        return seeds

    def _add_direct_evidence(self, scores: dict[str, float], seeds: dict[str, float]) -> None:
        """Add a bonus for seeds on a table or its columns: PPR dilutes direct evidence.

        The scale is the best table score before any bonus. Mutates ``scores`` in place.
        """
        graph = self.schema_graph.graph
        scale = max(scores.values(), default=1.0) or 1.0
        for node, weight in seeds.items():
            attrs = graph.nodes[node]
            if attrs.get("ntype") == "table":
                bonus = TABLE_SEED_BONUS
            elif attrs.get("ntype") == "column":
                bonus = COLUMN_SEED_BONUS
            else:
                continue
            capped = min(weight, SEED_BONUS_CAP)
            scores[attrs["fqn"]] = scores.get(attrs["fqn"], 0.0) + bonus * scale * capped

    def _rank(self, question: str, opts: LinkOptions) -> _Ranking:
        """Rank tables for a question (docs/DESIGN.md, stages 1-3).

        Activate, run PPR, aggregate to tables, add the direct lexical bonus, optionally route
        by schema, then rank by PPR alone, BM25F alone or their reciprocal-rank fusion.
        """
        activation = self._activate(question, opts)
        seeds = self._ppr_seeds(activation, question, opts)
        node_scores = personalized_pagerank(
            self.schema_graph,
            seeds,
            alpha=opts.ppr_alpha,
            specificity=self._spec,
            matrix=self._matrix(opts.ppr_edge_attr),
        )
        scores = table_scores(self.schema_graph, node_scores, agg=opts.agg)
        self._add_direct_evidence(scores, activation.seeds)
        if opts.schema_routing > 0:
            scores = self._route_by_schema(scores, opts.schema_routing)
        evidence = _relative(scores)
        if opts.ranker in {"bm25", "rrf"}:
            sparse = bm25_scores(self._bm25_index(), question)
            if opts.ranker == "bm25":
                scores = sparse
                evidence = _relative(sparse)
            else:
                scores = _fuse_with_bm25(scores, sparse, evidence, opts.rrf_k)
        return _Ranking(
            activation=activation,
            node_scores=node_scores,
            table_scores=scores,
            ranked=_sorted_positive(scores),
            evidence=evidence,
        )

    # ---------------------------------------------------------------- public
    def link(self, question: str, opts: LinkOptions | None = None) -> LinkResult:
        """Link a question to a join-complete sub-schema (the whole pipeline, docs/DESIGN.md).

        Args:
            question: The natural-language question.
            opts: Pipeline options (defaults when None).

        Returns:
            The linked tables and columns, join paths, anchors, matched terms, stats and,
            with ``opts.render``, the annotated DDL.
        """
        opts = opts or LinkOptions()
        start = time.perf_counter()
        schema_graph = self.schema_graph
        if opts.small_schema_bypass and len(schema_graph.tables) <= opts.small_schema_bypass:
            return self._everything(question, opts, start)
        opts = self._effective_options(opts)

        ranking = self._rank(question, opts)
        activation, ranked = ranking.activation, ranking.ranked
        anchors, sources, destinations = self._pick_anchors(
            question,
            ranked,
            ranking.evidence,
            opts,
        )
        paths, union, join_paths = self._join_paths(anchors, sources, destinations, ranking, opts)
        kept_tables = self._choose_tables(anchors, join_paths, ranking, opts)
        tables = self._select_columns(
            kept_tables,
            anchors,
            join_paths,
            ranking.node_scores,
            activation,
            ranking.table_scores,
            opts,
            rank={fqn: i + 1 for i, (fqn, _) in enumerate(ranked)},
        )
        values_matched = [f"value:{value}" for value in activation.matched_values]
        result = LinkResult(
            question=question,
            tables=tables,
            join_paths=join_paths,
            anchors=anchors,
            terms_matched=activation.matched_terms + values_matched,
            glossary=self._glossary_matches(activation),
            stats={
                "seeds": len(activation.seeds),
                "candidate_paths": len(paths),
                "union_tables": len(union),
                "ranked_tables": len(ranked),
                "ms": _elapsed_ms(start),
                "llm": "yes" if (opts.use_llm and self.llm) else "no",
            },
        )
        if opts.debug:
            limited = ranked[: opts.ranking_limit] if opts.ranking_limit else ranked
            result.ranking = [(fqn, round(score, 6)) for fqn, score in limited]
        if opts.render:
            result.ddl = render_ddl(schema_graph, result)
        return result

    def explain(self, question: str, opts: LinkOptions | None = None) -> dict:
        """Seeds, PPR node scores and the table ranking exactly as ``link`` computes them."""
        opts = self._effective_options(opts or LinkOptions())
        ranking = self._rank(question, opts)
        activation = ranking.activation
        top_seeds = _by_score(activation.seeds)[:EXPLAIN_TOP]
        top_nodes = _by_score(ranking.node_scores)[:EXPLAIN_TOP]
        return {
            "tokens": activation.tokens,
            "seeds": {
                node: {"weight": round(weight, 3), "why": activation.reasons.get(node, [])}
                for node, weight in top_seeds
            },
            "ppr_top": [(node, round(score, 5)) for node, score in top_nodes],
            "tables": [(fqn, round(score, 6)) for fqn, score in ranking.ranked[:EXPLAIN_TABLES]],
        }

    # ---------------------------------------------------------------- internals
    def _effective_options(self, opts: LinkOptions) -> LinkOptions:
        """``opts`` with the large-schema budget applied when ``adaptive_budget`` asks for it."""
        if opts.adaptive_budget and len(self.schema_graph.tables) > opts.large_threshold:
            return replace(
                opts,
                max_tables=max(opts.max_tables, opts.max_tables_large),
                anchor_k=max(opts.anchor_k, opts.anchor_k_large),
            )
        return opts

    def _route_by_schema(self, scores: dict[str, float], routing: float) -> dict[str, float]:
        """Blend each table's score with its dataset's top-3 score sum.

        A dataset whose tables collectively carry the activation mass is more likely the right
        one. ``routing`` in (0, 1] is the blend weight; with a single dataset nothing changes.
        """
        by_schema: dict[str, list[float]] = {}
        for fqn, score in scores.items():
            schema_key = fqn.rsplit(".", 1)[0] if "." in fqn else ""
            by_schema.setdefault(schema_key, []).append(score)
        if len(by_schema) <= 1:
            return scores
        mass = {k: sum(sorted(v, reverse=True)[:3]) for k, v in by_schema.items()}
        best = max(mass.values()) or 1.0
        routed = {}
        for fqn, score in scores.items():
            schema_key = fqn.rsplit(".", 1)[0] if "." in fqn else ""
            routed[fqn] = score * ((1 - routing) + routing * mass[schema_key] / best)
        return routed

    def _pick_anchors(
        self,
        question: str,
        ranked: list[tuple[str, float]],
        evidence: dict[str, float],
        opts: LinkOptions,
    ) -> tuple[list[str], list[str], list[str] | None]:
        """Choose the anchor tables paths are searched between (docs/DESIGN.md, stage 4).

        Candidates are ranked tables with enough relative evidence. With ``use_llm`` and an
        LLM, it picks sources and destinations among them; otherwise (or when it fails or
        picks nothing known) the top ``anchor_k`` candidates are the anchors.

        Args:
            question: The natural-language question (for the LLM).
            ranked: Tables with a positive score, best first.
            evidence: Relative evidence per table.
            opts: The linking options.

        Returns:
            ``(anchors, sources, destinations)``; ``destinations`` is None when paths should
            run between every pair of sources.
        """
        if not ranked:
            return [], [], None
        tables = self.schema_graph.tables
        qualified = [fqn for fqn, _ in ranked if evidence.get(fqn, 0.0) >= opts.min_anchor_ratio]
        n_candidates = max(opts.anchor_k * ANCHOR_CANDIDATE_FACTOR, MIN_ANCHOR_CANDIDATES)
        candidates = qualified[:n_candidates]
        if opts.use_llm and self.llm is not None:
            try:
                sources, destinations = self.llm.anchor_tables(
                    question,
                    [tables[fqn.lower()] for fqn in candidates],
                )
                sources = [fqn for fqn in sources if fqn.lower() in tables]
                destinations = [fqn for fqn in destinations if fqn.lower() in tables]
                if sources or destinations:
                    anchors = list(dict.fromkeys(sources + destinations))
                    return anchors, sources or destinations, destinations or None
            except Exception as e:  # pragma: no cover - LLM failure must not break linking
                self.last_llm_error = str(e)
        anchors = candidates[: opts.anchor_k]
        return anchors, anchors, None

    def _join_paths(
        self,
        anchors: list[str],
        sources: list[str],
        destinations: list[str] | None,
        ranking: _Ranking,
        opts: LinkOptions,
    ) -> tuple[list, set, list[JoinPath]]:
        """Find the shortest paths between anchors and prune them (docs/DESIGN.md, stages 5-6).

        Args:
            anchors: The anchor tables.
            sources: Path sources.
            destinations: Path destinations (None: between every pair of sources).
            ranking: The question's ranking; the evidence of its top :data:`PRIOR_WINDOW`
                tables is the pruning node prior (a magnitude, not the flat fused score).
            opts: The linking options.

        Returns:
            ``(candidate paths, tables on any candidate path, pruned join paths)``.
        """
        if opts.paths:
            paths, union = union_of_shortest_paths(
                self.schema_graph,
                sources,
                destinations,
                max_extra=opts.path_extra,
                cutoff=opts.path_cutoff,
            )
        else:
            paths, union = [], set()
        prior = {
            tnode(fqn): ranking.evidence.get(fqn, 0.0) for fqn, _ in ranking.ranked[:PRIOR_WINDOW]
        }
        join_paths = prune_paths(
            self.schema_graph,
            paths,
            [tnode(anchor) for anchor in anchors],
            alpha=opts.prune_alpha,
            theta=opts.prune_theta,
            top_k=opts.prune_top_k,
            node_prior=prior,
        )
        return paths, union, join_paths

    def _choose_tables(
        self,
        anchors: list[str],
        join_paths: list[JoinPath],
        ranking: _Ranking,
        opts: LinkOptions,
    ) -> set[str]:
        """Anchors and path tables, filled up to the budget (docs/DESIGN.md, stage 7).

        When the whole schema fits ``max_tables`` (and ``bypass_if_fits``), every table is
        kept, so a question with no schema vocabulary never returns no tables. Otherwise the
        next-best ranked tables with at least ``fill_ratio`` evidence fill the budget
        (recall-first, but not noise). The result is cut to the best
        ``max(max_tables, len(anchors))`` by score.
        """
        kept_tables = set(anchors)
        for join_path in join_paths:
            kept_tables.update(join_path.tables)
        if opts.bypass_if_fits and len(self.schema_graph.tables) <= opts.max_tables:
            kept_tables.update(table.fqn for table in self.schema_graph.tables.values())
        else:
            for fqn, _ in ranking.ranked:
                if len(kept_tables) >= opts.max_tables:
                    break
                if ranking.evidence.get(fqn, 0.0) >= opts.fill_ratio:
                    kept_tables.add(fqn)
        scores = ranking.table_scores
        by_score = sorted(kept_tables, key=lambda fqn: (-scores.get(fqn, 0.0), fqn))
        return set(by_score[: max(opts.max_tables, len(anchors))])

    def _glossary_matches(self, activation: Activation) -> dict[str, list[str]]:
        """Each matched glossary phrase -> the fqns (or node ids) its term points at."""
        graph = self.schema_graph.graph
        matches: dict[str, list[str]] = {}
        for term in activation.matched_terms:
            if term not in self.index.phrases:
                continue
            term_node = self.index.phrases[term]
            matches[term] = [
                graph.nodes[neighbor].get("fqn", neighbor)
                for neighbor in graph.neighbors(term_node)
                if graph[term_node][neighbor].get("etype") == "glossary"
            ]
        return matches

    def _select_columns(
        self,
        kept: set[str],
        anchors: list[str],
        join_paths: list[JoinPath],
        node_scores: dict[str, float],
        activation: Activation,
        scores: dict[str, float],
        opts: LinkOptions,
        rank: dict[str, int] | None = None,
    ) -> list[LinkedTable]:
        """Pick the columns of every kept table (docs/DESIGN.md, stage 8).

        Tables come out best first. Each keeps its linked columns (:func:`_keep_reason`) in
        declaration order; tables ranked below ``columns_top_uncapped`` are capped at
        ``max_columns_per_table`` (keys first, then by score). A table left with no column
        gets its first ``max_columns_per_table`` columns.

        Args:
            kept: Fqns of the kept tables.
            anchors: The anchor tables.
            join_paths: The pruned join paths (their key columns are always kept).
            node_scores: PPR score of every node.
            activation: The question's activation.
            scores: Final table scores (table order and ``LinkedTable.score``).
            opts: The linking options.
            rank: 1-based rank of each ranked table.

        Returns:
            The linked tables.
        """
        rank = rank or {}
        join_columns = _join_columns(self.schema_graph, join_paths)
        linked_tables: list[LinkedTable] = []
        for fqn in sorted(kept, key=lambda f: (-scores.get(f, 0.0), f)):
            table = self.schema_graph.table(fqn)
            if table is None:
                continue
            is_anchor = fqn in anchors
            columns = self._linked_columns(
                table,
                join_keys=join_columns.get(table.fqn.lower(), set()),
                is_anchor=is_anchor,
                node_scores=node_scores,
                activation=activation,
                opts=opts,
            )
            uncapped = rank.get(fqn, UNCAPPED) <= opts.columns_top_uncapped
            columns = _cap_columns(columns, UNCAPPED if uncapped else opts.max_columns_per_table)
            columns = _declaration_order(columns, table)
            if not columns and table.columns:
                columns = _plain_columns(table.columns[: opts.max_columns_per_table])
            linked_tables.append(
                LinkedTable(
                    fqn=table.fqn,
                    score=round(scores.get(fqn, 0.0), 6),
                    is_anchor=is_anchor,
                    columns=columns,
                    description=table.description,
                    kind=table.kind,
                )
            )
        return linked_tables

    def _linked_columns(
        self,
        table: Table,
        *,
        join_keys: set[str],
        is_anchor: bool,
        node_scores: dict[str, float],
        activation: Activation,
        opts: LinkOptions,
    ) -> list[LinkedColumn]:
        """Every column of ``table`` that :func:`_keep_reason` keeps, scored, uncapped."""
        columns: list[LinkedColumn] = []
        for column in table.columns:
            column_node = cnode(table.fqn, column.name)
            ppr_score = node_scores.get(column_node, 0.0)
            score = ppr_score * COLUMN_PPR_SCALE + activation.seeds.get(column_node, 0.0)
            keep, reason = _keep_reason(
                table,
                column,
                score,
                join_keys=join_keys,
                is_anchor=is_anchor,
                activation=activation,
                node_scores=node_scores,
                opts=opts,
            )
            if keep:
                columns.append(
                    LinkedColumn(
                        name=column.name,
                        data_type=column.data_type,
                        description=column.description,
                        score=round(score, 4),
                        reason=reason,
                    )
                )
        return columns

    def _everything(self, question: str, opts: LinkOptions, start: float) -> LinkResult:
        """The small-schema bypass: every table with every column, no ranking."""
        tables = [
            LinkedTable(
                fqn=table.fqn,
                score=0.0,
                is_anchor=False,
                columns=_plain_columns(table.columns),
                description=table.description,
                kind=table.kind,
            )
            for table in self.schema_graph.tables.values()
        ]
        result = LinkResult(
            question=question,
            tables=tables,
            join_paths=[],
            anchors=[],
            terms_matched=[],
            stats={"bypass": "small schema", "ms": _elapsed_ms(start)},
        )
        if opts.render:
            result.ddl = render_ddl(self.schema_graph, result)
        return result
