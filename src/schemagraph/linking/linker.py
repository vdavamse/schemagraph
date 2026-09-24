"""The schema-linking pipeline: activate -> PPR -> anchors -> path union -> prune -> columns -> render.

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
from schemagraph.model import JoinPath, LinkedColumn, LinkedTable, LinkResult


@dataclass
class LinkOptions:
    max_tables: int = 20  # 12 -> 20 after the Spider2-Lite loop: +5 strict recall for ~2 extra tables
    anchor_k: int = 6  # top tables by PPR treated as anchors when no LLM pass
    min_anchor_ratio: float = 0.15  # anchors must score >= ratio * best
    fill_ratio: float = 0.06  # non-path tables need >= ratio * best to be included
    bypass_if_fits: bool = True  # if the whole schema fits max_tables, return all of it ("Death of Schema Linking")
    ppr_alpha: float = 0.85
    paths: bool = True  # union of shortest paths between anchors (off = anchors + budget fill only; the ablation for bridge tables)
    path_extra: float = 0.0  # allow paths this much longer than the shortest
    path_cutoff: int = 6
    prune_alpha: float = 0.8
    prune_theta: float = 0.05
    prune_top_k: int = 8
    columns: str = "relevant"  # "relevant" | "all"
    max_columns_per_table: int = 40  # column cap for tables ranked below columns_top_uncapped (60 -> 40 on 2026-09-17: same col_strict, fewer tokens)
    columns_top_uncapped: int = 1  # tables at rank <= this keep every column: Lite col_strict 81.2 -> 91.2 at -0.5 % tokens (the cap was the whole column-level miss; table rank, not column evidence, predicts gold columns; top-3 gives 94.6 at +19 % tokens)
    small_schema_bypass: int = 0  # if > 0 and the graph has <= N tables, return everything
    use_llm: bool = False
    render: bool = True
    debug: bool = False  # include the table ranking in the result
    ranking_limit: int = 60  # with debug: keep this many ranked tables (0 = all; the benchmark uses 0 so gold rank is never ambiguous)
    idf: bool = True  # scale single-token evidence by rarity across the schema
    agg: str = "top3"  # column->table aggregation: "top3" | "sum"
    min_numeric_len: int = 4  # ignore digit-only tokens shorter than this
    adaptive_budget: bool = True  # widen max_tables / anchor_k on large schemas
    large_threshold: int = 30  # a schema with more tables than this is "large"
    max_tables_large: int = 20
    anchor_k_large: int = 6
    schema_routing: float = 0.0  # 0 = off (hurt on Spider2-Lite: -6 strict on large schemas); blend table score with its dataset's activation mass (DBCopilot-style)
    ngram_stop: bool = True  # n-gram name matches survive stopwords inside names ("date of birth" -> date_of_birth)
    ppr_edge_attr: str = "weight"  # edge attribute PPR reads as transition mass: "weight" (join cost, historical) | "affinity" (PPR_AFFINITY by kind)
    ranker: str = "rrf"  # "rrf" (reciprocal-rank fusion of the PPR ranking with BM25F over one document per table; Lite 95.85 strict, +3.7 strict@7) | "ppr" (activation + PPR alone, 95.66) | "bm25" (BM25F alone, 94.91)
    rrf_k: int = 60  # reciprocal-rank fusion constant: score = sum 1/(k + rank); smaller k weights the head of each ranking more
    seed_bm25: bool = False  # SPRIG seed-side fusion: top seed_k BM25F tables also seed PPR (personalization only). Rejected 2026-09-23: +1-2 strict, -0.3 strict@7 with embed
    seed_k: int = 5  # BM25F tables used as PPR seeds (SPRIG: 5-10)
    seed_w: float = 1.0  # BM25F seed weight at rank r (0-based) = seed_w / (r + 1); a name hit is 1.0
    embed: bool = False  # add paraphrase seeds from a small static embedding model (linking/embed.py; needs the `embed` extra)
    embed_model: str = "minishlab/potion-base-8M"
    embed_threshold: float = 0.5  # cosine floor for an embedding seed
    embed_top_k: int = 5  # objects considered per question phrase
    embed_weight: float = 0.8  # seed weight = embed_weight * cosine (a description hit is 0.35, a name hit 1.0)


class Linker:
    def __init__(self, schema_graph: SchemaGraph, index: LexicalIndex | None = None, llm=None):
        self.schema_graph = schema_graph
        self.index = index or build_index(schema_graph)
        self.llm = llm  # object with .anchor_tables(question, candidates) -> (sources, destinations)
        self._spec = specificity_weights(schema_graph)
        self._matrices: dict[str, PPRMatrix] = {}  # per edge attribute; the graph must not change after the index is built
        self._bm25: BM25Index | None = None
        self._embedder = None  # EmbeddingActivator, built on first use with embed=True

    def _matrix(self, edge_attr: str) -> PPRMatrix:
        m = self._matrices.get(edge_attr)
        if m is None:
            m = self._matrices[edge_attr] = PPRMatrix(self.schema_graph, edge_attr)
        return m

    def embedder(self, model_name: str):
        """The embedding activator for ``model_name``, built (model load + encoding every object) on first use."""
        if self._embedder is None or self._embedder.model_name != model_name:
            from schemagraph.linking.embed import EmbeddingActivator

            self._embedder = EmbeddingActivator(self.schema_graph, model_name)
        return self._embedder

    def _rank(self, question: str, opts: LinkOptions) -> tuple[Activation, dict[str, float], dict[str, float], list[tuple[str, float]], dict[str, float]]:
        """Activate, run PPR, aggregate to tables, add the direct lexical bonus, fuse with BM25F if asked.

        Returns (activation, node scores, table scores, ranking, evidence). ``evidence`` is each
        table's score relative to the best table under each ranker (max across rankers, in [0, 1]);
        the anchor and fill gates read it instead of the fused score, whose rank-based scale is flat.
        """
        schema_graph = self.schema_graph
        act = activate(schema_graph, self.index, question, idf=opts.idf, min_numeric_len=opts.min_numeric_len, ngram_stop=opts.ngram_stop)
        if opts.embed:
            for node, w, why in self.embedder(opts.embed_model).activate(question, threshold=opts.embed_threshold, top_k=opts.embed_top_k, weight=opts.embed_weight):
                act.bump(node, w, why)
        seeds = act.seeds
        if opts.seed_bm25:
            if self._bm25 is None:
                self._bm25 = build_bm25(schema_graph)
            top = sorted(bm25_scores(self._bm25, question).items(), key=lambda x: (-x[1], x[0]))[: opts.seed_k]
            seeds = dict(act.seeds)
            for r, (fqn, _) in enumerate(top):
                seeds[tnode(fqn)] = seeds.get(tnode(fqn), 0.0) + opts.seed_w / (r + 1)
        node_scores = personalized_pagerank(schema_graph, seeds, alpha=opts.ppr_alpha, specificity=self._spec, matrix=self._matrix(opts.ppr_edge_attr))
        tscores = table_scores(schema_graph, node_scores, agg=opts.agg)
        # direct lexical evidence on the table itself counts extra (PPR dilutes it)
        scale = max(tscores.values(), default=1.0) or 1.0
        for n, w in act.seeds.items():
            d = schema_graph.graph.nodes[n]
            if d.get("ntype") == "table":
                tscores[d["fqn"]] = tscores.get(d["fqn"], 0.0) + 0.35 * scale * min(w, 2.0)
            elif d.get("ntype") == "column":
                tscores[d["fqn"]] = tscores.get(d["fqn"], 0.0) + 0.10 * scale * min(w, 2.0)
        if opts.schema_routing > 0:
            tscores = self._route_by_schema(tscores, opts.schema_routing)
        best = max(tscores.values(), default=0.0) or 1.0
        evidence = {f: s / best for f, s in tscores.items()}
        if opts.ranker in {"bm25", "rrf"}:
            if self._bm25 is None:
                self._bm25 = build_bm25(schema_graph)
            sparse = bm25_scores(self._bm25, question)
            sbest = max(sparse.values(), default=0.0) or 1.0
            if opts.ranker == "bm25":
                tscores = sparse
                evidence = {f: s / sbest for f, s in sparse.items()}
            else:
                ppr_rank = [f for f, s in sorted(tscores.items(), key=lambda x: (-x[1], x[0])) if s > 0]
                sparse_rank = [f for f, s in sorted(sparse.items(), key=lambda x: (-x[1], x[0]))]
                tscores = reciprocal_rank_fusion(ppr_rank, sparse_rank, k=opts.rrf_k)
                for f, s in sparse.items():
                    evidence[f] = max(evidence.get(f, 0.0), s / sbest)
        ranked = sorted(tscores.items(), key=lambda x: (-x[1], x[0]))  # name breaks ties so runs are reproducible
        ranked = [(f, s) for f, s in ranked if s > 0]
        return act, node_scores, tscores, ranked, evidence

    # ---------------------------------------------------------------- public
    def link(self, question: str, opts: LinkOptions | None = None) -> LinkResult:
        opts = opts or LinkOptions()
        t0 = time.perf_counter()
        schema_graph = self.schema_graph
        if opts.small_schema_bypass and len(schema_graph.tables) <= opts.small_schema_bypass:
            return self._everything(question, opts, t0)
        if opts.adaptive_budget and len(schema_graph.tables) > opts.large_threshold:
            opts = replace(opts, max_tables=max(opts.max_tables, opts.max_tables_large), anchor_k=max(opts.anchor_k, opts.anchor_k_large))

        act, node_scores, tscores, ranked, evidence = self._rank(question, opts)

        anchors, sources, destinations = self._pick_anchors(question, ranked, evidence, opts)
        paths, union = union_of_shortest_paths(schema_graph, sources, destinations, max_extra=opts.path_extra, cutoff=opts.path_cutoff) if opts.paths else ([], set())
        prior = {tnode(f): evidence.get(f, 0.0) for f, _ in ranked[:50]}  # a magnitude: evidence, not the flat fused score
        join_paths = prune_paths(schema_graph, paths, [tnode(a) for a in anchors], alpha=opts.prune_alpha, theta=opts.prune_theta, top_k=opts.prune_top_k, node_prior=prior)
        kept_tables = set(anchors)
        for jp in join_paths:
            kept_tables.update(jp.tables)
        if opts.bypass_if_fits and len(schema_graph.tables) <= opts.max_tables:
            # the whole schema fits the budget: never risk missing a table, even when the
            # question activated nothing (a question with no schema vocabulary must not return no tables)
            kept_tables.update(t.fqn for t in schema_graph.tables.values())
        else:
            # fill remaining budget with next-best ranked tables that carry some evidence (recall-first, but not noise)
            for f, _s in ranked:
                if len(kept_tables) >= opts.max_tables:
                    break
                if evidence.get(f, 0.0) >= opts.fill_ratio:
                    kept_tables.add(f)
        kept_tables = set(sorted(kept_tables, key=lambda f: (-tscores.get(f, 0.0), f))[: max(opts.max_tables, len(anchors))])

        tables = self._select_columns(kept_tables, anchors, join_paths, node_scores, act, tscores, opts, rank={f: i + 1 for i, (f, _) in enumerate(ranked)})
        glossary = {term: [schema_graph.graph.nodes[m].get("fqn", m) for m in schema_graph.graph.neighbors(self.index.phrases[term]) if schema_graph.graph[self.index.phrases[term]][m].get("etype") == "glossary"] for term in act.matched_terms if term in self.index.phrases}
        result = LinkResult(
            question=question,
            tables=tables,
            join_paths=join_paths,
            anchors=anchors,
            terms_matched=act.matched_terms + [f"value:{v}" for v in act.matched_values],
            glossary=glossary,
            stats={
                "seeds": len(act.seeds),
                "candidate_paths": len(paths),
                "union_tables": len(union),
                "ranked_tables": len(ranked),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "llm": "yes" if (opts.use_llm and self.llm) else "no",
            },
        )
        if opts.debug:
            result.ranking = [(f, round(s, 6)) for f, s in (ranked[: opts.ranking_limit] if opts.ranking_limit else ranked)]
        if opts.render:
            result.ddl = render_ddl(schema_graph, result)
        return result

    def explain(self, question: str, opts: LinkOptions | None = None) -> dict:
        """Seeds, PPR node scores and the table ranking exactly as ``link`` computes them."""
        opts = opts or LinkOptions()
        if opts.adaptive_budget and len(self.schema_graph.tables) > opts.large_threshold:
            opts = replace(opts, max_tables=max(opts.max_tables, opts.max_tables_large), anchor_k=max(opts.anchor_k, opts.anchor_k_large))
        act, node_scores, _tscores, ranked, _evidence = self._rank(question, opts)
        top = sorted(node_scores.items(), key=lambda x: (-x[1], x[0]))[:40]
        return {
            "tokens": act.tokens,
            "seeds": {n: {"weight": round(w, 3), "why": act.reasons.get(n, [])} for n, w in sorted(act.seeds.items(), key=lambda x: (-x[1], x[0]))[:40]},
            "ppr_top": [(n, round(s, 5)) for n, s in top],
            "tables": [(f, round(s, 6)) for f, s in ranked[:20]],
        }

    # ---------------------------------------------------------------- internals
    def _route_by_schema(self, tscores: dict[str, float], r: float) -> dict[str, float]:
        """Schema routing: a dataset whose tables collectively carry the activation mass is
        more likely the right one; blend each table's score with its dataset's top-3 sum."""
        by_schema: dict[str, list[float]] = {}
        for f, s in tscores.items():
            key = f.rsplit(".", 1)[0] if "." in f else ""
            by_schema.setdefault(key, []).append(s)
        if len(by_schema) <= 1:
            return tscores
        mass = {k: sum(sorted(v, reverse=True)[:3]) for k, v in by_schema.items()}
        best = max(mass.values()) or 1.0
        out = {}
        for f, s in tscores.items():
            key = f.rsplit(".", 1)[0] if "." in f else ""
            out[f] = s * ((1 - r) + r * mass[key] / best)
        return out

    def _pick_anchors(self, question: str, ranked: list[tuple[str, float]], evidence: dict[str, float], opts: LinkOptions) -> tuple[list[str], list[str], list[str] | None]:
        if not ranked:
            return [], [], None
        cands = [f for f, _s in ranked if evidence.get(f, 0.0) >= opts.min_anchor_ratio][: max(opts.anchor_k * 3, 8)]
        if opts.use_llm and self.llm is not None:
            try:
                sources, destinations = self.llm.anchor_tables(question, [self.schema_graph.tables[f.lower()] for f in cands])
                sources = [f for f in sources if f.lower() in self.schema_graph.tables]
                destinations = [f for f in destinations if f.lower() in self.schema_graph.tables]
                if sources or destinations:
                    anchors = list(dict.fromkeys(sources + destinations))
                    return anchors, sources or destinations, destinations or None
            except Exception as e:  # pragma: no cover - LLM failure must not break linking
                self.last_llm_error = str(e)
        anchors = cands[: opts.anchor_k]
        return anchors, anchors, None

    def _select_columns(self, kept: set[str], anchors: list[str], join_paths: list[JoinPath], node_scores: dict[str, float], act: Activation, tscores: dict[str, float], opts: LinkOptions, rank: dict[str, int] | None = None) -> list[LinkedTable]:
        schema_graph = self.schema_graph
        rank = rank or {}
        join_cols: dict[str, set[str]] = {}
        for jp in join_paths:
            for step in jp.steps:
                rels = schema_graph.relations(step.from_table, step.to_table)
                for r in rels:
                    join_cols.setdefault(r.from_table.lower(), set()).update(c.lower() for c in r.from_columns)
                    join_cols.setdefault(r.to_table.lower(), set()).update(c.lower() for c in r.to_columns)
        out: list[LinkedTable] = []
        for fqn in sorted(kept, key=lambda f: (-tscores.get(f, 0.0), f)):
            t = schema_graph.table(fqn)
            if t is None:
                continue
            is_anchor = fqn in anchors
            cols: list[LinkedColumn] = []
            for c in t.columns:
                n = cnode(t.fqn, c.name)
                s = node_scores.get(n, 0.0) * 1000 + act.seeds.get(n, 0.0)
                reason = None
                keep = False
                if c.is_primary_key or c.name in t.primary_key:
                    keep, reason = True, "primary key"
                elif c.name.lower() in join_cols.get(t.fqn.lower(), set()):
                    keep, reason = True, "join key"
                elif n in act.seeds:
                    keep, reason = True, "; ".join(act.reasons.get(n, [])[:2])
                elif opts.columns == "all" or is_anchor or not act.seeds:
                    # no lexical evidence at all: there is nothing to select by, keep every column
                    keep = True
                elif s > 0 and node_scores.get(n, 0.0) > 0:
                    keep = True
                if keep:
                    cols.append(LinkedColumn(name=c.name, data_type=c.data_type, description=c.description, score=round(s, 4), reason=reason))
            # cap width: keep keys + top scored; the best-ranked tables keep everything
            cap = 10**9 if rank.get(fqn, 10**9) <= opts.columns_top_uncapped else opts.max_columns_per_table
            if len(cols) > cap:
                keys = [c for c in cols if c.reason in {"primary key", "join key"}]
                rest = sorted([c for c in cols if c.reason not in {"primary key", "join key"}], key=lambda c: -c.score)
                cols = keys + rest[: max(0, cap - len(keys))]
            # preserve declaration order
            order = {c.name: i for i, c in enumerate(t.columns)}
            cols.sort(key=lambda c: order.get(c.name, 0))
            if not cols and t.columns:
                cols = [LinkedColumn(name=c.name, data_type=c.data_type, description=c.description) for c in t.columns[: opts.max_columns_per_table]]
            out.append(LinkedTable(fqn=t.fqn, score=round(tscores.get(fqn, 0.0), 6), is_anchor=is_anchor, columns=cols, description=t.description, kind=t.kind))
        return out

    def _everything(self, question: str, opts: LinkOptions, t0: float) -> LinkResult:
        tables = [
            LinkedTable(fqn=t.fqn, score=0.0, is_anchor=False, columns=[LinkedColumn(name=c.name, data_type=c.data_type, description=c.description) for c in t.columns], description=t.description, kind=t.kind)
            for t in self.schema_graph.tables.values()
        ]
        res = LinkResult(question=question, tables=tables, join_paths=[], anchors=[], terms_matched=[], stats={"bypass": "small schema", "ms": round((time.perf_counter() - t0) * 1000, 1)})
        if opts.render:
            res.ddl = render_ddl(self.schema_graph, res)
        return res
