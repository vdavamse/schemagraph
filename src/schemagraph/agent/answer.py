"""The answer loop: link, generate, guard and execute, check, judge, score; then the final pick.

:mod:`schemagraph.agent.search` decides which nodes to generate.

The schema comes from a schemagraph MCP server: the generator calls its tools, and the
orchestrator reads it through a :class:`~schemagraph.agent.schema_client.SchemaClient`. Blocking
work (execution, catalog counts) runs in worker threads, so a batch's nodes run concurrently; the
server's tools link in worker threads, so the Engine lock is never held across an ``await``.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field, replace
from itertools import permutations
from typing import Any

from sqlglot import exp

from schemagraph.agent import agents, prompts
from schemagraph.agent.checks import det_of, join_keys, result_checks, static_checks, tables_read
from schemagraph.agent.execute import Executor
from schemagraph.agent.guard import GuardedSQL, GuardError, guard_sql
from schemagraph.agent.models import AgentModels, reasoning_level, reasons
from schemagraph.agent.results import (
    Action,
    AgentConfig,
    AnswerResult,
    Candidate,
    CheckReport,
    ExecResult,
    Finding,
    Judgement,
    RubricBase,
    SqlCandidate,
    Transcript,
    UsageRecord,
    UsageSummary,
    rubric_type,
    stored_prompt,
)
from schemagraph.agent.schema_client import LinkedSchema, SchemaClient
from schemagraph.agent.score import combine, feedback
from schemagraph.agent.search import new_node, run_search, select_final, split_action
from schemagraph.model import Edge

# Output tokens of one generator response.
GEN_MAX_TOKENS = 4096
# Timeout of one generator or critic request, in seconds.
GEN_TIMEOUT_S = 120.0
# Timeout of one judge or selector request, in seconds.
JUDGE_TIMEOUT_S = 30.0
# Output tokens of the critic's advice.
CRITIC_MAX_TOKENS = 800
# Extra output tokens and request timeout of a role whose model reasons: reasoning counts
# against ``max_tokens``, and a reasoning request takes longer (the node timeout still bounds
# the whole node, ``AgentConfig.node_timeout_s``). The extra tokens are at least this many.
REASONING_MAX_TOKENS = 8192
REASONING_TIMEOUT_S = 180.0
# Percent of ``max_tokens`` OpenRouter gives to thinking at each reasoning effort, for models that
# take a token budget (Gemini, Anthropic; openrouter.ai/docs, reasoning tokens). At ``high`` a
# flat 8192 extra left Gemini about 2450 tokens to answer in, and it often answered nothing;
# the extra tokens are sized so the answer keeps its own cap at every effort.
REASONING_BUDGET_PERCENT = {"low": 20, "medium": 50, "high": 80}
# Timeout of counting one join key's rows and distinct values for the judge, in seconds.
KEY_COUNT_TIMEOUT_S = 10.0
# Join keys of tables with more rows than this (or an unknown count) are not measured for the
# judge: the count is a full-table hash aggregate per key, and the judge then just lacks that
# statistic. The largest Spider2 local table has 540,800 rows.
KEY_STATS_MAX_ROWS = 10_000_000
# Budget of all the row and key counts behind one judge call's schema, in seconds; past it the
# judge sees the tables without counts, so a slow database cannot time the whole node out.
JUDGE_COUNTS_TIMEOUT_S = 30.0
# Usage roles whose records are attached to the node they ran for.
_NODE_ROLES = frozenset({"generator", "judge"})


def reasoning_headroom(answer_tokens: int) -> int:
    """Return the extra output tokens a reasoning model gets on top of its answer's cap.

    Sized so the answer keeps ``answer_tokens`` after OpenRouter's thinking share
    (:data:`REASONING_BUDGET_PERCENT` at :func:`reasoning_level`), and never below
    :data:`REASONING_MAX_TOKENS`.
    """
    percent = REASONING_BUDGET_PERCENT[reasoning_level()]
    return max(REASONING_MAX_TOKENS, math.ceil(answer_tokens * percent / (100 - percent)))


@dataclass
class _JoinLookups:
    """Pre-fetched join lookups between catalog keys, for the synchronous checks.

    Attributes:
        relations: (key a, key b) -> join-capable relations between the two tables.
        paths: (key a, key b) -> shortest join paths, for pairs with no direct relation.
    """

    relations: dict[tuple[str, str], list[Edge]] = field(default_factory=dict)
    paths: dict[tuple[str, str], list[list[str]]] = field(default_factory=dict)

    def join_relations(self, a: str, b: str) -> list[Edge]:
        """Return the relations between two catalog keys (none when not fetched)."""
        return self.relations.get((a, b), [])

    def join_path(self, a: str, b: str) -> list[list[str]]:
        """Return the join paths from ``a`` to ``b`` (none when not fetched)."""
        return self.paths.get((a, b), [])


class Answerer:
    """Answers one question at a time over an MCP schema server and a read-only executor.

    An Answerer can be reused: :meth:`prepare` resets the per-question state.

    Attributes:
        schema: The orchestrator's schema lookups; its ``mcp_url`` also serves the generator.
        executor: Where queries run.
        cfg: The answer's settings.
        models: The four roles' models.
        records: Every model call for the current question, including failed and cancelled
            ones.
        question: The question being answered (set by :meth:`prepare`).
        evidence: Its external knowledge, if any.
    """

    def __init__(
        self,
        schema: SchemaClient,
        executor: Executor,
        cfg: AgentConfig | None = None,
        models: AgentModels | None = None,
    ):
        self.schema = schema
        self.executor = executor
        self.cfg = cfg or AgentConfig()
        self.models = models or AgentModels.resolve(self.cfg)
        self.records: list[UsageRecord] = []
        self.transcripts: list[Transcript] | None = [] if self.cfg.trace else None
        self._key_counts: dict[tuple[str, str], tuple[int, int] | None] = {}
        self.question = ""
        self.evidence: str | None = None
        self._toolset = agents.schema_toolset(schema.mcp_url)
        self._link_text = ""
        self._wide = LinkedSchema(ddl="", tables=())
        self._advice_locks: dict[str, asyncio.Lock] = {}
        self._refinements: dict[str, list[Candidate]] = {}  # parent id -> its finished refinements
        self._contexts: dict[str, str] = {}  # context key -> the DDL the prompts showed
        self._prompts: dict[str, tuple[str, str]] = {}  # node id -> (stored prompt, context key)

    # ----------------------------------------------------------- entry points
    async def prepare(self, question: str, *, evidence: str | None = None) -> None:
        """Set the question, reset the per-question state and link its wide context.

        :meth:`answer` calls this; the judge study calls it directly before re-judging stored
        candidates.
        """
        self.records = []
        self.transcripts = [] if self.cfg.trace else None
        self._advice_locks = {}
        self._refinements = {}
        self._contexts = {}
        self._prompts = {}
        self.question = question
        self.evidence = evidence
        self._link_text = question
        if evidence:
            self._link_text += f"\n\n{evidence[: self.cfg.evidence_chars]}"
        self._wide = await self._link("wide")

    async def answer(self, question: str, *, evidence: str | None = None) -> AnswerResult:
        """Search for SQL that answers ``question`` and pick the final query.

        One MCP session serves the orchestrator and one serves every generator run, both open
        for the whole answer.
        """
        started = time.perf_counter()
        async with self.schema, self._toolset:
            await self.prepare(question, evidence=evidence)
            trace = await run_search(self.generate, self.search_config())
            self._restore_prompts(trace.candidates)
            pick = self._pick if self.cfg.selector else None
            chosen, chosen_by, matrix = await select_final(trace.candidates, self.cfg, pick)
        return AnswerResult(
            question=question,
            sql=chosen.sql if chosen and chosen.sql else None,
            result=chosen.exec if chosen else None,
            chosen_id=chosen.id if chosen else None,
            chosen_by=chosen_by,  # type: ignore[arg-type]
            score=chosen.score if chosen else 0.0,
            strategy=self.cfg.strategy,
            budget=1 if self.cfg.strategy == "single" else self.cfg.budget,
            nodes=trace.nodes,
            stopped_early=trace.stopped_early,
            candidates=trace.candidates,
            selector_matrix=matrix,
            usage=UsageSummary.of(self.records),
            transcripts=self.transcripts or [],
            linked_tables=list(self._wide.tables),
            models=self.models.names,
            ms=(time.perf_counter() - started) * 1000,
            contexts=dict(self._contexts),
            instructions=prompts.generator_instructions(
                self.executor.dialect, self.cfg.probe_limit
            ),
        )

    def _limits(
        self, role: str, max_tokens: int | None, timeout: float, *, model_name: str | None = None
    ) -> dict[str, Any]:
        """Return ``role``'s output-token cap and request timeout, widened when its model reasons.

        Args:
            role: The role whose model the call uses.
            max_tokens: The output-token cap without reasoning; None leaves it to the provider.
            timeout: The request timeout in seconds without reasoning.
            model_name: The model, when the role has several (the generators).

        Returns:
            The ``timeout`` and, when capped, ``max_tokens`` model settings.
        """
        reasoning = reasons(model_name) if model_name else self.models.reasoning(role)
        if reasoning:
            max_tokens = (max_tokens or 0) + reasoning_headroom(max_tokens or 0)
            timeout = max(timeout, REASONING_TIMEOUT_S)
        limits: dict[str, Any] = {"timeout": timeout}
        if max_tokens is not None:
            limits["max_tokens"] = max_tokens
        return limits

    async def _link(self, action: Action) -> LinkedSchema:
        """Link the question for a context action; ``tight`` keeps its small budget as is."""
        tight = action == "tight"
        return await self.schema.link(
            self._link_text,
            max_tables=self.cfg.tight_tables if tight else self.cfg.wide_tables,
            adaptive_budget=not tight,
        )

    # ----------------------------------------------------------- one node
    def search_config(self) -> AgentConfig:
        """Return the search settings, with the longer node timeout when the generator reasons.

        A refinement at high reasoning is a critic call and a multi-request generator run; under
        the plain node timeout it was cut off, and the tree read the score-0 node as "refining
        does not pay".
        """
        reasoning_timeout = self.cfg.reasoning_node_timeout_s
        if reasoning_timeout is None or not self.models.reasoning("generator"):
            return self.cfg
        return replace(self.cfg, node_timeout_s=max(self.cfg.node_timeout_s, reasoning_timeout))

    async def generate(self, node_id: str, parent: Candidate | None, action: str) -> Candidate:
        """Generate and score one search node: a draft, or a refinement of ``parent``.

        ``action`` is the context width, or the generator model when several are searched.
        """
        started = time.perf_counter()
        generator, context = split_action(self.cfg, action)
        linked = await self._link(context)
        if parent is not None:
            await self._ensure_advice(parent)
        candidate = new_node(
            node_id, parent, context, generator=generator, linked_tables=list(linked.tables)
        )
        try:
            prompt = self._generator_prompt(parent, context, linked)
            candidate.prompt, candidate.context_key = stored_prompt(prompt, linked.ddl)
            self._contexts[candidate.context_key] = linked.ddl.strip()
            self._prompts[node_id] = (candidate.prompt, candidate.context_key)
            output = await self._write_sql(node_id, prompt, parent, generator)
        except Exception as error:
            candidate.error = f"{type(error).__name__}: {error}"[: agents.FAILURE_CHARS]
            candidate.feedback = [f"generation failed: {candidate.error}"]
        else:
            candidate.sql = output.sql
            candidate.rationale = output.rationale
            candidate.tables_used = output.tables_used
            await self.score(candidate)
        candidate.usage = [
            record
            for record in self.records
            if record.node_id == node_id and record.role in _NODE_ROLES
        ]
        candidate.ms = (time.perf_counter() - started) * 1000
        if parent is not None:
            self._refinements.setdefault(parent.id, []).append(candidate)
        return candidate

    def _restore_prompts(self, candidates: list[Candidate]) -> None:
        """Give each node that timed out the prompt it was sent, in place.

        The search replaces a node that ran past ``node_timeout_s`` with an error node, which
        drops the prompt :meth:`generate` had already recorded.
        """
        for candidate in candidates:
            if not candidate.prompt and candidate.id in self._prompts:
                candidate.prompt, candidate.context_key = self._prompts[candidate.id]

    def _generator_prompt(
        self, parent: Candidate | None, context: Action, linked: LinkedSchema
    ) -> str:
        """Build the generator's prompt for a draft, or a refinement of ``parent``.

        Args:
            parent: The node to refine; None for a fresh draft.
            context: The width of the linked context, named in the prompt.
            linked: The schema context the generator sees.

        Returns:
            The prompt, schema DDL included.
        """
        # earlier refinements of the same parent, so this one does not repeat them
        siblings = list(self._refinements.get(parent.id, [])) if parent else None
        return prompts.generator_prompt(
            self.question,
            self.evidence,
            context,
            linked.ddl,
            len(linked.tables),
            parent,
            evidence_chars=self.cfg.evidence_chars,
            siblings=siblings,
        )

    async def _write_sql(
        self,
        node_id: str,
        prompt: str,
        parent: Candidate | None,
        generator: str | None,
    ) -> SqlCandidate:
        """Run the generator for one node.

        Args:
            node_id: The node's id, for usage records and traces.
            prompt: The generator's prompt (:meth:`_generator_prompt`).
            parent: The node to refine; None for a fresh draft (it sets the temperature).
            generator: The generator model; None is the default one.

        Returns:
            The generator's query, rationale and tables.

        Raises:
            Exception: The generator failed; the node records the error.
        """
        model, model_name = self.models.generator(generator)
        temperature = self.cfg.refine_temperature if parent else self.cfg.draft_temperature
        output, _, _ = await agents.run_agent(
            "generator",
            agents.generator(),
            prompt,
            model=model,
            model_name=model_name,
            node_id=node_id,
            sink=self.records,
            transcripts=self.transcripts,
            deps=agents.AgentDeps(self.executor, self.cfg, probes_left=self.cfg.probe_limit),
            toolsets=[self._toolset],
            model_settings={
                "temperature": temperature,
                **self._limits(
                    "generator", GEN_MAX_TOKENS, GEN_TIMEOUT_S, model_name=model_name
                ),
            },
            usage_limits=agents.generator_limits(self.cfg.output_retries),
            retries={"tools": 1, "output": self.cfg.output_retries},
        )
        return output

    async def score(self, candidate: Candidate) -> Candidate:
        """Guard, execute, check, judge and combine. Mutates and returns ``candidate``."""
        try:
            guarded = guard_sql(candidate.sql, self.executor.dialect)
        except GuardError as error:
            finding = Finding(code="guard", severity="error", message=str(error))
            candidate.checks = CheckReport(parsed=False, findings=[finding], det=0.0)
            candidate.score, candidate.score_parts = combine(
                candidate.checks, None, None, self.cfg.weights
            )
            candidate.feedback = feedback(candidate.checks, None)
            return candidate
        candidate.exec = await asyncio.to_thread(
            self.executor.execute,
            guarded.sql,
            limit=self.cfg.exec_limit,
            timeout_s=self.cfg.exec_timeout_s,
            count_cap=self.cfg.count_cap,
            cost_gate=self.cfg.cost_gate,
        )
        candidate.checks = await self._check(guarded, candidate.exec)
        if self.cfg.judge and candidate.exec.ok:
            candidate.judgement = await self.judge(candidate)
        candidate.score, candidate.score_parts = combine(
            candidate.checks, candidate.exec, candidate.judgement, self.cfg.weights
        )
        candidate.feedback = feedback(candidate.checks, candidate.judgement)
        return candidate

    async def _check(self, guarded: GuardedSQL, result: ExecResult) -> CheckReport:
        """Run the static checks and the checks on the result."""
        catalog = await asyncio.to_thread(self.executor.catalog)
        lookups = await self._join_lookups(guarded)
        report = static_checks(
            guarded,
            catalog,
            self.executor.resolve_table,
            lookups.join_relations,
            lookups.join_path,
        )
        base_rows = await asyncio.to_thread(self._row_counts, report.tables) if result.ok else {}
        report.findings += result_checks(
            result, base_rows, count_cap=self.cfg.count_cap, guarded=guarded
        )
        report.det = det_of(report.findings)
        return report

    async def _join_lookups(self, guarded: GuardedSQL) -> _JoinLookups:
        """Fetch the join relations and paths between every two tables the query reads.

        The checks see executor catalog keys (bare ``orders`` on SQLite), so each key is first
        resolved to its graph FQN. Paths are fetched only for pairs with no direct relation,
        the only ones the join check suggests a path for.
        """
        keys = await asyncio.to_thread(tables_read, guarded, self.executor.resolve_table)
        fqns = await asyncio.gather(*(self.schema.resolve(key) for key in keys))
        fqn_of = {key: fqn for key, fqn in zip(keys, fqns, strict=True) if fqn}
        pairs = [(a, b) for a, b in permutations(fqn_of, 2)]
        relations = await asyncio.gather(
            *(self.schema.join_relations(fqn_of[a], fqn_of[b]) for a, b in pairs)
        )
        lookups = _JoinLookups(relations=dict(zip(pairs, relations, strict=True)))
        unrelated = [pair for pair, found in lookups.relations.items() if not found]
        paths = await asyncio.gather(
            *(self.schema.join_path(fqn_of[a], fqn_of[b]) for a, b in unrelated)
        )
        lookups.paths = dict(zip(unrelated, paths, strict=True))
        return lookups

    def _row_counts(self, tables: list[str]) -> dict[str, int | None]:
        """Row counts of the tables a query reads (blocking; the executor memoises them)."""
        return {table: self.executor.row_count(table) for table in tables}

    # ----------------------------------------------------------- judge, critic, selector
    async def missing_options(self, candidate: Candidate) -> tuple[str, ...]:
        """Return the tables the judge may call missing.

        They are the wide link plus the neighbours of the tables used, minus the tables used,
        in link-rank order.
        """
        read = candidate.checks.tables if candidate.checks else []
        resolved = await asyncio.gather(*(self.schema.resolve(table) for table in read))
        used = {fqn for fqn in resolved if fqn}
        options = list(self._wide.tables)
        for table in sorted(used):
            options += await self.schema.neighbours(table)
        return tuple(dict.fromkeys(option for option in options if option not in used))

    async def _judge_schema(self, candidate: Candidate) -> str:
        """Describe the tables ``candidate`` reads, their row counts and its join keys.

        The counts run within :data:`JUDGE_COUNTS_TIMEOUT_S`; past it they are left out.
        """
        read = candidate.checks.tables if candidate.checks else []
        fetched = await asyncio.gather(*map(self.schema.table, read))
        details = [detail for detail in fetched if detail]
        unique = list({detail["fqn"]: detail for detail in details}.values())
        try:
            counts, measured = await asyncio.wait_for(
                self._judge_counts(candidate, [detail["fqn"] for detail in unique]),
                JUDGE_COUNTS_TIMEOUT_S,
            )
        except TimeoutError:  # the count thread runs on; what it finishes is memoised
            counts, measured = {}, []
        return prompts.judge_schema(unique, counts, measured)

    async def _judge_counts(
        self, candidate: Candidate, tables: list[str]
    ) -> tuple[dict[str, int | None], list[tuple[str, str, str, str, int, int, int, int]]]:
        """Row counts of ``tables`` and the measured join keys of ``candidate``'s query."""
        counts = await asyncio.to_thread(self._row_counts, tables)
        try:
            guarded = guard_sql(candidate.sql, self.executor.dialect)
        except GuardError:
            return counts, []
        joins = await asyncio.to_thread(join_keys, guarded, self.executor.resolve_table)
        return counts, await asyncio.to_thread(self._join_key_stats, joins)

    def _join_key_stats(
        self, joins: list[tuple[str, str, str, str]]
    ) -> list[tuple[str, str, str, str, int, int, int, int]]:
        """Rows and distinct values of each join key (blocking; memoised per column)."""
        measured = []
        for table_a, column_a, table_b, column_b in joins:
            side_a = self._key_stats(table_a, column_a)
            side_b = self._key_stats(table_b, column_b)
            if side_a and side_b:
                measured.append((table_a, column_a, table_b, column_b, *side_a, *side_b))
        return measured

    def _key_stats(self, table: str, column: str) -> tuple[int, int] | None:
        """Return (non-NULL rows, distinct values) of a table column, or None when not counted.

        NULLs are left out of both: they match no row of an equality join. A table larger than
        :data:`KEY_STATS_MAX_ROWS`, or of unknown size, is not counted.
        """
        key = (table, column.lower())
        if key not in self._key_counts:
            rows = self.executor.row_count(table)
            if rows is None or rows > KEY_STATS_MAX_ROWS:
                self._key_counts[key] = None
                return None
            quoted = exp.column(column, quoted=True).sql(self.executor.dialect)
            result = self.executor.execute(
                f"SELECT COUNT({quoted}), COUNT(DISTINCT {quoted}) "
                f"FROM {self.executor.quoted_name(table)}",
                timeout_s=KEY_COUNT_TIMEOUT_S,
            )
            self._key_counts[key] = (
                (int(result.rows[0][0]), int(result.rows[0][1])) if result.ok else None
            )
        return self._key_counts[key]

    async def judge(self, candidate: Candidate) -> Judgement | None:
        """Judge an executed candidate; None when the judge call failed (it is in the records)."""
        rubric = rubric_type(
            await self.missing_options(candidate), readings=self.cfg.judge_ambiguity
        )
        findings = candidate.checks.findings if candidate.checks else []
        material = prompts.judge_material(
            self.question,
            candidate.sql,
            candidate.exec,
            rows=self.cfg.preview_rows,
            evidence=self.evidence,
            evidence_chars=self.cfg.judge_evidence_chars,
            schema=await self._judge_schema(candidate) if self.cfg.judge_schema else None,
            findings=[finding.message for finding in findings] if self.cfg.judge_findings else None,
            stats=self.cfg.judge_stats,
        )
        model_name = self.models.names["judge"]
        try:
            output, result, record = await agents.run_agent(
                "judge",
                agents.judge(),
                material,
                model=self.models.judge,
                model_name=model_name,
                node_id=candidate.id,
                sink=self.records,
                transcripts=self.transcripts,
                output_type=rubric,
                model_settings=self._limits("judge", None, JUDGE_TIMEOUT_S),
            )
        except Exception:
            return None  # the candidate is scored on the checks alone
        fields = {
            name: float(getattr(output, name)) for name in rubric.model_fields if name != "missing"
        }
        return Judgement(
            model=model_name,
            fields=fields,
            missing=_missing_probabilities(output, result),
            mean=sum(fields.values()) / len(fields),
            ms=record.ms,
        )

    async def _ensure_advice(self, parent: Candidate) -> None:
        """Run the critic once per expanded parent; its children share the advice."""
        lock = self._advice_locks.setdefault(parent.id, asyncio.Lock())
        async with lock:
            if parent.advice is not None:
                return
            linked = await self._link(parent.action)
            prompt = prompts.critic_prompt(self.question, parent, linked.ddl)
            try:
                output, _, _ = await agents.run_agent(
                    "critic",
                    agents.critic(),
                    prompt,
                    model=self.models.critic,
                    model_name=self.models.names["critic"],
                    node_id=parent.id,
                    sink=self.records,
                    transcripts=self.transcripts,
                    model_settings=self._limits("critic", CRITIC_MAX_TOKENS, GEN_TIMEOUT_S),
                )
            except Exception:
                parent.advice = ""  # refine from the deterministic feedback alone
            else:
                parent.advice = str(output).strip()

    async def _pick_material(self, a: Candidate, b: Candidate) -> str:
        """Build the selector's prompt for a pair, with the judge's context when configured."""
        if not self.cfg.selector_context:
            return prompts.pick_material(self.question, a, b)
        schema_a, schema_b = await asyncio.gather(self._judge_schema(a), self._judge_schema(b))
        return prompts.pick_material(
            self.question,
            a,
            b,
            evidence=self.evidence,
            evidence_chars=self.cfg.judge_evidence_chars,
            schema_a=schema_a,
            schema_b=schema_b,
            rows=self.cfg.preview_rows,
            stats=True,
            findings=True,
        )

    async def _pick(self, a: Candidate, b: Candidate) -> float:
        """Return the selector's p(``a`` is better than ``b``); 0.5 when the call failed."""
        try:
            output, _, _ = await agents.run_agent(
                "selector",
                agents.selector(),
                await self._pick_material(a, b),
                model=self.models.selector,
                model_name=self.models.names["selector"],
                sink=self.records,
                transcripts=self.transcripts,
                model_settings=self._limits("selector", None, JUDGE_TIMEOUT_S),
            )
        except Exception:
            return 0.5  # no preference; the failure is in the records
        return float(output.a_is_better)


def _missing_probabilities(output: RubricBase, result: Any) -> dict[str, float]:
    """Return p(missing) per table: Jev's probabilities, else 1.0 per table the output names."""
    probabilities = agents.provider_details(result).get("probabilities", {}).get("missing")
    if isinstance(probabilities, dict):
        return {str(table): float(p) for table, p in probabilities.items()}
    return {table: 1.0 for table in output.model_dump().get("missing", [])}
