"""Records of the agentic answer loop.

Core dependencies only (pydantic), so the executor, checks, score and benchmark comparator can be
used and tested without the ``agent`` extra. The models the agents answer with (`SqlCandidate`,
`RubricBase`, `Pick`) carry their instructions in their docstrings and field descriptions, which
pydantic turns into the output tool's JSON schema.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

Strategy = Literal["single", "best_of_n", "refine", "abmcts"]
Action = Literal["tight", "wide"]

# Jev: at most 255 options per question, and a Literal needs at least 2.
MAX_OPTIONS = 255
_MIN_OPTIONS = 2

ErrorKind = Literal["guard", "syntax", "runtime", "timeout", "row_cap"]
FindingCode = Literal[
    "guard",
    "parse",
    "unknown_table",
    "unknown_column",
    "cartesian",
    "join_off_graph",
    "ungrouped_column",
    "exec_error",
    "timeout",
    "empty_result",
    "null_column",
    "row_explosion",
]


class ExecResult(BaseModel):
    """The outcome of running one query.

    Attributes:
        ok: Whether the query ran to completion.
        error: The error message when it did not.
        error_kind: Which stage rejected it.
        columns: Result column names.
        rows: The first ``limit`` rows; JSON-safe unless fetched raw.
        row_count: Rows seen, up to ``count_cap``.
        row_count_capped: Counting stopped at ``count_cap``.
        truncated: ``row_count`` exceeds ``len(rows)``.
        elapsed_ms: Wall time of the execution.
    """

    ok: bool
    error: str | None = None
    error_kind: ErrorKind | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    row_count_capped: bool = False
    truncated: bool = False
    elapsed_ms: float = 0.0


class SqlCandidate(BaseModel):
    """The generator's answer."""

    sql: str = Field(
        description=(
            "One read-only SELECT (or WITH ... SELECT) query in the target dialect; "
            "no markdown fences, no comments, no trailing semicolon"
        )
    )
    rationale: str = Field(
        default="",
        description=(
            "One or two sentences: which tables, joins and filters answer the question, and why"
        ),
    )
    tables_used: list[str] = Field(
        default_factory=list,
        description="The tables the query reads",
    )


def _probability(question: str) -> Any:
    """A judge field: the probability in [0, 1] that the answer to ``question`` is yes."""
    return Field(ge=0, le=1, description=question)


class RubricBase(BaseModel):
    """A SQL query was written to answer a question about a database, and was run.

    Judge the query and its result against the question only. Answer each field independently.
    """

    answers_question: float = _probability(
        "Does this query, given the result shown, answer the question that was asked?"
    )
    right_columns: float = _probability(
        "Does the result return the values the question asks for, "
        "with none missing and no unrelated extra quantities?"
    )
    right_filters: float = _probability(
        "Does the query apply every condition stated in the question, "
        "and no condition the question does not state?"
    )
    right_grain: float = _probability(
        "Is the result at the grain the question asks for "
        "(one row per requested entity or group, aggregated as asked)?"
    )
    right_order: float = _probability(
        "If the question asks for an ordering, a top-N or a limit, "
        "does the query apply it correctly? If it asks for none, answer yes."
    )
    plausible: float = _probability(
        "Do the result values look plausible for this question "
        "(not empty, not all NULL, no absurd magnitudes)?"
    )


RUBRIC_FIELDS: tuple[str, ...] = tuple(RubricBase.model_fields)
MISSING_DESCRIPTION = "Which of these tables does the question need that the query does not use?"
# Opt-in rubric field (AgentConfig.judge_ambiguity): a question with two readings should not be
# judged against only one of them.
READINGS_FIELD = "covers_readings"
READINGS_DESCRIPTION = (
    "If the question can reasonably be read in more than one way (for example, a list of rows "
    "and a figure per group), does the result give what every reasonable reading asks for? "
    "If the question has only one reading, answer yes."
)


@lru_cache(maxsize=256)
def rubric_type(
    missing_options: tuple[str, ...] = (), *, readings: bool = False
) -> type[RubricBase]:
    """Build the judge's output type.

    Jev picks from candidates better than it extracts, so the rubric asks which of the linked and
    neighbouring tables the query skips, as a ``missing`` field over those names.

    Args:
        missing_options: Candidate table names; blanks and duplicates are dropped and at most
            `MAX_OPTIONS` are kept.
        readings: Add the :data:`READINGS_FIELD` probability, which counts in the judge's mean.

    Returns:
        `RubricBase` itself when fewer than two options remain and ``readings`` is off, else a
        ``Rubric`` subclass with a ``missing: list[Literal[...]]`` field and/or the readings
        field.
    """
    options = tuple(dict.fromkeys(option for option in missing_options if option))[:MAX_OPTIONS]
    extra: dict[str, Any] = {}
    if readings:
        extra[READINGS_FIELD] = (float, _probability(READINGS_DESCRIPTION))
    if len(options) >= _MIN_OPTIONS:
        missing_field = Field(default_factory=list, description=MISSING_DESCRIPTION)
        extra["missing"] = (list[Literal[options]], missing_field)  # type: ignore[valid-type]
    if not extra:
        return RubricBase
    return create_model("Rubric", __base__=RubricBase, __doc__=RubricBase.__doc__, **extra)


class Pick(BaseModel):
    """Two SQL queries were written for the same question and run. Compare them."""

    a_is_better: float = _probability(
        "Is candidate A a more correct answer to the question than candidate B?"
    )


class Judgement(BaseModel):
    """One judge verdict on a candidate.

    Attributes:
        model: The judge model's name.
        fields: The rubric probabilities, by field name.
        missing: Table -> probability that the query needs it (Jev), or 1.0 when chosen (LLM).
        mean: Mean of ``fields``.
        ms: Wall time of the judge call.
    """

    model: str
    fields: dict[str, float]
    missing: dict[str, float] = Field(default_factory=dict)
    mean: float
    ms: float = 0.0


class Finding(BaseModel):
    """One deterministic problem found in a candidate.

    Attributes:
        code: What kind of problem it is.
        severity: How much it matters.
        message: Human-readable text, copied verbatim into refine feedback.
        penalty: What it subtracts from the deterministic score.
    """

    code: FindingCode
    severity: Literal["error", "warn", "info"]
    message: str
    penalty: float = 0.0


class CheckReport(BaseModel):
    """The deterministic checks' verdict on one query.

    Attributes:
        parsed: Whether the query parsed.
        findings: Every problem found.
        tables: Catalog keys of the base tables the query reads.
        det: ``1 - sum(penalties)``, clipped at 0.
    """

    parsed: bool
    findings: list[Finding] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    det: float = 1.0


class UsageRecord(BaseModel):
    """Model usage of one agent run, or a sum of runs.

    Attributes:
        role: The agent's role (``generator``, ``judge``, ...); ``total`` or ``*`` in sums.
        model: The model's name.
        node_id: The search node the run belongs to, if any.
        calls: Agent runs.
        requests: Model requests.
        input_tokens: Prompt tokens.
        output_tokens: Completion tokens, reasoning included.
        reasoning_tokens: The part of ``output_tokens`` spent reasoning, where reported.
        cache_read_tokens: Prompt tokens read from the provider's cache.
        cache_write_tokens: Prompt tokens written to the provider's cache.
        tool_calls: Tool calls the model made.
        cost_usd: Billed cost in USD: what the provider reported (OpenRouter), else the
            fallback price of the model; failed and retried requests included.
        unpriced: Model responses with neither a reported cost nor a fallback price.
        ms: Wall time.
        ok: Whether every run succeeded.
        error: The failure message of a failed run.
    """

    role: str
    model: str = ""
    node_id: str | None = None
    calls: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    unpriced: int = 0
    ms: float = 0.0
    ok: bool = True
    error: str | None = None

    def add(self, other: UsageRecord) -> None:
        """Accumulate ``other``'s counters into this record, in place."""
        self.calls += other.calls
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.tool_calls += other.tool_calls
        self.cost_usd += other.cost_usd
        self.unpriced += other.unpriced
        self.ms += other.ms
        self.ok = self.ok and other.ok


class Transcript(BaseModel):
    """The messages of one model call attempt, kept when ``AgentConfig.trace`` is on.

    Attributes:
        role: The agent's role (``generator``, ``judge``, ``selector`` or ``critic``).
        model: The model's name.
        node_id: The search node the call belongs to, if any.
        attempt: 1 for the first try, higher for retries after a rate limit or server error.
        ok: Whether the attempt succeeded.
        messages: pydantic-ai's messages in JSON form: the prompt, the reasoning, the tool
            calls and their results, and the output.
    """

    role: str
    model: str = ""
    node_id: str | None = None
    attempt: int = 1
    ok: bool = True
    messages: list[dict[str, Any]] = Field(default_factory=list)


class UsageSummary(BaseModel):
    """Usage records summed per role, per model and overall.

    Attributes:
        by_role: Sums keyed by role.
        by_model: Sums keyed by model name.
        total: The sum of every record.
    """

    by_role: dict[str, UsageRecord] = Field(default_factory=dict)
    by_model: dict[str, UsageRecord] = Field(default_factory=dict)
    total: UsageRecord = Field(default_factory=lambda: UsageRecord(role="total"))

    @classmethod
    def of(cls, records: Iterable[UsageRecord]) -> UsageSummary:
        """Sum ``records`` into a new summary."""
        summary = cls()
        for record in records:
            role_sum = UsageRecord(role=record.role, model=record.model)
            summary.by_role.setdefault(record.role, role_sum).add(record)
            model_sum = UsageRecord(role="*", model=record.model)
            summary.by_model.setdefault(record.model, model_sum).add(record)
            summary.total.add(record)
        return summary


class Candidate(BaseModel):
    """One node of the search: a generated query and everything learnt about it.

    Attributes:
        id: ``"n0"``, ``"n1"``, ... issued by the search loop.
        parent_id: The node this one refines, if any.
        depth: Refinement depth; drafts are 0.
        action: Which schema context the generator saw.
        sql: The generated query.
        rationale: The generator's explanation.
        tables_used: Tables the generator says the query reads.
        linked_tables: The tables of the context the generator saw.
        exec: The execution result.
        checks: The deterministic checks' report.
        judgement: The judge's verdict.
        score: The combined score.
        score_parts: The score's components, by name.
        feedback: Deterministic and judge feedback lines.
        advice: The critic's advice, filled lazily when this node is expanded.
        error: The generation failure, if any.
        usage: Model usage attributed to this node.
        ms: Wall time spent on this node.
    """

    id: str
    parent_id: str | None = None
    depth: int = 0
    action: Action = "wide"
    sql: str = ""
    rationale: str = ""
    tables_used: list[str] = Field(default_factory=list)
    linked_tables: list[str] = Field(default_factory=list)
    exec: ExecResult | None = None
    checks: CheckReport | None = None
    judgement: Judgement | None = None
    score: float = 0.0
    score_parts: dict[str, float | None] = Field(default_factory=dict)
    feedback: list[str] = Field(default_factory=list)
    advice: str | None = None
    error: str | None = None
    usage: list[UsageRecord] = Field(default_factory=list)
    ms: float = 0.0


class AnswerResult(BaseModel):
    """The answer to one question and the search that produced it.

    Attributes:
        question: The question asked.
        sql: The chosen query, or None when no candidate was usable.
        result: The chosen query's execution result.
        chosen_id: The chosen candidate's id.
        chosen_by: How it was chosen: best score, the pairwise selector, or the only candidate.
        score: The chosen candidate's score.
        strategy: The search strategy used.
        budget: The generator-node budget.
        nodes: Generator nodes actually run.
        stopped_early: The search stopped before its budget ran out.
        candidates: Every node.
        selector_matrix: Probability that the row candidate beats the column candidate.
        usage: Model usage summed over the search.
        linked_tables: Tables of the wide context linked for the question.
        models: Model name by role.
        ms: Wall time of the answer.
        transcripts: The messages of every model call, when ``AgentConfig.trace`` is on.
    """

    question: str
    sql: str | None
    result: ExecResult | None
    chosen_id: str | None
    chosen_by: Literal["score", "selector", "only"] | None
    score: float
    strategy: str
    budget: int
    nodes: int
    stopped_early: bool
    candidates: list[Candidate]
    selector_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    usage: UsageSummary
    linked_tables: list[str] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)
    ms: float = 0.0
    transcripts: list[Transcript] = Field(default_factory=list)


@dataclass(frozen=True)
class ScoreWeights:
    """Weights of the candidate score (see `schemagraph.agent.score`).

    Attributes:
        exec_fail: Score of a parsed candidate that failed to execute; below ``floor``, so an
            executed candidate always outranks it.
        floor: Lowest score of an executed candidate.
        det_w: Weight of the deterministic checks.
        judge_w: Weight of the judge's mean.
        version: Recorded in the benchmark config; bump it when the formula changes.
    """

    exec_fail: float = 0.05
    floor: float = 0.15
    det_w: float = 0.4
    judge_w: float = 0.6
    version: str = "v1"


@dataclass
class AgentConfig:
    """Settings of one agentic answer.

    Attributes:
        strategy: The search strategy.
        budget: Generator nodes; judge, critic and selector calls count in usage, not here.
        batch_size: Nodes generated per search step.
        actions: Schema contexts the search may pick from.
        tight_tables: ``max_tables`` of the tight context.
        wide_tables: ``max_tables`` of the wide context.
        early_stop: Stop the search once a candidate scores at least this.
        early_stop_agree: Nodes scoring at least ``early_stop`` that must return the same
            result before the search stops; 1 stops on one high score.
        top_k: Candidates the pairwise selector compares.
        selector: Run the pairwise selector over the top candidates.
        judge: Run the judge. Off, the score is the checks alone, so the first clean executed
            candidate scores 1.0 and stops the search.
        seed: Random seed of the search.
        gen_model: Generator model; None reads the environment, then the default.
        judge_model: Judge and selector model; None reads the environment, then the default.
        critic_model: Critic model; None reads the environment, then uses the generator's.
        probe_limit: ``run_query`` probes the generator may make per node.
        preview_rows: Result rows shown to the judge.
        evidence_chars: External-knowledge characters shown to the generator.
        judge_evidence_chars: External-knowledge characters shown to the judge.
        judge_schema: Show the judge the tables the query reads: columns, keys, relations and
            row counts, so it can see a join that repeats rows.
        judge_findings: Show the judge the deterministic checks' findings.
        judge_stats: Show the judge per-column statistics of the fetched result (distinct
            values, NULLs, range), so it can see the result's grain.
        judge_ambiguity: Ask the judge whether the result covers every reasonable reading of
            the question (an extra rubric field, counted in its mean).
        exec_limit: Rows fetched per execution.
        exec_timeout_s: Execution timeout in seconds.
        count_cap: Rows counted per execution before counting stops.
        draft_temperature: Generator temperature for fresh drafts.
        refine_temperature: Generator temperature for refinements.
        node_timeout_s: Timeout of one generator node in seconds.
        output_retries: Retries when a model's structured output fails validation.
        weights: Weights of the candidate score.
        mcp_url: URL of a schemagraph MCP server (streamable HTTP); None starts one in-process
            for the call.
        trace: Keep every model call's messages on ``AnswerResult.transcripts``.
    """

    strategy: Strategy = "abmcts"
    budget: int = 16
    batch_size: int = 4
    actions: tuple[Action, ...] = ("tight", "wide")
    tight_tables: int = 7
    wide_tables: int = 20
    early_stop: float = 0.9
    early_stop_agree: int = 1
    top_k: int = 4
    selector: bool = True
    judge: bool = True
    seed: int = 0
    gen_model: str | None = None
    judge_model: str | None = None
    critic_model: str | None = None
    probe_limit: int = 3
    preview_rows: int = 10
    evidence_chars: int = 4000
    judge_evidence_chars: int = 1000
    judge_schema: bool = False
    judge_findings: bool = False
    judge_stats: bool = False
    judge_ambiguity: bool = False
    exec_limit: int = 1000
    exec_timeout_s: float = 30.0
    count_cap: int = 100_000
    draft_temperature: float = 0.8
    refine_temperature: float = 0.4
    node_timeout_s: float = 300.0
    output_retries: int = 2
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    mcp_url: str | None = None
    trace: bool = False
