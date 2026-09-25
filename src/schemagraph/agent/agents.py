"""The four agents: generator (with tools), judge and selector (typed judgements), critic (text).

Agents are built once with ``defer_model_check=True`` and no model; the model is passed per run,
so building never needs a key and tests swap models per run. The generator's schema tools are
schemagraph's own MCP server (:func:`schema_toolset`), attached per run; its local tools are
the two that execute SQL, which the MCP server deliberately does not serve.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from functools import cache
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext, capture_run_messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.mcp import CallToolFunc, MCPToolset, ToolResult
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from schemagraph.agent import prompts
from schemagraph.agent.execute import Executor
from schemagraph.agent.guard import GuardError, guard_sql
from schemagraph.agent.results import AgentConfig, Pick, RubricBase, SqlCandidate, UsageRecord
from schemagraph.linking.linker import LinkOptions

# The MCP tools the generator may call; the server's other tools repeat these for other clients.
SCHEMA_TOOLS = frozenset(
    {"link_schema", "search_tables", "get_table", "find_join_path", "list_glossary"}
)
# Most tables a generator's ``link_schema`` call may ask for: the linker's default budget.
GENERATOR_MAX_TABLES = LinkOptions.max_tables
# Characters of one tool result the generator sees.
TOOL_RESULT_CHARS = 12_000
# Rows a ``run_query`` probe shows.
PROBE_ROWS = 20
# Most distinct values ``sample_values`` returns.
SAMPLE_VALUES_MAX = 50
# Timeout of one probe or sample query, in seconds.
TOOL_QUERY_TIMEOUT_S = 10.0
# Seconds before each retry of a rate-limited or failed model call (plus up to 25 % jitter).
RETRY_DELAYS = (1.0, 4.0, 16.0)
# Model requests and tool calls one generator node may make.
GENERATOR_REQUEST_LIMIT = 10
GENERATOR_TOOL_CALLS_LIMIT = 8
# Characters of a failure message kept on a usage record or a search node.
FAILURE_CHARS = 500
# USD per million (input, output) tokens of models whose responses carry no billed cost, by a
# substring of the model name. Jev's published price, output free; OpenRouter chat models report
# their cost and never need an entry.
FALLBACK_PRICES = {"jev": (0.042, 0.0)}
_TRUNCATED = "\n… (truncated)"


@dataclass
class AgentDeps:
    """What the generator's local tools and output validator need.

    Attributes:
        executor: The read-only database the query runs on.
        cfg: The answer's settings.
        probes_left: ``run_query`` calls left in this node.
    """

    executor: Executor
    cfg: AgentConfig
    probes_left: int


def _cut(text: str) -> str:
    """Truncate ``text`` to :data:`TOOL_RESULT_CHARS`, marking the cut."""
    if len(text) <= TOOL_RESULT_CHARS:
        return text
    return text[:TOOL_RESULT_CHARS] + _TRUNCATED


def _truncated(result: ToolResult) -> ToolResult:
    """Return an MCP tool result unchanged, or as truncated JSON text when it is too long."""
    if isinstance(result, str):
        return _cut(result)
    text = json.dumps(result, default=str)
    return result if len(text) <= TOOL_RESULT_CHARS else _cut(text)


def _pinned_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The arguments of a generator tool call, with ``link_schema``'s cost knobs pinned.

    ``use_llm`` is forced off, because the agent layer makes no Anthropic calls, and
    ``max_tables`` is clamped to ``[1, GENERATOR_MAX_TABLES]``. This holds for an external
    ``--mcp-url`` server too, which the agent layer does not build.
    """
    if name != "link_schema":
        return arguments
    requested = arguments.get("max_tables", GENERATOR_MAX_TABLES)
    if not isinstance(requested, int):
        requested = GENERATOR_MAX_TABLES
    max_tables = max(1, min(GENERATOR_MAX_TABLES, requested))
    return {**arguments, "use_llm": False, "max_tables": max_tables}


async def _process_call(
    run_context: RunContext[Any],
    call_tool: CallToolFunc,
    name: str,
    arguments: dict[str, Any],
) -> ToolResult:
    """Call an MCP tool with pinned arguments and cap its result at :data:`TOOL_RESULT_CHARS`."""
    return _truncated(await call_tool(name, _pinned_arguments(name, arguments)))


def _is_schema_tool(run_context: RunContext[Any], tool_def: ToolDefinition) -> bool:
    return tool_def.name in SCHEMA_TOOLS


def schema_toolset(mcp_url: str) -> AbstractToolset[AgentDeps]:
    """Return the generator's schema tools: :data:`SCHEMA_TOOLS` of the MCP server at ``mcp_url``.

    Pass it to each generator run (``toolsets=[...]``). Entering it (``async with``) opens one
    MCP session that every run inside the block shares; a run outside such a block opens its
    own. ``link_schema`` calls run with ``use_llm`` off and a bounded ``max_tables``.
    """
    server: MCPToolset[AgentDeps] = MCPToolset(mcp_url, process_tool_call=_process_call)
    return server.filtered(_is_schema_tool)


@cache
def generator() -> Agent[AgentDeps, SqlCandidate]:
    """Build the generator: one query as a :class:`SqlCandidate`, checked by the guard and EXPLAIN.

    Its schema tools are not part of the agent; pass :func:`schema_toolset` to each run.
    """
    agent: Agent[AgentDeps, SqlCandidate] = Agent(
        None,
        output_type=SqlCandidate,
        deps_type=AgentDeps,
        defer_model_check=True,
        retries={"tools": 1, "output": 2},
    )

    @agent.instructions
    def _instructions(run_context: RunContext[AgentDeps]) -> str:
        deps = run_context.deps
        return prompts.generator_instructions(deps.executor.dialect, deps.cfg.probe_limit)

    agent.tool(sample_values)
    agent.tool(run_query)
    agent.output_validator(_validate_output)
    return agent


def _resolve_column(executor: Executor, table: str, column: str) -> tuple[str, str] | str:
    """Return (catalog key, column name) for ``table.column``, or why not (blocking).

    The catalog is read on first use, so this runs in a worker thread.
    """
    key = executor.resolve_table(table)
    if key is None:
        return f"unknown table {table!r}"
    columns = executor.catalog()[key]
    by_lower_name = {name.lower(): name for name in columns}
    if column.lower() not in by_lower_name:
        return f"unknown column {column!r} in {key}; columns: {', '.join(columns)}"
    return key, by_lower_name[column.lower()]


async def sample_values(
    run_context: RunContext[AgentDeps],
    table: str,
    column: str,
    limit: int = 10,
) -> str:
    """Distinct non-null values of one column, to check spellings and formats before filtering.

    Args:
        run_context: The run context.
        table: The table name.
        column: The column name.
        limit: How many values (1-50).
    """
    executor = run_context.deps.executor
    resolved = await asyncio.to_thread(_resolve_column, executor, table, column)
    if isinstance(resolved, str):
        return resolved
    key, name = resolved
    quoted_column = '"' + name.replace('"', '""') + '"'
    count = max(1, min(SAMPLE_VALUES_MAX, limit))
    sql = (
        f"SELECT DISTINCT {quoted_column} FROM {executor.quoted_name(key)} "
        f"WHERE {quoted_column} IS NOT NULL LIMIT {count}"
    )
    result = await asyncio.to_thread(
        executor.execute, sql, limit=count, timeout_s=TOOL_QUERY_TIMEOUT_S
    )
    if not result.ok:
        return f"error: {result.error}"
    return f"{key}.{name}: " + ", ".join(repr(row[0]) for row in result.rows)


async def run_query(run_context: RunContext[AgentDeps], sql: str) -> str:
    """Run a small read-only probe query and see up to 20 rows.

    Args:
        run_context: The run context.
        sql: One SELECT query.
    """
    if run_context.deps.probes_left <= 0:
        return "probe budget exhausted; return your final query now"
    run_context.deps.probes_left -= 1
    result = await asyncio.to_thread(
        run_context.deps.executor.execute,
        sql,
        limit=PROBE_ROWS,
        timeout_s=TOOL_QUERY_TIMEOUT_S,
    )
    return _cut(prompts.preview(result, PROBE_ROWS))


async def _validate_output(
    run_context: RunContext[AgentDeps],
    output: SqlCandidate,
) -> SqlCandidate:
    """Send the query back to the model unless the guard and the database's EXPLAIN accept it."""
    executor = run_context.deps.executor
    try:
        guarded = guard_sql(output.sql, executor.dialect)
    except GuardError as error:
        raise ModelRetry(f"{error}. Return one read-only SELECT query.") from error
    result = await asyncio.to_thread(executor.explain, guarded.sql)
    if not result.ok:
        raise ModelRetry(f"The database rejected the query: {result.error}. Fix the query.")
    return output.model_copy(update={"sql": guarded.sql})


@cache
def judge() -> Agent[None, RubricBase]:
    """Build the judge.

    The output type is passed per run (:func:`~schemagraph.agent.results.rubric_type`). There are
    no validators, because Jev cannot revise an answer.
    """
    return Agent(
        None,
        output_type=RubricBase,
        instructions=prompts.JUDGE_INSTRUCTIONS,
        defer_model_check=True,
    )


@cache
def selector() -> Agent[None, Pick]:
    """Build the pairwise selector."""
    return Agent(
        None,
        output_type=Pick,
        instructions=prompts.JUDGE_INSTRUCTIONS,
        defer_model_check=True,
    )


@cache
def critic() -> Agent[None, str]:
    """Build the critic, which writes refinement advice as text."""
    return Agent(
        None,
        output_type=str,
        instructions=prompts.CRITIC_INSTRUCTIONS,
        defer_model_check=True,
    )


def _retryable(error: BaseException) -> bool:
    """Whether a model call failed on a rate limit, a server error or the connection."""
    if isinstance(error, ModelHTTPError):
        return error.status_code == 429 or error.status_code >= 500
    return isinstance(error, ModelAPIError)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:FAILURE_CHARS]


async def _run_with_backoff(
    agent: Agent[Any, Any],
    prompt: str,
    *,
    model: Any,
    usage: RunUsage,
    record: UsageRecord,
    run_options: dict[str, Any],
) -> Any:
    """Run ``agent``, retrying after each of :data:`RETRY_DELAYS` on a retryable error.

    Raises:
        Exception: The last error, once it is not retryable or the retries are spent; its
            message is also stored on ``record``.
    """
    for delay in RETRY_DELAYS:
        try:
            return await _priced_run(
                agent, prompt, model=model, usage=usage, record=record, **run_options
            )
        except Exception as error:
            if not _retryable(error):
                record.error = _error_text(error)
                raise
        await asyncio.sleep(delay * (1 + random.random() / 4))
    try:
        return await _priced_run(
            agent, prompt, model=model, usage=usage, record=record, **run_options
        )
    except Exception as error:
        record.error = _error_text(error)
        raise


async def _priced_run(
    agent: Agent[Any, Any], prompt: str, *, record: UsageRecord, **options: Any
) -> Any:
    """Run ``agent`` once and add the cost of every response it got to ``record``.

    The messages are captured so that a run that fails or is cancelled after some responses is
    still charged for them.
    """
    with capture_run_messages() as messages:
        try:
            return await agent.run(prompt, **options)
        finally:
            _add_cost(record, messages)


def _add_cost(record: UsageRecord, messages: list[ModelMessage]) -> None:
    """Add the billed cost of the responses in ``messages`` to ``record``."""
    fallback = next(
        (price for key, price in FALLBACK_PRICES.items() if key in record.model.lower()), None
    )
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        cost = (message.provider_details or {}).get("cost")
        if isinstance(cost, int | float):
            record.cost_usd += float(cost)
        elif fallback is not None:
            usage = message.usage
            record.cost_usd += (
                usage.input_tokens * fallback[0] + usage.output_tokens * fallback[1]
            ) / 1e6
        else:
            record.unpriced += 1


async def run_agent(
    role: str,
    agent: Agent[Any, Any],
    prompt: str,
    *,
    model: Any,
    model_name: str,
    node_id: str | None = None,
    sink: list[UsageRecord] | None = None,
    **run_options: Any,
) -> tuple[Any, Any, UsageRecord]:
    """Run one agent call with backoff on rate limits and server errors.

    Tokens are counted into one ``RunUsage`` across retries, and the record goes to ``sink`` on
    every exit (success, failure, cancellation by the node timeout), so spend is never lost.

    Args:
        role: The usage role: ``generator``, ``judge``, ``selector`` or ``critic``.
        agent: The agent to run.
        prompt: The user prompt.
        model: The model to run it on.
        model_name: The model's name, for the usage record.
        node_id: The search node the call belongs to, if any.
        sink: Where the usage record is appended.
        **run_options: Passed to ``agent.run`` (deps, toolsets, model settings, ...).

    Returns:
        The output, the run result and the usage record.

    Raises:
        Exception: The final failure, for the caller to record on the node.
    """
    started = time.perf_counter()
    usage = RunUsage()
    record = UsageRecord(
        role=role, model=model_name, node_id=node_id, calls=1, ok=False, error="cancelled"
    )
    try:
        result = await _run_with_backoff(
            agent, prompt, model=model, usage=usage, record=record, run_options=run_options
        )
        record.ok = True
        record.error = None
        return result.output, result, record
    finally:
        record.requests = usage.requests
        record.input_tokens = usage.input_tokens or 0
        record.output_tokens = usage.output_tokens or 0
        record.reasoning_tokens = usage.details.get("reasoning_tokens", 0)
        record.cache_read_tokens = usage.cache_read_tokens or 0
        record.cache_write_tokens = usage.cache_write_tokens or 0
        record.tool_calls = usage.tool_calls
        record.ms = (time.perf_counter() - started) * 1000
        if sink is not None:
            sink.append(record)


def generator_limits() -> UsageLimits:
    """Return the per-node request and tool-call limits of the generator."""
    return UsageLimits(
        request_limit=GENERATOR_REQUEST_LIMIT,
        tool_calls_limit=GENERATOR_TOOL_CALLS_LIMIT,
    )


def provider_details(result: Any) -> dict[str, Any]:
    """Return the provider details of a run's last response (Jev's probabilities), or ``{}``."""
    return result.response.provider_details or {}
