"""The answer loop with scripted models over a real localhost MCP server (no model call, no network).

The scripted generator calls the schema tools with their real parameters, so these tests prove
the agents read the schema over MCP.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("fastmcp")

from pydantic_ai.messages import (  # noqa: E402
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402
from pydantic_ai.usage import RequestUsage  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from schemagraph.agent import agents  # noqa: E402
from schemagraph.agent import models as agent_models  # noqa: E402
from schemagraph.agent.answer import (  # noqa: E402
    REASONING_MAX_TOKENS,
    REASONING_TIMEOUT_S,
    Answerer,
)
from schemagraph.agent.execute import AgentError, DuckDBExecutor, SQLiteExecutor  # noqa: E402
from schemagraph.agent.models import AgentModels, model_names  # noqa: E402
from schemagraph.agent.results import (  # noqa: E402
    AgentConfig,
    AnswerResult,
    Candidate,
    CheckReport,
    ExecResult,
    Finding,
    UsageRecord,
    UsageSummary,
    expand_prompt,
)
from schemagraph.agent.schema_client import SchemaClient  # noqa: E402
from schemagraph.agent.search import (  # noqa: E402
    _abmcts_algorithm,
    _refine_action,
    new_node,
    run_search,
    select_final,
    split_action,
)
from schemagraph.cli import app as cli_app  # noqa: E402
from schemagraph.connectors.ddl import DDLConfig, parse_ddl  # noqa: E402
from schemagraph.connectors.duckdb_conn import DuckDBConfig, introspect_duckdb  # noqa: E402
from schemagraph.engine import Engine  # noqa: E402
from schemagraph.graph.build import build_graph  # noqa: E402
from schemagraph.linking.linker import Linker  # noqa: E402
from schemagraph.mcp import LinkerSource, create_server, serve_http  # noqa: E402

from .conftest import STORE_DDL  # noqa: E402

QUESTION = "total order amount by customer state"
GOOD = (
    "select c.state, sum(o.total_amount) as total from orders o "
    "join customer c on c.id = o.customer_id group by c.state"
)
BAD = (  # `totl` does not exist: EXPLAIN rejects it and the generator retries
    "select c.state, sum(o.totl) from orders o join customer c on c.id = o.customer_id "
    "group by c.state"
)
NAMES = {"generator": "g", "judge": "j", "selector": "j", "critic": "c"}
# Tests that take tens of seconds (AB-MCTS-M's MCMC fits) run only when this is set.
SLOW = os.environ.get("SCHEMAGRAPH_SLOW_TESTS") == "1"


# ------------------------------------------------------------------ scripted models


def _last_part_types(messages) -> set[type]:
    return {type(part) for part in messages[-1].parts}


def _tool_returns(messages) -> dict[str, object]:
    """Tool name -> content of every tool return in the conversation."""
    return {
        part.tool_name: part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


def _output(info: AgentInfo, **arguments) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, arguments)])


def _lock_is_free(lock: threading.RLock) -> bool:
    """Whether another thread could take ``lock`` right now."""
    acquired: list[bool] = []

    def probe() -> None:
        got = lock.acquire(blocking=False)
        if got:
            lock.release()
        acquired.append(got)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return acquired[0]


class Script:
    """Generator: link_schema over MCP, then BAD (EXPLAIN rejects it), then ``good``."""

    def __init__(self, good: str = GOOD, lock: threading.RLock | None = None):
        self.good = good
        self.calls = 0
        self.lock = lock
        self.lock_free: list[bool] = []
        self.offered: set[str] = set()
        self.returns: dict[str, object] = {}
        self.prompts: list[str] = []  # the user prompt of each run, as the model read it

    def gen(self, messages, info: AgentInfo) -> ModelResponse:
        self.calls += 1
        if len(messages) == 1:  # a run's first request
            self.prompts += [p.content for p in messages[0].parts if isinstance(p, UserPromptPart)]
        if self.lock is not None:  # the Engine lock must be free while a model runs
            self.lock_free.append(_lock_is_free(self.lock))
        self.offered = {tool.name for tool in info.function_tools}
        kinds = _last_part_types(messages)
        if RetryPromptPart in kinds:
            return _output(info, sql=self.good, rationale="fixed")
        if ToolReturnPart in kinds:
            self.returns = _tool_returns(messages)
            return _output(info, sql=BAD)
        call = ToolCallPart(
            "link_schema",
            {"question": "customer state", "max_tables": 3, "adaptive_budget": False},
        )
        return ModelResponse(parts=[call])


def judge_fn(p: float = 0.9, missing: dict | None = None, schemas: list[str] | None = None):
    """Judge and selector: rubric fields at ``p``, Jev-style ``missing`` probabilities."""

    def judge(messages, info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        properties = tool.parameters_json_schema["properties"]
        if "a_is_better" in properties:  # the selector: always prefers the first candidate
            return _output(info, a_is_better=1.0)
        if schemas is not None:
            schemas.append(json.dumps(tool.parameters_json_schema))
        arguments = {name: ([] if name == "missing" else p) for name in properties}
        return ModelResponse(
            parts=[ToolCallPart(tool.name, arguments)],
            provider_details={"probabilities": {"missing": missing or {}}},
        )

    return judge


def critic_fn(calls: list[int]):
    def critic(messages, info) -> ModelResponse:
        calls.append(1)
        return ModelResponse(parts=[TextPart("- use total_amount")])

    return critic


def scripted(script: Script, critic_calls: list[int], judge=None) -> AgentModels:
    judge_model = FunctionModel(judge or judge_fn(missing={"main.products": 0.7}))
    return AgentModels(
        FunctionModel(script.gen),
        judge_model,
        judge_model,
        FunctionModel(critic_fn(critic_calls)),
        NAMES,
    )


def _models(generator, judge=None) -> AgentModels:
    judge_model = FunctionModel(judge or judge_fn())
    return AgentModels(
        FunctionModel(generator), judge_model, judge_model, FunctionModel(critic_fn([])), NAMES
    )


# ------------------------------------------------------------------ an MCP server over the store


@pytest.fixture
def store(store_duckdb):
    """(MCP URL, executor): the store database, its linker served over localhost HTTP."""
    snapshot = introspect_duckdb(DuckDBConfig(path=str(store_duckdb)), "db")
    source = LinkerSource(Linker(build_graph([snapshot])), dialect="duckdb")
    executor = DuckDBExecutor(store_duckdb)
    with serve_http(create_server(source)) as url:
        yield url, executor
    executor.close()


def _answerer(store, cfg: AgentConfig, models: AgentModels) -> Answerer:
    url, executor = store
    return Answerer(SchemaClient(url), executor, cfg, models)


def _answer(store, cfg: AgentConfig, models: AgentModels, question: str = QUESTION):
    return asyncio.run(_answerer(store, cfg, models).answer(question))


# ------------------------------------------------------------------ models and agents


def test_model_names_and_qwen_profile(monkeypatch):
    for name in ("SCHEMAGRAPH_GEN_MODEL", "SCHEMAGRAPH_JUDGE_MODEL", "SCHEMAGRAPH_CRITIC_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert model_names(AgentConfig()) == {
        "generator": "alibaba:qwen3.8-max",
        "judge": "typesafe:jev-1.13.0",
        "selector": "typesafe:jev-1.13.0",
        "critic": "alibaba:qwen3.8-max",
    }
    monkeypatch.setenv("SCHEMAGRAPH_JUDGE_MODEL", "alibaba:qwen3.8-max")
    assert model_names(AgentConfig(judge_model="test"))["judge"] == "test"  # config > env > default
    assert model_names(AgentConfig())["selector"] == "alibaba:qwen3.8-max"
    pytest.importorskip("openai")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "x")
    model = agent_models.resolve_model("alibaba:qwen3.8-max")
    assert model.model_name == "qwen3.8-max"
    assert model.profile["openai_supports_tool_choice_required"] is False  # pydantic-ai #1265


def test_openrouter_models_reason_and_report_their_cost(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.delenv("SCHEMAGRAPH_REASONING", raising=False)
    model = agent_models.resolve_model("openrouter:qwen/qwen3.8-max")
    assert model.model_name == "qwen/qwen3.8-max"
    assert model.settings["openrouter_reasoning"] == {"enabled": True, "effort": "medium"}
    assert model.settings["openrouter_usage"] == {"include": True}
    assert model.profile["openai_supports_tool_choice_required"] is False  # Qwen thinking
    names = {"generator": "openrouter:qwen/qwen3.8-max", "judge": "typesafe:jev-1.13.0"}
    models = AgentModels(None, None, None, None, names)
    assert models.reasoning("generator") and not models.reasoning("judge")
    widened = Answerer._limits(SimpleNamespace(models=models), "generator", 4096, 120.0)
    assert widened == {"max_tokens": 4096 + REASONING_MAX_TOKENS, "timeout": REASONING_TIMEOUT_S}
    assert Answerer._limits(SimpleNamespace(models=models), "judge", None, 30.0) == {
        "timeout": 30.0
    }

    monkeypatch.setenv("SCHEMAGRAPH_REASONING", "off")
    model = agent_models.resolve_model("openrouter:qwen/qwen3.8-max")
    assert model.settings["openrouter_reasoning"] == {"enabled": False}
    assert not models.reasoning("generator")
    monkeypatch.setenv("SCHEMAGRAPH_REASONING", "max")
    with pytest.raises(ValueError, match="SCHEMAGRAPH_REASONING"):
        agent_models.resolve_model("openrouter:qwen/qwen3.8-max")


def test_usage_records_billed_cost_reasoning_tokens_and_fallback_prices():
    def priced(messages, info):
        return ModelResponse(
            parts=[TextPart("advice")],
            usage=RequestUsage(input_tokens=100, output_tokens=50, details={"reasoning_tokens": 30}),
            provider_details={"cost": 0.0021},
        )

    records: list[UsageRecord] = []
    output, _, record = asyncio.run(
        agents.run_agent(
            "critic",
            agents.critic(),
            "p",
            model=FunctionModel(priced),
            model_name="openrouter:qwen/qwen3.8-max",
            sink=records,
        )
    )
    assert output == "advice" and records == [record]
    assert record.cost_usd == pytest.approx(0.0021) and record.unpriced == 0
    assert record.reasoning_tokens == 30

    unpriced = UsageRecord(role="critic", model="openai:gpt-x")
    agents._add_cost(unpriced, [ModelResponse(parts=[TextPart("a")])])
    assert unpriced.cost_usd == 0.0 and unpriced.unpriced == 1
    jev = UsageRecord(role="judge", model="typesafe:typesafe/jev-1.13")
    usage = RequestUsage(input_tokens=1_000_000, output_tokens=10)
    agents._add_cost(jev, [ModelResponse(parts=[TextPart("x")], usage=usage)])
    assert jev.cost_usd == pytest.approx(0.042) and jev.unpriced == 0  # Jev: input only
    jev.add(record)
    assert jev.cost_usd == pytest.approx(0.0441) and jev.reasoning_tokens == 30


def test_agents_build_without_keys(monkeypatch):
    for name in ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert agents.generator() and agents.judge() and agents.selector() and agents.critic()
    assert agents.schema_toolset("http://127.0.0.1:1/mcp")  # built lazily: nothing connects


def test_models_resolve_only_the_roles_in_use(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(agent_models, "resolve_model", lambda name: seen.append(name) or name)
    AgentModels.resolve(
        AgentConfig(
            strategy="best_of_n",
            judge=False,
            selector=False,
            gen_model="g",
            judge_model="j",
            critic_model="c",
        )
    )
    assert seen == ["g"]  # `ask --no-judge --no-selector` needs no Jev key; best_of_n never refines
    seen.clear()
    AgentModels.resolve(AgentConfig(gen_model="g", judge_model="j", critic_model="c"))
    assert seen == ["g", "j", "c"]


# ------------------------------------------------------------------ the generator over MCP


def test_generator_reads_the_schema_over_mcp(store):
    seen: dict[str, object] = {}
    offered: set[str] = set()

    def gen(messages, info):
        offered.update(tool.name for tool in info.function_tools)
        if ToolReturnPart in _last_part_types(messages):
            seen.update(_tool_returns(messages))
            return _output(info, sql=GOOD)
        calls = [
            ToolCallPart("link_schema", {"question": "customer state", "max_tables": 3}),
            ToolCallPart("get_table", {"fqn": "orders"}),
            ToolCallPart("find_join_path", {"from_table": "customer", "to_table": "products"}),
            ToolCallPart("search_tables", {"query": "order"}),
            ToolCallPart("list_glossary", {}),
        ]
        return ModelResponse(parts=calls)

    result = _answer(store, AgentConfig(strategy="single", judge=False), _models(gen))
    assert result.sql == GOOD
    assert offered == agents.SCHEMA_TOOLS | {"sample_values", "run_query"}  # no link_schema_json
    assert "CREATE TABLE" in seen["link_schema"] and "main.customer" in seen["link_schema"]
    assert seen["get_table"]["fqn"] == "main.orders"
    assert {relation["kind"] for relation in seen["get_table"]["relations"]} == {"foreign_key"}
    path = seen["find_join_path"]["paths"][0]["tables"]
    assert path == ["main.customer", "main.orders", "main.order_items", "main.products"]
    assert {table["fqn"] for table in seen["search_tables"]} == {"main.orders", "main.order_items"}
    assert seen["list_glossary"] == []


def test_tool_results_are_truncated(store, monkeypatch):
    monkeypatch.setattr(agents, "TOOL_RESULT_CHARS", 40)
    seen: dict[str, object] = {}

    def gen(messages, info):
        if ToolReturnPart in _last_part_types(messages):
            seen.update(_tool_returns(messages))
            return _output(info, sql=GOOD)
        return ModelResponse(parts=[ToolCallPart("get_table", {"fqn": "orders"})])

    _answer(store, AgentConfig(strategy="single", judge=False), _models(gen))
    assert isinstance(seen["get_table"], str) and seen["get_table"].endswith("(truncated)")
    assert len(seen["get_table"]) < 60


class _SpySource(LinkerSource):
    """A LinkerSource that records the options of every link."""

    def __init__(self, linker: Linker):
        super().__init__(linker, dialect="duckdb")
        self.links: list[tuple[str, dict]] = []

    def link(self, question: str, **overrides):
        self.links.append((question, overrides))
        return super().link(question, **overrides)


def test_generator_link_schema_never_uses_claude_or_a_huge_budget(store_duckdb):
    snapshot = introspect_duckdb(DuckDBConfig(path=str(store_duckdb)), "db")
    source = _SpySource(Linker(build_graph([snapshot])))
    executor = DuckDBExecutor(store_duckdb)

    def gen(messages, info):
        if ToolReturnPart in _last_part_types(messages):
            return _output(info, sql=GOOD)
        arguments = {"question": "spend check", "max_tables": 500, "use_llm": True}
        return ModelResponse(parts=[ToolCallPart("link_schema", arguments)])

    with serve_http(create_server(source)) as url:
        answerer = Answerer(SchemaClient(url), executor, AgentConfig(strategy="single"), _models(gen))
        asyncio.run(answerer.answer(QUESTION))
    executor.close()
    generator_links = [overrides for question, overrides in source.links if question == "spend check"]
    assert generator_links == [
        {"max_tables": 20, "columns": "relevant", "use_llm": False, "adaptive_budget": True}
    ]


def test_schema_client_merges_concurrent_lookups(store):
    url, _ = store
    client = SchemaClient(url)
    calls: list[tuple[str, str]] = []
    call_tool = client._call

    async def counting(tool: str, arguments: dict):
        calls.append((tool, str(arguments.get("fqn") or arguments.get("question"))))
        return await call_tool(tool, arguments)

    client._call = counting

    async def lookups() -> None:
        async with client:
            await asyncio.gather(*(client.table("orders") for _ in range(5)))
            assert (await client.table("main.orders"))["fqn"] == "main.orders"  # memoised by FQN
            names = ["customer", "orders", "products"]
            pairs = [(a, b) for a in names for b in names if a != b]
            await asyncio.gather(*(client.join_relations(a, b) for a, b in pairs))
            links = [client.link("state", max_tables=3, adaptive_budget=False) for _ in range(3)]
            await asyncio.gather(*links)

    asyncio.run(lookups())
    assert sorted(calls) == [
        ("get_table", "customer"),
        ("get_table", "orders"),
        ("get_table", "products"),
        ("link_schema_json", "state"),
    ]


def test_schema_client_retries_a_failed_lookup():
    client = SchemaClient("http://127.0.0.1:1/mcp")  # never contacted: _call is replaced
    attempts: list[int] = []

    async def flaky(tool: str, arguments: dict):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("down")
        return {"fqn": "main.orders", "relations": []}

    client._call = flaky

    async def lookups() -> dict | None:
        with pytest.raises(ConnectionError):
            await client.table("orders")
        return await client.table("orders")

    assert asyncio.run(lookups())["fqn"] == "main.orders"
    assert len(attempts) == 2


def test_schema_client_forgets_a_failure_no_waiter_saw(caplog):
    client = SchemaClient("http://127.0.0.1:1/mcp")  # never contacted: _call is replaced
    attempts: list[int] = []

    async def slow_then_ok(tool: str, arguments: dict):
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(0.05)
            raise ConnectionError("down")
        return {"fqn": "main.orders", "relations": []}

    client._call = slow_then_ok

    async def lookups() -> dict | None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.table("orders"), timeout=0.01)  # the only waiter
        await asyncio.sleep(0.1)  # the shared fetch fails with nobody awaiting it
        return await client.table("orders")

    with caplog.at_level("ERROR", logger="asyncio"):
        assert asyncio.run(lookups())["fqn"] == "main.orders"
    assert len(attempts) == 2
    assert not caplog.records  # the failure was retrieved, not logged


def test_schema_client_one_waiters_timeout_spares_the_others():
    client = SchemaClient("http://127.0.0.1:1/mcp")
    attempts: list[int] = []

    async def slow(tool: str, arguments: dict):
        attempts.append(1)
        await asyncio.sleep(0.05)
        return {"fqn": "main.orders", "relations": []}

    client._call = slow

    async def lookups() -> dict | None:
        impatient = asyncio.wait_for(client.table("orders"), timeout=0.01)
        patient = client.table("orders")
        timed_out, detail = await asyncio.gather(impatient, patient, return_exceptions=True)
        assert isinstance(timed_out, asyncio.TimeoutError)
        return detail

    assert asyncio.run(lookups())["fqn"] == "main.orders"
    assert len(attempts) == 1


def test_generate_retries_after_explain_and_scores(store):
    script, critic_calls, rubric_schemas = Script(), [], []
    judge = judge_fn(missing={"main.products": 0.7}, schemas=rubric_schemas)
    result = _answer(store, AgentConfig(strategy="single"), scripted(script, critic_calls, judge))
    candidate = result.candidates[0]
    assert result.sql == GOOD and candidate.rationale == "fixed"
    assert script.calls == 3  # the tool call, the rejected query, the fixed query
    assert "CREATE TABLE" in script.returns["link_schema"]
    assert candidate.exec.ok and candidate.exec.row_count == 2 and candidate.checks.det == 1.0
    assert candidate.judgement.missing == {"main.products": 0.7}  # from provider_details
    assert candidate.judgement.mean == pytest.approx(0.9)
    assert candidate.score == pytest.approx(0.15 + 0.85 * (0.4 + 0.6 * 0.9))
    assert "may need table(s): main.products (p=0.70)" in candidate.feedback
    # the judge may name the wide link's and the neighbours' tables, never the ones used
    assert "main.order_items" in rubric_schemas[0] and "main.orders" not in rubric_schemas[0]
    assert result.chosen_by == "only"
    assert result.usage.by_role["generator"].tool_calls >= 1
    assert result.usage.by_role["judge"].calls == 1
    # the prompt is kept with its schema DDL stored once, under the context key
    assert candidate.context_key in result.contexts
    assert f"<<schema {candidate.context_key}>>" in candidate.prompt
    assert "CREATE TABLE" not in candidate.prompt and "CREATE TABLE" in result.contexts[
        candidate.context_key
    ]
    assert "Question:" in candidate.prompt and "read-only" in result.instructions.lower()
    assert expand_prompt(candidate.prompt, result.contexts) == script.prompts[0]  # exact
    assert (candidate.asked_after, candidate.told) == (0, 0) and candidate.end_ms >= 0


def test_a_node_that_times_out_keeps_the_prompt_it_was_sent(store):
    async def stalls(messages, info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(30)  # cancelled at the node timeout
        return _output(info, sql=GOOD)

    cfg = AgentConfig(strategy="single", node_timeout_s=0.5, reasoning_node_timeout_s=None)
    result = _answer(store, cfg, _models(stalls))
    candidate = result.candidates[0]
    assert candidate.error == "node timed out" and not candidate.sql
    assert "Question:" in candidate.prompt and candidate.context_key in result.contexts


def test_judge_on_test_model_falls_back_to_the_output(store):
    models = AgentModels(FunctionModel(Script().gen), TestModel(), TestModel(), TestModel(), NAMES)
    result = _answer(store, AgentConfig(strategy="single"), models)
    judgement = result.candidates[0].judgement
    assert judgement is not None
    assert set(judgement.fields) == {
        "answers_question",
        "right_columns",
        "right_filters",
        "right_grain",
        "right_order",
        "plausible",
    }
    assert all(0 <= value <= 1 for value in judgement.fields.values())


def test_a_reused_answerer_reports_each_question_alone(store):
    script, critic_calls = Script(), []
    answerer = _answerer(store, AgentConfig(strategy="single"), scripted(script, critic_calls))
    first = asyncio.run(answerer.answer(QUESTION))
    second = asyncio.run(answerer.answer(QUESTION + " again"))
    assert second.question == QUESTION + " again"
    assert second.usage.total.calls == first.usage.total.calls > 0
    assert second.usage.total.input_tokens < 2 * first.usage.total.input_tokens
    assert len(answerer.records) == second.usage.total.calls  # the first question's are gone
    assert {record.node_id for record in answerer.records} == {"n0"}


def test_generation_failure_is_a_zero_node(store):
    def broken(messages, info):
        raise RuntimeError("provider down")

    cfg = AgentConfig(strategy="best_of_n", budget=2, batch_size=2)
    result = _answer(store, cfg, _models(broken), question="q")
    assert result.nodes == 2 and result.sql is None
    assert all(candidate.score == 0 and candidate.error for candidate in result.candidates)
    assert not result.usage.by_role["generator"].ok


def test_usage_survives_a_failed_generation(store):
    def always_bad(messages, info):  # valid calls with token usage, but EXPLAIN always rejects
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"sql": BAD})],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )

    models = AgentModels(FunctionModel(always_bad), None, None, None, NAMES)
    cfg = AgentConfig(strategy="single", judge=False, output_retries=1)
    result = _answer(store, cfg, models, question="q")
    usage = result.usage.by_role["generator"]
    assert result.candidates[0].error and not usage.ok
    assert usage.requests == 2 and usage.input_tokens == 200  # both attempts counted


def test_run_query_probe_budget_and_sample_values(store):
    seen: list[str] = []

    def gen(messages, info):
        returns = [part for message in messages for part in message.parts]
        returns = [part for part in returns if isinstance(part, ToolReturnPart)]
        seen.extend(str(part.content) for part in returns)
        if returns:
            return _output(info, sql=GOOD)
        calls = [
            ToolCallPart("run_query", {"sql": "select 1"}),
            ToolCallPart("run_query", {"sql": "select 2"}),
            ToolCallPart("sample_values", {"table": "customer", "column": "state"}),
            ToolCallPart("sample_values", {"table": "customer", "column": "nope"}),
        ]
        return ModelResponse(parts=calls)

    _answer(store, AgentConfig(strategy="single", probe_limit=1), _models(gen), question="q")
    assert any("1 rows" in text for text in seen)
    assert any("probe budget exhausted" in text for text in seen)
    assert any("'CA'" in text and "'NY'" in text for text in seen)
    assert any("unknown column 'nope'" in text for text in seen)


# ------------------------------------------------------------------ checks over MCP lookups


def test_join_check_maps_sqlite_catalog_keys_to_graph_tables(store_sqlite):
    """SQLite's catalog keys are bare (``orders``); the graph's FQNs are ``public.orders``."""
    snapshot = parse_ddl(DDLConfig(ddl=STORE_DDL, dialect="postgres", default_schema="public"), "s")
    source = LinkerSource(Linker(build_graph([snapshot])), dialect="sqlite")
    executor = SQLiteExecutor(store_sqlite)
    off_graph = "select * from orders o join customer c on c.id = o.total_amount"
    unrelated = (
        "select * from customer c join products p on p.id = c.id "
        "join orders o on o.customer_id = c.id"
    )
    answerer_cfg = AgentConfig(strategy="single", judge=False)

    async def score(url: str, sql: str) -> list[str]:
        answerer = Answerer(SchemaClient(url), executor, answerer_cfg, _models(Script().gen))
        async with answerer.schema:
            candidate = await answerer.score(Candidate(id="n0", sql=sql))
        return [f.message for f in candidate.checks.findings if f.code == "join_off_graph"]

    try:
        with serve_http(create_server(source)) as url:
            known = asyncio.run(score(url, off_graph))
            suggested = asyncio.run(score(url, unrelated))
    finally:
        executor.close()
    assert known == [
        "join `customer.id = orders.total_amount` is not a known relation; known: "
        "`public.orders.customer_id -> public.customer.id` (foreign_key)"
    ]
    assert suggested == [
        "no known relation between `products` and `customer`; shortest known path: "
        "public.products -> public.order_items -> public.orders -> public.customer"
    ]


# ------------------------------------------------------------------ search


def fake_generate(scores: list[float], log: list[tuple]):
    remaining = iter(scores)

    async def generate(node_id, parent, action):
        log.append((node_id, parent.id if parent else None, action))
        return Candidate(
            id=node_id,
            parent_id=parent.id if parent else None,
            depth=parent.depth + 1 if parent else 0,
            action=action,
            sql=f"select {node_id}",
            score=next(remaining),
            exec=ExecResult(ok=True, rows=[[node_id]], row_count=1),
        )

    return generate


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("single", 1), ("best_of_n", 8), ("refine", 8), ("abmcts", 8)],
)
def test_strategies_spend_the_budget(strategy, expected):
    pytest.importorskip("treequest")
    log: list[tuple] = []
    scores = [0.3 + 0.01 * i for i in range(8)]
    cfg = AgentConfig(strategy=strategy, budget=8, batch_size=4)
    trace = asyncio.run(run_search(fake_generate(scores, log), cfg))
    assert trace.nodes == expected == len(log)
    assert [candidate.id for candidate in trace.candidates] == [f"n{i}" for i in range(expected)]
    if strategy == "best_of_n":
        assert all(parent is None for _, parent, _ in log)
        assert [action for *_, action in log] == ["tight", "wide"] * 4
    if strategy == "refine":
        assert [parent for _, parent, _ in log] == [None] + [f"n{i}" for i in range(7)]


def test_abmcts_is_seeded_and_refines():
    pytest.importorskip("treequest")
    scores = [0.5, 0.6, 0.2, 0.7, 0.8, 0.1, 0.4, 0.3, 0.5, 0.6, 0.2, 0.7]
    runs = []
    for _ in range(2):
        log: list[tuple] = []
        cfg = AgentConfig(budget=12, batch_size=4, seed=3)
        asyncio.run(run_search(fake_generate(scores, log), cfg))
        runs.append(log)
    assert runs[0] == runs[1]
    assert any(parent is not None for _, parent, _ in runs[0])  # some trials refined a node


def test_abmcts_stops_early():
    pytest.importorskip("treequest")
    log: list[tuple] = []
    trace = asyncio.run(run_search(fake_generate([0.95] * 16, log), AgentConfig(budget=16)))
    assert trace.nodes == 4 and trace.stopped_early  # the first batch already passed 0.9


def test_early_stop_and_failures():
    log: list[tuple] = []
    scores = [0.2, 0.95, 0.1, 0.1, 0.1, 0.1]
    cfg = AgentConfig(strategy="best_of_n", budget=6, batch_size=2)
    trace = asyncio.run(run_search(fake_generate(scores, log), cfg))
    assert trace.nodes == 2 and trace.stopped_early

    async def boom(node_id, parent, action):
        raise ValueError("bad")

    trace = asyncio.run(run_search(boom, AgentConfig(strategy="best_of_n", budget=3, batch_size=2)))
    assert trace.nodes == 3
    assert all(candidate.score == 0 and "ValueError" in candidate.error for candidate in trace.candidates)


def _executed(index: int, score: float, rows: list) -> Candidate:
    return Candidate(
        id=f"n{index}", score=score, exec=ExecResult(ok=True, rows=rows, row_count=len(rows))
    )


def test_select_final_dedupes_and_cancels_position_bias():
    candidates = [
        _executed(0, 0.5, [[1]]),
        _executed(1, 0.6, [[2]]),
        _executed(2, 0.4, [[2]]),
        _executed(3, 0.1, [[3]]),
    ]
    calls: list[tuple[str, str]] = []

    async def always_first(a, b):
        calls.append((a.id, b.id))
        return 1.0

    best, chosen_by, matrix = asyncio.run(select_final(candidates, AgentConfig(top_k=4), always_first))
    assert chosen_by == "selector" and len(calls) == 6  # 3 distinct results: 3 pairs, both orders
    assert all(p == 0.5 for row in matrix.values() for p in row.values())  # the bias cancels
    assert best.id == "n1"  # a tie goes to the larger result group (n1 and n2 agree), then score
    only, chosen_by, _ = asyncio.run(select_final(candidates[:1], AgentConfig(), always_first))
    assert chosen_by == "only" and only.id == "n0"
    top, chosen_by, _ = asyncio.run(select_final(candidates, AgentConfig(selector=False), None))
    assert chosen_by == "score" and top.id == "n1"


def test_select_final_top_k_and_nothing_executed():
    candidates = [_executed(i, 0.1 * i, [[i]]) for i in range(6)]
    calls: list[int] = []

    async def indifferent(a, b):
        calls.append(1)
        return 0.5

    best, _, matrix = asyncio.run(select_final(candidates, AgentConfig(top_k=2), indifferent))
    assert set(matrix) == {"n5", "n4"} and len(calls) == 2 and best.id == "n5"
    failed = [
        Candidate(id="n0", score=0.05, exec=ExecResult(ok=False, error_kind="runtime")),
        Candidate(id="n1", score=0.0),
    ]
    best, chosen_by, _ = asyncio.run(select_final(failed, AgentConfig(), indifferent))
    assert best.id == "n0" and chosen_by == "score" and len(calls) == 2  # no selector call


def test_refine_widens_after_unknown_names():
    finding = Finding(code="unknown_column", severity="error", message="m")
    parent = Candidate(id="n0", action="tight", checks=CheckReport(parsed=True, findings=[finding]))
    assert _refine_action(parent) == "wide"
    assert _refine_action(Candidate(id="n0", action="tight")) == "tight"


# ------------------------------------------------------------------ the critic


def test_critic_runs_once_per_expanded_parent(store):
    critic_calls: list[int] = []
    cfg = AgentConfig(strategy="refine", budget=3, early_stop=1.01)
    result = _answer(store, cfg, scripted(Script(), critic_calls))
    assert result.nodes == 3 and len(critic_calls) == 2
    assert result.candidates[0].advice == "- use total_amount"
    assert result.usage.by_role["critic"].calls == 2


def test_concurrent_children_share_one_critic_call(store):
    critic_calls: list[int] = []
    answerer = _answerer(store, AgentConfig(strategy="abmcts"), scripted(Script(), critic_calls))
    parent = Candidate(id="n0", sql=BAD, score=0.3)

    async def expand_three_times():
        async with answerer.schema:
            await answerer.prepare("q")
            await asyncio.gather(*(answerer._ensure_advice(parent) for _ in range(3)))

    asyncio.run(expand_three_times())
    assert len(critic_calls) == 1 and parent.advice == "- use total_amount"


# ------------------------------------------------------------------ Engine and CLI


@pytest.fixture
def engine(tmp_path, store_duckdb):
    """An engine with the executable ``shop`` database and an unrelated ``other`` DDL schema."""
    engine = Engine(tmp_path / "home", llm=None, embed=False)
    engine.add_connection("shop", "duckdb", {"path": str(store_duckdb)})
    engine.add_connection(
        "other",
        "ddl",
        {
            "ddl": STORE_DDL.replace("customer", "client"),
            "dialect": "postgres",
            "default_schema": "crm",
        },
    )
    yield engine
    engine.close()


def _patch_models(monkeypatch, script: Script) -> None:
    by_name = {
        "g": FunctionModel(script.gen),
        "j": FunctionModel(judge_fn()),
        "c": FunctionModel(critic_fn([])),
    }
    monkeypatch.setattr(agent_models, "resolve_model", lambda name: by_name[name])
    monkeypatch.setenv("SCHEMAGRAPH_GEN_MODEL", "g")
    monkeypatch.setenv("SCHEMAGRAPH_JUDGE_MODEL", "j")
    monkeypatch.setenv("SCHEMAGRAPH_CRITIC_MODEL", "c")


def test_engine_answer_scopes_to_the_connection_and_frees_the_lock(engine, monkeypatch):
    script = Script(lock=engine._lock)
    _patch_models(monkeypatch, script)
    # one node at a time: a concurrent node may legitimately hold the lock inside engine.link
    cfg = AgentConfig(strategy="best_of_n", budget=2, batch_size=1, early_stop=1.01)
    result = engine.answer(QUESTION, config=cfg)
    assert result.sql == GOOD and result.result.ok and result.nodes == 2
    assert result.models["generator"] == "g"
    # the other connection's crm.* tables are out of scope, for the orchestrator and the tools
    assert result.linked_tables and all(t.startswith("main.") for t in result.linked_tables)
    assert "main." in script.returns["link_schema"] and "crm." not in script.returns["link_schema"]
    assert script.lock_free and all(script.lock_free)


# Longest pause a ticker may see on the loop during answer_async. Measured on WSL: about 0.01 s
# with the server started and stopped in a thread, 0.12 to 0.15 s when it ran on the loop.
MAX_LOOP_GAP_S = 0.05
_TICK_S = 0.005


async def _max_loop_gap(work) -> tuple[object, float]:
    """Run ``work`` with a ticker beside it; return its result and the longest gap between ticks."""
    gaps: list[float] = []
    done = asyncio.Event()

    async def ticker() -> None:
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(_TICK_S)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    ticking = asyncio.create_task(ticker())
    try:
        result = await work
    finally:
        done.set()
        await ticking
    return result, max(gaps)


def test_engine_answer_async_never_stalls_the_event_loop(engine, monkeypatch):
    _patch_models(monkeypatch, Script())
    cfg = AgentConfig(strategy="single", judge=False, selector=False)
    engine.answer(QUESTION, config=cfg)  # warm imports and the linker outside the measurement
    result, gap = asyncio.run(_max_loop_gap(engine.answer_async(QUESTION, config=cfg)))
    assert result.sql == GOOD
    assert gap < MAX_LOOP_GAP_S


def test_engine_answer_uses_a_configured_mcp_url(engine, monkeypatch):
    """With ``mcp_url`` the agents read that server's schema; the connection only executes."""
    _patch_models(monkeypatch, Script())
    snapshot = parse_ddl(DDLConfig(ddl=STORE_DDL, dialect="postgres", default_schema="public"), "s")
    source = LinkerSource(Linker(build_graph([snapshot])), dialect="duckdb")
    with serve_http(create_server(source)) as url:
        cfg = AgentConfig(strategy="single", mcp_url=url)
        result = engine.answer(QUESTION, connection="shop", config=cfg)
    assert result.result.ok and all(t.startswith("public.") for t in result.linked_tables)


def test_engine_answer_errors(engine):
    with pytest.raises(AgentError, match="duckdb"):
        engine.answer("q", connection="other")
    with pytest.raises(AgentError, match="unknown connection 'nope'"):
        engine.answer("q", connection="nope")

    async def inside_a_loop():
        return engine.answer("q")

    with pytest.raises(RuntimeError, match="answer_async"):
        asyncio.run(inside_a_loop())


def test_engine_answer_on_a_sqlite_file_without_a_connection(tmp_path, store_sqlite, monkeypatch):
    engine = Engine(tmp_path / "h2", llm=None, embed=False)
    engine.add_connection(
        "store", "ddl", {"ddl": STORE_DDL.replace("public.", ""), "dialect": "postgres"}
    )
    _patch_models(monkeypatch, Script())
    result = engine.answer(QUESTION, db=store_sqlite, config=AgentConfig(strategy="single"))
    engine.close()
    assert result.result.ok and result.result.row_count == 2  # ran on SQLite; schema from the ddl


def test_cli_ask(engine, monkeypatch):
    _patch_models(monkeypatch, Script())
    home = str(engine.home)
    engine.close()
    runner = CliRunner()
    args = ["ask", QUESTION, "--connection", "shop", "--strategy", "best_of_n", "--budget", "2"]
    out = runner.invoke(cli_app, [*args, "--json", "--home", home])
    assert out.exit_code == 0, out.output
    body = json.loads(out.stdout[out.stdout.index("{") :])
    assert body["sql"] == GOOD and body["chosen_by"] in {"score", "selector"}
    text = runner.invoke(cli_app, [*args, "--home", home])
    assert text.exit_code == 0 and text.stdout.startswith(GOOD) and "score " in text.stdout
    assert runner.invoke(cli_app, ["ask", "q", "--strategy", "nope", "--home", home]).exit_code != 0
    missing = runner.invoke(cli_app, ["ask", "q", "--connection", "nope", "--home", home])
    assert missing.exit_code == 1 and isinstance(missing.exception, SystemExit)  # no traceback
    assert "error: unknown connection 'nope'" in missing.output


def test_cli_ask_with_an_mcp_url(engine, monkeypatch):
    _patch_models(monkeypatch, Script())
    home = str(engine.home)
    engine.close()
    snapshot = parse_ddl(DDLConfig(ddl=STORE_DDL, dialect="postgres", default_schema="public"), "s")
    source = LinkerSource(Linker(build_graph([snapshot])), dialect="duckdb")
    with serve_http(create_server(source)) as url:
        args = ["ask", QUESTION, "-c", "shop", "--strategy", "single", "--mcp-url", url]
        out = CliRunner().invoke(cli_app, [*args, "--json", "--home", home])
    assert out.exit_code == 0, out.output
    body = json.loads(out.stdout[out.stdout.index("{") :])
    assert body["sql"] == GOOD and all(t.startswith("public.") for t in body["linked_tables"])


def test_cli_rejects_an_unknown_strategy(tmp_path):
    arguments = ["ask", "q", "--home", str(tmp_path), "--strategy", "greedy"]
    result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 2 and "'greedy' is not one of" in result.output


def test_bench_rejects_an_only_that_names_no_task(tmp_path):
    result = CliRunner().invoke(cli_app, ["bench-spider2-exec", str(tmp_path), "--only", " "])
    assert result.exit_code == 2 and "names no task" in result.output  # not every task


@pytest.mark.parametrize("command", [["ask", "q", "--home", "{home}"], ["bench-spider2-exec", "/x"]])
def test_cli_without_the_agent_extra_prints_the_hint(tmp_path, monkeypatch, command):
    from importlib.util import find_spec

    from schemagraph import cli

    def without_treequest(name: str, *args):
        return None if name == "treequest" else find_spec(name, *args)

    monkeypatch.setattr(cli, "find_spec", without_treequest)
    arguments = [argument.format(home=tmp_path) for argument in command]
    result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 1
    assert "missing treequest; install the agent extra" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("command", [["ask", "q", "--home", "{home}"], ["bench-spider2-exec", "/x"]])
def test_algorithm_m_without_pymc_prints_the_hint(tmp_path, monkeypatch, command):
    from importlib.util import find_spec

    from schemagraph import cli

    def without_pymc(name: str, *args):
        return None if name == "pymc" else find_spec(name, *args)

    monkeypatch.setattr(cli, "find_spec", without_pymc)
    arguments = [argument.format(home=tmp_path) for argument in command] + ["--algorithm", "m"]
    result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 1  # before any task runs, not an error row per task
    # naming only abmcts-m, an exact uv sync would uninstall the agent extra
    assert "needs the abmcts-m extra: uv sync --extra agent --extra abmcts-m" in result.output


def test_answer_errors_without_pydantic_ai(monkeypatch):
    import builtins

    from schemagraph import cli

    real_import = builtins.__import__

    def no_pydantic_ai(name, *args, **kwargs):
        if name.startswith("pydantic_ai"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pydantic_ai)
    assert ImportError in cli._answer_errors()


def test_early_stop_can_require_agreeing_results():
    from schemagraph.agent.search import _agreed

    def node(node_id: str, score: float, value: int, sql: str | None = None) -> Candidate:
        result = ExecResult(ok=True, columns=["v"], rows=[[value]], row_count=1)
        return Candidate(id=node_id, sql=sql or f"select {value} -- {node_id}", score=score, exec=result)

    split = [node("n0", 0.95, 1), node("n1", 0.93, 2), node("n2", 0.5, 1)]
    assert _agreed(split, AgentConfig())  # one high score stops, as before
    assert not _agreed(split, AgentConfig(early_stop_agree=2))  # the high scores disagree
    agreed = [*split, node("n3", 0.91, 1)]
    assert _agreed(agreed, AgentConfig(early_stop_agree=2))  # n0 and n3 return the same result
    echo = [*split, node("n3", 0.91, 1, sql=" select 1  -- n0")]
    assert not _agreed(echo, AgentConfig(early_stop_agree=2))  # n3 repeats n0's SQL: one vote


def test_early_stop_waits_for_the_minimum_nodes():
    pytest.importorskip("treequest")
    log: list[tuple] = []
    cfg = AgentConfig(budget=16, early_stop_min_nodes=8)
    trace = asyncio.run(run_search(fake_generate([0.95] * 16, log), cfg))
    assert trace.nodes == 8 and trace.stopped_early  # two batches, not one


def staggered(generate):
    """Delay each node by 20-100 ms, so nodes finish one at a time and the rolling loop rolls."""

    async def slow(node_id, parent, action):
        delay_ms = 20 * (int(node_id[1:]) * 7 % 5 + 1)  # n0 20, n1 60, n2 100, n3 40, n4 80, ...
        await asyncio.sleep(delay_ms / 1000)
        return await generate(node_id, parent, action)

    return slow


def test_rolling_abmcts_spends_the_budget_and_refines(monkeypatch):
    pytest.importorskip("treequest")
    from schemagraph.agent import search

    sizes: list[int] = []
    ask = search._ask

    def recording_ask(algorithm, cfg, state, count, actions):
        sizes.append(count)
        return ask(algorithm, cfg, state, count, actions)

    monkeypatch.setattr(search, "_ask", recording_ask)
    log: list[tuple] = []
    scores = [0.5, 0.6, 0.2, 0.7, 0.8, 0.1, 0.4, 0.3, 0.5, 0.6, 0.2, 0.7]
    cfg = AgentConfig(budget=12, batch_size=4, rolling=True, seed=3)
    trace = asyncio.run(run_search(staggered(fake_generate(scores, log)), cfg))
    assert trace.nodes == 12 == len(log)
    assert sorted(c.id for c in trace.candidates) == sorted(f"n{i}" for i in range(12))
    ids = {c.id for c in trace.candidates}
    assert all(parent in ids for _, parent, _ in log if parent)  # refinements of real nodes
    assert any(parent is not None for _, parent, _ in log)
    assert sizes[0] == 4 and 1 in sizes[1:] and sum(sizes) == 12  # asks as each node lands

    stopped = asyncio.run(
        run_search(fake_generate([0.95] * 16, []), AgentConfig(budget=16, rolling=True))
    )
    assert stopped.nodes == 4 and stopped.stopped_early  # the in-flight nodes are kept


def test_rolling_early_stop_counts_the_nodes_in_flight():
    pytest.importorskip("treequest")
    cfg = AgentConfig(budget=16, batch_size=4, rolling=True, early_stop_min_nodes=8)
    trace = asyncio.run(run_search(staggered(fake_generate([0.95] * 16, [])), cfg))
    assert trace.nodes == 8 and trace.stopped_early  # not up to a batch past the floor
    low = asyncio.run(run_search(staggered(fake_generate([0.5] * 16, [])), cfg))
    assert low.nodes == 16 and not low.stopped_early


@pytest.mark.parametrize("stop", ["cancel", "ask fails"])
def test_a_rolling_search_that_ends_abruptly_cancels_its_nodes(monkeypatch, stop):
    pytest.importorskip("treequest")
    from schemagraph.agent import search

    started: list[str] = []
    finished: list[str] = []

    async def slow(node_id, parent, action):
        started.append(node_id)
        await asyncio.sleep(0.2 if node_id == "n0" else 0.4)
        finished.append(node_id)
        return search.new_node(node_id, parent, action, score=0.3, sql="select 1")

    ask = search._ask
    asks: list[int] = []

    def failing_ask(algorithm, cfg, state, count, actions):
        asks.append(count)
        if len(asks) > 1:  # the first batch launches; the next ask fails
            raise RuntimeError("sampling failed")
        return ask(algorithm, cfg, state, count, actions)

    if stop == "ask fails":
        monkeypatch.setattr(search, "_ask", failing_ask)

    async def end_abruptly():
        running = asyncio.ensure_future(run_search(slow, AgentConfig(budget=8, rolling=True)))
        for _ in range(500):  # until the first batch is in flight
            if len(started) == 4:
                break
            await asyncio.sleep(0.01)
        if stop == "cancel":
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        else:
            with pytest.raises(RuntimeError, match="sampling failed"):
                await running
        await asyncio.sleep(0.5)

    asyncio.run(end_abruptly())
    assert finished == ([] if stop == "cancel" else ["n0"])  # none finish after the end


def test_abmcts_m_asks_one_search_at_a_time():
    from schemagraph.agent import search

    inside = {"now": 0, "peak": 0}
    count_lock = threading.Lock()

    class Fitting:
        def ask_batch(self, state, count, actions):
            with count_lock:
                inside["now"] += 1
                inside["peak"] = max(inside["peak"], inside["now"])
            time.sleep(0.05)
            with count_lock:
                inside["now"] -= 1
            return state, []

    async def two_searches(algorithm):
        cfg = AgentConfig(abmcts_algorithm=algorithm)
        asks = (asyncio.to_thread(search._ask, Fitting(), cfg, None, 1, ["wide"]) for _ in range(2))
        await asyncio.gather(*asks)

    asyncio.run(two_searches("a"))
    assert inside["peak"] == 2
    inside["peak"] = 0
    asyncio.run(two_searches("m"))
    assert inside["peak"] == 1  # TreeQuest's M frees every live JAX array in the process


@pytest.mark.skipif(not SLOW, reason="MCMC fits, about 25 s; set SCHEMAGRAPH_SLOW_TESTS=1")
def test_abmcts_m_runs_when_installed():
    pytest.importorskip("pymc")
    log: list[tuple] = []
    # batch 1 fits in this process; a larger batch starts JAX worker processes (about a minute)
    cfg = AgentConfig(budget=2, batch_size=1, abmcts_algorithm="m")  # each step fits by MCMC
    trace = asyncio.run(run_search(fake_generate([0.4, 0.6], log), cfg))
    assert trace.nodes == 2


def test_abmcts_m_starts_one_worker_process_per_batch_slot():
    pytest.importorskip("pymc")
    batch = AgentConfig(batch_size=4, abmcts_algorithm="m")
    assert _abmcts_algorithm(batch).max_process_workers == 4  # not one per CPU


@pytest.mark.parametrize(("selection", "strategy"), [(1, "multiarm_bandit_thompson"), (2, "stack")])
@pytest.mark.parametrize("algorithm", ["a", "m"])
def test_generator_selection_picks_treequests_strategy(algorithm, selection, strategy):
    pytest.importorskip("pymc" if algorithm == "m" else "treequest")
    cfg = AgentConfig(abmcts_algorithm=algorithm, generator_selection=selection)
    assert _abmcts_algorithm(cfg).model_selection_strategy == strategy


def test_an_unknown_generator_selection_is_rejected():
    pytest.importorskip("treequest")
    cfg = AgentConfig(budget=1, generator_selection=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown generator selection 3"):
        asyncio.run(run_search(fake_generate([0.5], []), cfg))


def test_an_unknown_abmcts_algorithm_is_rejected():
    pytest.importorskip("treequest")
    cfg = AgentConfig(budget=1, abmcts_algorithm="M")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown AB-MCTS algorithm 'M'"):
        asyncio.run(run_search(fake_generate([0.5], []), cfg))  # not an unlocked ABMCTSM


def test_abmcts_imports_treequest_off_the_event_loop(monkeypatch):
    pytest.importorskip("treequest")
    from schemagraph.agent import search

    on_loop_thread: list[bool] = []
    loop_thread = threading.current_thread()  # asyncio.run runs the loop on this thread
    build = search._abmcts_algorithm

    def recording_build(cfg):
        on_loop_thread.append(threading.current_thread() is loop_thread)
        return build(cfg)

    monkeypatch.setattr(search, "_abmcts_algorithm", recording_build)
    asyncio.run(run_search(fake_generate([0.5], []), AgentConfig(budget=1)))
    assert on_loop_thread == [False]  # with the abmcts-m extra the import loads JAX and PyMC


def test_abmcts_seeds_the_rng_after_building_the_algorithm(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("treequest")
    from schemagraph.agent import search

    build, ask = search._abmcts_algorithm, search._ask
    at_first_ask: list[tuple] = []

    def drawing_build(cfg):  # as importing TreeQuest with the abmcts-m extra does
        np.random.random(10)
        return build(cfg)

    def recording_ask(*args):
        if not at_first_ask:
            at_first_ask.append(np.random.get_state())
        return ask(*args)

    monkeypatch.setattr(search, "_abmcts_algorithm", drawing_build)
    monkeypatch.setattr(search, "_ask", recording_ask)
    np.random.seed(3)
    seeded = np.random.get_state()
    asyncio.run(run_search(fake_generate([0.5], []), AgentConfig(budget=1, seed=3)))
    state = at_first_ask[0]
    assert state[2] == seeded[2] and (state[1] == seeded[1]).all()  # the seed's state, undrawn


def test_a_reasoning_generator_gets_the_longer_node_timeout(monkeypatch):
    monkeypatch.delenv("SCHEMAGRAPH_REASONING", raising=False)
    names = {"generator": "openrouter:qwen/qwen3.8-max", "judge": "typesafe:jev"}
    reasoning = SimpleNamespace(cfg=AgentConfig(), models=AgentModels(None, None, None, None, names))
    assert Answerer.search_config(reasoning).node_timeout_s == 600.0
    plain = SimpleNamespace(cfg=AgentConfig(), models=AgentModels(None, None, None, None, NAMES))
    assert Answerer.search_config(plain).node_timeout_s == 300.0
    off = SimpleNamespace(
        cfg=AgentConfig(reasoning_node_timeout_s=None),
        models=AgentModels(None, None, None, None, names),
    )
    assert Answerer.search_config(off).node_timeout_s == 300.0


def test_a_refinement_sees_the_rubric_the_rationale_and_earlier_siblings():
    from schemagraph.agent.prompts import generator_prompt
    from schemagraph.agent.results import Judgement

    judged = Judgement(model="jev", fields={"right_grain": 0.24, "answers_question": 0.9}, mean=0.57)
    result = ExecResult(ok=True, columns=["a"], rows=[[1]], row_count=600)
    parent = Candidate(id="n2", sql="select a from t", score=0.8, exec=result,
                       judgement=judged, rationale="one row per actor and film")  # fmt: skip
    sibling = Candidate(id="n5", parent_id="n2", sql="select a from t group by a", score=0.9,
                        exec=ExecResult(ok=True, columns=["a"], rows=[[1]], row_count=200))  # fmt: skip
    prompt = generator_prompt("q", None, "tight", "create table t (a int);", 1, parent,
                              evidence_chars=0, siblings=[sibling])  # fmt: skip
    assert "Judge, weakest first (1 = yes): right_grain 0.24, answers_question 0.90" in prompt
    assert "Its author's reasoning: one row per actor and film" in prompt
    assert "already tried (do not repeat them)" in prompt
    assert "score 0.90; 200 rows" in prompt and "select a from t group by a" in prompt
    assert "already tried" not in generator_prompt(
        "q", None, "tight", "ddl", 1, parent, evidence_chars=0
    )


# ------------------------------------------------------------------ several generator models
MODELS = ("qwen", "glm", "grok")


def fake_mixed_generate(log: list[tuple], cfg: AgentConfig, scores: dict[str, float] | None = None):
    """Like fake_generate, for actions that name generator models; ``scores`` by model (0.5)."""

    async def generate(node_id, parent, action):
        log.append((node_id, parent.id if parent else None, action))
        generator, context = split_action(cfg, action)
        score = (scores or {}).get(generator, 0.5)
        node = new_node(node_id, parent, context, generator=generator, score=score)
        node.sql = f"select {node_id}"
        node.exec = ExecResult(ok=True, rows=[[node_id]], row_count=1)
        return node

    return generate


def test_an_action_names_the_generator_when_several_are_searched():
    several = AgentConfig(gen_models=MODELS)
    assert split_action(several, "glm") == ("glm", "wide")  # every model links the wide context
    assert split_action(several, "tight") == ("qwen", "tight")  # single, refine: the first model
    assert split_action(AgentConfig(), "tight") == (None, "tight")  # one generator: a width


def test_best_of_n_cycles_the_generators():
    cfg = AgentConfig(strategy="best_of_n", budget=6, batch_size=3, gen_models=MODELS)
    log: list[tuple] = []
    asyncio.run(run_search(fake_mixed_generate(log, cfg), cfg))
    assert [action for _, _, action in log] == list(MODELS) * 2  # round-robin, no context action


def test_abmcts_draws_every_node_from_the_generators():
    pytest.importorskip("treequest")
    tree = AgentConfig(budget=8, batch_size=2, gen_models=MODELS)
    log: list[tuple] = []
    trace = asyncio.run(run_search(fake_mixed_generate(log, tree), tree))
    assert trace.nodes == 8 and {action for _, _, action in log} <= set(MODELS)
    assert {node.generator for node in trace.candidates} <= set(MODELS)
    assert {node.action for node in trace.candidates} == {"wide"}


@pytest.mark.parametrize("selection", [1, 2])
def test_rewards_steer_the_search_towards_the_better_generator(selection):
    pytest.importorskip("treequest")
    cfg = AgentConfig(
        budget=24, batch_size=1, gen_models=MODELS, generator_selection=selection,
        early_stop=2.0,  # never stop early: every node counts
    )  # fmt: skip
    scores = {"qwen": 0.1, "glm": 0.9, "grok": 0.1}
    trace = asyncio.run(run_search(fake_mixed_generate([], cfg, scores), cfg))
    counts = Counter(node.generator for node in trace.candidates)
    assert counts["glm"] > trace.nodes / 2  # 17-23 of 24 over seeds 0-9, either selection


def test_a_failed_node_keeps_its_generator():
    cfg = AgentConfig(strategy="best_of_n", budget=2, batch_size=2, gen_models=MODELS[:2])

    async def failing(node_id, parent, action):
        raise RuntimeError("provider down")

    trace = asyncio.run(run_search(failing, cfg))
    assert [(node.generator, node.action) for node in trace.candidates] == [
        ("qwen", "wide"), ("glm", "wide"),
    ]  # fmt: skip
    assert all("provider down" in node.error for node in trace.candidates)


@pytest.mark.skipif(not SLOW, reason="MCMC fits, about a minute; set SCHEMAGRAPH_SLOW_TESTS=1")
@pytest.mark.parametrize("selection", [1, 2])
def test_abmcts_m_draws_every_node_from_the_generators(selection):
    pytest.importorskip("pymc")
    cfg = AgentConfig(
        budget=4, batch_size=1, abmcts_algorithm="m", gen_models=MODELS[:2],
        generator_selection=selection,
    )  # fmt: skip
    log: list[tuple] = []
    trace = asyncio.run(run_search(fake_mixed_generate(log, cfg), cfg))
    assert trace.nodes == 4 and {action for _, _, action in log} <= set(MODELS[:2])


def test_each_node_is_written_by_the_model_its_action_names(store):
    scripts = {name: Script() for name in MODELS[:2]}
    generators = {name: FunctionModel(script.gen) for name, script in scripts.items()}
    judge = FunctionModel(judge_fn())
    names = {**NAMES, "generator": "qwen"}
    models = AgentModels(generators["qwen"], judge, judge, None, names, generators)
    cfg = AgentConfig(
        strategy="best_of_n", budget=2, batch_size=2, gen_models=MODELS[:2], selector=False
    )
    result = _answer(store, cfg, models)
    by_id = {node.id: node for node in result.candidates}
    assert (by_id["n0"].generator, by_id["n1"].generator) == ("qwen", "glm")
    assert all(script.calls > 0 for script in scripts.values())  # both models wrote a node
    assert {"qwen", "glm"} <= set(result.usage.by_model)  # usage records each model by name


def test_several_generators_resolve_once_each(monkeypatch):
    resolved: list[str] = []

    def resolve(name):
        resolved.append(name)
        return TestModel()

    monkeypatch.setattr(agent_models, "resolve_model", resolve)
    monkeypatch.delenv("SCHEMAGRAPH_REASONING", raising=False)
    cfg = AgentConfig(gen_models=("openrouter:z-ai/glm-5.3", "alibaba:qwen"), judge_model="j")
    models = AgentModels.resolve(cfg)
    assert models.names["generator"] == models.names["critic"] == "openrouter:z-ai/glm-5.3"
    assert sorted(resolved) == sorted(["openrouter:z-ai/glm-5.3", "alibaba:qwen", "j"])
    assert models.generator("alibaba:qwen")[1] == "alibaba:qwen"
    assert models.generator(None)[1] == "openrouter:z-ai/glm-5.3"
    assert models.reasoning("generator")  # one openrouter generator reasons, so the node waits


def test_repeating_gen_model_searches_several_generators():
    from schemagraph.cli import _generators

    assert _generators(None) == {}
    assert _generators(["qwen"]) == {"gen_model": "qwen"}  # one model: the setting runs had
    assert _generators(["qwen", "glm", "qwen"]) == {"gen_models": ("qwen", "glm")}


def test_an_unknown_generator_is_an_error_not_the_default():
    names = {**NAMES, "generator": "qwen"}
    models = AgentModels("qwen-model", None, None, None, names, {"glm": "glm-model"})
    assert models.generator(None) == models.generator("qwen") == ("qwen-model", "qwen")
    assert models.generator("glm") == ("glm-model", "glm")
    with pytest.raises(KeyError):  # recording "grok" on a node qwen wrote would skew the stats
        models.generator("grok")


def test_only_strategies_that_search_the_generators_resolve_them(monkeypatch):
    resolved: list[str] = []

    def resolve(name):
        resolved.append(name)
        return name

    monkeypatch.setattr(agent_models, "resolve_model", resolve)
    single = AgentConfig(strategy="single", gen_models=MODELS, judge=False, selector=False)
    assert AgentModels.resolve(single).generators == {} and resolved == ["qwen"]  # first only
    resolved.clear()
    tree = replace(single, strategy="abmcts")
    assert list(AgentModels.resolve(tree).generators) == list(MODELS)


def test_each_generator_gets_its_own_reasoning_headroom(monkeypatch):
    monkeypatch.delenv("SCHEMAGRAPH_REASONING", raising=False)
    names = {**NAMES, "generator": "alibaba:qwen"}
    answerer = SimpleNamespace(models=AgentModels(None, None, None, None, names))
    reasoning = Answerer._limits(answerer, "generator", 4096, 120.0, model_name="openrouter:z-ai/glm")
    plain = Answerer._limits(answerer, "generator", 4096, 120.0, model_name="alibaba:qwen")
    assert reasoning == {"timeout": REASONING_TIMEOUT_S, "max_tokens": 4096 + REASONING_MAX_TOKENS}
    assert plain == {"timeout": 120.0, "max_tokens": 4096}


def test_a_refinement_is_written_by_its_model_and_advised_by_the_critic(store):
    """The critic is a separate model here; the default critic is the first generator
    (test_several_generators_resolve_once_each)."""
    scripts = {name: Script() for name in MODELS[:2]}
    generators = {name: FunctionModel(script.gen) for name, script in scripts.items()}
    judge = FunctionModel(judge_fn())
    critic_calls: list[int] = []
    names = {**NAMES, "generator": "qwen"}
    critic = FunctionModel(critic_fn(critic_calls))
    models = AgentModels(generators["qwen"], judge, judge, critic, names, generators)
    answerer = _answerer(store, AgentConfig(gen_models=MODELS[:2]), models)

    async def draft_then_refine():
        async with answerer.schema, answerer._toolset:
            await answerer.prepare(QUESTION)
            draft = await answerer.generate("n0", None, "qwen")
            return draft, await answerer.generate("n1", draft, "glm")

    draft, refinement = asyncio.run(draft_then_refine())
    assert (draft.generator, refinement.generator) == ("qwen", "glm")
    assert refinement.parent_id == "n0" and scripts["glm"].calls > 0
    assert critic_calls == [1]  # the critic ran once, on the critic model
    critic_models = {record.model for record in answerer.records if record.role == "critic"}
    assert critic_models == {"c"}


def test_action_restricts_the_context_widths():
    from schemagraph.cli import _search_actions

    assert _search_actions(None) == {}  # both widths, the setting runs had
    assert _search_actions(["wide", "wide"]) == {"actions": ("wide",)}
    result = CliRunner().invoke(cli_app, ["bench-spider2-exec", "/x", "--action", "narrow"])
    assert result.exit_code == 2 and "is not tight or wide" in result.output


def test_generator_selection_takes_the_papers_two_algorithms():
    result = CliRunner().invoke(cli_app, ["bench-spider2-exec", "/x", "--generator-selection", "3"])
    assert result.exit_code == 2 and "--generator-selection" in result.output


def test_only_best_of_n_and_abmcts_search_several_generators():
    assert AgentConfig(strategy="abmcts", gen_models=MODELS).searched_generators() == MODELS
    assert AgentConfig(strategy="best_of_n", gen_models=MODELS).searched_generators() == MODELS
    assert AgentConfig(strategy="refine", gen_models=MODELS).searched_generators() == ()
    assert AgentConfig(gen_models=MODELS[:1]).searched_generators() == ()  # one model: widths


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--gen-model", "a", "--gen-model", "b", "--action", "wide"], "models are the actions"),
        (["--strategy", "best_of_n", "--generator-selection", "2"], "only --strategy abmcts"),
        (["--strategy", "refine", "--algorithm", "m"], "only --strategy abmcts runs AB-MCTS"),
    ],
)
def test_options_the_run_would_ignore_are_refused(options, message):
    result = CliRunner().invoke(cli_app, ["bench-spider2-exec", "/x", *options])
    unboxed = " ".join(result.output.replace("│", " ").split())  # the error box wraps lines
    assert result.exit_code == 2 and message in unboxed


def test_ask_prints_each_generator_models_usage(capsys):
    from schemagraph.cli import _print_answer

    records = [
        UsageRecord(role="generator", model="qwen", calls=1, cost_usd=0.02),
        UsageRecord(role="generator", model="glm", calls=2, cost_usd=0.01),
    ]
    candidates = [Candidate(id="n0", generator="qwen"), Candidate(id="n1", generator="glm")]
    result = AnswerResult(
        question="q", sql=None, result=None, chosen_id=None, chosen_by=None, score=0.0,
        strategy="best_of_n", budget=2, nodes=2, stopped_early=False, candidates=candidates,
        usage=UsageSummary.of(records),
    )  # fmt: skip
    _print_answer(result)
    lines = capsys.readouterr().out.splitlines()
    model_lines = [line for line in lines if line.strip().startswith("model ")]
    assert [line.split()[1] for line in model_lines] == ["glm", "qwen"]  # one line per model
