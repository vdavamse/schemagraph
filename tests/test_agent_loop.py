"""The answer loop with scripted models over a real localhost MCP server (no model call, no network).

The scripted generator calls the schema tools with their real parameters, so these tests prove
the agents read the schema over MCP.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
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
    Candidate,
    CheckReport,
    ExecResult,
    Finding,
    UsageRecord,
)
from schemagraph.agent.schema_client import SchemaClient  # noqa: E402
from schemagraph.agent.search import _refine_action, run_search, select_final  # noqa: E402
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

    def gen(self, messages, info: AgentInfo) -> ModelResponse:
        self.calls += 1
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
