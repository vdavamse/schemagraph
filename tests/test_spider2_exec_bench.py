"""The execution benchmark and the judge study on a tiny fake Spider2 tree (scripted models).

Each database is served by its own localhost MCP server; the scripted generator reads the table
over MCP before answering.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path

import pytest

from schemagraph.bench.spider2_exec import (
    McpServers,
    auroc,
    load_tasks,
    resolve_schema_dir,
    sqlite_path,
)
from schemagraph.graph.build import build_graph
from schemagraph.linking.linker import Linker

COUNT_2001 = "How many movies came out in 2001?"
TITLES = "List the movie titles."
# Per question: the tight-context query, then the wide-context one.
SQL = {
    COUNT_2001: ["select count(*) as n from movie where year = 2001", "select count(*) from movie"],
    TITLES: ["select title from movie", "select title from movie"],
}
TASKS = [
    {"instance_id": "local901", "db": "Db-X", "question": COUNT_2001, "external_knowledge": None},
    {"instance_id": "local902", "db": "Db-X", "question": TITLES, "external_knowledge": None},
]
NAMES = {"generator": "g", "judge": "j", "selector": "j", "critic": "c"}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records))


def _spider2(root: Path) -> Path:
    """Two local tasks on db ``Db-X``, whose schema folder is spelled ``DB_X`` (as Db-IMDB)."""
    lite = root / "spider2-lite"
    local_db = lite / "resource/databases/spider2-localdb"
    local_db.mkdir(parents=True)
    con = sqlite3.connect(local_db / "Db-X.sqlite")
    con.executescript(
        "CREATE TABLE movie (id INT PRIMARY KEY, title TEXT, year INT);"
        "INSERT INTO movie VALUES (1, 'A', 1999), (2, 'B', 2001), (3, 'C', 2001);"
    )
    con.commit()
    con.close()
    folder = lite / "resource/databases/sqlite/DB_X"
    folder.mkdir(parents=True)
    movie = {
        "table_name": "movie",
        "column_names": ["id", "title", "year"],
        "column_types": ["INT", "TEXT", "INT"],
        "sample_rows": [{"id": 1, "title": "A", "year": 1999}],
    }
    (folder / "movie.json").write_text(json.dumps(movie))
    _write_jsonl(lite / "spider2-lite.jsonl", TASKS)
    gold = root / "methods/gold-tables"
    gold.mkdir(parents=True)
    _write_jsonl(
        gold / "spider2-lite-gold-tables.jsonl",
        [{"instance_id": task["instance_id"], "gold_tables": ["movie"]} for task in TASKS],
    )
    evaluation = lite / "evaluation_suite/gold"
    (evaluation / "exec_result").mkdir(parents=True)
    (evaluation / "exec_result/local901_a.csv").write_text("n\n2\n")
    (evaluation / "exec_result/local902.csv").write_text("title\nA\nB\nC\n")
    _write_jsonl(
        evaluation / "spider2lite_eval.jsonl",
        [
            {"instance_id": task["instance_id"], "condition_cols": [], "ignore_order": True}
            for task in TASKS
        ],
    )
    return root


def test_resolvers(tmp_path):
    root = _spider2(tmp_path)
    base = root / "spider2-lite/resource/databases/sqlite"
    assert resolve_schema_dir(base, "Db-X").name == "DB_X"
    (base / "db-x").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_schema_dir(base, "Db_X")  # two candidates
    assert sqlite_path(root, "Db-X").name == "Db-X.sqlite"
    with pytest.raises(FileNotFoundError, match="local_sqlite.zip"):
        sqlite_path(root, "Nope")
    tasks, standard = load_tasks(root)
    assert [task.instance_id for task in tasks] == ["local901", "local902"]
    assert standard["local901"]["ignore_order"]


def test_auroc():
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert auroc([0.5, 0.5], [0, 1]) == 0.5  # ties count half
    assert auroc([1.0, 2.0], [1, 1]) is None


class _Models:
    """Scripted models; the generator reads ``movie`` over MCP before answering."""

    def __init__(self):
        pytest.importorskip("pydantic_ai")
        pytest.importorskip("fastmcp")
        self.tables_seen: list[str] = []

    def generator(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("get_table", {"fqn": "movie"})])
        self.tables_seen.append(returns[0].content["fqn"])
        # by context action (tight: the first query), not call order: a batch runs concurrently
        prompt = messages[0].parts[-1].content
        question = next(text for text in SQL if text in prompt)
        sql = SQL[question][0 if "tight context" in prompt else 1]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"sql": sql})])

    @staticmethod
    def judge(messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        tool = info.output_tools[0]
        properties = tool.parameters_json_schema["properties"]
        if "a_is_better" in properties:
            return ModelResponse(parts=[ToolCallPart(tool.name, {"a_is_better": 0.5})])
        material = messages[0].parts[-1].content
        p = 0.2 if "count" in material and "where" not in material else 0.9  # doubts the unfiltered
        arguments = {name: ([] if name == "missing" else p) for name in properties}
        return ModelResponse(parts=[ToolCallPart(tool.name, arguments)])

    def agent_models(self, generator=None):
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        from schemagraph.agent.models import AgentModels

        judge = FunctionModel(self.judge)
        critic = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("- fix")]))
        return AgentModels(FunctionModel(generator or self.generator), judge, judge, critic, NAMES)


def _provider_down(messages, info):
    raise RuntimeError("provider down")


def _config(**settings):
    from schemagraph.agent.results import AgentConfig

    return AgentConfig(**{"strategy": "best_of_n", "budget": 2, "batch_size": 2, **settings})


def test_run_scores_with_the_official_comparison_and_resumes(tmp_path):
    from schemagraph.bench import spider2_exec

    scripted = _Models()
    models = scripted.agent_models()
    root, out = _spider2(tmp_path / "s2"), tmp_path / "out"
    cfg = _config(early_stop=1.01)
    result = spider2_exec.run(root, cfg=cfg, models=models, out_dir=out, tag="t")
    assert scripted.tables_seen and set(scripted.tables_seen) == {"movie"}  # read over MCP
    rows = {row["instance_id"]: row for row in result["rows"]}
    assert rows["local901"]["oracle"] == 1
    assert rows["local901"]["candidate_ex"] == {"n0": 1, "n1": 0}
    assert rows["local901"]["ex"] == 1 and rows["local901"]["ex_by_score"] == 1  # judge: filtered
    assert rows["local902"]["ex"] == 1 and rows["local902"]["table_recall"] == 1.0
    assert rows["local902"]["cost_usd"] == 0.0 and rows["local902"]["unpriced"] > 0  # scripted
    assert result["summary"]["overall"]["ex"] == 100.0 and result["summary"]["overall"]["n"] == 2
    assert "mcp_url" not in result["config"]["agent_config"]
    assert (out / "spider2_exec_t.json").exists() and (out / "spider2_exec_t.csv").exists()
    candidates = (out / "spider2_exec_t_candidates.jsonl").read_text().splitlines()
    assert len(candidates) == 4 and {json.loads(line)["ex"] for line in candidates} == {0, 1}

    again = spider2_exec.run(root, cfg=cfg, models=models, out_dir=out, tag="t")
    assert len(again["rows"]) == 2  # resume: nothing to do
    assert len((out / "spider2_exec_t.rows.jsonl").read_text().splitlines()) == 2
    with (out / "spider2_exec_t_candidates.jsonl").open("a") as handle:
        handle.write('{"instance_id": "local901", "sq')  # a run killed mid-write
    spider2_exec.run(root, cfg=cfg, models=models, out_dir=out, tag="t")
    assert len((out / "spider2_exec_t_candidates.jsonl").read_text().splitlines()) == 4
    spider2_exec.append_record(out / "cut.jsonl", {"instance_id": "a"})
    with (out / "cut.jsonl").open("a") as handle:
        handle.write('{"instance_id": "b", "e')
    spider2_exec.append_record(out / "cut.jsonl", {"instance_id": "c"})
    assert [row["instance_id"] for row in spider2_exec.read_rows(out / "cut.jsonl")] == ["a", "c"]
    with pytest.raises(ValueError, match="another configuration"):  # one configuration per tag
        spider2_exec.run(root, cfg=_config(selector=False), models=models, out_dir=out, tag="t")


def test_trace_writes_every_model_call_and_the_candidates_keep_the_advice(tmp_path):
    from schemagraph.bench import spider2_exec

    models = _Models().agent_models()
    root, out = _spider2(tmp_path / "s2"), tmp_path / "out"
    cfg = _config(strategy="refine", budget=3, batch_size=1, early_stop=1.01, trace=True)
    result = spider2_exec.run(root, cfg=cfg, models=models, out_dir=out, tag="t")
    messages = [
        json.loads(line)
        for line in (out / "spider2_exec_t_messages.jsonl").read_text().splitlines()
    ]
    roles = {record["role"] for record in messages}
    assert {"generator", "critic"} <= roles
    assert all(record["messages"] and record["attempt"] >= 1 for record in messages)
    assert {record["instance_id"] for record in messages} == {"local901", "local902"}
    candidates = [
        json.loads(line)
        for line in (out / "spider2_exec_t_candidates.jsonl").read_text().splitlines()
    ]
    assert any(candidate["advice"] == "- fix" for candidate in candidates)  # the critic's text
    untraced = _run_config_hash(cfg, models)
    assert result["config"]["config_hash"] == untraced  # tracing changes no answer


def _run_config_hash(cfg, models):
    from dataclasses import replace

    from schemagraph.bench.spider2_exec import _run_config

    return _run_config(replace(cfg, trace=False), models, seed=0, use_docs=True, concurrency=1)[
        "config_hash"
    ]


# config_hash of AgentConfig() with the NAMES models, seed 0 and docs on, computed on the
# pre-refactor code (backup/agentic-sql-6-pre-refactor, 54828c6): rows written before the
# refactor resume only while this holds.
PRE_REFACTOR_DEFAULT_HASH = "396f29d01a5b"


def _legacy_config(**settings):
    """AgentConfig with the judge material those runs had (before the schema context)."""
    from schemagraph.agent.results import AgentConfig

    legacy = dict(preview_rows=10, judge_evidence_chars=1000, judge_schema=False,
                  judge_findings=False, judge_stats=False)  # fmt: skip
    return AgentConfig(**{**legacy, **settings})


def test_config_hash_ignores_the_mcp_url_and_concurrency():
    from schemagraph.agent.results import AgentConfig
    from schemagraph.bench.spider2_exec import _run_config

    models = _Models().agent_models()
    plain = _run_config(_legacy_config(), models, seed=0, use_docs=True, concurrency=1)
    served = _run_config(
        _legacy_config(mcp_url="http://127.0.0.1:1/mcp"), models, seed=0, use_docs=True,
        concurrency=4,
    )  # fmt: skip
    assert plain["config_hash"] == served["config_hash"] == PRE_REFACTOR_DEFAULT_HASH
    current = _run_config(AgentConfig(), models, seed=0, use_docs=True, concurrency=1)
    assert current["config_hash"] != PRE_REFACTOR_DEFAULT_HASH  # the judge context changes answers


def test_config_records_the_reasoning_effort_only_for_openrouter_models(monkeypatch):
    from schemagraph.agent.results import AgentConfig
    from schemagraph.bench.spider2_exec import _run_config

    models = _Models().agent_models()
    monkeypatch.setenv("SCHEMAGRAPH_REASONING", "high")
    plain = _run_config(_legacy_config(), models, seed=0, use_docs=True, concurrency=1)
    assert "reasoning" not in plain and plain["config_hash"] == PRE_REFACTOR_DEFAULT_HASH
    models.names = {**models.names, "generator": "openrouter:qwen/qwen3.8-max"}
    high = _run_config(AgentConfig(), models, seed=0, use_docs=True, concurrency=1)
    monkeypatch.setenv("SCHEMAGRAPH_REASONING", "low")
    low = _run_config(AgentConfig(), models, seed=0, use_docs=True, concurrency=1)
    assert high["reasoning"] == "high" and high["config_hash"] != low["config_hash"]


def test_summary_reports_the_cost_per_task():
    from schemagraph.bench.spider2_exec import summarize

    rows = [
        {
            "instance_id": f"t{i}",
            "db": "d",
            "ex": 1,
            "cost_usd": cost,
            "unpriced": i % 2,
            "tokens_reasoning": 100,
            "usage": {"generator": {"cost_usd": cost, "input_tokens": 10}},
        }
        for i, cost in enumerate([0.001, 0.002, 0.003, 0.010])
    ]
    overall = summarize(rows)["overall"]
    assert overall["avg_cost_usd"] == 0.004 and overall["total_cost_usd"] == 0.016
    assert overall["p50_cost_usd"] == 0.002 and overall["p90_cost_usd"] == 0.01
    assert overall["max_cost_usd"] == 0.01 and overall["unpriced"] == 2
    assert overall["avg_tokens_reasoning"] == 100
    assert overall["per_task_by_role"]["generator"] == {"cost_usd": 0.004, "input_tokens": 10.0}


def test_failed_tasks_are_error_rows_and_are_retried(tmp_path):
    from schemagraph.bench import spider2_exec

    scripted = _Models()
    good = scripted.agent_models()
    down = scripted.agent_models(generator=_provider_down)
    root, out = _spider2(tmp_path / "s2"), tmp_path / "out"
    cfg = _config(early_stop=1.01)
    first = spider2_exec.run(root, cfg=cfg, models=down, out_dir=out, tag="r")
    assert all(row["error"].startswith("all 2 nodes failed") for row in first["rows"])
    assert first["summary"]["overall"]["errors"] == 2
    second = spider2_exec.run(root, cfg=cfg, models=good, out_dir=out, tag="r")  # retries them
    assert [row["error"] for row in second["rows"]] == [None, None]
    assert second["summary"]["overall"]["ex"] == 100.0
    candidates = (out / "spider2_exec_r_candidates.jsonl").read_text().splitlines()
    assert len(candidates) == 4  # the failed attempts' candidates are gone


def test_judge_only_reports_auroc(tmp_path):
    from pydantic_ai.models.function import FunctionModel

    from schemagraph.agent.results import AgentConfig
    from schemagraph.bench import spider2_judge

    scripted = _Models()
    report = spider2_judge.judge_only(
        _spider2(tmp_path / "s2"),
        judges=["j"],
        judge_models={"j": FunctionModel(scripted.judge)},
        gen_models=scripted.agent_models(),
        cfg=AgentConfig(),
        pool_size=2,
        out_dir=tmp_path / "out",
        tag="jt",
    )
    assert report["tasks"] == 2 and report["candidates"] == 4 and report["positives"] == 3
    judge = report["judges"]["j"]
    assert judge["n"] == 4 and judge["auroc_mean"] == 1.0 and judge["pick_accuracy"] == 100.0
    assert (tmp_path / "out/spider2_judge_jt_pool.jsonl").exists()


def test_judge_pool_retries_failed_generations(tmp_path):
    from pydantic_ai.models.function import FunctionModel

    from schemagraph.agent.models import AgentModels
    from schemagraph.agent.results import AgentConfig
    from schemagraph.bench import spider2_judge

    scripted = _Models()
    root = _spider2(tmp_path / "s2")
    down = AgentModels(FunctionModel(_provider_down), None, None, None, NAMES)
    settings = {
        "judges": ["j"],
        "judge_models": {"j": FunctionModel(scripted.judge)},
        "cfg": AgentConfig(),
        "pool_size": 2,
        "out_dir": tmp_path / "out",
        "tag": "p",
    }
    first = spider2_judge.judge_only(root, gen_models=down, **settings)
    assert first["tasks"] == 0 and first["pool_errors"] == 2
    second = spider2_judge.judge_only(root, gen_models=scripted.agent_models(), **settings)
    assert second["tasks"] == 2 and second["pool_errors"] == 0
    assert second["judges"]["j"]["n"] == 4


def test_a_database_build_blocks_neither_another_server_nor_an_executor(tmp_path, store_snapshot):
    from schemagraph.bench.spider2_exec import Runner

    runner = Runner(_spider2(tmp_path / "s2"))
    linker = Linker(build_graph([store_snapshot]))
    release = threading.Event()

    def get(dialect: str, db: str):
        if db == "slow":
            release.wait(10)
        return linker, None, None, None

    runner.graphs.get = get
    servers = McpServers(runner, Counter({"Db-X": 2, "slow": 1}))

    async def scenario() -> None:
        fast_url = await servers.url("Db-X")
        slow = asyncio.create_task(servers.url("slow"))
        await asyncio.sleep(0.1)  # the slow build now holds the graph lock
        try:
            assert await asyncio.wait_for(servers.url("Db-X"), timeout=1.0) == fast_url
            executor = await asyncio.wait_for(asyncio.to_thread(runner.executor, "Db-X"), 1.0)
            assert executor.dialect == "sqlite"
            assert not slow.done()
        finally:
            release.set()
            await slow
            await servers.aclose()
            runner.close()

    asyncio.run(scenario())
