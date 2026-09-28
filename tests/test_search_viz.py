"""The search-tree page (``agent/viz.py``), its benchmark loader and the two CLI surfaces.

No model is called: the trees come from fixture candidates files (``fixtures/viz``) and from a
hand-built `AnswerResult`.
"""

from __future__ import annotations

import json
import re
import shutil
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from schemagraph.agent import viz
from schemagraph.agent.results import (
    AnswerResult,
    Candidate,
    CheckReport,
    ExecResult,
    Finding,
    Judgement,
    UsageSummary,
    candidate_record,
    exec_error_text,
)
from schemagraph.agent.viz import (
    MODEL_PALETTE,
    UNKNOWN_GENERATOR,
    check_html_path,
    model_colors,
    render_index_html,
    render_search_html,
    short_model_name,
    tree_from_answer,
    tree_from_records,
)
from schemagraph.bench.spider2_exec import load_search_trees, run_files
from schemagraph.cli import app as cli_app
from tests.conftest import FIXTURES

VIZ = FIXTURES / "viz"
HOSTILE_ID = "n\"><b onmouseover='x'>"
MIX = VIZ / "spider2_exec_mix_candidates.jsonl"
LEGACY = VIZ / "spider2_exec_legacy_candidates.jsonl"
GLM = "openrouter:z-ai/glm-5.3"
QWEN = "openrouter:qwen/qwen3.8-max"
GEMINI = "openrouter:google/gemini-3.8-flash"


def _mix_trees(**kwargs):
    rows, report = run_files(MIX)
    return {tree.key: tree for tree in load_search_trees(MIX, rows, report, **kwargs)}


def _row_of(page: str, node_id: str) -> str:
    """Return the outline row (the ``<a>`` element) of one node."""
    match = re.search(rf'<a href="#node-{node_id}">(.*?)</a>', page)
    assert match, node_id
    return match.group(1)


# ----------------------------------------------------------------------------- 1. single model
def test_a_single_model_run_names_the_run_generator_for_every_node():
    rows, report = run_files(LEGACY)
    (tree,) = load_search_trees(LEGACY, rows, report)
    assert {node.model for node in tree.nodes} == {QWEN} and tree.models == (QWEN,)
    assert tree.chosen_id is None  # no rows file: the choice is unknown, not guessed
    (unknown,) = load_search_trees(LEGACY)  # no report either
    assert {node.model for node in unknown.nodes} == {UNKNOWN_GENERATOR}
    page = render_search_html(tree)
    assert "qwen3.8-max" in _row_of(page, "n0") and MODEL_PALETTE[0] in _row_of(page, "n0")


def test_a_record_generator_wins_over_the_default_model():
    records = [{"id": "n0", "generator": GLM}, {"id": "n1", "generator": None}]
    tree = tree_from_records(records, default_model=QWEN)
    assert [node.model for node in tree.nodes] == [GLM, QWEN]


def test_short_model_names_drop_the_provider_and_vendor():
    assert short_model_name("openrouter:qwen/qwen3.8-max") == "qwen3.8-max"
    assert short_model_name("typesafe:jev") == "jev" and short_model_name("plain") == "plain"
    assert short_model_name("odd/") == "odd/"  # all prefix: kept whole


# ----------------------------------------------------------------------------- 2. several models
def test_several_generators_get_their_configured_colours_and_a_legend_row_each():
    tree = _mix_trees()["local901"]
    assert tree.models == (GLM, QWEN, GEMINI)  # configured order, gemini with no node
    colors = model_colors(tree.models)
    assert list(colors.values()) == list(MODEL_PALETTE[:3])
    page = render_search_html(tree)
    assert f'fill="{MODEL_PALETTE[0]}"' in _row_of(page, "n0")  # glm
    assert f'stroke="{MODEL_PALETTE[1]}"' in _row_of(page, "n1")  # qwen, hollow
    legend = page[page.index('class="legend"') : page.index("Search tree")]
    assert all(model in legend for model in (GLM, QWEN, GEMINI))
    gemini_row = legend[legend.index(GEMINI) :].split("</tr>")[0]
    assert "<td>0</td>" in gemini_row  # listed, with no nodes
    assert "EX ✓" in legend  # a benchmark run shows the correct nodes per model


def test_a_model_outside_the_configuration_is_listed_after_it_sorted():
    records = [{"id": "n0", "generator": "z"}, {"id": "n1", "generator": "a"}, {"id": "n2"}]
    tree = tree_from_records(records, generators=["m"], default_model="b")
    assert tree.models == ("m", "a", "b", "z")


# ----------------------------------------------------------------------------- 3. palette
def test_the_palette_cycles_after_its_last_colour():
    models = [f"model{index}" for index in range(len(MODEL_PALETTE) + 1)]
    colors = model_colors(models)
    assert colors["model0"] == colors[f"model{len(MODEL_PALETTE)}"] == MODEL_PALETTE[0]
    assert len(set(colors.values())) == len(MODEL_PALETTE)


# ----------------------------------------------------------------------------- 4. marks
def test_marks_show_the_choice_failures_exec_errors_and_ex():
    tree = _mix_trees()["local901"]
    page = render_search_html(tree)
    chosen = _row_of(page, "n10")
    assert 'class="ring"' in chosen and "★ chosen (selector)" in chosen and "EX ✓" in chosen
    failed = _row_of(page, "n2")
    assert "dashed" in failed and "⚠ failed" in failed and "EX" not in failed
    exec_error = _row_of(page, "n1")
    assert "hollow" in exec_error and "exec error" in exec_error and "EX ✗" in exec_error
    assert 'class="ring"' not in _row_of(page, "n0")
    section = page[page.index('id="node-n1"') :].split("</section>")[0]
    assert "runtime: no such column: x" in section  # the recorded exec_error
    assert "chosen</dt><dd>n10 by selector" in page


def test_a_legacy_record_that_did_not_run_is_marked_without_a_message():
    rows, report = run_files(LEGACY)
    (tree,) = load_search_trees(LEGACY, rows, report)
    page = render_search_html(tree)
    row = _row_of(page, "n1")  # ok: false, no exec_error key
    assert "hollow" in row and "exec error" in row
    section = page[page.index('id="node-n1"') :].split("</section>")[0]
    assert "<dt>exec error</dt>" not in section and "<dt>error</dt>" not in section


def test_an_ask_tree_has_no_ex_marks():
    tree = tree_from_records([{"id": "n0", "ok": True, "score": 0.5}], chosen_id="n0")
    page = render_search_html(tree)
    assert "EX ✓" not in page and "EX ✗" not in page and "★ chosen" in _row_of(page, "n0")


# ----------------------------------------------------------------------------- 5. layout
def test_the_outline_is_depth_first_in_ask_order_with_orphans_on_the_root():
    tree = _mix_trees()["local901"]
    assert [node.id for node in tree.nodes] == ["n0", "n10", "n1", "n3", "n4", "n2", "n5"]
    levels = {node.id: node.level for node in tree.nodes}
    assert levels == {"n0": 1, "n10": 2, "n1": 1, "n3": 2, "n4": 3, "n2": 1, "n5": 1}
    page = render_search_html(tree)
    ys = [float(re.search(r'<circle cx="[^"]+" cy="([^"]+)"', _row_of(page, node.id)).group(1))
          for node in tree.nodes]  # fmt: skip
    assert ys == sorted(ys) and len(set(ys)) == len(ys)  # one row each, top to bottom
    x_of = {
        node.id: float(re.search(r'<circle cx="([^"]+)"', _row_of(page, node.id)).group(1))
        for node in tree.nodes
    }
    assert x_of["n4"] - x_of["n3"] == x_of["n3"] - x_of["n1"] == viz.INDENT


def test_numeric_ask_order_and_a_parent_cycle_keep_every_node():
    records = [
        {"id": "n10"}, {"id": "n2"}, {"id": "x"},
        {"id": "a", "parent_id": "b"}, {"id": "b", "parent_id": "a"},
    ]  # fmt: skip
    tree = tree_from_records(records)
    # the drafts in ask order (n2 before n10, other ids after), then the cycle from its first id
    assert [node.id for node in tree.nodes] == ["n2", "n10", "x", "a", "b"]
    assert [node.level for node in tree.nodes] == [1, 1, 1, 1, 2]
    edges = re.findall(r'<path class="edge" d="M([\d.]+) ([\d.]+) V([\d.]+)', render_search_html(tree))
    root_start = (viz._node_x(0), viz._row_y(0) + viz.NODE_RADIUS)
    starts = [(float(x), float(y)) for x, y, _ in edges]
    assert starts[3] == root_start  # a: its parent b is drawn below it, so it hangs off the root
    assert starts[4] == (viz._node_x(1), viz._row_y(4) + viz.NODE_RADIUS)  # b: under a
    assert all(float(y) < float(end) for _, y, end in edges)  # every edge runs downwards


def test_a_self_parent_hangs_off_the_root():
    page = render_search_html(tree_from_records([{"id": "n0", "parent_id": "n0"}]))
    (edge,) = re.findall(r'<path class="edge" d="M[\d.]+ ([\d.]+) V', page)
    assert float(edge) == viz._row_y(0) + viz.NODE_RADIUS


# ----------------------------------------------------------------------------- 6. escaping
def test_every_string_is_escaped_and_the_page_has_no_script():
    evil = '<script>alert("x")</script>'
    records = [
        {
            "id": "n0", "generator": evil, "sql": evil, "rationale": evil, "advice": evil,
            "feedback": [evil], "findings": [evil], "error": None, "exec_error": evil,
            "rubric": {evil: 0.5}, "missing": {evil: 0.4}, "score_parts": {"det": 1.0},
            "action": evil,
        },
        {"id": "n1", "parent_id": "n0", "error": evil},
        {"id": HOSTILE_ID, "parent_id": HOSTILE_ID + "p", "ok": True},
        {"id": HOSTILE_ID + "p", "parent_id": 7},
    ]  # fmt: skip
    tree = tree_from_records(
        records, title=evil, chosen_id="n0", chosen_by=evil, facts=[(evil, evil)], key=evil
    )
    for page in (render_search_html(tree), render_index_html(evil, [(evil, tree)])):
        assert "<script" not in page.lower()
        assert "&lt;script&gt;" in page
        assert "default-src &#x27;none&#x27;" in page or "default-src 'none'" in page
        assert 'http-equiv="Content-Security-Policy"' in page
    assert viz.page_filename(tree) == "_script_alert_x_script_.html"
    page = render_search_html(tree)
    assert HOSTILE_ID not in page and '"><b' not in page  # no attribute is closed early
    assert 'href="#node-n&quot;&gt;&lt;b onmouseover=&#x27;x&#x27;&gt;"' in page
    assert 'id="node-n&quot;&gt;&lt;b onmouseover=&#x27;x&#x27;&gt;p"' in page
    assert tree.node(HOSTILE_ID + "p").parent_id == "7"  # coerced to text


def test_exec_error_text_never_prints_none():
    assert exec_error_text(None) is None and exec_error_text(ExecResult(ok=True)) is None
    assert exec_error_text(ExecResult(ok=False)) == "error"
    assert exec_error_text(ExecResult(ok=False, error_kind="timeout")) == "timeout"
    assert exec_error_text(ExecResult(ok=False, error="boom")) == "error: boom"


# ----------------------------------------------------------------------------- 7. paths
def test_the_html_path_is_checked(tmp_path, monkeypatch):
    assert check_html_path(tmp_path / "tree.html") == tmp_path / "tree.html"
    assert check_html_path(tmp_path / "tree.HTM").suffix == ".HTM"
    with pytest.raises(ValueError, match="give a .html path"):
        check_html_path(tmp_path / "tree.png")
    with pytest.raises(ValueError, match="does not exist"):
        check_html_path(tmp_path / "missing" / "tree.html")
    (tmp_path / "dir.html").mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        check_html_path(tmp_path / "dir.html")
    existing = tmp_path / "old.html"
    existing.write_text("", encoding="utf-8")
    monkeypatch.setattr(viz.os, "access", lambda path, mode: path != existing)
    with pytest.raises(ValueError, match="old.html is not writable"):
        check_html_path(existing)  # its directory is writable, the file is not
    monkeypatch.setattr(viz.os, "access", lambda path, mode: False)
    with pytest.raises(ValueError, match="not writable"):
        check_html_path(tmp_path / "tree.html")


def test_write_search_html_writes_the_page(tmp_path):
    tree = tree_from_records([{"id": "n0"}], title="q")
    written = viz.write_search_html(tree, tmp_path / "t.html")
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")


# ----------------------------------------------------------------------------- 8. loader
def test_the_loader_keeps_the_last_record_skips_cut_lines_and_reads_the_rows():
    trees = _mix_trees()
    assert list(trees) == ["local901", "local902"]
    tree = trees["local901"]
    assert tree.node("n3").score == 0.7 and tree.node("n3").feedback  # the later record
    assert len(trees["local902"].nodes) == 1  # the cut-short n1 line is skipped
    assert (tree.chosen_id, tree.chosen_by, tree.strategy) == ("n10", "selector", "abmcts")
    facts = dict(tree.facts)
    assert facts["database"] == "shop" and facts["EX of the pick"] == "1"
    assert facts["cost"] == "$0.1234" and facts["stopped early"] == "yes"
    assert list(_mix_trees(tasks={"local902"})) == ["local902"]


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_without_a_report_a_model_keeps_its_colour_on_every_page(tmp_path):
    candidates = _write_jsonl(tmp_path / "run_candidates.jsonl", [
        {"instance_id": "a", "id": "n0", "generator": QWEN, "ok": True},
        {"instance_id": "a", "id": "n1", "generator": GLM, "ok": True},
        {"instance_id": "b", "id": "n0", "generator": QWEN, "ok": True},
        {"instance_id": "b", "generator": GLM},  # no id: skipped, not a traceback
    ])  # fmt: skip
    first, second = load_search_trees(candidates)
    assert first.models == second.models == (QWEN, GLM)  # the whole file's models, sorted
    assert [node.id for node in second.nodes] == ["n0"]
    qwen = model_colors(first.models)[QWEN]
    for tree in (first, second):
        assert f'fill="{qwen}"' in _row_of(render_search_html(tree), "n0")


def test_a_report_that_is_not_a_run_report_is_an_error(tmp_path):
    for text in ("[1, 2]", '{"config": "x"}', "not json"):
        report = tmp_path / "report.json"
        report.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_search_trees(MIX, report_path=report)


def test_run_files_finds_the_siblings(tmp_path):
    rows, report = run_files(tmp_path / "spider2_exec_t_candidates.jsonl")
    assert (rows.name, report.name) == ("spider2_exec_t.rows.jsonl", "spider2_exec_t.json")
    assert run_files(tmp_path / "pool.jsonl")[0].name == "pool.jsonl.rows.jsonl"


def test_the_index_counts_pick_and_any_node_ex():
    trees = _mix_trees()
    pages = [(f"{key}.html", tree) for key, tree in trees.items()]
    index = render_index_html("run", pages)
    assert "2 tasks, pick EX 1/2, any node correct 1/2" in index
    assert '<a href="local901.html">local901</a>' in index and "n10 (selector)" in index


# ----------------------------------------------------------------------------- 9. round trip
def _candidates() -> list[Candidate]:
    judged = Judgement(model="jev", fields={"answers_question": 0.9}, missing={"t": 0.4}, mean=0.9)
    checks = CheckReport(parsed=True, findings=[Finding(code="empty_result", severity="warn",
                                                        message="no rows")])  # fmt: skip
    ran = ExecResult(ok=True, columns=["a"], rows=[[1]], row_count=1)
    broken = ExecResult(ok=False, error="no such column: x", error_kind="runtime")
    return [
        Candidate(id="n0", generator=GLM, sql="SELECT 1", exec=ran, checks=checks,
                  judgement=judged, score=0.7, score_parts={"det": 0.9, "judge": 0.9, "x": 0.9},
                  feedback=["no rows"], rationale="r"),
        Candidate(id="n1", parent_id="n0", depth=1, generator=QWEN, sql="SELECT x", exec=broken,
                  score=0.05, advice="- fix"),
        Candidate(id="n2", generator=GLM, error="node timed out"),
    ]  # fmt: skip


def _answer(candidates: list[Candidate]) -> AnswerResult:
    return AnswerResult(
        question="how many orders?", sql="SELECT 1", result=None, chosen_id="n0",
        chosen_by="score", score=0.7, strategy="abmcts", budget=4, nodes=len(candidates),
        stopped_early=False, candidates=candidates, usage=UsageSummary(),
        models={"generator": GLM},
    )  # fmt: skip


def test_the_candidates_file_record_and_the_answer_give_the_same_tree():
    pytest.importorskip("pydantic_ai")
    from schemagraph.bench.spider2_exec import _candidate_record

    candidates = _candidates()
    task = SimpleNamespace(instance_id="local1")
    records = [_candidate_record(task, candidate, None) for candidate in candidates]
    assert [record["exec_error"] for record in records] == [None, "runtime: no such column: x", None]
    shared = list(candidate_record(candidates[0]))
    assert shared == [
        "id", "parent_id", "depth", "action", "generator", "sql", "rationale", "advice",
        "feedback", "score", "score_parts", "rubric", "missing", "findings", "ok", "row_count",
        "error", "exec_error",
    ]  # fmt: skip
    assert list(records[0]) == ["instance_id", *shared, "fingerprint", "ex"]  # the bench adds 3
    from_file = tree_from_records(records, default_model=GLM, generators=(GLM, QWEN))
    from_answer = tree_from_answer(_answer(candidates), generators=(GLM, QWEN))
    assert from_file.nodes == from_answer.nodes and from_file.models == from_answer.models
    assert from_answer.title == "how many orders?" and from_answer.chosen_id == "n0"
    assert dict(from_answer.facts)["nodes"] == "3 of 4"


def test_the_answer_tree_falls_back_to_the_run_generator():
    candidates = [Candidate(id="n0", sql="SELECT 1")]  # a single-generator run names none
    tree = tree_from_answer(_answer(candidates))
    assert tree.nodes[0].model == GLM and tree.models == (GLM,)


# ----------------------------------------------------------------------------- 10. viz-search
def test_viz_search_writes_a_page_per_task_and_an_index(tmp_path):
    out = tmp_path / "pages"
    result = CliRunner().invoke(cli_app, ["viz-search", str(MIX), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in out.iterdir()) == [
        "index.html", "local901.html", "local902.html",
    ]  # fmt: skip
    assert str(out / "index.html") in result.stdout
    page = (out / "local901.html").read_text(encoding="utf-8")
    assert 'class="legend"' in page and "<script" not in page


def test_viz_search_writes_one_task_to_a_file_and_warns_without_rows(tmp_path):
    runner = CliRunner()
    target = tmp_path / "one.html"
    result = runner.invoke(cli_app, ["viz-search", str(LEGACY), "--out", str(target)])
    assert result.exit_code == 0, result.output
    assert "warning: no rows file" in result.stderr and target.exists()
    several = runner.invoke(cli_app, ["viz-search", str(MIX), "--out", str(target)])
    assert several.exit_code == 2 and "pick one with --task" in several.output
    one = runner.invoke(cli_app, ["viz-search", str(MIX), "-t", "local902", "--out", str(target)])
    assert one.exit_code == 0 and "local902" in target.read_text(encoding="utf-8")
    none = runner.invoke(cli_app, ["viz-search", str(MIX), "-t", "nope", "--out", str(target)])
    assert none.exit_code == 2 and "no candidates of these tasks" in none.output


def test_viz_search_defaults_to_a_directory_next_to_the_run(tmp_path):
    for path in VIZ.glob("spider2_exec_mix*"):
        shutil.copy(path, tmp_path / path.name)
    candidates = tmp_path / MIX.name
    result = CliRunner().invoke(cli_app, ["viz-search", str(candidates), "-t", "local901"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "spider2_exec_mix_viz" / "index.html").exists()
    assert "warning" not in result.stderr


def test_viz_search_gives_every_task_its_own_safe_page(tmp_path):
    keys = ["a b", "a_b", "A_B", "CON", "nul.x", "local1"]
    candidates = _write_jsonl(
        tmp_path / "run_candidates.jsonl", [{"instance_id": key, "id": "n0"} for key in keys]
    )
    out = tmp_path / "pages"
    result = CliRunner().invoke(cli_app, ["viz-search", str(candidates), "--out", str(out)])
    assert result.exit_code == 0, result.output
    expected = ["a_b.html", "a_b-2.html", "A_B-3.html", "_CON.html", "_nul.x.html", "local1.html"]
    assert sorted(path.name for path in out.iterdir()) == sorted([*expected, "index.html"])
    index = (out / "index.html").read_text(encoding="utf-8")
    for key, name in zip(keys, expected, strict=True):
        assert f'<a href="{name}">{key}</a>' in index
        assert f"<h1>{key}</h1>" in (out / name).read_text(encoding="utf-8")


def test_viz_search_reads_the_rows_and_report_given(tmp_path):
    candidates = tmp_path / "pool.jsonl"  # no siblings under the run's names
    shutil.copy(MIX, candidates)
    target = tmp_path / "one.html"
    arguments = ["viz-search", str(candidates), "-t", "local901", "--out", str(target)]
    arguments += ["--rows", str(VIZ / "spider2_exec_mix.rows.jsonl")]
    arguments += ["--report", str(VIZ / "spider2_exec_mix.json")]
    result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 0, result.output
    assert "warning" not in result.stderr and "search tree written to" in result.stdout
    page = target.read_text(encoding="utf-8")
    assert "★ chosen (selector)" in _row_of(page, "n10") and GEMINI in page


def test_viz_search_reports_a_bad_report_and_an_empty_file(tmp_path):
    runner = CliRunner()
    report = tmp_path / "bad.json"
    report.write_text("not json", encoding="utf-8")
    bad = runner.invoke(cli_app, ["viz-search", str(MIX), "--report", str(report)])
    assert bad.exit_code == 1 and bad.stderr.count("error:") == 1
    assert isinstance(bad.exception, SystemExit)  # no traceback
    empty = tmp_path / "empty_candidates.jsonl"
    empty.write_text("", encoding="utf-8")
    result = runner.invoke(cli_app, ["viz-search", str(empty), "--out", str(tmp_path / "o")])
    assert result.exit_code == 2 and "no candidates in" in result.output
    assert "--task" not in result.output


# ----------------------------------------------------------------------------- 11. ask --viz
@pytest.fixture
def answered(monkeypatch):
    """Patch ``Engine.answer`` to return a fixed answer; the calls land in the returned list."""
    for module in ("pydantic_ai", "treequest", "fastmcp"):  # ask checks the agent extra first
        pytest.importorskip(module)
    from schemagraph.engine import Engine

    calls: list[dict] = []

    def answer(self, question, **kwargs):
        calls.append({"question": question, **kwargs})
        return _answer(_candidates())

    monkeypatch.setattr(Engine, "answer", answer)
    return calls


def test_ask_writes_the_search_tree_and_keeps_json_clean(tmp_path, answered):
    target = tmp_path / "tree.html"
    arguments = ["ask", "q", "--json", "--viz", str(target), "--home", str(tmp_path / "home")]
    arguments += ["--gen-model", QWEN, "--gen-model", GLM, "--strategy", "best_of_n"]
    result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["chosen_id"] == "n0"  # nothing else on stdout
    assert f"search tree written to {target}" in result.stderr
    page = target.read_text(encoding="utf-8")
    assert "how many orders?" in page and "<script" not in page
    assert f'fill="{MODEL_PALETTE[1]}"' in _row_of(page, "n0")  # glm is the second --gen-model


@pytest.mark.parametrize("name", ["tree.txt", "missing/tree.html"])
def test_ask_checks_the_viz_path_before_answering(tmp_path, answered, name):
    arguments = ["ask", "q", "--viz", str(tmp_path / name), "--home", str(tmp_path / "home")]
    result = CliRunner().invoke(cli_app, arguments)
    assert result.exit_code == 2 and "--viz" in result.output
    assert answered == []  # no search, no spend


def test_ask_reports_a_failed_write_after_the_answer(tmp_path, answered, monkeypatch):
    def full_disk(tree, path):
        raise OSError("disk full")

    monkeypatch.setattr(viz, "write_search_html", full_disk)
    arguments = ["ask", "q", "--json", "--viz", str(tmp_path / "t.html")]
    result = CliRunner().invoke(cli_app, [*arguments, "--home", str(tmp_path / "home")])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert json.loads(result.stdout)["chosen_id"] == "n0"  # the answer is still printed
    assert result.stderr.strip().splitlines() == ["error: search tree not written: disk full"]
