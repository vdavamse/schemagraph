"""The search replay page (``agent/viz.py``), its benchmark loader and the two CLI surfaces.

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
    expand_prompt,
    stored_prompt,
)
from schemagraph.agent.viz import (
    MODEL_FAMILIES,
    OTHER_COLOR,
    UNKNOWN_GENERATOR,
    check_html_path,
    drawn_parents,
    layout_tree,
    model_colors,
    model_family,
    render_index_html,
    render_search_html,
    replay_events,
    script_hash,
    short_model_name,
    tree_from_answer,
    tree_from_records,
)
from schemagraph.agent.viz_assets import PLAYER_JS
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
CODER = "openrouter:qwen/qwen3.8-coder"  # a second qwen: it sorts before QWEN
FAMILY_COLOR = {family: color for family, _, color in MODEL_FAMILIES}


def _mix_trees(**kwargs):
    rows, report, contexts = run_files(MIX)
    trees = load_search_trees(MIX, rows, report, contexts_path=contexts, **kwargs)
    return {tree.key: tree for tree in trees}


def _index(tree, node_id: str) -> int:
    """Return a node's index, which names its elements on the page."""
    return next(index for index, node in enumerate(tree.nodes) if node.id == node_id)


def _group_of(page: str, tree, node_id: str) -> str:
    """Return the drawn group (the ``<a>`` element's content) of one node."""
    index = _index(tree, node_id)
    match = re.search(rf'<a href="#node-{index}" class="node-g" id="g-{index}">(.*?)</a>', page)
    assert match, node_id
    return match.group(1)


def _section_of(page: str, tree, node_id: str) -> str:
    """Return a node's section in the panel."""
    start = page.index(f'id="node-{_index(tree, node_id)}"')
    return page[start:].split("</section>")[0]


def _events_data(page: str) -> dict:
    """Return the replay's JSON data block."""
    block = page.split('<script type="application/json" id="search-events">')[1]
    return json.loads(block.split("</script>")[0])


# ----------------------------------------------------------------------------- 1. single model
def test_a_single_model_run_names_the_run_generator_for_every_node():
    rows, report, _ = run_files(LEGACY)
    (tree,) = load_search_trees(LEGACY, rows, report)
    assert {node.model for node in tree.nodes} == {QWEN} and tree.models == (QWEN,)
    assert tree.chosen_id is None  # no rows file: the choice is unknown, not guessed
    (unknown,) = load_search_trees(LEGACY)  # no report either
    assert {node.model for node in unknown.nodes} == {UNKNOWN_GENERATOR}
    page = render_search_html(tree)
    assert "qwen3.8-max" in _section_of(page, tree, "n0")
    assert f"--c:{FAMILY_COLOR['qwen']}" in _group_of(page, tree, "n0")


def test_a_record_generator_wins_over_the_default_model():
    records = [{"id": "n0", "generator": GLM}, {"id": "n1", "generator": None}]
    tree = tree_from_records(records, default_model=QWEN)
    assert [node.model for node in tree.nodes] == [GLM, QWEN]


def test_short_model_names_drop_the_provider_and_vendor():
    assert short_model_name("openrouter:qwen/qwen3.8-max") == "qwen3.8-max"
    assert short_model_name("typesafe:jev") == "jev" and short_model_name("plain") == "plain"
    assert short_model_name("odd/") == "odd/"  # all prefix: kept whole


# ----------------------------------------------------------------------------- 2. several models
def test_several_generators_get_their_family_colours_and_a_legend_row_each():
    tree = _mix_trees()["local901"]
    assert tree.models == (GLM, QWEN, GEMINI)  # configured order, gemini with no node
    colors = model_colors(tree.models)
    assert list(colors.values()) == [FAMILY_COLOR[name] for name in ("glm", "qwen", "gemini")]
    page = render_search_html(tree)
    assert f"--c:{FAMILY_COLOR['glm']}" in _group_of(page, tree, "n0")
    assert f"--c:{FAMILY_COLOR['qwen']}" in _group_of(page, tree, "n1")  # hollow: an outline
    legend = page[page.index('class="legend"') : page.index('class="player"')]
    assert all(model in legend for model in (GLM, QWEN, GEMINI))
    gemini_row = legend[legend.index(GEMINI) :].split("</tr>")[0]
    assert "<td>gemini</td>" in gemini_row and "<td>0</td>" in gemini_row  # listed, no nodes
    assert "EX ✓" in legend  # a benchmark run shows the correct nodes per model


def test_a_model_outside_the_configuration_is_listed_after_it_sorted():
    records = [{"id": "n0", "generator": "z"}, {"id": "n1", "generator": "a"}, {"id": "n2"}]
    tree = tree_from_records(records, generators=["m"], default_model="b")
    assert tree.models == ("m", "a", "b", "z")


# ----------------------------------------------------------------------------- 3. palette
def test_a_model_keeps_its_family_colour_and_a_family_shares_shades():
    assert [model_family(name) for name in (GLM, QWEN, GEMINI)] == ["glm", "qwen", "gemini"]
    assert model_family("anthropic/claude-opus-5-5") == "claude"
    assert model_family("openrouter:openai/gpt-6-sol") == "gpt"
    assert model_family("mystery-7b") == "other"
    alone = model_colors([QWEN])[QWEN]
    mixed = model_colors([GLM, QWEN, "openrouter:qwen/qwen3.8-coder", "mystery-7b"])
    assert mixed[QWEN] == alone == FAMILY_COLOR["qwen"]  # whatever else the run holds
    assert mixed["openrouter:qwen/qwen3.8-coder"] not in (FAMILY_COLOR["qwen"], mixed[GLM])
    assert mixed["mystery-7b"] == OTHER_COLOR
    shades = model_colors([f"qwen-{index}" for index in range(len(viz.SHADE_STEPS) + 1)])
    assert len(set(shades.values())) == len(viz.SHADE_STEPS)  # the shades cycle
    assert shades["qwen-0"] == shades[f"qwen-{len(viz.SHADE_STEPS)}"]


# ----------------------------------------------------------------------------- 4. marks
def test_marks_show_the_choice_failures_exec_errors_and_ex():
    tree = _mix_trees()["local901"]
    page = render_search_html(tree)
    chosen = _group_of(page, tree, "n10")
    assert 'class="chosen-ring"' in chosen and "★ chosen (selector)" in chosen
    assert 'class="ex good"' in chosen
    failed = _group_of(page, tree, "n2")
    assert "node dashed" in failed and "⚠ failed" in failed and 'class="score"' not in failed
    exec_error = _group_of(page, tree, "n1")
    assert "node hollow" in exec_error and "exec error" in exec_error
    assert 'class="ex bad"' in exec_error
    assert 'class="chosen-ring"' not in _group_of(page, tree, "n0")
    assert "runtime: no such column: x" in _section_of(page, tree, "n1")  # the recorded error
    assert "chosen</dt><dd>n10 by selector" in page


def test_a_legacy_record_that_did_not_run_is_marked_without_a_message():
    rows, report, _ = run_files(LEGACY)
    (tree,) = load_search_trees(LEGACY, rows, report)
    page = render_search_html(tree)
    group = _group_of(page, tree, "n1")  # ok: false, no exec_error key
    assert "node hollow" in group and "exec error" in group
    section = _section_of(page, tree, "n1")
    assert "<dt>exec error</dt>" not in section and "<dt>error</dt>" not in section


def test_an_ask_tree_has_no_ex_marks():
    tree = tree_from_records([{"id": "n0", "ok": True, "score": 0.5}], chosen_id="n0")
    page = render_search_html(tree)
    assert "EX ✓" not in page and "EX ✗" not in page and 'class="ex ' not in page
    assert "★ chosen" in _group_of(page, tree, "n0")


# ----------------------------------------------------------------------------- 5. layout
def test_the_tree_is_top_down_with_parents_over_their_children_and_orphans_on_the_root():
    tree = _mix_trees()["local901"]
    assert [node.id for node in tree.nodes] == ["n0", "n10", "n1", "n3", "n4", "n2", "n5"]
    levels = {node.id: node.level for node in tree.nodes}
    assert levels == {"n0": 1, "n10": 2, "n1": 1, "n3": 2, "n4": 3, "n2": 1, "n5": 1}
    layout = layout_tree(tree)
    assert layout.parents["n5"] is None  # its parent n99 was never saved
    x = {node_id: position[0] for node_id, position in layout.positions.items()}
    y = {node_id: position[1] for node_id, position in layout.positions.items()}
    assert y["n0"] == y["n1"] == y["n2"] < y["n10"] == y["n3"] < y["n4"]  # one row per level
    leaves = ["n10", "n4", "n2", "n5"]  # side by side, in pre-order
    assert [x[leaf] for leaf in leaves] == sorted(x[leaf] for leaf in leaves)
    assert x["n0"] == x["n10"] and x["n1"] == x["n3"] == x["n4"]  # over their only child
    assert layout.root[0] == (x["n0"] + x["n5"]) / 2  # the question over the drafts
    assert layout.root[1] < y["n0"]


def test_numeric_ask_order_and_a_parent_cycle_keep_every_node():
    records = [
        {"id": "n10"}, {"id": "n2"}, {"id": "x"},
        {"id": "a", "parent_id": "b"}, {"id": "b", "parent_id": "a"},
    ]  # fmt: skip
    tree = tree_from_records(records)
    # the drafts in ask order (n2 before n10, other ids after), then the cycle from its first id
    assert [node.id for node in tree.nodes] == ["n2", "n10", "x", "a", "b"]
    assert [node.level for node in tree.nodes] == [1, 1, 1, 1, 2]
    # a hangs off the question (its parent b is drawn below it), b under a
    assert drawn_parents(tree) == {"n2": None, "n10": None, "x": None, "a": None, "b": "a"}
    page = render_search_html(tree)
    assert page.count('class="edge"') == len(records)  # one edge into every node


def test_a_self_parent_hangs_off_the_root():
    tree = tree_from_records([{"id": "n0", "parent_id": "n0"}])
    assert drawn_parents(tree) == {"n0": None}
    (event_ask, event_tell) = replay_events(tree)
    assert event_ask.path == event_tell.path == ()


# ----------------------------------------------------------------------------- 6. escaping
def test_every_string_is_escaped_and_only_the_hashed_player_script_runs():
    evil = '<script>alert("x")</script>'
    records = [
        {
            "id": "n0", "generator": evil, "sql": evil, "rationale": evil, "advice": evil,
            "feedback": [evil], "findings": [evil], "error": None, "exec_error": evil,
            "rubric": {evil: 0.5}, "missing": {evil: 0.4}, "score_parts": {"det": 1.0},
            "action": evil, "prompt": evil + "<<schema 00ff>>",
        },
        {"id": "n1", "parent_id": "n0", "error": evil},
        {"id": HOSTILE_ID, "parent_id": HOSTILE_ID + "p", "ok": True},
        {"id": HOSTILE_ID + "p", "parent_id": 7},
    ]  # fmt: skip
    tree = tree_from_records(
        records, title=evil, chosen_id="n0", chosen_by=evil, facts=[(evil, evil)], key=evil,
        contexts={"00ff": evil}, instructions=evil,
    )  # fmt: skip
    page = render_search_html(tree)
    index = render_index_html(evil, [(evil, tree)])
    assert "<script" not in index.lower() and "default-src &#x27;none&#x27;" in index
    # the page's only scripts: the JSON data block and the constant player, admitted by hash
    assert page.count("<script") == 2 and f"<script>{PLAYER_JS}</script>" in page
    assert f"script-src {script_hash(PLAYER_JS)}".replace("'", "&#x27;") in page
    for text in (page, index):
        assert "&lt;script&gt;" in text and 'http-equiv="Content-Security-Policy"' in text
    data = page.split('id="search-events">')[1].split("</script>")[0]
    assert "<" not in data and ">" not in data  # a string cannot close the data block
    assert _events_data(page)["events"][-1]["caption"].startswith("Final pick: n0 by <script>")
    assert viz.page_filename(tree) == "_script_alert_x_script_.html"
    assert HOSTILE_ID not in page and '"><b onmouseover' not in page  # nothing closed early
    assert "n&quot;&gt;&lt;b onmouseover=&#x27;x&#x27;&gt;" in page  # the id, as text
    assert tree.node(HOSTILE_ID + "p").parent_id == "7"  # coerced to text


def test_a_stored_prompt_keeps_the_schema_once_and_expands_back():
    ddl = "CREATE TABLE t (a INT);\n"
    prompt = f"Question: q\n\nSchema (wide context, 1 tables):\n{ddl.strip()}\n\nPrevious: x"
    stored, key = stored_prompt(prompt, ddl)
    assert ddl.strip() not in stored and f"<<schema {key}>>" in stored
    assert expand_prompt(stored, {key: ddl}) == prompt
    assert expand_prompt(stored, {}) == stored  # an unknown key keeps its marker
    assert stored_prompt("no schema here", "")[0] == "no schema here"


def test_the_panel_shows_the_prompt_with_its_schema_folded_and_the_instructions():
    tree = _mix_trees()["local901"]
    page = render_search_html(tree)
    section = _section_of(page, tree, "n0")
    assert "Question: how many orders?" in section and "<<schema" not in section
    key = tree.node("n0").prompt.split("<<schema ")[1].split(">>")[0]
    assert f'<a class="ctx-link" href="#ctx-{key}">schema DDL (1 lines)</a>' in section
    assert "CREATE TABLE orders (id INTEGER);" not in section  # linked, written once below
    assert f'<details class="schema-context" id="ctx-{key}">' in page
    assert page.count("CREATE TABLE orders (id INTEGER);") == 1
    assert "generator instructions (system prompt)" in page
    assert "You write one read-only SQLite query." in page
    unsaved = tree_from_records([{"id": "n0", "prompt": "Schema:\n<<schema abc123>>"}])
    assert "[schema abc123: not saved with this run]" in render_search_html(unsaved)


def test_a_schema_shared_by_every_node_is_written_once():
    ddl = "CREATE TABLE wide (a INT);\nCREATE TABLE wider (b INT);"
    records = [
        {"id": f"n{i}", "parent_id": f"n{i - 1}" if i else None, "prompt": "S:\n<<schema 0a1b>>"}
        for i in range(8)
    ]
    page = render_search_html(tree_from_records(records, contexts={"0a1b": ddl, "ffff": "unused"}))
    assert page.count("CREATE TABLE wide (a INT);") == 1 and "unused" not in page
    assert page.count('href="#ctx-0a1b"') == 8
    assert "<summary>schema DDL 0a1b (2 lines)</summary>" in page


def test_a_node_hides_its_outcome_while_the_replay_has_it_generating():
    tree = _mix_trees()["local901"]
    page = render_search_html(tree)
    node = tree.node("n0")
    section = _section_of(page, tree, "n0")
    heading = section.split("</h3>")[0]
    # the score, facts, SQL and judgement sit in outcome blocks; the id, model and prompt do not
    assert f'<span class="outcome"> · {node.score:.2f}' in heading
    assert section.count('<div class="outcome">') == 2 and 'class="pending-note' in section
    assert '</div><details open><summary>generator prompt</summary>' in section  # outside both
    assert '.js .node-detail.pending .outcome { display: none; }' in page
    # the tooltip the player shows while the node generates carries no score
    title = _group_of(page, tree, "n0").split("</title>")[0]
    assert f'data-pending="n0 · {node.model} · generating"' in title
    assert f"score {node.score:.2f}" in title


def test_exec_error_text_never_prints_none():
    assert exec_error_text(None) is None and exec_error_text(ExecResult(ok=True)) is None
    assert exec_error_text(ExecResult(ok=False)) == "error"
    assert exec_error_text(ExecResult(ok=False, error_kind="timeout")) == "timeout"
    assert exec_error_text(ExecResult(ok=False, error="boom")) == "error: boom"
    ends_in_a_colon = ExecResult(ok=False, error_kind="runtime", error="syntax error at : ")
    assert exec_error_text(ends_in_a_colon) == "runtime: syntax error at : "  # nothing cut off


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
        {"instance_id": "a", "id": "n1", "generator": CODER, "ok": True},
        {"instance_id": "b", "id": "n0", "generator": QWEN, "ok": True},
        {"instance_id": "b", "generator": CODER},  # no id: skipped, not a traceback
    ])  # fmt: skip
    first, second = load_search_trees(candidates)
    assert first.models == second.models == (CODER, QWEN)  # the whole file's models, sorted
    assert [node.id for node in second.nodes] == ["n0"]
    qwen = model_colors(first.models)[QWEN]
    assert qwen != FAMILY_COLOR["qwen"]  # the family's second model: a shade
    for tree in (first, second):
        assert f"--c:{qwen}" in _group_of(render_search_html(tree), tree, "n0")


def test_without_a_report_the_task_filter_keeps_the_colours(tmp_path):
    candidates = _write_jsonl(tmp_path / "run_candidates.jsonl", [
        {"instance_id": "x", "id": "n0", "generator": CODER, "ok": True},
        {"instance_id": "y", "id": "n0", "generator": QWEN, "ok": True},
    ])  # fmt: skip
    (only_y,) = load_search_trees(candidates, tasks={"y"})
    assert only_y.models == (CODER, QWEN)  # CODER comes from task x, which was filtered out
    (full_x, full_y) = load_search_trees(candidates)
    assert model_colors(only_y.models)[QWEN] == model_colors(full_y.models)[QWEN]  # as in full


def test_a_report_that_is_not_a_run_report_is_an_error(tmp_path):
    for text in ("[1, 2]", '{"config": "x"}', "not json"):
        report = tmp_path / "report.json"
        report.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_search_trees(MIX, report_path=report)


def test_run_files_finds_the_siblings(tmp_path):
    rows, report, contexts = run_files(tmp_path / "spider2_exec_t_candidates.jsonl")
    assert (rows.name, report.name) == ("spider2_exec_t.rows.jsonl", "spider2_exec_t.json")
    assert contexts.name == "spider2_exec_t_contexts.jsonl"
    assert run_files(tmp_path / "pool.jsonl").rows.name == "pool.jsonl.rows.jsonl"


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
                  feedback=["no rows"], rationale="r", prompt="Schema: <<schema 0a1b>>",
                  context_key="0a1b"),
        Candidate(id="n1", parent_id="n0", depth=1, generator=QWEN, sql="SELECT x", exec=broken,
                  score=0.05, advice="- fix"),
        Candidate(id="n2", generator=GLM, error="node timed out"),
    ]  # fmt: skip


def _answer(candidates: list[Candidate]) -> AnswerResult:
    return AnswerResult(
        question="how many orders?", sql="SELECT 1", result=None, chosen_id="n0",
        chosen_by="score", score=0.7, strategy="abmcts", budget=4, nodes=len(candidates),
        stopped_early=False, candidates=candidates, usage=UsageSummary(),
        models={"generator": GLM}, contexts={"0a1b": "CREATE TABLE t (a INT);"},
        instructions="Write SQL.",
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
        "error", "exec_error", "prompt", "context_key", "asked_after", "told", "start_ms",
        "end_ms",
    ]  # fmt: skip
    assert list(records[0]) == ["instance_id", *shared, "fingerprint", "ex"]  # the bench adds 3
    from_file = tree_from_records(records, default_model=GLM, generators=(GLM, QWEN))
    from_answer = tree_from_answer(_answer(candidates), generators=(GLM, QWEN))
    assert from_file.nodes == from_answer.nodes and from_file.models == from_answer.models
    assert from_answer.title == "how many orders?" and from_answer.chosen_id == "n0"
    assert dict(from_answer.facts)["nodes"] == "3 of 4"
    assert from_answer.contexts == {"0a1b": "CREATE TABLE t (a INT);"}
    assert from_answer.instructions == "Write SQL."


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
    assert 'class="legend"' in page and 'id="search-events"' in page
    assert "CREATE TABLE orders" in page  # the contexts file next to the run was read
    assert "<script" not in (out / "index.html").read_text(encoding="utf-8")


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
    keys = ["a b", "a_b", "A_B", "CON", "nul.x", "local1", "index", "INDEX"]
    candidates = _write_jsonl(
        tmp_path / "run_candidates.jsonl", [{"instance_id": key, "id": "n0"} for key in keys]
    )
    out = tmp_path / "pages"
    result = CliRunner().invoke(cli_app, ["viz-search", str(candidates), "--out", str(out)])
    assert result.exit_code == 0, result.output
    expected = ["a_b.html", "a_b-2.html", "A_B-3.html", "_CON.html", "_nul.x.html", "local1.html"]
    expected += ["index-2.html", "INDEX-3.html"]  # index.html is the run's index page
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
    tree = _mix_trees()["local901"]
    assert "★ chosen (selector)" in _group_of(page, tree, "n10") and GEMINI in page


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


@pytest.mark.parametrize("option", ["--rows", "--report"])
def test_viz_search_refuses_a_directory_for_the_rows_or_report(tmp_path, option):
    result = CliRunner().invoke(cli_app, ["viz-search", str(MIX), option, str(tmp_path)])
    unboxed = " ".join(result.output.replace("│", " ").split())  # the error box wraps lines
    assert result.exit_code == 2 and "is a directory" in unboxed


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
    assert "how many orders?" in page and 'id="search-events"' in page
    tree = tree_from_answer(_answer(_candidates()), generators=(QWEN, GLM))
    assert f"--c:{FAMILY_COLOR['glm']}" in _group_of(page, tree, "n0")
    assert "Schema:" in page and "CREATE TABLE t (a INT);" in page  # prompt and its schema


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


# ----------------------------------------------------------------------------- 12. replay
def _kinds(tree) -> list[str]:
    """Return the replay as ``ask n0``, ``tell n0``, ... lines."""
    return [f"{event.kind} {tree.nodes[event.node].id}" for event in replay_events(tree)]


def test_a_lockstep_batch_is_asked_together_and_told_in_order():
    records = [
        {"id": "n0", "asked_after": 0, "told": 0}, {"id": "n1", "asked_after": 0, "told": 1},
        {"id": "n2", "parent_id": "n0", "asked_after": 2, "told": 3},
        {"id": "n3", "asked_after": 2, "told": 2},
    ]  # fmt: skip
    tree = tree_from_records(records, strategy="abmcts")
    assert _kinds(tree) == [
        "ask n0", "ask n1", "tell n0", "tell n1", "ask n2", "ask n3", "tell n3", "tell n2",
    ]  # fmt: skip


def test_a_rolling_search_asks_again_as_each_result_lands():
    records = [
        {"id": "n0", "asked_after": 0, "told": 1}, {"id": "n1", "asked_after": 0, "told": 0},
        {"id": "n2", "parent_id": "n1", "asked_after": 1, "told": 2},
    ]  # fmt: skip
    tree = tree_from_records(records, strategy="abmcts", chosen_id="n2", chosen_by="score")
    assert _kinds(tree) == ["ask n0", "ask n1", "tell n1", "ask n2", "tell n0", "tell n2", "pick n2"]


def test_older_records_replay_one_node_at_a_time_in_ask_order():
    tree = _mix_trees()["local901"]  # no asked_after or told
    kinds = _kinds(tree)
    order = ["n0", "n1", "n2", "n3", "n4", "n5", "n10"]
    assert kinds[:-1] == [f"{kind} {node}" for node in order for kind in ("ask", "tell")]
    assert kinds[-1] == "pick n10"


def test_the_steps_carry_the_selection_path_and_say_what_happens():
    records = [
        {"id": "n0", "generator": GLM, "asked_after": 0, "told": 0, "start_ms": 26000.0,
         "end_ms": 30000.0, "score": 0.4, "ok": True},
        {"id": "n1", "parent_id": "n0", "generator": QWEN, "asked_after": 1, "told": 1,
         "start_ms": 30000.0, "end_ms": 34500.0, "score": 0.9, "ok": True, "ex": 1},
        {"id": "n2", "parent_id": "n1", "generator": GLM, "asked_after": 2, "told": 2,
         "error": "node timed out"},
    ]  # fmt: skip
    tree = tree_from_records(records, strategy="abmcts", chosen_id="n1", chosen_by="selector")
    events = replay_events(tree)
    ask_n2 = events[4]
    assert (ask_n2.kind, ask_n2.path) == ("ask", (_index(tree, "n0"), _index(tree, "n1")))
    assert "Select question → n0 → n1 (CONT), then GEN under n1: n2 with glm-5.3" in ask_n2.caption
    assert events[0].caption == "0.0 s · GEN at the question: a new draft n0 with glm-5.3"
    tell_n1 = events[3]
    assert tell_n1.caption == (
        "8.5 s · n1 (qwen3.8-max) scored 0.90 · EX ✓ (matches gold); the reward backs up "
        "n0 → question"
    )  # the clock starts at the first node (TreeQuest's import is not search time)
    assert "generation failed" in events[5].caption
    assert events[-1].caption == "Final pick: n1 by selector, score 0.90 · EX ✓ (matches gold)"
    plain = tree_from_records(records, strategy="refine")
    assert replay_events(plain)[2].caption.endswith("Refine n0 into n1 with qwen3.8-max")
    page = render_search_html(tree)
    data = _events_data(page)
    assert data["nodes"] == 3 and len(data["events"]) == len(events)
    assert data["events"][4] == {
        "kind": "ask", "node": _index(tree, "n2"), "path": list(ask_n2.path),
        "caption": ask_n2.caption,
    }  # fmt: skip


def _fake_generate():
    """Return a generate function whose nodes finish at once, scored 0.5."""
    async def generate(node_id, parent, action):
        return Candidate(id=node_id, parent_id=parent.id if parent else None, score=0.5)

    return generate


def test_the_search_loops_record_when_each_node_was_asked_and_told():
    import asyncio

    from schemagraph.agent.results import AgentConfig
    from schemagraph.agent.search import run_search

    lockstep = AgentConfig(strategy="best_of_n", budget=4, batch_size=2, selector=False)
    trace = asyncio.run(run_search(_fake_generate(), lockstep))
    assert [(node.asked_after, node.told) for node in trace.candidates] == [
        (0, 0), (0, 1), (2, 2), (2, 3),
    ]  # fmt: skip
    assert all(0 <= node.start_ms <= node.end_ms for node in trace.candidates)
    chain = asyncio.run(run_search(_fake_generate(), AgentConfig(strategy="refine", budget=3)))
    assert [(node.asked_after, node.told) for node in chain.candidates] == [(0, 0), (1, 1), (2, 2)]


def test_a_rolling_abmcts_search_records_a_consistent_order():
    pytest.importorskip("treequest")
    import asyncio

    from schemagraph.agent.results import AgentConfig
    from schemagraph.agent.search import run_search

    cfg = AgentConfig(strategy="abmcts", budget=6, batch_size=2, rolling=True, early_stop=2.0)
    trace = asyncio.run(run_search(_fake_generate(), cfg))
    assert [node.told for node in trace.candidates] == list(range(6))
    assert all(node.asked_after <= node.told for node in trace.candidates)
    tree = tree_from_records(candidate_record(node) for node in trace.candidates)
    kinds = _kinds(tree)
    for node in trace.candidates:  # every node is asked before it is told
        assert kinds.index(f"ask {node.id}") < kinds.index(f"tell {node.id}")
