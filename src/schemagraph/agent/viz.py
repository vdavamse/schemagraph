"""Draw one answer's search tree as an HTML page: which node refined which, by which model.

The page is an indented outline drawn in inline SVG, one row per node, depth-first: each node
sits under the node it refines, and siblings follow the order the search asked for them. Rows
are coloured by the generator model, so a Multi-LLM search shows which model wrote what. Each
row links to a section with the node's score parts, judge rubric, findings, feedback, critic
advice, errors and SQL. The page has no script: a Content-Security-Policy meta forbids every
script, and every string is HTML-escaped.

The tree is built from our own candidates, not TreeQuest's state (which the search discards and
the baselines never have), in the one record form `candidate_record` defines:
:func:`tree_from_answer` reads an `AnswerResult`, and :func:`tree_from_records` reads those
records, as the exec benchmark's ``*_candidates.jsonl`` stores them
(:func:`schemagraph.bench.spider2_exec.load_search_trees`). Core dependencies only (the stdlib
and `schemagraph.agent.results`), so the offline viewer runs without the ``agent`` extra.
"""

from __future__ import annotations

import html
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from schemagraph.agent.results import candidate_record

if TYPE_CHECKING:
    from schemagraph.agent.results import AnswerResult

# Okabe-Ito colours, told apart under the common colour-vision deficiencies, in the order models
# get them; black is swapped for grey so the eighth model shows on a dark background. The palette
# cycles after eight models.
MODEL_PALETTE = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999",
)  # fmt: skip
# The model of a node whose record names none and whose run names no default generator.
UNKNOWN_GENERATOR = "unknown"
# Suffixes the page may be written to.
HTML_SUFFIXES = (".html", ".htm")
# Outline geometry, in SVG pixels: row height, indent per tree level, node radius, the width of
# one character of the 12 px monospace font, the score bar's full width, the gap between columns
# and the padding around the outline.
ROW_HEIGHT = 26
INDENT = 20
NODE_RADIUS = 6
CHAR_WIDTH = 7.3
SCORE_BAR_WIDTH = 80
COLUMN_GAP = 16
PADDING = 12
# Characters of the score number ("0.66"), which the score bar follows.
SCORE_TEXT_CHARS = 4

_CSP = "default-src 'none'; style-src 'unsafe-inline'"
_ASK_ORDER_ID = re.compile(r"n(\d+)")
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"{port}{n}" for port in ("COM", "LPT") for n in range(1, 10)}
)
_ROOT_LABEL = "question"
# The fill of the example circles in the marks key; any model colour would do.
_KEY_FILL = MODEL_PALETTE[1]
_CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1b1f24; --muted: #5b6470; --line: #c3c9d1; --panel: #f3f4f6;
  --accent: #b35900; --good: #1a7f37; --bad: #cf222e; --bar: #7d8590; --code: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa4b1; --line: #3b424c; --panel: #1b1f26;
    --accent: #ffb454; --good: #56d364; --bad: #ff7b72; --bar: #8b949e; --code: #161b22;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 72rem; padding: 16px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; overflow-wrap: anywhere;
}
a { color: inherit; }
h1 { font-size: 1.3rem; margin: 0 0 .5rem; }
h2 { font-size: 1.1rem; margin: 1.5rem 0 .5rem; }
h3 { font-size: 1rem; margin: 0 0 .5rem; }
.facts { display: grid; grid-template-columns: max-content 1fr; gap: .15rem 1rem; margin: 0; }
.facts dt { color: var(--muted); }
.facts dd { margin: 0; }
.scroll { overflow-x: auto; max-width: 100%; }
.outline { border: 1px solid var(--line); border-radius: 6px; background: var(--bg); }
.outline svg { display: block; font: 12px ui-monospace, "SFMono-Regular", Consolas, monospace; }
.outline text { fill: var(--fg); dominant-baseline: middle; }
.outline .muted { fill: var(--muted); }
.outline .edge { stroke: var(--line); fill: none; stroke-width: 1.5; }
.outline .filled { stroke: var(--fg); stroke-width: 1; }
.outline .hollow { stroke-width: 2.5; }
.outline .dashed { stroke-width: 2; stroke-dasharray: 3 2; }
.outline .ring { stroke: var(--accent); fill: none; stroke-width: 2; }
.outline .root { fill: var(--muted); }
.outline .bar-bg { fill: var(--panel); }
.outline .bar { fill: var(--bar); }
.outline .hit { fill: transparent; }
.outline a:hover .hit, .outline a:focus .hit { fill: var(--panel); }
.outline .good { fill: var(--good); }
.outline .bad { fill: var(--bad); }
.outline .chosen { fill: var(--accent); }
table { border-collapse: collapse; }
th, td { text-align: left; padding: .2rem .75rem .2rem 0; vertical-align: top; }
th { color: var(--muted); font-weight: 600; white-space: nowrap; }
.legend { margin: .5rem 0 1rem; }
.swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 50%;
  border: 1px solid var(--fg); vertical-align: -1px; margin-right: .4rem; }
.key { display: flex; flex-wrap: wrap; gap: .25rem 1.25rem; color: var(--muted); margin: .5rem 0; }
.key svg { vertical-align: -3px; margin-right: .3rem; }
.key .node { stroke: var(--fg); }
.key .ring { stroke: var(--accent); fill: none; stroke-width: 2; }
.node-detail { border-top: 1px solid var(--line); padding: .75rem 0; }
.node-detail:target { background: var(--panel); }
.marks { color: var(--accent); font-weight: 600; }
.good { color: var(--good); }
.bad { color: var(--bad); }
.muted { color: var(--muted); }
pre { background: var(--code); border: 1px solid var(--line); border-radius: 6px; padding: .6rem;
  overflow-x: auto; font: 12.5px/1.45 ui-monospace, "SFMono-Regular", Consolas, monospace;
  overflow-wrap: normal; }
ul { margin: .25rem 0; padding-left: 1.25rem; }
"""


@dataclass(frozen=True)
class TreeNode:
    """One search node, normalised from a candidate or a candidates-file record.

    Attributes:
        id: The node id (``"n0"``, ``"n1"``, ... in ask order).
        parent_id: The node it refines, None for a fresh draft.
        level: Its level in the drawn outline: 1 for a draft, one more per refinement; a node
            whose parent is missing hangs off the root at level 1.
        depth: The refinement depth the search recorded.
        model: The generator model that wrote it.
        action: The action the search chose (a context width, or the model in a Multi-LLM run).
        score: The combined score.
        score_parts: The score's components, by name.
        rubric: The judge's probabilities, by rubric field.
        missing: Table -> probability that the query needs it, from the judge.
        findings: The deterministic checks' finding codes.
        feedback: The feedback lines a refinement would see.
        advice: The critic's advice, when the node was expanded.
        error: The generation failure, if any.
        exec_error: The execution error of a generated query, when known.
        executed: Whether the query ran to completion.
        row_count: Rows the query returned.
        sql: The query.
        rationale: The generator's explanation.
        ex: The execution match against the gold result (benchmark only), else None.
    """

    id: str
    parent_id: str | None = None
    level: int = 1
    depth: int = 0
    model: str = UNKNOWN_GENERATOR
    action: str = ""
    score: float = 0.0
    score_parts: Mapping[str, float | None] = field(default_factory=dict)
    rubric: Mapping[str, float] = field(default_factory=dict)
    missing: Mapping[str, float] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    feedback: tuple[str, ...] = ()
    advice: str | None = None
    error: str | None = None
    exec_error: str | None = None
    executed: bool = False
    row_count: int | None = None
    sql: str = ""
    rationale: str = ""
    ex: int | None = None

    @property
    def failed(self) -> bool:
        """Whether generation failed, so there is no query."""
        return bool(self.error)

    @property
    def exec_failed(self) -> bool:
        """Whether a query was generated but did not run to completion."""
        return not self.failed and not self.executed


@dataclass(frozen=True)
class SearchTree:
    """One answer's search, ready to render.

    Attributes:
        key: A short identifier: the task id in the benchmark, ``answer`` for ``ask``.
        title: The page title (the question, or the task id).
        nodes: Every node, in depth-first pre-order with children in ask order.
        models: The generator models, in legend and colour order.
        chosen_id: The node the answer picked.
        chosen_by: How it was picked (``score``, ``selector`` or ``only``).
        strategy: The search strategy.
        facts: Label and value pairs shown under the title.
    """

    key: str
    title: str
    nodes: tuple[TreeNode, ...]
    models: tuple[str, ...]
    chosen_id: str | None = None
    chosen_by: str | None = None
    strategy: str | None = None
    facts: tuple[tuple[str, str], ...] = ()

    def node(self, node_id: str | None) -> TreeNode | None:
        """Return the node with this id, or None."""
        return next((node for node in self.nodes if node.id == node_id), None)


# --------------------------------------------------------------------------- building the tree
def tree_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    key: str = "answer",
    title: str = "",
    default_model: str | None = None,
    generators: Sequence[str] = (),
    chosen_id: str | None = None,
    chosen_by: str | None = None,
    strategy: str | None = None,
    facts: Iterable[tuple[str, str]] = (),
) -> SearchTree:
    """Build a search tree from candidate records, the one place records are normalised.

    Args:
        records: Candidate records as the exec benchmark's candidates file stores them; a later
            record with the same id replaces an earlier one.
        key: The tree's identifier.
        title: The page title.
        default_model: The model of a record whose ``generator`` is empty (a single-generator
            run records none); None leaves it `UNKNOWN_GENERATOR`.
        generators: The configured generator models; they lead the legend, in this order, and
            keep their colours from task to task.
        chosen_id: The node the answer picked.
        chosen_by: How it was picked.
        strategy: The search strategy.
        facts: Label and value pairs shown under the title.

    Returns:
        The tree, with nodes in depth-first pre-order and children in ask order.
    """
    by_id: dict[str, TreeNode] = {}
    for record in records:
        node = _node_from_record(record, default_model)
        by_id[node.id] = node
    nodes = tuple(_preorder(by_id))
    return SearchTree(
        key=key,
        title=title or key,
        nodes=nodes,
        models=_legend_order(nodes, generators),
        chosen_id=chosen_id,
        chosen_by=chosen_by,
        strategy=strategy,
        facts=tuple((str(label), str(value)) for label, value in facts),
    )


def tree_from_answer(result: AnswerResult, *, generators: Sequence[str] = ()) -> SearchTree:
    """Build the search tree of one ``ask`` answer.

    Args:
        result: The answer.
        generators: The configured generator models (``AgentConfig.gen_models``), for the
            legend order; with a single generator the candidates name none and every node is
            the run's generator (``result.models["generator"]``).

    Returns:
        The tree, titled with the question.
    """
    stopped = ", stopped early" if result.stopped_early else ""
    facts = [
        ("strategy", result.strategy),
        ("nodes", f"{result.nodes} of {result.budget}{stopped}"),
        ("score", f"{result.score:.2f}"),
        ("cost", f"${result.usage.total.cost_usd:.4f}"),
        ("time", f"{result.ms / 1000:.1f}s"),
    ]
    return tree_from_records(
        (candidate_record(candidate) for candidate in result.candidates),
        key="answer",
        title=result.question,
        default_model=result.models.get("generator"),
        generators=generators,
        chosen_id=result.chosen_id,
        chosen_by=result.chosen_by,
        strategy=result.strategy,
        facts=facts,
    )


def _node_from_record(record: Mapping[str, Any], default_model: str | None) -> TreeNode:
    """Normalise one record: missing keys (older candidates files) get neutral values."""
    return TreeNode(
        id=str(record["id"]),
        parent_id=None if record.get("parent_id") is None else str(record["parent_id"]),
        depth=int(record.get("depth") or 0),
        model=record.get("generator") or default_model or UNKNOWN_GENERATOR,
        action=str(record.get("action") or ""),
        score=float(record.get("score") or 0.0),
        score_parts=dict(record.get("score_parts") or {}),
        rubric=dict(record.get("rubric") or {}),
        missing=dict(record.get("missing") or {}),
        findings=tuple(record.get("findings") or ()),
        feedback=tuple(record.get("feedback") or ()),
        advice=record.get("advice"),
        error=record.get("error"),
        exec_error=record.get("exec_error"),
        executed=bool(record.get("ok")),
        row_count=record.get("row_count"),
        sql=record.get("sql") or "",
        rationale=record.get("rationale") or "",
        ex=record.get("ex"),
    )


def _ask_order(node_id: str) -> tuple[int, int, str]:
    """Sort key of node ids in ask order: ``n2`` before ``n10``, other ids after, by name."""
    match = _ASK_ORDER_ID.fullmatch(node_id)
    return (0, int(match.group(1)), "") if match else (1, 0, node_id)


def _preorder(by_id: dict[str, TreeNode]) -> Iterator[TreeNode]:
    """Yield the nodes depth-first, children in ask order, each with its outline ``level``.

    A node whose parent is not among the records (a run killed mid-write) hangs off the root.
    """
    children: defaultdict[str | None, list[str]] = defaultdict(list)
    for node in by_id.values():
        known_parent = node.parent_id in by_id and node.parent_id != node.id
        children[node.parent_id if known_parent else None].append(node.id)
    visited: set[str] = set()
    # the drafts first; any node left over sits on a parent cycle and is drawn from the root
    for start in [*sorted(children[None], key=_ask_order), *sorted(by_id, key=_ask_order)]:
        stack = [(start, 1)]
        while stack:
            node_id, level = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            yield replace(by_id[node_id], level=level)
            for child in sorted(children[node_id], key=_ask_order, reverse=True):
                stack.append((child, level + 1))


def _legend_order(nodes: Sequence[TreeNode], generators: Sequence[str]) -> tuple[str, ...]:
    """Return the configured generators in order, then any other model seen, sorted."""
    configured = tuple(dict.fromkeys(generators))
    seen = {node.model for node in nodes} - set(configured)
    return (*configured, *sorted(seen))


def model_colors(models: Sequence[str]) -> dict[str, str]:
    """Map each model to its `MODEL_PALETTE` colour, in order, cycling after the palette ends."""
    return {model: MODEL_PALETTE[index % len(MODEL_PALETTE)] for index, model in enumerate(models)}


def short_model_name(name: str) -> str:
    """Return a model name without its provider and vendor prefixes.

    ``openrouter:qwen/qwen3.8-max`` becomes ``qwen3.8-max``; a name that is all prefix stays.
    """
    short = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return short or name


def page_filename(tree: SearchTree) -> str:
    """Return a safe file name for the tree's page, from its key.

    Characters outside ``[A-Za-z0-9._-]`` become ``_``, and a name Windows reserves for a
    device (``CON``, ``NUL``, ``COM1``, ..., with or without an extension) gets a leading ``_``.
    """
    stem = _UNSAFE_FILENAME.sub("_", tree.key).strip(".") or "search"
    if stem.split(".", 1)[0].upper() in _RESERVED_FILENAMES:
        stem = "_" + stem
    return f"{stem}.html"


def page_filenames(trees: Sequence[SearchTree]) -> list[str]:
    """Return one distinct page file name per tree, in order.

    Names are compared without case (Windows and macOS file systems ignore it); a repeat
    gets ``-2``, ``-3``, ... before its suffix.
    """
    taken: set[str] = set()
    names = []
    for tree in trees:
        stem = page_filename(tree).removesuffix(".html")
        name, count = f"{stem}.html", 1
        while name.lower() in taken:
            count += 1
            name = f"{stem}-{count}.html"
        taken.add(name.lower())
        names.append(name)
    return names


# --------------------------------------------------------------------------- files
def check_html_path(path: str | Path) -> Path:
    """Check that the page can be written to ``path`` before any work (or money) is spent.

    Raises:
        ValueError: The path does not end in ``.html``/``.htm``, is a directory, is an
            existing file that is not writable, or its directory does not exist or is not
            writable.
    """
    target = Path(path)
    if target.suffix.lower() not in HTML_SUFFIXES:
        raise ValueError(f"{target}: the search view is an HTML page; give a .html path")
    if target.is_dir():
        raise ValueError(f"{target} is a directory")
    if target.exists() and not os.access(target, os.W_OK):
        raise ValueError(f"{target} is not writable")
    directory = target.parent
    if not directory.is_dir():
        raise ValueError(f"{directory} does not exist")
    if not os.access(directory, os.W_OK):
        raise ValueError(f"{directory} is not writable")
    return target


def write_search_html(tree: SearchTree, path: str | Path) -> Path:
    """Render ``tree`` and write it to ``path`` (checked by :func:`check_html_path`).

    Returns:
        The path written.

    Raises:
        ValueError: The path cannot hold the page.
        OSError: Writing failed.
    """
    target = check_html_path(path)
    target.write_text(render_search_html(tree), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- the search page
def render_search_html(tree: SearchTree) -> str:
    """Render one search tree as a standalone, script-free HTML page.

    The page has the title and facts, a legend (each model's colour, node count, best score
    and correct nodes when known; the key to the marks), the outline, and one section per node.
    """
    colors = model_colors(tree.models)
    body = "".join(
        [
            f"<h1>{_escape(tree.title)}</h1>",
            _facts(tree),
            _legend(tree, colors),
            '<h2>Search tree</h2><div class="scroll outline">',
            _outline_svg(tree, colors),
            "</div><h2>Nodes</h2>",
            "".join(_node_section(node, tree, colors) for node in tree.nodes),
        ]
    )
    return _page(tree.title, body)


def _escape(value: object) -> str:
    """Escape any value as HTML text or an attribute value."""
    return html.escape(str(value), quote=True)


def _page(title: str, body: str) -> str:
    """Wrap ``body`` in a document with the CSP meta, the viewport and the style sheet."""
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{_escape(_CSP)}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title)}</title><style>{_CSS}</style></head>"
        f'<body id="top">{body}</body></html>\n'
    )


def _facts(tree: SearchTree) -> str:
    """Render the tree's facts, and which node was chosen, as a definition list."""
    facts = list(tree.facts)
    if tree.chosen_id:
        how = f" by {tree.chosen_by}" if tree.chosen_by else ""
        facts.append(("chosen", f"{tree.chosen_id}{how}"))
    if not facts:
        return ""
    items = "".join(f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>" for label, value in facts)
    return f'<dl class="facts">{items}</dl>'


def _marks(node: TreeNode, tree: SearchTree) -> list[str]:
    """Return the node's marks: chosen, failed or exec error, and its EX when known."""
    marks = []
    if node.id == tree.chosen_id:
        marks.append(f"★ chosen ({tree.chosen_by})" if tree.chosen_by else "★ chosen")
    if node.failed:
        marks.append("⚠ failed")
    elif node.exec_failed:
        marks.append("exec error")
    if node.ex is not None:
        marks.append("EX ✓" if node.ex else "EX ✗")
    return marks


# --------------------------------------------------------------------------- legend
def _legend(tree: SearchTree, colors: Mapping[str, str]) -> str:
    """Render the model table (colour, nodes, best score, correct nodes) and the marks key."""
    known_ex = any(node.ex is not None for node in tree.nodes)
    header = "<th>model</th><th>nodes</th><th>best score</th>" + (
        "<th>EX ✓</th>" if known_ex else ""
    )
    rows = "".join(_legend_row(tree, model, colors[model], known_ex) for model in tree.models)
    return (
        '<div class="legend"><div class="scroll"><table>'
        f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"
        f"{_marks_key()}</div>"
    )


def _legend_row(tree: SearchTree, model: str, color: str, known_ex: bool) -> str:
    """Render one model's legend row."""
    nodes = [node for node in tree.nodes if node.model == model]
    best = max((node.score for node in nodes), default=None)
    best_text = f"{best:.2f}" if best is not None else "–"
    correct = f"<td>{sum(1 for node in nodes if node.ex)}</td>" if known_ex else ""
    return (
        f'<tr><td><span class="swatch" style="background:{_escape(color)}"></span>'
        f"{_escape(model)}</td><td>{len(nodes)}</td><td>{best_text}</td>{correct}</tr>"
    )


def _marks_key() -> str:
    """Render the key to the node marks, each with a small drawing of its circle."""
    radius, size = NODE_RADIUS, 2 * NODE_RADIUS + 8
    center = size / 2

    def swatch(shape: str) -> str:
        return f'<svg width="{size}" height="{size}" aria-hidden="true">{shape}</svg>'

    circle = f'<circle cx="{center}" cy="{center}" r="{radius}" class="node"'
    entries = [
        (swatch(f'{circle} fill="{_KEY_FILL}"/>'), "ran (fill: model)"),
        (swatch(f'{circle} fill="none" stroke-width="2.5"/>'), "exec error (hollow)"),
        (swatch(f'{circle} fill="none" stroke-dasharray="3 2"/>'), "⚠ generation failed"),
        (
            swatch(
                f'{circle} fill="{_KEY_FILL}"/>'
                f'<circle cx="{center}" cy="{center}" r="{radius + 3}" class="ring"/>'
            ),
            "★ chosen (ring)",
        ),
    ]
    items = "".join(f"<span>{shape}{_escape(label)}</span>" for shape, label in entries)
    return f'<div class="key">{items}</div>'


# --------------------------------------------------------------------------- outline
@dataclass(frozen=True)
class _Columns:
    """The x positions of the outline's aligned columns, and its size.

    Attributes:
        model: Left edge of the model name column.
        score: Left edge of the score number.
        bar: Left edge of the score bar.
        marks: Left edge of the marks column.
        width: The SVG's width.
        height: The SVG's height.
    """

    model: float
    score: float
    bar: float
    marks: float
    width: float
    height: float


def _node_x(level: int) -> float:
    """Return the x of the centre of a node circle at outline ``level`` (the root is 0)."""
    return PADDING + level * INDENT + NODE_RADIUS


def _row_y(row: int) -> float:
    """Return the y of the middle of outline row ``row`` (the root is row 0)."""
    return PADDING + row * ROW_HEIGHT + ROW_HEIGHT / 2


def _columns(tree: SearchTree) -> _Columns:
    """Lay out the aligned columns after the deepest node's id label."""
    labels = [(node.level, node.id) for node in tree.nodes] or [(0, _ROOT_LABEL)]
    tree_right = max(
        _node_x(level) + NODE_RADIUS + 6 + len(label) * CHAR_WIDTH for level, label in labels
    )
    model_x = tree_right + COLUMN_GAP
    model_chars = max((len(short_model_name(model)) for model in tree.models), default=0)
    score_x = model_x + model_chars * CHAR_WIDTH + COLUMN_GAP
    bar_x = score_x + (SCORE_TEXT_CHARS + 1) * CHAR_WIDTH
    marks_x = bar_x + SCORE_BAR_WIDTH + COLUMN_GAP
    marks_chars = max((len("  ".join(_marks(node, tree))) for node in tree.nodes), default=0)
    return _Columns(
        model=model_x,
        score=score_x,
        bar=bar_x,
        marks=marks_x,
        width=round(marks_x + marks_chars * CHAR_WIDTH + PADDING),
        height=round(2 * PADDING + (len(tree.nodes) + 1) * ROW_HEIGHT),
    )


def _outline_svg(tree: SearchTree, colors: Mapping[str, str]) -> str:
    """Render the outline: the root row, the edges, then one linked row per node."""
    columns = _columns(tree)
    rows = {node.id: row for row, node in enumerate(tree.nodes, start=1)}
    edges = "".join(_edge(node, rows) for node in tree.nodes)
    node_rows = "".join(
        _node_row(node, row, tree, colors[node.model], columns)
        for row, node in enumerate(tree.nodes, start=1)
    )
    root_y = _row_y(0)
    root = (
        f'<rect class="root" x="{_node_x(0) - NODE_RADIUS + 1}" y="{root_y - NODE_RADIUS + 1}" '
        f'width="{2 * NODE_RADIUS - 2}" height="{2 * NODE_RADIUS - 2}"/>'
        f'<text class="muted" x="{_node_x(0) + NODE_RADIUS + 6}" y="{root_y}">{_ROOT_LABEL}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{columns.width}" '
        f'height="{columns.height}" viewBox="0 0 {columns.width} {columns.height}" '
        f'role="img" aria-label="search tree">{edges}{root}{node_rows}</svg>'
    )


def _edge(node: TreeNode, rows: Mapping[str, int]) -> str:
    """Draw the elbow from the node's parent down and across to the node.

    A draft, an orphan and the first node drawn of a parent cycle hang off the root: their
    parent has no row, or a row that is not above theirs.
    """
    parent_row = rows.get(node.parent_id or "", 0)
    if parent_row >= rows[node.id]:
        parent_row = 0
    parent_x, parent_y = _node_x(node.level - 1), _row_y(parent_row)
    child_x, child_y = _node_x(node.level), _row_y(rows[node.id])
    return (
        f'<path class="edge" d="M{parent_x} {parent_y + NODE_RADIUS} '
        f'V{child_y} H{child_x - NODE_RADIUS}"/>'
    )


def _node_row(node: TreeNode, row: int, tree: SearchTree, color: str, columns: _Columns) -> str:
    """Render one node's row, linked to its section, with a hover tooltip.

    Args:
        node: The node.
        row: Its outline row (1 for the first node).
        tree: The tree, for the chosen node.
        color: The node's model colour.
        columns: The column layout.

    Returns:
        An ``<a>`` element holding the row's shapes and text.
    """
    y = _row_y(row)
    x = _node_x(node.level)
    marks = _marks(node, tree)
    score = min(max(node.score, 0.0), 1.0)
    parts = [
        f"<title>{_escape(_tooltip(node, marks))}</title>",
        f'<rect class="hit" x="0" y="{y - ROW_HEIGHT / 2}" width="{columns.width}" '
        f'height="{ROW_HEIGHT}"/>',
        _node_circle(node, x, y, color),
        f'<circle class="ring" cx="{x}" cy="{y}" r="{NODE_RADIUS + 3}"/>'
        if node.id == tree.chosen_id
        else "",
        f'<text x="{x + NODE_RADIUS + 6}" y="{y}">{_escape(node.id)}</text>',
        f'<text class="muted" x="{columns.model}" y="{y}">'
        f"{_escape(short_model_name(node.model))}</text>",
        f'<text x="{columns.score}" y="{y}">{node.score:.2f}</text>',
        f'<rect class="bar-bg" x="{columns.bar}" y="{y - 4}" width="{SCORE_BAR_WIDTH}" '
        'height="8" rx="2"/>',
        f'<rect class="bar" x="{columns.bar}" y="{y - 4}" width="{score * SCORE_BAR_WIDTH:.1f}" '
        'height="8" rx="2"/>',
        _marks_text(marks, columns.marks, y),
    ]
    return f'<a href="#node-{_escape(node.id)}">{"".join(parts)}</a>'


def _node_circle(node: TreeNode, x: float, y: float, color: str) -> str:
    """Draw the node: filled when it ran, hollow on an exec error, dashed when it failed."""
    if node.failed:
        style = f'class="node dashed" fill="none" stroke="{_escape(color)}"'
    elif node.exec_failed:
        style = f'class="node hollow" fill="none" stroke="{_escape(color)}"'
    else:
        style = f'class="node filled" fill="{_escape(color)}"'
    return f'<circle cx="{x}" cy="{y}" r="{NODE_RADIUS}" {style}/>'


def _marks_text(marks: Sequence[str], x: float, y: float) -> str:
    """Render the marks as one text line, each coloured by what it says."""
    spans = []
    for mark in marks:
        css = "chosen" if mark.startswith("★") else "good" if mark == "EX ✓" else "bad"
        spans.append(f'<tspan class="{css}">{_escape(mark)}</tspan>')
    return f'<text x="{x}" y="{y}" xml:space="preserve">{"  ".join(spans)}</text>' if spans else ""


def _tooltip(node: TreeNode, marks: Sequence[str]) -> str:
    """Return the hover text of a node row."""
    parent = f"refines {node.parent_id}" if node.parent_id else "draft"
    parts = [node.id, node.model, f"score {node.score:.2f}", parent]
    if node.action and node.action != node.model:
        parts.append(node.action)
    return " · ".join([*parts, *marks])


# --------------------------------------------------------------------------- node sections
def _node_section(node: TreeNode, tree: SearchTree, colors: Mapping[str, str]) -> str:
    """Render one node's section: heading, facts, judge rubric, feedback and SQL."""
    marks = _marks(node, tree)
    marks_html = f' <span class="marks">{_escape("  ".join(marks))}</span>' if marks else ""
    heading = (
        f'<h3><span class="swatch" style="background:{_escape(colors[node.model])}"></span>'
        f"{_escape(node.id)} · {_escape(short_model_name(node.model))} · "
        f"{node.score:.2f}{marks_html}</h3>"
    )
    return (
        f'<section class="node-detail" id="node-{_escape(node.id)}">{heading}'
        f"{_node_facts(node)}{_rubric(node)}{_text_list('feedback', node.feedback)}"
        f"{_prose('advice', node.advice)}{_prose('rationale', node.rationale)}"
        f"{_sql(node.sql)}"
        '<p><a href="#top">↑ top</a></p></section>'
    )


def _node_facts(node: TreeNode) -> str:
    """Render the node's facts, leaving out the empty ones."""
    parts = ", ".join(
        f"{name} {value:.2f}" for name, value in node.score_parts.items() if value is not None
    )
    facts = [
        ("model", node.model),
        ("refines", node.parent_id),
        ("depth", str(node.depth)),
        ("action", node.action if node.action != node.model else None),
        ("score parts", parts or None),
        ("rows", None if node.row_count is None else str(node.row_count)),
        ("findings", ", ".join(node.findings) or None),
        ("error", node.error),
        ("exec error", node.exec_error),
    ]
    items = "".join(
        f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>" for label, value in facts if value
    )
    return f'<dl class="facts">{items}</dl>'


def _rubric(node: TreeNode) -> str:
    """Render the judge's rubric probabilities and the tables it thinks are missing."""
    if not node.rubric and not node.missing:
        return ""
    rows = "".join(
        f"<tr><td>{_escape(name)}</td><td>{float(value):.2f}</td></tr>"
        for name, value in node.rubric.items()
    )
    rows += "".join(
        f"<tr><td>missing {_escape(table)}</td><td>{float(value):.2f}</td></tr>"
        for table, value in node.missing.items()
    )
    return f'<div class="scroll"><table><tbody>{rows}</tbody></table></div>'


def _text_list(label: str, lines: Sequence[str]) -> str:
    """Render labelled lines as a list; nothing when there are none."""
    if not lines:
        return ""
    items = "".join(f"<li>{_escape(line)}</li>" for line in lines)
    return f'<p class="muted">{_escape(label)}</p><ul>{items}</ul>'


def _prose(label: str, text: str | None) -> str:
    """Render a labelled paragraph; nothing when it is empty."""
    if not text:
        return ""
    return f'<p><span class="muted">{_escape(label)}:</span> {_escape(text)}</p>'


def _sql(sql: str) -> str:
    """Render the query in a block that scrolls sideways inside itself."""
    return f"<pre><code>{_escape(sql)}</code></pre>" if sql else ""


# --------------------------------------------------------------------------- the index page
def render_index_html(title: str, pages: Sequence[tuple[str, SearchTree]]) -> str:
    """Render an index of several search pages, one table row per tree.

    Args:
        title: The page title.
        pages: (link, tree) pairs, in display order.

    Returns:
        A standalone, script-free HTML page: tasks, nodes, models, best score, the chosen node,
        the pick's EX and whether any node was correct (when EX is known).
    """
    rows = "".join(_index_row(href, tree) for href, tree in pages)
    picked = [_pick_ex(tree) for _, tree in pages]
    known = [ex for ex in picked if ex is not None]
    summary = f"{len(pages)} tasks"
    if known:
        oracle = sum(1 for _, tree in pages if any(node.ex for node in tree.nodes))
        summary += f", pick EX {sum(known)}/{len(known)}, any node correct {oracle}/{len(pages)}"
    header = "".join(
        f"<th>{name}</th>"
        for name in ("task", "nodes", "models", "best", "chosen", "pick EX", "any EX ✓")
    )
    body = (
        f'<h1>{_escape(title)}</h1><p class="muted">{_escape(summary)}</p>'
        f'<div class="scroll"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )
    return _page(title, body)


def _pick_ex(tree: SearchTree) -> int | None:
    """Return the EX of the chosen node, or None when unknown."""
    chosen = tree.node(tree.chosen_id)
    return chosen.ex if chosen else None


def _index_row(href: str, tree: SearchTree) -> str:
    """Render one tree's index row."""
    best = max((node.score for node in tree.nodes), default=None)
    used = {node.model for node in tree.nodes}
    models = ", ".join(short_model_name(model) for model in tree.models if model in used)
    pick = _pick_ex(tree)
    any_ex = [node.ex for node in tree.nodes if node.ex is not None]
    cells = [
        f'<a href="{_escape(href)}">{_escape(tree.key)}</a>',
        str(len(tree.nodes)),
        _escape(models),
        f"{best:.2f}" if best is not None else "–",
        _escape(f"{tree.chosen_id} ({tree.chosen_by})" if tree.chosen_id else "–"),
        _ex_cell(pick),
        _ex_cell(int(any(any_ex)) if any_ex else None),
    ]
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _ex_cell(ex: int | None) -> str:
    """Render an EX value as a coloured tick or cross, or a dash when unknown."""
    if ex is None:
        return "–"
    return '<span class="good">✓</span>' if ex else '<span class="bad">✗</span>'
