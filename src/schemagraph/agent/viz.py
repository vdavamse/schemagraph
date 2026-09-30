"""Replay one answer's search as an HTML page: which node the search selected, refined and scored.

The page draws the search tree top-down in inline SVG, the question at the top and each node
under the node it refines, coloured by the generator model's family (qwen, glm, gemini, ...).
A player steps through the search as it happened:

* **ask**: the selection path lights up from the question down to the node being expanded
  (AB-MCTS moves only from a node to one of its children, so the path of a step is the chain of
  ancestors of the expanded node: CONT at each ancestor, GEN at the parent), and the new node
  appears, dashed, while its generator runs;
* **tell**: the node fills with its score, and the path lights up again as the reward backs up;
* **pick**: the final choice gets its ring.

Asks and results are replayed in the order they happened (``Candidate.asked_after`` and
``told``), so lockstep batches and rolling searches both replay truthfully; records written
before those fields existed replay one node at a time in ask order. A panel shows the selected
node's SQL, generator prompt (each schema DDL is written once per page, and the prompts link to
it), score parts, judge rubric, feedback and errors.

Everything is server-rendered and HTML-escaped, and readable without script: the final tree and
every node's section. The one script (:data:`schemagraph.agent.viz_assets.PLAYER_JS`) is a
constant admitted by its hash in the page's Content-Security-Policy; it reads the replay steps
from a JSON data block and only toggles classes and sets ``textContent``.

The tree is built from our own candidates, not TreeQuest's state, in the one record form
`candidate_record` defines: :func:`tree_from_answer` reads an `AnswerResult`, and
:func:`tree_from_records` reads those records as the exec benchmark's ``*_candidates.jsonl``
stores them (:func:`schemagraph.bench.spider2_exec.load_search_trees`). Core dependencies only
(the stdlib and `schemagraph.agent.results`), so the offline viewer runs without the ``agent``
extra.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from schemagraph.agent.results import SCHEMA_MARKER, candidate_record
from schemagraph.agent.viz_assets import CSS, PLAYER_JS

if TYPE_CHECKING:
    from schemagraph.agent.results import AnswerResult

# LLM families: (family, fragments of a model name that identify it, the family's colour). A
# model belongs to the first family with a fragment in its lowercased full name, so it keeps its
# family's colour in every run. The colours start from Okabe-Ito, told apart under the common
# colour-vision deficiencies.
MODEL_FAMILIES = (
    ("qwen", ("qwen",), "#7B5CD6"),
    ("glm", ("glm", "z-ai/", "zhipu"), "#0072B2"),
    ("gemini", ("gemini", "gemma", "google/"), "#009E73"),
    ("gpt", ("gpt", "openai/"), "#E69F00"),
    ("claude", ("claude", "anthropic"), "#D55E00"),
    ("deepseek", ("deepseek",), "#56B4E9"),
    ("kimi", ("kimi", "moonshot"), "#CC79A7"),
    ("llama", ("llama",), "#A08C00"),
    ("mistral", ("mistral", "codestral", "devstral"), "#8C564B"),
    ("grok", ("grok", "x-ai/"), "#B03060"),
)
# The family of a model no fragment names, and its colour.
OTHER_FAMILY = "other"
OTHER_COLOR = "#999999"
# The shade of each further model of one family, in legend order: how far its colour moves
# towards white (positive) or black (negative). The steps cycle after the last one.
SHADE_STEPS = (0.0, 0.45, -0.35, 0.7, -0.55)
# The model of a node whose record names none and whose run names no default generator.
UNKNOWN_GENERATOR = "unknown"
# Suffixes the page may be written to.
HTML_SUFFIXES = (".html", ".htm")
INDEX_FILENAME = "index.html"  # a run directory's index page; no task page may take it
# Tree geometry, in SVG units: the distance between neighbouring leaves and between levels, the
# node radius, the question box and the padding around the tree.
SLOT_WIDTH = 34
LEVEL_HEIGHT = 64
NODE_RADIUS = 11
ROOT_WIDTH = 76
ROOT_HEIGHT = 22
PADDING = 24
# The gap between a node and its id label below it.
LABEL_GAP = 12
# How much larger than its SVG units the tree is shown (readable on a slide); a tree wider than
# the page shrinks to fit.
DISPLAY_SCALE = 1.6

_CSP = "default-src 'none'; style-src 'unsafe-inline'"
_ASK_ORDER_ID = re.compile(r"n(\d+)")
# What a node's section says, in the replay, while its generator is still running.
_PENDING_NOTE = (
    '<p class="pending-note muted">Generating: the search has not been told this node\'s result '
    "yet.</p>"
)
_SCHEMA_MARKER = re.compile(re.escape(SCHEMA_MARKER).replace(re.escape("{key}"), "([0-9a-f]+)"))
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"{port}{n}" for port in ("COM", "LPT") for n in range(1, 10)}
)
_ROOT_LABEL = "question"
# The fill of the example circles in the marks key; any model colour would do.
_KEY_FILL = MODEL_FAMILIES[1][2]
_SPEEDS = ("0.5", "1", "2", "4")


@dataclass(frozen=True)
class TreeNode:
    """One search node, normalised from a candidate or a candidates-file record.

    Attributes:
        id: The node id (``"n0"``, ``"n1"``, ... in ask order).
        parent_id: The node it refines, None for a fresh draft.
        level: Its level in the drawn tree: 1 for a draft, one more per refinement; a node
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
        prompt: The generator's prompt, its schema DDL replaced by a marker.
        asked_after: Nodes finished when the search asked for this one, when recorded.
        told: Its place in the order results came back, when recorded.
        start_ms: When it started, in milliseconds since the search started, when recorded.
        end_ms: When it finished, on the same clock, when recorded.
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
    prompt: str = ""
    asked_after: int | None = None
    told: int | None = None
    start_ms: float | None = None
    end_ms: float | None = None

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
        contexts: The schema DDL of each context key the prompts name.
        instructions: The generator's system instructions, when recorded.
    """

    key: str
    title: str
    nodes: tuple[TreeNode, ...]
    models: tuple[str, ...]
    chosen_id: str | None = None
    chosen_by: str | None = None
    strategy: str | None = None
    facts: tuple[tuple[str, str], ...] = ()
    contexts: Mapping[str, str] = field(default_factory=dict)
    instructions: str = ""

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
    contexts: Mapping[str, str] | None = None,
    instructions: str = "",
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
            keep their shades from task to task.
        chosen_id: The node the answer picked.
        chosen_by: How it was picked.
        strategy: The search strategy.
        facts: Label and value pairs shown under the title.
        contexts: The schema DDL by context key, to put back into the prompts.
        instructions: The generator's system instructions.

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
        contexts={str(name): str(ddl) for name, ddl in (contexts or {}).items()},
        instructions=instructions,
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
        contexts=result.contexts,
        instructions=result.instructions,
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
        prompt=str(record.get("prompt") or ""),
        asked_after=_optional_int(record.get("asked_after")),
        told=_optional_int(record.get("told")),
        start_ms=_optional_float(record.get("start_ms")),
        end_ms=_optional_float(record.get("end_ms")),
    )


def _optional_int(value: Any) -> int | None:
    """Return ``value`` as an int, or None when it is missing or not a number."""
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    """Return ``value`` as a float, or None when it is missing or not a number."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _ask_order(node_id: str) -> tuple[int, int, str]:
    """Sort key of node ids in ask order: ``n2`` before ``n10``, other ids after, by name."""
    match = _ASK_ORDER_ID.fullmatch(node_id)
    return (0, int(match.group(1)), "") if match else (1, 0, node_id)


def _preorder(by_id: dict[str, TreeNode]) -> Iterator[TreeNode]:
    """Yield the nodes depth-first, children in ask order, each with its drawn ``level``.

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


# --------------------------------------------------------------------------- colours
def model_family(model: str) -> str:
    """Return the LLM family of a model name (see `MODEL_FAMILIES`), else `OTHER_FAMILY`."""
    name = model.lower()
    for family, fragments, _ in MODEL_FAMILIES:
        if any(fragment in name for fragment in fragments):
            return family
    return OTHER_FAMILY


def model_colors(models: Sequence[str]) -> dict[str, str]:
    """Map each model to its family's colour, the family's further models to its shades.

    Args:
        models: The models in legend order; within a family, the first gets the family colour
            and the next ones the `SHADE_STEPS` shades, in this order.

    Returns:
        Model -> ``#rrggbb`` colour.
    """
    base = {family: color for family, _, color in MODEL_FAMILIES}
    seen: defaultdict[str, int] = defaultdict(int)
    colors = {}
    for model in models:
        family = model_family(model)
        step = SHADE_STEPS[seen[family] % len(SHADE_STEPS)]
        seen[family] += 1
        colors[model] = _shade(base.get(family, OTHER_COLOR), step)
    return colors


def _shade(color: str, amount: float) -> str:
    """Move a ``#rrggbb`` colour towards white (``amount`` > 0) or black (< 0) by ``|amount|``."""
    channels = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    target = 255 if amount > 0 else 0
    mixed = [round(channel + (target - channel) * abs(amount)) for channel in channels]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def short_model_name(name: str) -> str:
    """Return a model name without its provider and vendor prefixes.

    ``openrouter:qwen/qwen3.8-max`` becomes ``qwen3.8-max``; a name that is all prefix stays.
    """
    short = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return short or name


# --------------------------------------------------------------------------- file names
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

    Names are compared without case (Windows and macOS file systems ignore it); a repeat,
    or a tree whose name is the index page's, gets ``-2``, ``-3``, ... before its suffix.
    """
    taken = {INDEX_FILENAME}
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


# --------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class TreeLayout:
    """Where the tree is drawn.

    Attributes:
        parents: Node id -> the id of the node drawn above it (None: the question).
        positions: Node id -> the centre of its circle.
        root: The centre of the question box.
        width: The SVG's width.
        height: The SVG's height.
    """

    parents: Mapping[str, str | None]
    positions: Mapping[str, tuple[float, float]]
    root: tuple[float, float]
    width: float
    height: float


def drawn_parents(tree: SearchTree) -> dict[str, str | None]:
    """Return each node's parent as drawn: the nearest node one level up before it in pre-order.

    This is the recorded parent, except for an orphan or a node on a parent cycle, which hang
    off the question (None).
    """
    parents: dict[str, str | None] = {}
    path: list[str] = []  # the drawn path from the question to the previous node
    for node in tree.nodes:
        del path[node.level - 1 :]
        parents[node.id] = path[-1] if path else None
        path.append(node.id)
    return parents


def layout_tree(tree: SearchTree) -> TreeLayout:
    """Lay the tree out top-down: leaves side by side in pre-order, parents over their children."""
    parents = drawn_parents(tree)
    children: defaultdict[str | None, list[str]] = defaultdict(list)
    for node in tree.nodes:
        children[parents[node.id]].append(node.id)
    x_of: dict[str, float] = {}
    leaves = 0
    for node in tree.nodes:  # leaves left to right in pre-order
        if not children[node.id]:
            x_of[node.id] = PADDING + (leaves + 0.5) * SLOT_WIDTH
            leaves += 1
    for node in reversed(tree.nodes):  # a node's children come after it in pre-order
        if children[node.id]:
            x_of[node.id] = (x_of[children[node.id][0]] + x_of[children[node.id][-1]]) / 2
    width = max(2 * PADDING + leaves * SLOT_WIDTH, 2 * PADDING + ROOT_WIDTH)
    tops = children[None]
    root_x = (x_of[tops[0]] + x_of[tops[-1]]) / 2 if tops else width / 2
    root_y = PADDING + ROOT_HEIGHT / 2
    positions = {node.id: (x_of[node.id], _level_y(node.level)) for node in tree.nodes}
    deepest = max((node.level for node in tree.nodes), default=0)
    height = _level_y(deepest) + NODE_RADIUS + LABEL_GAP + PADDING if tree.nodes else 2 * root_y
    return TreeLayout(parents, positions, (root_x, root_y), width, height)


def _level_y(level: int) -> float:
    """Return the y of the centre of a node at tree ``level`` (the question is level 0)."""
    return PADDING + ROOT_HEIGHT / 2 + level * LEVEL_HEIGHT


# --------------------------------------------------------------------------- replay
EventKind = Literal["ask", "tell", "pick"]


@dataclass(frozen=True)
class ReplayEvent:
    """One step of the replay.

    Attributes:
        kind: ``ask`` (the search selects a node and starts a child), ``tell`` (the child's
            score comes back) or ``pick`` (the final choice).
        node: The index in `SearchTree.nodes` of the node the step is about.
        path: The indexes of its drawn ancestors, the question's side first: the selection path
            of an ask, the backup path of a tell.
        caption: What the step shows, in words.
    """

    kind: EventKind
    node: int
    path: tuple[int, ...]
    caption: str


def replay_events(tree: SearchTree) -> list[ReplayEvent]:
    """Return the replay's steps: every ask and tell in the order they happened, then the pick.

    With ``asked_after`` and ``told`` on every node, an ask comes after the results that had
    come back when it was made; otherwise the nodes replay one at a time in ask order.
    """
    index = {node.id: position for position, node in enumerate(tree.nodes)}
    parents = drawn_parents(tree)

    def ancestors(node: TreeNode) -> list[TreeNode]:
        chain = []
        parent = parents[node.id]
        while parent is not None:
            chain.append(tree.nodes[index[parent]])
            parent = parents[parent]
        return chain[::-1]

    clock_zero = min((node.start_ms for node in tree.nodes if node.start_ms is not None), default=0)
    events = []
    for kind, node in _replay_order(tree.nodes):
        path = ancestors(node)
        caption = (
            _ask_caption(node, path, tree.strategy, clock_zero)
            if kind == "ask"
            else _tell_caption(node, path, tree.strategy, clock_zero)
        )
        events.append(ReplayEvent(kind, index[node.id], tuple(index[n.id] for n in path), caption))
    chosen = tree.node(tree.chosen_id)
    if chosen is not None:
        path = ancestors(chosen)
        how = f" by {tree.chosen_by}" if tree.chosen_by else ""
        caption = f"Final pick: {chosen.id}{how}, score {chosen.score:.2f}{_ex_words(chosen)}"
        events.append(
            ReplayEvent("pick", index[chosen.id], tuple(index[n.id] for n in path), caption)
        )
    return events


def _replay_order(nodes: Sequence[TreeNode]) -> list[tuple[EventKind, TreeNode]]:
    """Return every node's ask and tell, in the order they happened."""
    keyed: list[tuple[tuple[int, int, tuple[int, int, str]], EventKind, TreeNode]] = []
    if all(node.asked_after is not None and node.told is not None for node in nodes):
        for node in nodes:
            keyed.append(((node.asked_after or 0, 0, _ask_order(node.id)), "ask", node))
            keyed.append(((node.told or 0, 1, _ask_order(node.id)), "tell", node))
    else:  # older records: one node at a time, in ask order
        for position, node in enumerate(sorted(nodes, key=lambda node: _ask_order(node.id))):
            keyed.append(((position, 0, _ask_order(node.id)), "ask", node))
            keyed.append(((position, 1, _ask_order(node.id)), "tell", node))
    keyed.sort(key=lambda item: item[0])
    return [(kind, node) for _, kind, node in keyed]


def _clock(ms: float | None, zero: float) -> str:
    """Return a caption's time prefix (``12.3 s · ``), or nothing when the time is unknown."""
    return "" if ms is None else f"{max(0.0, ms - zero) / 1000:.1f} s · "


def _ask_caption(
    node: TreeNode, path: Sequence[TreeNode], strategy: str | None, zero: float
) -> str:
    """Describe an ask: the selection path and the child started, in AB-MCTS terms for abmcts."""
    model = short_model_name(node.model)
    context = f" ({node.action} context)" if node.action and node.action != node.model else ""
    with_model = f"with {model}{context}"
    abmcts = strategy == "abmcts"
    if not path:
        what = f"GEN at the question: a new draft {node.id} {with_model}"
        return _clock(node.start_ms, zero) + (what if abmcts else f"Draft {node.id} {with_model}")
    parent = path[-1].id
    if not abmcts:
        return _clock(node.start_ms, zero) + f"Refine {parent} into {node.id} {with_model}"
    route = " → ".join([_ROOT_LABEL, *(step.id for step in path)])
    return _clock(node.start_ms, zero) + (
        f"Select {route} (CONT), then GEN under {parent}: {node.id} {with_model}"
    )


def _tell_caption(
    node: TreeNode, path: Sequence[TreeNode], strategy: str | None, zero: float
) -> str:
    """Describe a tell: the node's score and marks, and the path its reward backs up."""
    status = " · generation failed" if node.failed else " · exec error" if node.exec_failed else ""
    text = (
        f"{node.id} ({short_model_name(node.model)}) scored {node.score:.2f}"
        f"{status}{_ex_words(node)}"
    )
    if strategy == "abmcts":
        route = " → ".join([*(step.id for step in reversed(path)), _ROOT_LABEL])
        text += f"; the reward backs up {route}"
    return _clock(node.end_ms, zero) + text


def _ex_words(node: TreeNode) -> str:
    """Return a node's EX for a caption, or nothing when unknown."""
    if node.ex is None:
        return ""
    return " · EX ✓ (matches gold)" if node.ex else " · EX ✗"


def _events_json(tree: SearchTree, events: Sequence[ReplayEvent]) -> str:
    """Return the player's data as JSON that is safe inside a ``<script>`` data block."""
    models = len({node.model for node in tree.nodes})
    noun = "model" if models == 1 else "models"
    data = {
        "nodes": len(tree.nodes),
        "start": (
            f"Press ▶ Play (or Space) to replay the search: {len(tree.nodes)} nodes by {models} "
            f"{noun}. ← and → step through it; click a node for its prompt and SQL."
        ),
        "events": [
            {"kind": event.kind, "node": event.node, "path": list(event.path),
             "caption": event.caption}
            for event in events
        ],
    }  # fmt: skip
    text = json.dumps(data, ensure_ascii=False)
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


# --------------------------------------------------------------------------- the search page
def render_search_html(tree: SearchTree) -> str:
    """Render one search tree as a standalone HTML page with the replay player.

    The page has the title and facts, a legend (each model's family colour, node count, best
    score and correct nodes when known; the key to the marks), the player (controls, caption and
    the tree) beside the node panel, and the replay steps as a JSON data block.
    """
    colors = model_colors(tree.models)
    layout = layout_tree(tree)
    events = replay_events(tree)
    body = "".join(
        [
            f"<h1>{_escape(tree.title)}</h1>",
            _facts(tree),
            _legend(tree, colors),
            '<div class="player"><div class="player-main" id="player">',
            _controls(),
            '<p class="caption" id="caption" aria-live="polite"></p>',
            f'<div class="stage">{_tree_svg(tree, colors, layout)}</div></div>',
            _panel(tree, colors),
            "</div>",
            '<script type="application/json" id="search-events">',
            _events_json(tree, events),
            "</script>",
            f"<script>{PLAYER_JS}</script>",
        ]
    )
    return _page(tree.title, body, script=PLAYER_JS)


def _escape(value: object) -> str:
    """Escape any value as HTML text or an attribute value."""
    return html.escape(str(value), quote=True)


def script_hash(script: str) -> str:
    """Return the CSP source that admits exactly this inline script (``'sha256-...'``)."""
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    return f"'sha256-{digest}'"


def _page(title: str, body: str, *, script: str | None = None) -> str:
    """Wrap ``body`` in a document with the CSP meta, the viewport and the style sheet.

    The CSP forbids every script, or admits only ``script`` (by its hash) when one is given.
    """
    csp = _CSP + (f"; script-src {script_hash(script)}" if script is not None else "")
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{_escape(csp)}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title)}</title><style>{CSS}</style></head>"
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


def _controls() -> str:
    """Render the player's controls (shown only when the script runs)."""
    speeds = "".join(
        f'<option value="{speed}"{" selected" if speed == "1" else ""}>{speed}×</option>'
        for speed in _SPEEDS
    )
    return (
        '<div class="controls" role="toolbar" aria-label="replay">'
        '<button type="button" id="first" title="first step (Home)">|◀</button>'
        '<button type="button" id="back" title="step back (←)">◀ step</button>'
        '<button type="button" id="play" title="play or pause (Space)">▶ Play</button>'
        '<button type="button" id="forward" title="step forward (→)">step ▶</button>'
        '<button type="button" id="last" title="last step (End)">▶|</button>'
        '<input type="range" id="step" min="0" max="0" value="0" aria-label="replay step">'
        '<span class="counter" id="counter"></span>'
        f'<label>speed <select id="speed">{speeds}</select></label></div>'
    )


# --------------------------------------------------------------------------- legend
def _legend(tree: SearchTree, colors: Mapping[str, str]) -> str:
    """Render the model table (colour, family, nodes, best score, correct nodes) and the key."""
    known_ex = any(node.ex is not None for node in tree.nodes)
    header = "<th>model</th><th>family</th><th>nodes</th><th>best score</th>" + (
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
        f"{_escape(model)}</td><td>{_escape(model_family(model))}</td>"
        f"<td>{len(nodes)}</td><td>{best_text}</td>{correct}</tr>"
    )


def _marks_key() -> str:
    """Render the key to the node marks and the replay's colours, each with a small drawing."""
    radius, size = 6, 20
    center = size / 2

    def swatch(shape: str) -> str:
        return f'<svg width="{size}" height="{size}" aria-hidden="true">{shape}</svg>'

    circle = f'<circle cx="{center}" cy="{center}" r="{radius}" class="node"'
    line = f'<line x1="2" y1="{center}" x2="{size - 2}" y2="{center}" stroke-width="3.5" '
    entries = [
        (swatch(f'{circle} fill="{_KEY_FILL}"/>'), "ran (fill: model family)"),
        (swatch(f'{circle} fill="none" stroke-width="3"/>'), "exec error (hollow)"),
        (swatch(f'{circle} fill="none" stroke-dasharray="3 2"/>'), "⚠ generation failed"),
        (
            swatch(
                f'{circle} fill="{_KEY_FILL}"/>'
                f'<circle cx="{center}" cy="{center}" r="{radius + 3}" class="chosen-ring"/>'
            ),
            "★ chosen (ring)",
        ),
        (swatch(f'{line}style="stroke:var(--select)"/>'), "selection path"),
        (swatch(f'{line}style="stroke:var(--backup)"/>'), "reward backup"),
    ]
    items = "".join(f"<span>{shape}{_escape(label)}</span>" for shape, label in entries)
    return f'<div class="key">{items}</div>'


# --------------------------------------------------------------------------- the tree
def _tree_svg(tree: SearchTree, colors: Mapping[str, str], layout: TreeLayout) -> str:
    """Render the tree: the edges, the question box, then one linked group per node."""
    edges = "".join(_edge(index, node, layout) for index, node in enumerate(tree.nodes))
    groups = "".join(
        _node_group(index, node, tree, colors[node.model], layout)
        for index, node in enumerate(tree.nodes)
    )
    root_x, root_y = layout.root
    root = (
        f'<g class="root" id="root-node"><rect x="{root_x - ROOT_WIDTH / 2:.1f}" '
        f'y="{root_y - ROOT_HEIGHT / 2:.1f}" width="{ROOT_WIDTH}" height="{ROOT_HEIGHT}" rx="6"/>'
        f'<text x="{root_x:.1f}" y="{root_y:.1f}">{_ROOT_LABEL}</text></g>'
    )
    width, height = round(layout.width), round(layout.height)
    shown_width, shown_height = round(width * DISPLAY_SCALE), round(height * DISPLAY_SCALE)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{shown_width}" height="{shown_height}" role="img" aria-label="search tree">'
        f"{edges}{root}{groups}</svg>"
    )


def _edge(index: int, node: TreeNode, layout: TreeLayout) -> str:
    """Draw the curve from the node's drawn parent (or the question box) down to the node."""
    parent = layout.parents[node.id]
    if parent is None:
        start_x, start_y = layout.root[0], layout.root[1] + ROOT_HEIGHT / 2
    else:
        start_x, start_y = layout.positions[parent]
        start_y += NODE_RADIUS
    end_x, end_y = layout.positions[node.id]
    end_y -= NODE_RADIUS
    middle = (start_y + end_y) / 2
    return (
        f'<path class="edge" id="e-{index}" d="M{start_x:.1f} {start_y:.1f} '
        f'C{start_x:.1f} {middle:.1f} {end_x:.1f} {middle:.1f} {end_x:.1f} {end_y:.1f}"/>'
    )


def _node_group(
    index: int, node: TreeNode, tree: SearchTree, color: str, layout: TreeLayout
) -> str:
    """Render one node: its circle, score, id label, EX mark and chosen ring, linked to its section.

    Args:
        index: The node's index in `SearchTree.nodes`, which names its elements.
        node: The node.
        tree: The tree, for the chosen node.
        color: The node's model colour.
        layout: Where the node sits.

    Returns:
        An ``<a>`` element holding the node's shapes and text.
    """
    x, y = layout.positions[node.id]
    marks = _marks(node, tree)
    parts = [
        f'<title data-pending="{_escape(_pending_tooltip(node))}">'
        f"{_escape(_tooltip(node, marks))}</title>",
        f'<circle class="halo" cx="{x:.1f}" cy="{y:.1f}" r="{NODE_RADIUS + 5}"/>',
        _node_circle(node, x, y, color),
        f'<circle class="chosen-ring" cx="{x:.1f}" cy="{y:.1f}" r="{NODE_RADIUS + 4}"/>'
        if node.id == tree.chosen_id
        else "",
        ""
        if node.failed
        else f'<text class="score" x="{x:.1f}" y="{y:.1f}">{_score_text(node.score)}</text>',
        f'<text class="label" x="{x:.1f}" y="{y + NODE_RADIUS + LABEL_GAP:.1f}">'
        f"{_escape(node.id)}</text>",
        _ex_mark(node, x, y),
    ]
    return f'<a href="#node-{index}" class="node-g" id="g-{index}">{"".join(parts)}</a>'


def _node_circle(node: TreeNode, x: float, y: float, color: str) -> str:
    """Draw the node: filled when it ran, hollow on an exec error, dashed when it failed.

    The model colour is the ``--c`` custom property, which the style sheet uses for the fill
    or the outline, and the replay for the outline of a node still running.
    """
    shape = "dashed" if node.failed else "hollow" if node.exec_failed else "filled"
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{NODE_RADIUS}" class="node {shape}" '
        f'style="--c:{_escape(color)}"/>'
    )


def _score_text(score: float) -> str:
    """Return a score as the node shows it: ``.82``, ``1.0`` (it must fit in the circle)."""
    clamped = min(max(score, 0.0), 1.0)
    return "1.0" if clamped >= 0.995 else f"{clamped:.2f}".removeprefix("0")


def _ex_mark(node: TreeNode, x: float, y: float) -> str:
    """Draw a tick or a cross right of the node when its EX is known."""
    if node.ex is None:
        return ""
    css, mark = ("good", "✓") if node.ex else ("bad", "✗")
    position = f'x="{x + NODE_RADIUS + 2:.1f}" y="{y - NODE_RADIUS:.1f}"'
    return f'<text class="ex {css}" {position}>{mark}</text>'


def _tooltip(node: TreeNode, marks: Sequence[str]) -> str:
    """Return the hover text of a node."""
    parent = f"refines {node.parent_id}" if node.parent_id else "draft"
    parts = [node.id, node.model, f"score {node.score:.2f}", parent]
    if node.action and node.action != node.model:
        parts.append(node.action)
    return " · ".join([*parts, *marks])


def _pending_tooltip(node: TreeNode) -> str:
    """Return the hover text the player shows while the node is generating: no outcome yet."""
    return " · ".join([node.id, node.model, "generating"])


# --------------------------------------------------------------------------- node panel
def _panel(tree: SearchTree, colors: Mapping[str, str]) -> str:
    """Render the node panel: every node's section (the player shows one), then the instructions."""
    sections = "".join(
        _node_section(index, node, tree, colors) for index, node in enumerate(tree.nodes)
    )
    schemas = "".join(_schema_context(key, tree.contexts[key]) for key in _context_keys(tree))
    instructions = (
        "<details><summary>generator instructions (system prompt)</summary>"
        f"<pre>{_escape(tree.instructions)}</pre></details>"
        if tree.instructions
        else ""
    )
    return (
        '<aside class="panel" aria-label="node details">'
        '<p class="panel-empty" id="panel-empty">Play the replay, or click a node, to see its '
        f"SQL and generator prompt.</p>{sections}{schemas}{instructions}</aside>"
    )


def _node_section(index: int, node: TreeNode, tree: SearchTree, colors: Mapping[str, str]) -> str:
    """Render one node's section: heading, facts, SQL, rationale, prompt, rubric and feedback.

    Everything but the id, model and prompt is the node's outcome (class ``outcome``), which the
    player hides while the node is still generating.
    """
    marks = _marks(node, tree)
    marks_html = f' <span class="marks">{_escape("  ".join(marks))}</span>' if marks else ""
    heading = (
        f'<h3><span class="swatch" style="background:{_escape(colors[node.model])}"></span>'
        f"{_escape(node.id)} · {_escape(short_model_name(node.model))}"
        f'<span class="outcome"> · {node.score:.2f}{marks_html}</span></h3>'
    )
    outcome = f"{_node_facts(node)}{_sql(node.sql)}{_prose('rationale', node.rationale)}"
    judged = (
        f"{_rubric(node)}{_text_list('feedback', node.feedback)}{_prose('advice', node.advice)}"
    )
    return (
        f'<section class="node-detail" id="node-{index}">{heading}{_PENDING_NOTE}'
        f'<div class="outcome">{outcome}</div>{_prompt(node, tree.contexts)}'
        f'<div class="outcome">{judged}</div></section>'
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
        ("time", _duration(node)),
    ]
    items = "".join(
        f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>" for label, value in facts if value
    )
    return f'<dl class="facts">{items}</dl>'


def _duration(node: TreeNode) -> str | None:
    """Return how long the node took, when both ends are recorded."""
    if node.start_ms is None or node.end_ms is None:
        return None
    return f"{(node.end_ms - node.start_ms) / 1000:.1f} s"


def _prompt(node: TreeNode, contexts: Mapping[str, str]) -> str:
    """Render the generator prompt, its schema DDL folded away in its own disclosure."""
    if not node.prompt:
        return ""
    parts = []
    for position, piece in enumerate(_SCHEMA_MARKER.split(node.prompt)):
        if position % 2 == 0:  # prompt text between markers
            if piece.strip():
                parts.append(f'<pre class="prompt-text">{_escape(piece.strip(chr(10)))}</pre>')
        elif piece in contexts:
            parts.append(
                f'<p><a class="ctx-link" href="#ctx-{_escape(piece)}">schema DDL '
                f"({_line_count(contexts[piece])} lines)</a></p>"
            )
        else:
            parts.append(f'<p class="muted">[schema {_escape(piece)}: not saved with this run]</p>')
    return f"<details open><summary>generator prompt</summary>{''.join(parts)}</details>"


def _context_keys(tree: SearchTree) -> list[str]:
    """Return the saved schema contexts the nodes' prompts name, in the order they first appear."""
    keys: dict[str, None] = {}
    for node in tree.nodes:
        for position, piece in enumerate(_SCHEMA_MARKER.split(node.prompt or "")):
            if position % 2 == 1 and piece in tree.contexts:
                keys.setdefault(piece)
    return list(keys)


def _schema_context(key: str, ddl: str) -> str:
    """Render one schema DDL, folded away, as the target of the prompts' links to it."""
    return (
        f'<details class="schema-context" id="ctx-{_escape(key)}">'
        f"<summary>schema DDL {_escape(key)} ({_line_count(ddl)} lines)</summary>"
        f"<pre>{_escape(ddl)}</pre></details>"
    )


def _line_count(text: str) -> int:
    """Return the number of lines in ``text``."""
    return text.count("\n") + 1


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
    """Render the query in a block that wraps inside itself."""
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
