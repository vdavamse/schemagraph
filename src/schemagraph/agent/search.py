"""Where to spend the generation budget: TreeQuest AB-MCTS and three baselines on one interface.

Every strategy calls the same ``generate(node_id, parent, action)`` (already scored), so at equal
budget the strategy is the only variable:

* ``single``    one draft;
* ``best_of_n`` N independent drafts, breadth only (BestOfN; drafts cycle through the actions);
* ``refine``    a chain, each node refining the previous one with its feedback (deep only; the
  pattern of DSPy's Refine, implemented here; DSPy is not a dependency);
* ``abmcts``    TreeQuest AB-MCTS, ``ABMCTSA`` or ``ABMCTSM`` (``AgentConfig.abmcts_algorithm``):
  decides between a new child (a draft at the root, "wider") and expanding an existing one (a
  refinement, "deeper"), and between the actions. Lockstep batches, or a rolling loop that asks
  for the next trial as each node finishes (``AgentConfig.rolling``).

The actions are the ``tight`` and ``wide`` schema contexts, or, with several generator models
(``AgentConfig.gen_models``), the models themselves: the paper's Multi-LLM AB-MCTS, where every
node links the wide context (:func:`split_action`).

The final pick is the same for every multi-node strategy: top-k by score, deduplicated by result,
then a round-robin both-order pairwise selector.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import threading
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, cast

from schemagraph.agent.results import Action, AgentConfig, Candidate

# generate(node id, parent or None, action): the action is a context width, or a generator
# model when several are searched (see split_action).
GenerateFn = Callable[[str, Candidate | None, str], Awaitable[Candidate]]
# p(first is better), from one selector call.
PickFn = Callable[[Candidate, Candidate], Awaitable[float]]
# Selector preferences: candidate id -> other candidate id -> p(the first is better).
PreferenceMatrix = dict[str, dict[str, float]]

# Characters of a failed node's error message.
NODE_ERROR_CHARS = 500
# A judge probability of a missing table at least this high makes a refinement widen its context.
WIDEN_MISSING_P = 0.5
# Finding codes after which a refinement widens its context.
WIDEN_CODES = frozenset({"unknown_table", "unknown_column"})

# One AB-MCTS-M fit at a time per process: every 10th fit, TreeQuest frees every live JAX array
# in the process, including those of a fit running in another thread.
_ABMCTS_M_LOCK = threading.Lock()
# TreeQuest's model selection strategy for each of the paper's generator selection algorithms
# (appendix D.1): 1 shares one GEN node and then samples the generator, 2 keeps one GEN node
# per generator.
_TREEQUEST_SELECTION = {1: "multiarm_bandit_thompson", 2: "stack"}
# How to install AB-MCTS-M. An exact sync drops unnamed extras, so the agent extra is named too.
ABMCTS_M_INSTALL = "uv sync --extra agent --extra abmcts-m"


@dataclass
class SearchTrace:
    """What a search produced.

    Attributes:
        candidates: Every node, in generation order.
        stopped_early: The search stopped before its budget because a node scored high enough.
    """

    candidates: list[Candidate] = field(default_factory=list)
    stopped_early: bool = False

    @property
    def nodes(self) -> int:
        """Nodes generated so far."""
        return len(self.candidates)

    @property
    def best(self) -> float:
        """The best score so far (0 before any node)."""
        return max((candidate.score for candidate in self.candidates), default=0.0)


class _NodeIds:
    """Hands out node ids ``n0``, ``n1``, ... in call order."""

    def __init__(self) -> None:
        self.issued = 0

    def __call__(self) -> str:
        self.issued += 1
        return f"n{self.issued - 1}"


def new_node(node_id: str, parent: Candidate | None, action: Action, **fields: Any) -> Candidate:
    """Return a candidate placed in the tree: under ``parent``, one level deeper, or at the root."""
    return Candidate(
        id=node_id,
        parent_id=parent.id if parent else None,
        depth=parent.depth + 1 if parent else 0,
        action=action,
        **fields,
    )


def split_action(cfg: AgentConfig, action: str) -> tuple[str | None, Action]:
    """Return the generator model a node's action picks and the context it links.

    With several generator models (``cfg.gen_models``) an action names one of them and every
    node links the wide context; a width, which the ``single`` and ``refine`` strategies pass,
    then goes to the first model. With one generator the model is None and the action is the
    context width.
    """
    if action in cfg.gen_models:
        return action, "wide"
    default = cfg.gen_models[0] if cfg.gen_models else None
    return default, cast(Action, action)


def _trial_node(
    generate: GenerateFn, cfg: AgentConfig, node_id: str, trial: Any
) -> Awaitable[Candidate]:
    """Generate the node of a TreeQuest trial, whose parent state is a Candidate (None at root)."""
    parent = cast(Candidate | None, trial.parent_state)
    return _safe(generate, cfg, node_id, parent, trial.action)


async def _safe(
    generate: GenerateFn,
    cfg: AgentConfig,
    node_id: str,
    parent: Candidate | None,
    action: str,
) -> Candidate:
    """Generate one node within ``cfg.node_timeout_s``; a failure or a timeout is a score-0 node.

    The budget then stays exact and the tree consistent.
    """
    try:
        return await asyncio.wait_for(generate(node_id, parent, action), cfg.node_timeout_s)
    except Exception as error:
        if isinstance(error, TimeoutError):
            message = "node timed out"
        else:
            message = f"{type(error).__name__}: {error}"
        generator, context = split_action(cfg, action)
        return new_node(
            node_id, parent, context, generator=generator, error=message[:NODE_ERROR_CHARS]
        )


async def run_search(generate: GenerateFn, cfg: AgentConfig) -> SearchTrace:
    """Spend ``cfg.budget`` generator nodes with ``cfg.strategy``.

    Raises:
        ValueError: The strategy, or the AB-MCTS algorithm, is unknown.
    """
    strategies = {
        "single": _single,
        "best_of_n": _best_of_n,
        "refine": _refine,
        "abmcts": _abmcts,
    }
    if cfg.strategy not in strategies:
        raise ValueError(f"unknown strategy {cfg.strategy!r}")
    trace = SearchTrace()
    budget = max(1, cfg.budget)
    await strategies[cfg.strategy](generate, cfg, trace, _NodeIds(), budget)
    trace.stopped_early = trace.nodes < budget and cfg.strategy != "single"
    return trace


def _actions(cfg: AgentConfig) -> list[str]:
    """Return the search's actions: the generator models when there are several, else widths."""
    if len(cfg.gen_models) > 1:
        return list(cfg.gen_models)
    return list(cfg.actions) or ["wide"]


def _searching(trace: SearchTrace, cfg: AgentConfig, budget: int) -> bool:
    """Whether budget is left and the early stop has not been reached (see :func:`_launchable`)."""
    return _launchable(trace, cfg, budget) > 0


def _launchable(trace: SearchTrace, cfg: AgentConfig, budget: int, in_flight: int = 0) -> int:
    """Return how many more nodes the search may start, with ``in_flight`` nodes still running.

    Before the early stop, that is the budget left. After it, it is what is left of the
    ``cfg.early_stop_min_nodes`` floor, often nothing. The early stop needs
    ``cfg.early_stop_agree`` nodes that score at least ``cfg.early_stop`` and return the same
    result (:func:`fingerprint`) from different SQL; with 1, one high score is enough.
    """
    launched = trace.nodes + in_flight
    room = budget - launched
    if _agreed(trace.candidates, cfg):
        room = min(room, cfg.early_stop_min_nodes - launched)
    return max(0, room)


def _agreed(candidates: list[Candidate], cfg: AgentConfig) -> bool:
    """Whether enough high-scoring nodes share one result to stop early."""
    high = [candidate for candidate in candidates if candidate.score >= cfg.early_stop]
    if cfg.early_stop_agree <= 1:
        return bool(high)
    # One vote per distinct query: a refinement that repeats its parent's SQL is no second opinion.
    votes = {(fingerprint(candidate), " ".join(candidate.sql.split())) for candidate in high}
    groups = Counter(result for result, _ in votes)
    groups.pop(None, None)
    return max(groups.values(), default=0) >= cfg.early_stop_agree


async def _single(
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """One wide draft."""
    trace.candidates.append(await _safe(generate, cfg, ids(), None, "wide"))


async def _best_of_n(
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """Independent drafts in batches, cycling through the actions (widths or models)."""
    actions = _actions(cfg)
    while _searching(trace, cfg, budget):
        size = min(cfg.batch_size, budget - trace.nodes)
        batch = [(ids(), actions[(trace.nodes + i) % len(actions)]) for i in range(size)]
        trace.candidates += await asyncio.gather(
            *(_safe(generate, cfg, node_id, None, action) for node_id, action in batch)
        )


async def _refine(
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """A chain: a wide draft, then each node refines the previous one."""
    parent: Candidate | None = None
    while _searching(trace, cfg, budget):
        action: Action = "wide" if parent is None else _refine_action(parent)
        parent = await _safe(generate, cfg, ids(), parent, action)
        trace.candidates.append(parent)


def _refine_action(parent: Candidate) -> Action:
    """Keep the parent's context, widening when it hit unknown names or a table seems missing."""
    codes = {finding.code for finding in parent.checks.findings} if parent.checks else set()
    missing = parent.judgement and any(
        p >= WIDEN_MISSING_P for p in parent.judgement.missing.values()
    )
    return "wide" if codes & WIDEN_CODES or missing else parent.action


async def _abmcts(
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """TreeQuest AB-MCTS: each trial is a draft or a refinement of a sampled node."""
    import numpy as np

    # in a thread: with the abmcts-m extra, importing TreeQuest loads JAX, PyMC and NumPyro
    # (about 26 s against 9 s without them on the Windows mount), even for AB-MCTS-A
    algorithm = await asyncio.to_thread(_abmcts_algorithm, cfg)
    # AB-MCTS-A samples from the global numpy RNG, seeded after the import because that import
    # draws from it. AB-MCTS-M is not seeded: its first choice uses Python's random module, PyMC
    # samples without a seed, and batches run in worker processes.
    np.random.seed(cfg.seed)
    search = _abmcts_rolling if cfg.rolling else _abmcts_lockstep
    await search(algorithm, generate, cfg, trace, ids, budget)


def _abmcts_algorithm(cfg: AgentConfig) -> Any:
    """Return TreeQuest's AB-MCTS-A or AB-MCTS-M.

    Raises:
        ValueError: The algorithm is neither ``a`` nor ``m`` (:func:`_ask` locks only for
            ``m``), or the generator selection is neither 1 nor 2.
        ImportError: ``m`` without the ``abmcts-m`` extra (PyMC, NumPyro).
    """
    import treequest as tq

    if cfg.generator_selection not in _TREEQUEST_SELECTION:
        raise ValueError(f"unknown generator selection {cfg.generator_selection!r}")
    selection = _TREEQUEST_SELECTION[cfg.generator_selection]
    if cfg.abmcts_algorithm == "a":
        return tq.ABMCTSA(model_selection_strategy=selection)
    if cfg.abmcts_algorithm != "m":
        raise ValueError(f"unknown AB-MCTS algorithm {cfg.abmcts_algorithm!r}")
    try:
        import pymc  # noqa: F401  # TreeQuest exports a placeholder ABMCTSM without it
    except ImportError as error:
        raise ImportError(f"AB-MCTS-M needs PyMC and NumPyro: {ABMCTS_M_INSTALL}") from error
    # a batch is chosen in worker processes that each load JAX; TreeQuest's default is one per CPU
    return tq.ABMCTSM(
        model_selection_strategy=selection, max_process_workers=max(1, cfg.batch_size)
    )


async def _abmcts_lockstep(
    algorithm: Any,
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """Ask for ``cfg.batch_size`` trials, run them all, tell every result, and repeat.

    The early stop is checked between batches: once it is reached, the search stops at the first
    batch boundary at or past the ``cfg.early_stop_min_nodes`` floor.
    """
    state = algorithm.init_tree()
    actions = _actions(cfg)
    while _searching(trace, cfg, budget):
        size = min(cfg.batch_size, budget - trace.nodes)
        state, trials = await asyncio.to_thread(_ask, algorithm, cfg, state, size, actions)
        batch = [(ids(), trial) for trial in trials]
        nodes = await asyncio.gather(
            *(_trial_node(generate, cfg, node_id, trial) for node_id, trial in batch)
        )
        for (_, trial), node in zip(batch, nodes, strict=True):
            state = _tell(algorithm, state, trial, node)
        trace.candidates += nodes


async def _abmcts_rolling(
    algorithm: Any,
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """Keep ``cfg.batch_size`` nodes in flight; tell each result and ask again as it lands.

    Every trial after the first batch is chosen knowing every finished node, where a lockstep
    batch knows only the nodes that finished before it started.

    * Nodes join ``trace`` in completion order; their ids are issued in ask order.
    * The ``cfg.early_stop_min_nodes`` floor counts nodes in flight, so a search whose nodes
      agree early stops at the floor rather than up to a batch past it.
    * Once the search stops, the nodes still in flight are awaited and kept: they are paid for.
    * If the search is cancelled or TreeQuest fails, the nodes in flight are cancelled.
    """
    width = max(1, cfg.batch_size)
    state = algorithm.init_tree()
    actions = _actions(cfg)
    in_flight: dict[asyncio.Future[Candidate], Any] = {}  # each running node's TreeQuest trial

    async def launch(count: int) -> None:
        nonlocal state
        state, trials = await asyncio.to_thread(_ask, algorithm, cfg, state, count, actions)
        for trial in trials:
            task = asyncio.ensure_future(_trial_node(generate, cfg, ids(), trial))
            in_flight[task] = trial

    try:
        await launch(min(width, budget))
        while in_flight:
            finished, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                trial = in_flight.pop(task)
                node = task.result()  # _safe turns every failure into a score-0 node
                state = _tell(algorithm, state, trial, node)
                trace.candidates.append(node)
            count = min(width - len(in_flight), _launchable(trace, cfg, budget, len(in_flight)))
            if count > 0:
                await launch(count)
    finally:
        for task in in_flight:
            task.cancel()
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)


def _ask(
    algorithm: Any,
    cfg: AgentConfig,
    state: Any,
    count: int,
    actions: list[str],
) -> tuple[Any, list[Any]]:
    """Ask TreeQuest for ``count`` trials; AB-MCTS-M fits under :data:`_ABMCTS_M_LOCK`.

    Args:
        algorithm: TreeQuest's ``ABMCTSA`` or ``ABMCTSM``.
        cfg: The search settings; ``cfg.abmcts_algorithm`` decides whether to take the lock.
        state: The algorithm's tree state.
        count: Trials to ask for.
        actions: The actions a trial may take: context widths, or generator models.

    Returns:
        The new tree state and the trials, as ``algorithm.ask_batch`` returns them.
    """
    lock = _ABMCTS_M_LOCK if cfg.abmcts_algorithm == "m" else contextlib.nullcontext()
    with lock:
        return algorithm.ask_batch(state, count, actions)


def _tell(algorithm: Any, state: Any, trial: Any, node: Candidate) -> Any:
    """Report a finished node to TreeQuest, with its score clamped to [0, 1] as the reward.

    Cheap for both algorithms (it adds a node to the tree; AB-MCTS-M fits in :func:`_ask`), so
    it runs on the event loop.

    Args:
        algorithm: TreeQuest's ``ABMCTSA`` or ``ABMCTSM``.
        state: The algorithm's tree state.
        trial: The trial the node ran for.
        node: The finished node; its score becomes the reward.

    Returns:
        The new tree state.
    """
    reward = min(1.0, max(0.0, node.score))
    return algorithm.tell(state, trial.trial_id, (node, reward))


def fingerprint(candidate: Candidate) -> str | None:
    """Return a hash of a result, so candidates with the same result vote together.

    Order-insensitive over the preview rows, plus the row and column counts. None when the
    candidate did not execute.
    """
    result = candidate.exec
    if not result or not result.ok:
        return None
    rows = sorted(repr(row) for row in result.rows)
    text = f"{result.row_count}|{len(result.columns)}|{'|'.join(rows)}"
    return hashlib.sha1(text.encode()).hexdigest()


def _group_key(candidate: Candidate) -> str:
    return fingerprint(candidate) or candidate.id


def _representatives(
    ranked: list[Candidate],
) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    """Return the best candidate per distinct result, in rank order, and every result group."""
    groups: dict[str, list[Candidate]] = {}
    representatives: list[Candidate] = []
    for candidate in ranked:
        key = _group_key(candidate)
        if key not in groups:
            representatives.append(candidate)
        groups.setdefault(key, []).append(candidate)
    return representatives, groups


async def _preferences(candidates: list[Candidate], pick: PickFn) -> PreferenceMatrix:
    """Ask the selector about every pair in both orders; the average cancels position bias."""
    pairs = list(combinations(candidates, 2))
    forward = await asyncio.gather(*(pick(a, b) for a, b in pairs))
    backward = await asyncio.gather(*(pick(b, a) for a, b in pairs))
    matrix: PreferenceMatrix = {candidate.id: {} for candidate in candidates}
    for (a, b), p_ab, p_ba in zip(pairs, forward, backward, strict=True):
        p = (p_ab + 1 - p_ba) / 2
        matrix[a.id][b.id] = p
        matrix[b.id][a.id] = 1 - p
    return matrix


async def select_final(
    candidates: list[Candidate],
    cfg: AgentConfig,
    pick: PickFn | None,
) -> tuple[Candidate | None, str | None, PreferenceMatrix]:
    """Pick the answer among the search's candidates.

    Args:
        candidates: Every node, in generation order.
        cfg: The answer's settings (``top_k``, ``selector``, ``strategy``).
        pick: The pairwise selector; None ranks by score alone.

    Returns:
        The pick, how it was chosen (``only``, ``score`` or ``selector``) and the selector's
        preference matrix (empty unless the selector ran).
    """
    if not candidates:
        return None, None, {}
    order = {candidate.id: i for i, candidate in enumerate(candidates)}

    def by_score(candidate: Candidate) -> tuple[float, int]:
        return (-candidate.score, order[candidate.id])

    executed = [c for c in candidates if c.exec and c.exec.ok]
    if not executed:  # nothing ran: no point asking the selector to choose among failures
        best = sorted(candidates, key=by_score)[0]
        return best, "only" if len(candidates) == 1 else "score", {}
    ranked = sorted(executed, key=by_score)
    representatives, groups = _representatives(ranked)
    representatives = representatives[: max(1, cfg.top_k)]
    if len(candidates) == 1:
        return ranked[0], "only", {}
    if len(representatives) < 2 or not cfg.selector or pick is None or cfg.strategy == "single":
        return representatives[0], "score", {}
    matrix = await _preferences(representatives, pick)

    def rank_key(candidate: Candidate) -> tuple[float, int, float, int]:
        wins = sum(matrix[candidate.id].values())
        votes = len(groups[_group_key(candidate)])
        return (wins, votes, candidate.score, -order[candidate.id])

    return max(representatives, key=rank_key), "selector", matrix
