"""Where to spend the generation budget: TreeQuest AB-MCTS and three baselines on one interface.

Every strategy calls the same ``generate(node_id, parent, action)`` (already scored), so at equal
budget the strategy is the only variable:

* ``single``    one draft;
* ``best_of_n`` N independent drafts, breadth only (BestOfN; drafts alternate the tight and wide
  context actions);
* ``refine``    a chain, each node refining the previous one with its feedback (deep only; the
  pattern of DSPy's Refine, implemented here; DSPy is not a dependency);
* ``abmcts``    TreeQuest ``ABMCTSA``: Thompson sampling per node decides between a new child (a
  draft at the root, "wider") and expanding an existing one (a refinement, "deeper"), and between
  the ``tight`` and ``wide`` context actions.

The final pick is the same for every multi-node strategy: top-k by score, deduplicated by result,
then a round-robin both-order pairwise selector.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, cast

from schemagraph.agent.results import Action, AgentConfig, Candidate

GenerateFn = Callable[[str, Candidate | None, Action], Awaitable[Candidate]]
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


def _trial_node(
    generate: GenerateFn,
    node_id: str,
    trial: Any,
    timeout_s: float,
) -> Awaitable[Candidate]:
    """Generate the node of a TreeQuest trial, whose parent state is a Candidate (None at root)."""
    parent = cast(Candidate | None, trial.parent_state)
    action = cast(Action, trial.action)
    return _safe(generate, node_id, parent, action, timeout_s)


async def _safe(
    generate: GenerateFn,
    node_id: str,
    parent: Candidate | None,
    action: Action,
    timeout_s: float,
) -> Candidate:
    """Generate one node; a failure or a timeout is a score-0 node.

    The budget then stays exact and the tree consistent.
    """
    try:
        return await asyncio.wait_for(generate(node_id, parent, action), timeout_s)
    except Exception as error:
        if isinstance(error, TimeoutError):
            message = "node timed out"
        else:
            message = f"{type(error).__name__}: {error}"
        return new_node(node_id, parent, action, error=message[:NODE_ERROR_CHARS])


async def run_search(generate: GenerateFn, cfg: AgentConfig) -> SearchTrace:
    """Spend ``cfg.budget`` generator nodes with ``cfg.strategy``.

    Raises:
        ValueError: The strategy is unknown.
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


def _actions(cfg: AgentConfig) -> list[Action]:
    return list(cfg.actions) or ["wide"]


def _searching(trace: SearchTrace, cfg: AgentConfig, budget: int) -> bool:
    """Whether budget is left and no node has reached ``cfg.early_stop`` yet."""
    return trace.nodes < budget and trace.best < cfg.early_stop


async def _single(
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """One wide draft."""
    trace.candidates.append(await _safe(generate, ids(), None, "wide", cfg.node_timeout_s))


async def _best_of_n(
    generate: GenerateFn,
    cfg: AgentConfig,
    trace: SearchTrace,
    ids: _NodeIds,
    budget: int,
) -> None:
    """Independent drafts in batches, cycling through the context actions."""
    actions = _actions(cfg)
    while _searching(trace, cfg, budget):
        size = min(cfg.batch_size, budget - trace.nodes)
        batch = [(ids(), actions[(trace.nodes + i) % len(actions)]) for i in range(size)]
        trace.candidates += await asyncio.gather(
            *(
                _safe(generate, node_id, None, action, cfg.node_timeout_s)
                for node_id, action in batch
            )
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
        parent = await _safe(generate, ids(), parent, action, cfg.node_timeout_s)
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
    """TreeQuest AB-MCTS-A in batches: each trial is a draft or a refinement of a sampled node."""
    import numpy as np
    import treequest as tq

    np.random.seed(cfg.seed)  # TreeQuest samples from the global numpy RNG
    algorithm = tq.ABMCTSA()
    state = algorithm.init_tree()
    actions = _actions(cfg)
    while _searching(trace, cfg, budget):
        size = min(cfg.batch_size, budget - trace.nodes)
        state, trials = algorithm.ask_batch(state, size, actions)
        batch = [(ids(), trial) for trial in trials]
        nodes = await asyncio.gather(
            *(_trial_node(generate, node_id, trial, cfg.node_timeout_s) for node_id, trial in batch)
        )
        for (_, trial), node in zip(batch, nodes, strict=True):
            reward = min(1.0, max(0.0, node.score))
            state = algorithm.tell(state, trial.trial_id, (node, reward))
        trace.candidates += nodes


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
