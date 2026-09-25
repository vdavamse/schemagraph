"""One reward in [0, 1] per candidate, from the deterministic checks, execution and the judge.

Additive with a floor: a candidate that executed always outranks one that failed, and one warning
with a mediocre judge does not collapse the reward to ~0 (a multiplicative ``det * judge`` would
give AB-MCTS a flat signal). A clean executed candidate needs a judge mean >= 0.804 to reach the 0.9
early stop: ``0.15 + 0.85 * (0.4 + 0.6 * j) >= 0.9``; one warning (det 0.7) cannot reach it.
The judge's ``missing`` tables are feedback, not score.
"""

from __future__ import annotations

from schemagraph.agent.results import (
    RUBRIC_FIELDS,
    CheckReport,
    ExecResult,
    Judgement,
    RubricBase,
    ScoreWeights,
)

# A judge "missing table" probability at or above this becomes a refine hint.
MISSING_FEEDBACK_P = 0.5
# A rubric field below this probability is reported as a reviewer doubt.
DOUBT_P = 0.5
# Most "may need table(s)" suggestions in one feedback line.
MISSING_TABLES_SHOWN = 5

_SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}

ScoreParts = dict[str, float | None]


def combine(
    checks: CheckReport | None,
    execution: ExecResult | None,
    judgement: Judgement | None,
    weights: ScoreWeights | None = None,
) -> tuple[float, ScoreParts]:
    """Combine checks, execution and judgement into one reward.

    Args:
        checks: The deterministic check report; None or unparsed scores 0.
        execution: The execution result; a guard refusal scores 0 and any other failure
            ``weights.exec_fail``.
        judgement: The judge's rubric; without it the deterministic score stands alone.
        weights: The formula's weights; the defaults when None.

    Returns:
        The reward and its parts: ``det`` (deterministic score), ``judge`` (judge mean, or None)
        and ``x`` (the blend the floor is applied to).
    """
    weights = weights or ScoreWeights()
    if checks is None or not checks.parsed:
        return 0.0, {"det": 0.0, "judge": None, "x": 0.0}
    if execution is not None and execution.error_kind == "guard":
        return 0.0, {"det": 0.0, "judge": None, "x": 0.0}
    if execution is None or not execution.ok:
        return weights.exec_fail, {"det": checks.det, "judge": None, "x": 0.0}
    if judgement is None:
        blend = checks.det
    else:
        blend = weights.det_w * checks.det + weights.judge_w * judgement.mean
    reward = weights.floor + (1 - weights.floor) * blend
    judge_mean = judgement.mean if judgement else None
    return reward, {"det": checks.det, "judge": judge_mean, "x": blend}


def feedback(checks: CheckReport | None, judgement: Judgement | None) -> list[str]:
    """Return refine-prompt lines: errors and warnings first, then info, then the judge's doubts."""
    lines: list[str] = []
    if checks is not None:
        by_severity = sorted(checks.findings, key=lambda finding: _SEVERITY_ORDER[finding.severity])
        lines.extend(finding.message for finding in by_severity)
    if judgement is not None:
        lines.extend(_judge_feedback(judgement))
    return lines


def _judge_feedback(judgement: Judgement) -> list[str]:
    """Return the rubric fields the judge doubted, then the tables it thinks are missing."""
    lines: list[str] = []
    for name in RUBRIC_FIELDS:
        probability = judgement.fields.get(name)
        if probability is not None and probability < DOUBT_P:
            description = RubricBase.model_fields[name].description
            lines.append(f"reviewer doubts: {description} (p={probability:.2f})")
    likely = sorted(
        (
            (table, probability)
            for table, probability in judgement.missing.items()
            if probability >= MISSING_FEEDBACK_P
        ),
        key=lambda item: -item[1],
    )
    if likely:
        shown = likely[:MISSING_TABLES_SHOWN]
        lines.append("may need table(s): " + ", ".join(f"{t} (p={p:.2f})" for t, p in shown))
    return lines
