"""Instructions and prompt material for the four agents (pure strings, no model imports).

The judge and selector run on Jev, which reads the prompt as the material being judged and takes
its questions from the output type; so their material carries no instructions and no DDL (Jev
degrades on irrelevant context), only the question, the SQL and a short result preview.
"""

from __future__ import annotations

from schemagraph.agent.results import Candidate, ExecResult

GENERATOR_INSTRUCTIONS = (
    "You write exactly one read-only SQL query in the {dialect} dialect that answers the user's "
    "question. "
    "Use only tables and columns that appear in the schema you are given or that you found with "
    "the tools. "
    "If the schema lacks a concept the question needs, call link_schema with that concept, or "
    "search_tables. "
    "Read a table's columns and relations with get_table, the join keys between two tables with "
    "find_join_path, and business terms with list_glossary. "
    "Prefer the join columns listed in the schema. "
    "Check literal values with sample_values before filtering on them. "
    "You may probe with run_query (at most {probes} times, small results only). "
    "Return the final query through the output tool: no markdown, no comments, no trailing "
    "semicolon. {notes}"
)

DIALECT_NOTES = {
    "sqlite": (
        "SQLite: use strftime()/date() for dates, || to concatenate, CAST(x AS REAL) before "
        "dividing integers; there is no ILIKE, no FULL JOIN before 3.39, no date_trunc."
    ),
    "duckdb": (
        "DuckDB: date_trunc(), strftime(), ILIKE and QUALIFY are available; integer division "
        "with / returns a double."
    ),
}

JUDGE_INSTRUCTIONS = "Judge only against the question."

CRITIC_INSTRUCTIONS = (
    "You review a SQL attempt that did not fully answer a question. "
    "Give 2 to 5 short, concrete fixes as bullets. Do not write the full query."
)

# Characters per cell in a result preview (generator, critic, judge and selector).
PREVIEW_CELL_CHARS = 60
# Result rows shown with a previous attempt, to the generator and to the critic.
ATTEMPT_PREVIEW_ROWS = 5
# SQL characters shown to the judge, and per candidate to the selector.
JUDGE_SQL_CHARS = 6000
PICK_SQL_CHARS = 4000
# Result rows shown per candidate to the selector.
PICK_PREVIEW_ROWS = 6
# Schema DDL characters shown to the critic.
CRITIC_SCHEMA_CHARS = 6000


def preview(result: ExecResult | None, rows: int, *, cell_chars: int = PREVIEW_CELL_CHARS) -> str:
    """Summarise an execution result: row count, columns and the first ``rows`` rows.

    Args:
        result: The execution result, or None when the query was not executed.
        rows: Rows to show; 0 shows only the count and columns.
        cell_chars: Characters kept per cell.

    Returns:
        One header line, followed by a ``|``-separated table when rows are shown.
    """
    if result is None:
        return "not executed"
    if not result.ok:
        return f"error ({result.error_kind}): {result.error}"
    count = f"{result.row_count:,}{'+' if result.row_count_capped else ''}"
    head = f"{count} rows, columns: {', '.join(result.columns)}"
    if not result.rows or rows <= 0:
        return head
    lines = [" | ".join(result.columns)]
    for row in result.rows[:rows]:
        cells = ("NULL" if value is None else str(value)[:cell_chars] for value in row)
        lines.append(" | ".join(cells))
    shown = min(rows, len(result.rows))
    return f"{head} (showing {shown})\n" + "\n".join(lines)


def generator_instructions(dialect: str, probes: int) -> str:
    """Fill the generator's instructions for ``dialect`` and a ``run_query`` probe budget."""
    notes = DIALECT_NOTES.get(dialect, "")
    return GENERATOR_INSTRUCTIONS.format(dialect=dialect, probes=probes, notes=notes)


def _problems(feedback: list[str]) -> str:
    """Render feedback lines as a bulleted "Problems found" section."""
    return "Problems found:\n" + "\n".join(f"- {line}" for line in feedback)


def _refine_sections(parent: Candidate) -> list[str]:
    """Show the generator a previous attempt and what was wrong with it."""
    sections = [
        f"Previous attempt (score {parent.score:.2f}):\n"
        f"SQL:\n{parent.sql or '(none)'}\n"
        f"Result: {preview(parent.exec, ATTEMPT_PREVIEW_ROWS)}"
    ]
    if parent.error:
        sections.append(f"The attempt failed: {parent.error}")
    if parent.feedback:
        sections.append(_problems(parent.feedback))
    if parent.advice:
        sections.append(f"Reviewer advice:\n{parent.advice}")
    sections.append("Write an improved query.")
    return sections


def generator_prompt(
    question: str,
    evidence: str | None,
    action: str,
    ddl: str,
    n_tables: int,
    parent: Candidate | None = None,
    *,
    evidence_chars: int,
) -> str:
    """Build the generator's user prompt for a fresh draft, or a refinement of ``parent``.

    Args:
        question: The user's question.
        evidence: External knowledge for the question, if any.
        action: The schema context's name (``tight`` or ``wide``).
        ddl: The linked schema's DDL.
        n_tables: Tables in ``ddl``.
        parent: The attempt to refine; None for a fresh draft.
        evidence_chars: Characters of ``evidence`` kept (``AgentConfig.evidence_chars``).

    Returns:
        The prompt's sections, separated by blank lines.
    """
    sections = [f"Question: {question}"]
    if evidence:
        sections.append(f"External knowledge:\n{evidence[:evidence_chars]}")
    sections.append(f"Schema ({action} context, {n_tables} tables):\n{ddl.strip()}")
    if parent is not None:
        sections.extend(_refine_sections(parent))
    return "\n\n".join(sections)


def judge_material(
    question: str,
    sql: str,
    result: ExecResult | None,
    *,
    rows: int,
    evidence: str | None = None,
    evidence_chars: int,
) -> str:
    """Build what the judge reads: the question, optional notes, the SQL and a result preview.

    Args:
        question: The user's question.
        sql: The candidate query.
        result: Its execution result.
        rows: Result rows shown (``AgentConfig.preview_rows``).
        evidence: External knowledge for the question, if any.
        evidence_chars: Characters of ``evidence`` kept; 0 leaves it out
            (``AgentConfig.judge_evidence_chars``).

    Returns:
        The material's sections, separated by blank lines.
    """
    sections = [f"Question: {question}"]
    if evidence and evidence_chars > 0:
        sections.append(f"Notes: {evidence[:evidence_chars]}")
    sections.append(f"SQL:\n{sql[:JUDGE_SQL_CHARS]}")
    sections.append(f"Result: {preview(result, rows)}")
    return "\n\n".join(sections)


def _pick_section(label: str, candidate: Candidate, rows: int) -> str:
    """Show one candidate of a pairwise comparison."""
    return (
        f"Candidate {label} SQL:\n{candidate.sql[:PICK_SQL_CHARS]}\n"
        f"Candidate {label} result: {preview(candidate.exec, rows)}"
    )


def pick_material(question: str, a: Candidate, b: Candidate) -> str:
    """Build what the selector reads to compare candidates ``a`` and ``b``."""
    sections = [
        f"Question: {question}",
        _pick_section("A", a, PICK_PREVIEW_ROWS),
        _pick_section("B", b, PICK_PREVIEW_ROWS),
    ]
    return "\n\n".join(sections)


def critic_prompt(question: str, candidate: Candidate, schema: str) -> str:
    """Build the critic's prompt: the question, the schema and the attempt with its problems."""
    sections = [
        f"Question: {question}",
        f"Schema:\n{schema[:CRITIC_SCHEMA_CHARS]}",
        f"SQL:\n{candidate.sql or '(none)'}",
        f"Result: {preview(candidate.exec, ATTEMPT_PREVIEW_ROWS)}",
    ]
    if candidate.error:
        sections.append(f"Generation error: {candidate.error}")
    if candidate.feedback:
        sections.append(_problems(candidate.feedback))
    return "\n\n".join(sections)
