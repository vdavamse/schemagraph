"""Render a LinkResult as LLM-ready annotated DDL (SignalPilot / Spider 2.0 leaders' format)."""

from __future__ import annotations

from schemagraph.graph.build import SchemaGraph
from schemagraph.model import JoinPath, LinkedTable, LinkResult, Table

# Upstream / downstream lineage neighbours named in a table's trailing comment.
MAX_LINEAGE_LISTED = 6
# Sample values listed per column.
MAX_SAMPLES_RENDERED = 5
# Table kinds that come from dbt and are rendered as "dbt <kind>".
_DBT_KINDS = frozenset({"model", "source", "seed", "snapshot"})
# Starts the header line with the linked-table count; the benchmark splits the DDL on it.
LINKED_TABLES_MARKER = "-- Linked tables:"


def _header_lines(result: LinkResult) -> list[str]:
    """The question, the linked-table count with anchors, and the business glossary."""
    anchors = ", ".join(result.anchors) or "none"
    lines = [
        f"-- Question: {result.question}",
        f"{LINKED_TABLES_MARKER} {len(result.tables)} (anchors: {anchors})",
    ]
    if result.glossary:
        lines.append("")
        lines.append("-- === Business Glossary ===")
        for term, targets in result.glossary.items():
            lines.append(f"-- {term} = {', '.join(targets)}")
    return lines


def _column_entries(linked: LinkedTable, table: Table | None) -> list[tuple[str, str | None]]:
    """Each column definition (and the primary key clause) with its comment, if any."""
    entries: list[tuple[str, str | None]] = []
    for column in linked.columns:
        piece = f"  {column.name}"
        if column.data_type:
            piece += f" {column.data_type}"
        entries.append((piece, column.description))
    if table and table.primary_key:
        entries.append((f"  PRIMARY KEY ({', '.join(table.primary_key)})", None))
    return entries


def _truncated(names: list[str]) -> str:
    """The first :data:`MAX_LINEAGE_LISTED` names, comma-separated, with ``, ...`` if cut."""
    more = ", ..." if len(names) > MAX_LINEAGE_LISTED else ""
    return f"{', '.join(names[:MAX_LINEAGE_LISTED])}{more}"


def _table_notes(schema_graph: SchemaGraph, linked: LinkedTable, table: Table | None) -> list[str]:
    """Description, row count, kind, lineage and sources for the table's trailing comment."""
    notes = []
    if linked.description:
        notes.append(linked.description)
    if table and table.row_count is not None:
        notes.append(f"{table.row_count:,} rows")
    if table and table.kind != "table":
        notes.append(f"dbt {table.kind}" if table.kind in _DBT_KINDS else table.kind)
    if table:
        upstream, downstream = schema_graph.lineage(table.fqn)
        if upstream:
            notes.append(f"built from {_truncated(upstream)}")
        if downstream:
            notes.append(f"feeds {_truncated(downstream)}")
    if table and table.source:
        notes.append(f"from {table.source}")
    return notes


def _table_block(schema_graph: SchemaGraph, linked: LinkedTable) -> list[str]:
    """A blank line, then the ``CREATE TABLE`` statement with its column and table comments."""
    table = schema_graph.table(linked.fqn)
    lines = ["", f"CREATE TABLE {linked.fqn} ("]
    entries = _column_entries(linked, table)
    for i, (piece, description) in enumerate(entries):
        comma = "," if i < len(entries) - 1 else ""
        comment = f"  -- {description}" if description else ""
        lines.append(f"{piece}{comma}{comment}")
    notes = _table_notes(schema_graph, linked, table)
    lines.append(");" + (f"  -- {'; '.join(notes)}" if notes else ""))
    return lines


def _join_path_lines(join_paths: list[JoinPath]) -> list[str]:
    """The join paths, least reliable first, each with its steps."""
    if not join_paths:
        return []
    # PathRAG serializes ascending by reliability: most reliable last, closest to the question
    lines = ["", "-- === Join paths (most reliable last) ==="]
    for join_path in sorted(join_paths, key=lambda p: p.reliability):
        lines.append(f"-- path [{join_path.reliability:.2f}]: {' -> '.join(join_path.tables)}")
        for step in join_path.steps:
            lines.append(f"--   {step.kind}: {step.on}")
    return lines


def _sample_value_lines(schema_graph: SchemaGraph, linked_tables: list[LinkedTable]) -> list[str]:
    """Sample values of every linked column that has some, grouped by table."""
    block: list[str] = []
    for linked in linked_tables:
        table = schema_graph.table(linked.fqn)
        if not table:
            continue
        values: list[tuple[str, list[str]]] = []
        for linked_column in linked.columns:
            column = table.column(linked_column.name)
            if column and column.sample_values:
                values.append((linked_column.name, column.sample_values))
        if values:
            block.append(f"-- Sample values for {linked.fqn}:")
            for name, samples in values:
                rendered = ", ".join(repr(v) for v in samples[:MAX_SAMPLES_RENDERED])
                block.append(f"--   {name}: {rendered}")
    if not block:
        return []
    return ["", "-- === Sample Values ===", *block]


def render_ddl(schema_graph: SchemaGraph, result: LinkResult, *, samples: bool = True) -> str:
    """Render a link result as annotated DDL for an LLM prompt.

    Header and glossary, one ``CREATE TABLE`` per linked table (column descriptions, row
    count, kind, lineage and sources as comments), the join paths, then sample values.

    Args:
        schema_graph: The graph the result was linked against (for keys, lineage, samples).
        result: The link result to render.
        samples: Append the sample-value section.

    Returns:
        The DDL text, newline-terminated.
    """
    lines = _header_lines(result)
    for linked in result.tables:
        lines.extend(_table_block(schema_graph, linked))
    lines.extend(_join_path_lines(result.join_paths))
    if samples:
        lines.extend(_sample_value_lines(schema_graph, result.tables))
    return "\n".join(lines) + "\n"
