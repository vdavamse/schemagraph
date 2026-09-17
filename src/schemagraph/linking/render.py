"""Render a LinkResult as LLM-ready annotated DDL (SignalPilot / Spider 2.0 leaders' format)."""

from __future__ import annotations

from schemagraph.graph.build import SchemaGraph
from schemagraph.model import LinkResult


def render_ddl(sg: SchemaGraph, result: LinkResult, *, samples: bool = True) -> str:
    lines: list[str] = []
    lines.append(f"-- Question: {result.question}")
    lines.append(f"-- Linked tables: {len(result.tables)} (anchors: {', '.join(result.anchors) or 'none'})")
    if result.glossary:
        lines.append("")
        lines.append("-- === Business Glossary ===")
        for term, targets in result.glossary.items():
            lines.append(f"-- {term} = {', '.join(targets)}")
    for lt in result.tables:
        t = sg.table(lt.fqn)
        lines.append("")
        head = f"CREATE TABLE {lt.fqn} ("
        lines.append(head)
        col_lines: list[str] = []
        for c in lt.columns:
            piece = f"  {c.name}"
            if c.data_type:
                piece += f" {c.data_type}"
            col_lines.append((piece, c.description))
        if t and t.primary_key:
            col_lines.append((f"  PRIMARY KEY ({', '.join(t.primary_key)})", None))
        for i, (piece, desc) in enumerate(col_lines):
            comma = "," if i < len(col_lines) - 1 else ""
            comment = f"  -- {desc}" if desc else ""
            lines.append(f"{piece}{comma}{comment}")
        tail = ");"
        notes = []
        if lt.description:
            notes.append(lt.description)
        if t and t.row_count is not None:
            notes.append(f"{t.row_count:,} rows")
        if t and t.kind not in {"table"}:
            notes.append(f"dbt {t.kind}" if t.kind in {"model", "source", "seed", "snapshot"} else t.kind)
        if t:
            up, down = sg.lineage(t.fqn)
            if up:
                notes.append(f"built from {', '.join(up[:6])}{', ...' if len(up) > 6 else ''}")
            if down:
                notes.append(f"feeds {', '.join(down[:6])}{', ...' if len(down) > 6 else ''}")
        if t and t.source:
            notes.append(f"from {t.source}")
        lines.append(tail + (f"  -- {'; '.join(notes)}" if notes else ""))
    if result.join_paths:
        lines.append("")
        lines.append("-- === Join paths (most reliable last) ===")
        for jp in sorted(result.join_paths, key=lambda p: p.reliability):
            lines.append(f"-- path [{jp.reliability:.2f}]: {' -> '.join(jp.tables)}")
            for s in jp.steps:
                lines.append(f"--   {s.kind}: {s.on}")
    if samples:
        block: list[str] = []
        for lt in result.tables:
            t = sg.table(lt.fqn)
            if not t:
                continue
            vals = [(c.name, t.column(c.name).sample_values) for c in lt.columns if t.column(c.name) and t.column(c.name).sample_values]
            if vals:
                block.append(f"-- Sample values for {lt.fqn}:")
                for name, sv in vals:
                    block.append(f"--   {name}: {', '.join(repr(v) for v in sv[:5])}")
        if block:
            lines.append("")
            lines.append("-- === Sample Values ===")
            lines.extend(block)
    return "\n".join(lines) + "\n"
