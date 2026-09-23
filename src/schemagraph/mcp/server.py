"""MCP server exposing the schema graph to any agent (Claude Code, Databao-style clients).

Tools are read-only. No SQL execution and no governance: the agent that calls
these owns SQL generation and execution against its own connection.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from schemagraph.engine import Engine


def create_server(engine: Engine | None = None) -> FastMCP:
    eng = engine or Engine()
    mcp = FastMCP("schemagraph", instructions="Schema context for text-to-SQL. Call link_schema first; it returns the tables, columns and join paths needed for a question as annotated DDL.")

    @mcp.tool()
    def link_schema(question: str, max_tables: int = 20, columns: str = "relevant", use_llm: bool = False) -> str:
        """Return the sub-schema (annotated DDL with join paths and glossary) needed to answer a natural-language question.

        Args:
            question: The user's question in natural language.
            max_tables: Upper bound on tables returned.
            columns: "relevant" (pruned) or "all".
            use_llm: Let a Claude call choose anchor tables (needs ANTHROPIC_API_KEY).
        """
        return eng.link(question, max_tables=max_tables, columns=columns, use_llm=use_llm).ddl

    @mcp.tool()
    def link_schema_json(question: str, max_tables: int = 20) -> dict[str, Any]:
        """Same as link_schema but structured: tables with scores, columns with reasons, join paths, matched glossary terms."""
        r = eng.link(question, max_tables=max_tables, render=False)
        return r.model_dump()

    @mcp.tool()
    def search_tables(query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find tables by name or description substring."""
        q = query.lower()
        out = []
        for t in eng.tables():
            if q in t.fqn.lower() or q in (t.description or "").lower() or any(q in c.name.lower() for c in t.columns):
                out.append({"fqn": t.fqn, "kind": t.kind, "description": t.description, "columns": [c.name for c in t.columns][:50], "source": t.source})
            if len(out) >= limit:
                break
        return out

    @mcp.tool()
    def get_table(fqn: str) -> dict[str, Any]:
        """Full detail for one table: columns (types, descriptions, samples), primary key, and every relation with provenance."""
        t = eng.table(fqn)
        if not t:
            return {"error": f"unknown table {fqn}"}
        rels = [e.model_dump() for e in eng.edges() if t.fqn.lower() in (e.from_table.lower(), e.to_table.lower())]
        return {**json.loads(t.model_dump_json(by_alias=True)), "fqn": t.fqn, "relations": rels}

    @mcp.tool()
    def find_join_path(from_table: str, to_table: str) -> dict[str, Any]:
        """Shortest join path(s) between two tables over foreign keys, catalog relations, dbt relationship tests, join hints and inferred keys; when none exists, the dbt lineage route (provenance, no declared join keys)."""
        try:
            paths = eng.join_path(from_table, to_table)
        except KeyError as e:
            return {"error": f"unknown table {e}"}
        detailed = []
        for p in paths:
            steps = []
            for a, b in zip(p, p[1:], strict=False):
                rels = eng.graph.relations(a, b)
                steps.append([r.model_dump() for r in rels])
            detailed.append({"tables": p, "steps": steps})
        return {"paths": detailed}

    @mcp.tool()
    def list_glossary() -> list[dict[str, Any]]:
        """Business terms with the tables/columns they map to."""
        return [t.model_dump() for t in eng.graph.terms.values()]

    @mcp.tool()
    def graph_stats() -> dict[str, Any]:
        """Counts of tables, columns, relations, terms and sources in the loaded graph."""
        return eng.stats()

    return mcp


def main() -> None:  # pragma: no cover
    create_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
