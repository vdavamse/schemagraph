"""MCP server exposing the schema graph to any agent (Claude Code, Databao-style clients).

Tools are read-only. No SQL execution and no governance: the agent that calls
these owns SQL generation and execution against its own connection.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from schemagraph.engine import Engine
from schemagraph.model import Table

# Instructions the MCP client shows the agent.
INSTRUCTIONS = (
    "Schema context for text-to-SQL. Call link_schema first; it returns the tables, columns and "
    "join paths needed for a question as annotated DDL."
)
# ``search_tables`` lists at most this many column names per table.
MAX_SEARCH_COLUMNS = 50


def _matches_query(table: Table, lower_query: str) -> bool:
    """Whether a lowercased query is in the table's FQN, description or any column name."""
    return (
        lower_query in table.fqn.lower()
        or lower_query in (table.description or "").lower()
        or any(lower_query in column.name.lower() for column in table.columns)
    )


def _search_tables(eng: Engine, query: str, limit: int) -> list[dict[str, Any]]:
    """Up to ``limit`` tables matching ``query`` (case-insensitive substring), in FQN order."""
    lower_query = query.lower()
    out = []
    for table in eng.tables():
        if _matches_query(table, lower_query):
            out.append(
                {
                    "fqn": table.fqn,
                    "kind": table.kind,
                    "description": table.description,
                    "columns": [c.name for c in table.columns][:MAX_SEARCH_COLUMNS],
                    "source": table.source,
                }
            )
        if len(out) >= limit:
            break
    return out


def _table_detail(eng: Engine, fqn: str) -> dict[str, Any]:
    """One table as JSON plus every relation touching it, or an ``error`` entry when unknown."""
    table = eng.table(fqn)
    if not table:
        return {"error": f"unknown table {fqn}"}
    lower_fqn = table.fqn.lower()
    relations = [
        edge.model_dump()
        for edge in eng.edges()
        if lower_fqn in (edge.from_table.lower(), edge.to_table.lower())
    ]
    return {
        **json.loads(table.model_dump_json(by_alias=True)),
        "fqn": table.fqn,
        "relations": relations,
    }


def _join_path_detail(eng: Engine, from_table: str, to_table: str) -> dict[str, Any]:
    """Join paths with every relation of each hop, or an ``error`` entry for an unknown table."""
    try:
        paths = eng.join_path(from_table, to_table)
    except KeyError as e:
        return {"error": f"unknown table {e}"}
    detailed = []
    for path in paths:
        steps = []
        for a, b in zip(path, path[1:], strict=False):
            relations = eng.graph.relations(a, b)
            steps.append([relation.model_dump() for relation in relations])
        detailed.append({"tables": path, "steps": steps})
    return {"paths": detailed}


def create_server(engine: Engine | None = None) -> FastMCP:
    """Build the FastMCP server with read-only tools over ``engine`` (a new Engine when None).

    Tool names, parameters and docstrings are what agents see; they are part of the public
    surface.
    """
    eng = engine or Engine()
    mcp = FastMCP("schemagraph", instructions=INSTRUCTIONS)

    @mcp.tool()
    def link_schema(
        question: str,
        max_tables: int = 20,
        columns: str = "relevant",
        use_llm: bool = False,
    ) -> str:
        """Return the sub-schema (annotated DDL with join paths and glossary) needed to answer a natural-language question.

        Args:
            question: The user's question in natural language.
            max_tables: Upper bound on tables returned.
            columns: "relevant" (pruned) or "all".
            use_llm: Let a Claude call choose anchor tables (needs ANTHROPIC_API_KEY).
        """  # noqa: E501
        return eng.link(question, max_tables=max_tables, columns=columns, use_llm=use_llm).ddl

    @mcp.tool()
    def link_schema_json(question: str, max_tables: int = 20) -> dict[str, Any]:
        """Same as link_schema but structured: tables with scores, columns with reasons, join paths, matched glossary terms."""  # noqa: E501
        r = eng.link(question, max_tables=max_tables, render=False)
        return r.model_dump()

    @mcp.tool()
    def search_tables(query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find tables by name or description substring."""
        return _search_tables(eng, query, limit)

    @mcp.tool()
    def get_table(fqn: str) -> dict[str, Any]:
        """Full detail for one table: columns (types, descriptions, samples), primary key, and every relation with provenance."""  # noqa: E501
        return _table_detail(eng, fqn)

    @mcp.tool()
    def find_join_path(from_table: str, to_table: str) -> dict[str, Any]:
        """Shortest join path(s) between two tables over foreign keys, catalog relations, dbt relationship tests, join hints and inferred keys; when none exists, the dbt lineage route (provenance, no declared join keys)."""  # noqa: E501
        return _join_path_detail(eng, from_table, to_table)

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
    """Run the server over stdio."""
    create_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
