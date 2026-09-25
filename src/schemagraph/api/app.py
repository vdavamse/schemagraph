"""FastAPI backend: connections, builds, graph browsing, linking, glossary, join hints.

Serves the built frontend from ``web/dist`` at ``/`` when present.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from schemagraph.engine import Engine
from schemagraph.model import BusinessTerm, Edge, LinkResult, Table

log = logging.getLogger("schemagraph.api")


class ConnectionIn(BaseModel):
    """A connection to register: connector type, config and optional merge priority."""

    name: str
    type: str
    config: dict[str, Any] = Field(default_factory=dict)
    build: bool = True
    # merge order: lower merges first and wins conflicting fields;
    # None keeps a stored one, else by source type
    priority: int | None = None
    clear_priority: bool = False  # reset a stored priority to the source-type order


class DDLIn(BaseModel):
    """Pasted DDL to register as a ``ddl`` connection."""

    name: str
    ddl: str
    dialect: str | None = None
    default_schema: str | None = None
    default_catalog: str | None = None
    build: bool = True


class LinkIn(BaseModel):
    """A question to link, with the linker options the UI exposes."""

    question: str
    max_tables: int = 20
    anchor_k: int = 6
    columns: str = "relevant"
    use_llm: bool = False
    prune_top_k: int = 8
    path_extra: float = 0.0


class TermIn(BaseModel):
    """A user glossary term: name, description, synonyms and the objects it refers to."""

    name: str
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)


class JoinHintIn(BaseModel):
    """A user join hint between two tables, optionally on named columns."""

    from_table: str
    to_table: str
    from_columns: list[str] = Field(default_factory=list)
    to_columns: list[str] = Field(default_factory=list)
    description: str | None = None


def create_app(engine: Engine | None = None, *, web_dist: str | Path | None = None) -> FastAPI:
    """Build the FastAPI app over ``engine`` (a new default Engine when None).

    Args:
        engine: The engine every route reads and writes.
        web_dist: Built frontend directory; default ``$SCHEMAGRAPH_WEB_DIST`` or the repo's
            ``web/dist``. The UI is mounted only when it exists.

    Returns:
        The app, with the engine at ``app.state.engine``.
    """
    engine = engine or Engine()
    app = FastAPI(title="schemagraph", version="0.1.0")
    app.state.engine = engine
    _add_health_route(app, engine)
    _add_connection_routes(app, engine)
    _add_graph_routes(app, engine)
    _add_linking_routes(app, engine)
    _add_glossary_routes(app, engine)
    _mount_frontend(app, _resolve_web_dist(web_dist))
    return app


def _add_health_route(app: FastAPI, engine: Engine) -> None:
    """Register ``GET /api/health``: status plus the engine's stats."""

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **engine.stats()}


def _register_connection(engine: Engine, body: ConnectionIn) -> dict[str, Any]:
    """Register (and optionally build) a connection; failures are HTTP 400."""
    try:
        snap = engine.add_connection(
            body.name,
            body.type,
            body.config,
            build=body.build,
            priority=body.priority,
            clear_priority=body.clear_priority,
        )
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # connector/config failure
        log.exception("add_connection failed")
        raise HTTPException(400, f"{type(e).__name__}: {e}") from e
    return {
        "name": body.name,
        "built": snap is not None,
        "tables": len(snap.tables) if snap else 0,
        "edges": len(snap.edges) if snap else 0,
        "warnings": snap.warnings if snap else [],
    }


def _check_connection(engine: Engine, name: str) -> dict[str, str]:
    """Run a connection's check; unknown is HTTP 404, a failing check HTTP 400."""
    try:
        return {"status": engine.check_connection(name)}
    except KeyError as e:
        raise HTTPException(404, f"unknown connection {name}") from e
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}") from e


def _build_connection(engine: Engine, name: str) -> dict[str, Any]:
    """Build one connection and return its counts; unknown is HTTP 404, a failure HTTP 400."""
    try:
        snap = engine.build(name)
    except KeyError as e:
        raise HTTPException(404, f"unknown connection {name}") from e
    except Exception as e:
        log.exception("build failed")
        raise HTTPException(400, f"{type(e).__name__}: {e}") from e
    return {
        "name": name,
        "tables": len(snap.tables),
        "edges": len(snap.edges),
        "terms": len(snap.terms),
        "warnings": snap.warnings,
    }


def _add_connection_routes(app: FastAPI, engine: Engine) -> None:
    """Register connector types, connection CRUD, checks and builds.

    Connector and config failures become HTTP 400; unknown connections 404 where looked up.
    """

    @app.get("/api/connector-types")
    def connector_types() -> list[dict[str, Any]]:
        return [{"type": t, "schema": engine.connector_schema(t)} for t in engine.connector_types()]

    @app.get("/api/connections")
    def list_connections() -> list[dict[str, Any]]:
        return engine.connections()

    @app.post("/api/connections")
    def add_connection(body: ConnectionIn) -> dict[str, Any]:
        return _register_connection(engine, body)

    @app.post("/api/ddl")
    def add_ddl(body: DDLIn) -> dict[str, Any]:
        cfg = {
            "ddl": body.ddl,
            "dialect": body.dialect,
            "default_schema": body.default_schema,
            "default_catalog": body.default_catalog,
        }
        try:
            snap = engine.add_connection(body.name, "ddl", cfg, build=body.build)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        return {
            "name": body.name,
            "tables": len(snap.tables) if snap else 0,
            "edges": len(snap.edges) if snap else 0,
            "warnings": snap.warnings if snap else [],
        }

    @app.delete("/api/connections/{name}")
    def delete_connection(name: str) -> dict[str, str]:
        engine.remove_connection(name)
        return {"deleted": name}

    @app.post("/api/connections/{name}/check")
    def check_connection(name: str) -> dict[str, str]:
        return _check_connection(engine, name)

    @app.post("/api/connections/{name}/build")
    def build_connection(name: str) -> dict[str, Any]:
        return _build_connection(engine, name)

    @app.post("/api/build")
    def build_all() -> dict[str, Any]:
        snaps = engine.build()
        return {"built": [s.source for s in snaps], **engine.stats()}


def _table_matches(table: Table, query: str | None) -> bool:
    """Whether ``query`` (case-insensitive) is empty or in the table's FQN or description."""
    if not query:
        return True
    lower_query = query.lower()
    return lower_query in table.fqn.lower() or lower_query in (table.description or "").lower()


def _table_summary(table: Table) -> dict[str, Any]:
    """One row of the table browser."""
    return {
        "fqn": table.fqn,
        "kind": table.kind,
        "columns": len(table.columns),
        "description": table.description,
        "source": table.source,
        "row_count": table.row_count,
        "tags": table.tags,
    }


def _export_edge(edge: Edge) -> dict[str, Any]:
    """One relation edge of the visualisation export, with its ``a=b`` join condition."""
    on = ", ".join(
        f"{a}={b}" for a, b in zip(edge.from_columns, edge.to_columns, strict=False)
    )
    return {
        "source": edge.from_table,
        "target": edge.to_table,
        "kind": edge.kind,
        "on": on,
        "provenance": edge.source,
    }


def _add_graph_routes(app: FastAPI, engine: Engine) -> None:
    """Register graph browsing: stats, tables, one table with its relations, edges, paths, export.

    Unknown tables are HTTP 404.
    """

    @app.get("/api/graph/stats")
    def graph_stats() -> dict[str, Any]:
        return engine.stats()

    @app.get("/api/graph/tables")
    def graph_tables(q: str | None = None) -> list[dict[str, Any]]:
        return [_table_summary(table) for table in engine.tables() if _table_matches(table, q)]

    @app.get("/api/graph/tables/{fqn}")
    def graph_table(fqn: str) -> dict[str, Any]:
        table = engine.table(fqn)
        if not table:
            raise HTTPException(404, f"unknown table {fqn}")
        lower_fqn = table.fqn.lower()
        relations = [
            edge.model_dump()
            for edge in engine.edges()
            if edge.from_table.lower() == lower_fqn or edge.to_table.lower() == lower_fqn
        ]
        return {**table.model_dump(by_alias=True), "fqn": table.fqn, "relations": relations}

    @app.get("/api/graph/edges")
    def graph_edges() -> list[dict[str, Any]]:
        return [e.model_dump() for e in engine.edges()]

    @app.get("/api/graph/path")
    def graph_path(a: str, b: str) -> dict[str, Any]:
        try:
            return {"paths": engine.join_path(a, b)}
        except KeyError as e:
            raise HTTPException(404, f"unknown table {e}") from e

    @app.get("/api/graph/export")
    def graph_export() -> dict[str, Any]:
        """Table-level graph for visualisation: nodes + relation edges."""
        nodes = [
            {
                "id": table.fqn,
                "kind": table.kind,
                "columns": len(table.columns),
                "source": table.source,
            }
            for table in engine.tables()
        ]
        edges = [_export_edge(edge) for edge in engine.edges()]
        return {"nodes": nodes, "edges": edges}


def _add_linking_routes(app: FastAPI, engine: Engine) -> None:
    """Register ``POST /api/link`` (a :class:`LinkResult`) and ``GET /api/explain``."""

    @app.post("/api/link", response_model=LinkResult)
    def link(body: LinkIn) -> LinkResult:
        return engine.link(
            body.question,
            max_tables=body.max_tables,
            anchor_k=body.anchor_k,
            columns=body.columns,
            use_llm=body.use_llm,
            prune_top_k=body.prune_top_k,
            path_extra=body.path_extra,
        )

    @app.get("/api/explain")
    def explain(question: str) -> dict[str, Any]:
        return engine.explain(question)


def _add_glossary_routes(app: FastAPI, engine: Engine) -> None:
    """Register the glossary (merged terms; user terms CRUD) and user join-hint routes."""

    @app.get("/api/glossary")
    def glossary() -> list[dict[str, Any]]:
        return [t.model_dump() for t in engine.graph.terms.values()]

    @app.post("/api/glossary")
    def upsert_term(body: TermIn) -> dict[str, str]:
        engine.upsert_term(
            BusinessTerm(
                name=body.name,
                description=body.description,
                synonyms=body.synonyms,
                targets=body.targets,
                source="user",
            )
        )
        return {"upserted": body.name}

    @app.delete("/api/glossary/{name}")
    def delete_term(name: str) -> dict[str, str]:
        engine.delete_term(name)
        return {"deleted": name}

    @app.get("/api/join-hints")
    def join_hints() -> list[dict[str, Any]]:
        return [{"id": i, **e.model_dump()} for i, e in engine.store.join_hints()]

    @app.post("/api/join-hints")
    def add_join_hint(body: JoinHintIn) -> dict[str, int]:
        hint_id = engine.add_join_hint(
            Edge(
                kind="join_hint",
                from_table=body.from_table,
                to_table=body.to_table,
                from_columns=body.from_columns,
                to_columns=body.to_columns,
                description=body.description,
                source="user",
            )
        )
        return {"id": hint_id}

    @app.delete("/api/join-hints/{hid}")
    def delete_join_hint(hid: int) -> dict[str, int]:
        engine.delete_join_hint(hid)
        return {"deleted": hid}


def _resolve_web_dist(web_dist: str | Path | None) -> Path:
    """Frontend build directory: ``web_dist``, else ``$SCHEMAGRAPH_WEB_DIST``, else ``web/dist``."""
    if web_dist:
        return Path(web_dist)
    default = Path(__file__).resolve().parents[3] / "web" / "dist"
    return Path(os.environ.get("SCHEMAGRAPH_WEB_DIST", default))


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built UI from ``dist`` when it exists; register last (its catch-all is greedy).

    ``/assets`` is mounted statically; any other path serves the file if it exists, else
    ``index.html`` (SPA routing). The catch-all is left out of the OpenAPI schema.
    """
    if not dist.exists():
        return
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = dist / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")
