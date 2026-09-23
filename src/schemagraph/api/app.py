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
from schemagraph.model import BusinessTerm, Edge, LinkResult

log = logging.getLogger("schemagraph.api")


class ConnectionIn(BaseModel):
    name: str
    type: str
    config: dict[str, Any] = Field(default_factory=dict)
    build: bool = True
    priority: int | None = None  # merge order: lower merges first and wins conflicting fields; None keeps a stored one, else by source type
    clear_priority: bool = False  # reset a stored priority to the source-type order


class DDLIn(BaseModel):
    name: str
    ddl: str
    dialect: str | None = None
    default_schema: str | None = None
    default_catalog: str | None = None
    build: bool = True


class LinkIn(BaseModel):
    question: str
    max_tables: int = 20
    anchor_k: int = 6
    columns: str = "relevant"
    use_llm: bool = False
    prune_top_k: int = 8
    path_extra: float = 0.0


class TermIn(BaseModel):
    name: str
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)


class JoinHintIn(BaseModel):
    from_table: str
    to_table: str
    from_columns: list[str] = Field(default_factory=list)
    to_columns: list[str] = Field(default_factory=list)
    description: str | None = None


def create_app(engine: Engine | None = None, *, web_dist: str | Path | None = None) -> FastAPI:
    eng = engine or Engine()
    app = FastAPI(title="schemagraph", version="0.1.0")
    app.state.engine = eng

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **eng.stats()}

    # ---------------------------------------------------------- connections
    @app.get("/api/connector-types")
    def connector_types() -> list[dict[str, Any]]:
        return [{"type": t, "schema": eng.connector_schema(t)} for t in eng.connector_types()]

    @app.get("/api/connections")
    def list_connections() -> list[dict[str, Any]]:
        return eng.connections()

    @app.post("/api/connections")
    def add_connection(body: ConnectionIn) -> dict[str, Any]:
        try:
            snap = eng.add_connection(body.name, body.type, body.config, build=body.build, priority=body.priority, clear_priority=body.clear_priority)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # connector/config failure
            log.exception("add_connection failed")
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        return {"name": body.name, "built": snap is not None, "tables": len(snap.tables) if snap else 0, "edges": len(snap.edges) if snap else 0, "warnings": snap.warnings if snap else []}

    @app.post("/api/ddl")
    def add_ddl(body: DDLIn) -> dict[str, Any]:
        cfg = {"ddl": body.ddl, "dialect": body.dialect, "default_schema": body.default_schema, "default_catalog": body.default_catalog}
        try:
            snap = eng.add_connection(body.name, "ddl", cfg, build=body.build)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        return {"name": body.name, "tables": len(snap.tables) if snap else 0, "edges": len(snap.edges) if snap else 0, "warnings": snap.warnings if snap else []}

    @app.delete("/api/connections/{name}")
    def delete_connection(name: str) -> dict[str, str]:
        eng.remove_connection(name)
        return {"deleted": name}

    @app.post("/api/connections/{name}/check")
    def check_connection(name: str) -> dict[str, str]:
        try:
            return {"status": eng.check_connection(name)}
        except KeyError as e:
            raise HTTPException(404, f"unknown connection {name}") from e
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e

    @app.post("/api/connections/{name}/build")
    def build_connection(name: str) -> dict[str, Any]:
        try:
            snap = eng.build(name)
        except KeyError as e:
            raise HTTPException(404, f"unknown connection {name}") from e
        except Exception as e:
            log.exception("build failed")
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        return {"name": name, "tables": len(snap.tables), "edges": len(snap.edges), "terms": len(snap.terms), "warnings": snap.warnings}

    @app.post("/api/build")
    def build_all() -> dict[str, Any]:
        snaps = eng.build()
        return {"built": [s.source for s in snaps], **eng.stats()}

    # ---------------------------------------------------------- graph
    @app.get("/api/graph/stats")
    def graph_stats() -> dict[str, Any]:
        return eng.stats()

    @app.get("/api/graph/tables")
    def graph_tables(q: str | None = None) -> list[dict[str, Any]]:
        out = []
        for t in eng.tables():
            if q and q.lower() not in t.fqn.lower() and q.lower() not in (t.description or "").lower():
                continue
            out.append({"fqn": t.fqn, "kind": t.kind, "columns": len(t.columns), "description": t.description, "source": t.source, "row_count": t.row_count, "tags": t.tags})
        return out

    @app.get("/api/graph/tables/{fqn}")
    def graph_table(fqn: str) -> dict[str, Any]:
        t = eng.table(fqn)
        if not t:
            raise HTTPException(404, f"unknown table {fqn}")
        rels = [e.model_dump() for e in eng.edges() if e.from_table.lower() == t.fqn.lower() or e.to_table.lower() == t.fqn.lower()]
        return {**t.model_dump(by_alias=True), "fqn": t.fqn, "relations": rels}

    @app.get("/api/graph/edges")
    def graph_edges() -> list[dict[str, Any]]:
        return [e.model_dump() for e in eng.edges()]

    @app.get("/api/graph/path")
    def graph_path(a: str, b: str) -> dict[str, Any]:
        try:
            return {"paths": eng.join_path(a, b)}
        except KeyError as e:
            raise HTTPException(404, f"unknown table {e}") from e

    @app.get("/api/graph/export")
    def graph_export() -> dict[str, Any]:
        """Table-level graph for visualisation: nodes + relation edges."""
        nodes = [{"id": t.fqn, "kind": t.kind, "columns": len(t.columns), "source": t.source} for t in eng.tables()]
        edges = [{"source": e.from_table, "target": e.to_table, "kind": e.kind, "on": ", ".join(f"{a}={b}" for a, b in zip(e.from_columns, e.to_columns, strict=False)), "provenance": e.source} for e in eng.edges()]
        return {"nodes": nodes, "edges": edges}

    # ---------------------------------------------------------- linking
    @app.post("/api/link", response_model=LinkResult)
    def link(body: LinkIn) -> LinkResult:
        return eng.link(body.question, max_tables=body.max_tables, anchor_k=body.anchor_k, columns=body.columns, use_llm=body.use_llm, prune_top_k=body.prune_top_k, path_extra=body.path_extra)

    @app.get("/api/explain")
    def explain(question: str) -> dict[str, Any]:
        return eng.explain(question)

    # ---------------------------------------------------------- glossary / hints
    @app.get("/api/glossary")
    def glossary() -> list[dict[str, Any]]:
        return [t.model_dump() for t in eng.graph.terms.values()]

    @app.post("/api/glossary")
    def upsert_term(body: TermIn) -> dict[str, str]:
        eng.upsert_term(BusinessTerm(name=body.name, description=body.description, synonyms=body.synonyms, targets=body.targets, source="user"))
        return {"upserted": body.name}

    @app.delete("/api/glossary/{name}")
    def delete_term(name: str) -> dict[str, str]:
        eng.delete_term(name)
        return {"deleted": name}

    @app.get("/api/join-hints")
    def join_hints() -> list[dict[str, Any]]:
        return [{"id": i, **e.model_dump()} for i, e in eng.store.join_hints()]

    @app.post("/api/join-hints")
    def add_join_hint(body: JoinHintIn) -> dict[str, int]:
        hid = eng.add_join_hint(Edge(kind="join_hint", from_table=body.from_table, to_table=body.to_table, from_columns=body.from_columns, to_columns=body.to_columns, description=body.description, source="user"))
        return {"id": hid}

    @app.delete("/api/join-hints/{hid}")
    def delete_join_hint(hid: int) -> dict[str, int]:
        eng.delete_join_hint(hid)
        return {"deleted": hid}

    # ---------------------------------------------------------- frontend
    dist = Path(web_dist) if web_dist else Path(os.environ.get("SCHEMAGRAPH_WEB_DIST", Path(__file__).resolve().parents[3] / "web" / "dist"))
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = dist / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
