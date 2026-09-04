"""Unity Catalog connector (Databricks REST API 2.1, also the OSS Unity Catalog server).

Pulls catalogs -> schemas -> tables (columns, comments, PK/FK ``table_constraints``,
owner, properties) and, when ``lineage=True`` on Databricks, table-level lineage
from the lineage-tracking API (``lineage`` edges).

Endpoints used (verify against your workspace's API version):

* ``GET  /api/2.1/unity-catalog/catalogs``
* ``GET  /api/2.1/unity-catalog/schemas?catalog_name=``
* ``GET  /api/2.1/unity-catalog/tables?catalog_name=&schema_name=&max_results=&page_token=``
* ``GET  /api/2.0/lineage-tracking/table-lineage?table_name=&include_entity_lineage=false``
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from schemagraph.connectors.base import register
from schemagraph.model import Column, Edge, SchemaSnapshot, Table


class UnityConfig(BaseModel):
    host: str = Field(description="Workspace URL, e.g. https://adb-123.azuredatabricks.net or http://localhost:8080 for OSS")
    token: str | None = Field(default=None, description="Personal access token (use ${DATABRICKS_TOKEN})")
    catalogs: list[str] | None = Field(default=None, description="Restrict to these catalogs (default: all)")
    schemas: list[str] | None = Field(default=None, description="Restrict to these schemas (default: all)")
    include_views: bool = True
    lineage: bool = Field(default=False, description="Also query table lineage (Databricks only)")
    api_base: str = Field(default="/api/2.1/unity-catalog", description="API prefix")
    timeout: float = 30.0
    verify_ssl: bool = True


class UnityClient:
    def __init__(self, cfg: UnityConfig, transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {cfg.token}"} if cfg.token else {}
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.host.rstrip("/"), headers=headers, timeout=cfg.timeout, verify=cfg.verify_ssl, transport=transport)

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        r = self.http.get(path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()

    def paged(self, path: str, key: str, **params: Any):
        token = None
        while True:
            data = self.get(path, page_token=token, **params)
            yield from data.get(key) or []
            token = data.get("next_page_token")
            if not token:
                break

    def catalogs(self) -> list[dict[str, Any]]:
        return list(self.paged(f"{self.cfg.api_base}/catalogs", "catalogs"))

    def schemas(self, catalog: str) -> list[dict[str, Any]]:
        return list(self.paged(f"{self.cfg.api_base}/schemas", "schemas", catalog_name=catalog))

    def tables(self, catalog: str, schema: str) -> list[dict[str, Any]]:
        return list(self.paged(f"{self.cfg.api_base}/tables", "tables", catalog_name=catalog, schema_name=schema, max_results=50))

    def lineage(self, full_name: str) -> dict[str, Any]:
        return self.get("/api/2.0/lineage-tracking/table-lineage", table_name=full_name, include_entity_lineage="false")


def _kind(table_type: str | None) -> str:
    tt = (table_type or "").upper()
    if "VIEW" in tt:
        return "view"
    if tt == "EXTERNAL":
        return "external"
    return "table"


def introspect_unity(cfg: UnityConfig, source: str = "unity", client: UnityClient | None = None) -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="unity")
    uc = client or UnityClient(cfg)
    known: dict[str, Table] = {}
    for cat in uc.catalogs():
        cname = cat.get("name")
        if not cname or (cfg.catalogs and cname not in cfg.catalogs):
            continue
        if cname in {"system", "__databricks_internal"}:
            continue
        for sch in uc.schemas(cname):
            sname = sch.get("name")
            if not sname or sname == "information_schema" or (cfg.schemas and sname not in cfg.schemas):
                continue
            for tbl in uc.tables(cname, sname):
                kind = _kind(tbl.get("table_type"))
                if kind == "view" and not cfg.include_views:
                    continue
                t = Table(
                    name=tbl["name"],
                    schema=sname,
                    catalog=cname,
                    kind=kind,
                    description=tbl.get("comment") or None,
                    owner=tbl.get("owner"),
                    properties={k: str(v) for k, v in (tbl.get("properties") or {}).items() if not k.startswith("delta.")},
                    source=source,
                )
                for col in sorted(tbl.get("columns") or [], key=lambda c: c.get("position", 0)):
                    t.columns.append(
                        Column(
                            name=col["name"],
                            data_type=col.get("type_text") or col.get("type_name"),
                            description=col.get("comment") or None,
                            nullable=col.get("nullable"),
                        )
                    )
                for cons in tbl.get("table_constraints") or []:
                    pk = cons.get("primary_key_constraint")
                    fk = cons.get("foreign_key_constraint")
                    if pk:
                        t.primary_key = list(pk.get("child_columns") or [])
                        for c in t.columns:
                            if c.name in t.primary_key:
                                c.is_primary_key = True
                    if fk:
                        snap.edges.append(
                            Edge(
                                kind="foreign_key",
                                from_table=t.fqn,
                                to_table=fk.get("parent_table", ""),
                                from_columns=list(fk.get("child_columns") or []),
                                to_columns=list(fk.get("parent_columns") or []),
                                description=fk.get("name"),
                                source=source,
                            )
                        )
                snap.tables.append(t)
                known[t.fqn.lower()] = t
    if cfg.lineage:
        for t in list(known.values()):
            try:
                lin = uc.lineage(t.fqn)
            except httpx.HTTPError as e:
                snap.warnings.append(f"lineage failed for {t.fqn}: {e}")
                continue
            for up in lin.get("upstreams") or []:
                info = up.get("tableInfo") or {}
                up_fqn = ".".join(p for p in (info.get("catalog_name"), info.get("schema_name"), info.get("name")) if p)
                if up_fqn:
                    snap.edges.append(Edge(kind="lineage", from_table=up_fqn, to_table=t.fqn, source=source))
    # drop FK edges pointing at tables we did not see (keeps the graph honest)
    snap.edges = [e for e in snap.edges if e.kind != "foreign_key" or e.to_table.lower() in known]
    return snap.stamp()


@register
class UnityConnector:
    type_name: ClassVar[str] = "unity_catalog"
    Config = UnityConfig

    def __init__(self, name: str, config: UnityConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        uc = UnityClient(self.config)
        cats = [c.get("name") for c in uc.catalogs()]
        return f"ok: {len(cats)} catalogs ({', '.join(cats[:5])}{'...' if len(cats) > 5 else ''})"

    def introspect(self) -> SchemaSnapshot:
        return introspect_unity(self.config, self.name)
