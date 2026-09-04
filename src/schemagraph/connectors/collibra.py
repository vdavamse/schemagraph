"""Collibra connector (REST API 2.0).

Collibra is the *metadata plane*: it rarely holds the physical schema better than the
engine catalogs do, but it holds what they lack - curated relations, business terms,
descriptions, classifications and ownership. This connector extracts:

* **Tables / Columns** from the Data Dictionary asset types (``Table``, ``Column``),
  hierarchy via ``contains`` relations (Schema > Table > Column).
* **Relations** between columns (foreign-key style, e.g. role ``references``) and
  between tables -> ``catalog_relation`` edges (``foreign_key`` when column-level).
* **Business Terms** related to tables/columns -> glossary ``BusinessTerm`` entries.
* **Descriptions** and tag-like attributes (e.g. data classification) as descriptions / tags.

Asset-type names, relation roles and attribute types differ per operating model, so
all of them are configurable. The defaults follow the out-of-the-box Collibra
Data Dictionary + Business Glossary operating model; verify them against
``GET /rest/2.0/assetTypes`` and ``GET /rest/2.0/relationTypes`` on your instance.

Endpoints used:

* ``GET /rest/2.0/assetTypes?name=&nameMatchMode=EXACT``
* ``GET /rest/2.0/assets?typeIds=&limit=&offset=``
* ``GET /rest/2.0/attributes?assetId=`` (or ``typeIds``)
* ``GET /rest/2.0/relations?sourceId=|targetId=&limit=&offset=``
* ``GET /rest/2.0/relationTypes``
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from schemagraph.connectors.base import register
from schemagraph.model import BusinessTerm, Column, Edge, SchemaSnapshot, Table


class CollibraConfig(BaseModel):
    host: str = Field(description="Collibra base URL, e.g. https://acme.collibra.com")
    username: str | None = Field(default=None, description="Basic-auth user (or use token)")
    password: str | None = Field(default=None, description="Basic-auth password (use ${COLLIBRA_PASSWORD})")
    token: str | None = Field(default=None, description="Bearer/JWT token, alternative to username/password")
    domains: list[str] | None = Field(default=None, description="Restrict to these domain names (default: all)")
    # operating-model names
    table_type: str = "Table"
    column_type: str = "Column"
    schema_type: str = "Schema"
    database_type: str = "Database"
    term_type: str = "Business Term"
    contains_role: str = Field(default="contains", description="Relation role for Schema>Table and Table>Column hierarchy")
    fk_roles: list[str] = Field(default_factory=lambda: ["references", "is reference to", "is foreign key of"], description="Column->Column relation roles that mean a foreign key")
    table_relation_roles: list[str] = Field(default_factory=lambda: ["is related to", "relates to"], description="Table->Table relation roles to import as catalog relations")
    term_roles: list[str] = Field(default_factory=lambda: ["represents", "is represented by", "is described by", "describes"], description="Business Term <-> Table/Column relation roles")
    description_attribute: str = "Description"
    tag_attributes: list[str] = Field(default_factory=lambda: ["Data Classification", "Personal Identifiable Information", "Security Classification"], description="Attribute types copied into tags as name=value")
    synonym_relation_roles: list[str] = Field(default_factory=lambda: ["is synonym of", "has synonym"])
    page_size: int = 500
    timeout: float = 60.0
    verify_ssl: bool = True


class CollibraClient:
    def __init__(self, cfg: CollibraConfig, transport: httpx.BaseTransport | None = None):
        headers = {"Accept": "application/json"}
        if cfg.token:
            headers["Authorization"] = f"Bearer {cfg.token}"
        elif cfg.username and cfg.password:
            headers["Authorization"] = "Basic " + base64.b64encode(f"{cfg.username}:{cfg.password}".encode()).decode()
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.host.rstrip("/") + "/rest/2.0", headers=headers, timeout=cfg.timeout, verify=cfg.verify_ssl, transport=transport)

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        r = self.http.get(path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()

    def paged(self, path: str, **params: Any):
        offset = 0
        while True:
            data = self.get(path, limit=self.cfg.page_size, offset=offset, **params)
            results = data.get("results") or []
            yield from results
            if len(results) < self.cfg.page_size:
                break
            offset += self.cfg.page_size

    def asset_type_id(self, name: str) -> str | None:
        data = self.get("/assetTypes", name=name, nameMatchMode="EXACT")
        res = data.get("results") or []
        return res[0]["id"] if res else None

    def assets_of_type(self, type_id: str) -> list[dict[str, Any]]:
        return list(self.paged("/assets", typeIds=type_id))

    def attributes(self, asset_id: str) -> list[dict[str, Any]]:
        return list(self.paged("/attributes", assetId=asset_id))

    def relations_from(self, asset_id: str) -> list[dict[str, Any]]:
        return list(self.paged("/relations", sourceId=asset_id))

    def relations_to(self, asset_id: str) -> list[dict[str, Any]]:
        return list(self.paged("/relations", targetId=asset_id))

    def relation_types(self) -> list[dict[str, Any]]:
        return list(self.paged("/relationTypes"))


def _short_name(asset: dict[str, Any]) -> str:
    name = asset.get("displayName") or asset.get("name") or ""
    for sep in (">", "."):
        if sep in name:
            name = name.split(sep)[-1]
    return name.strip()


def introspect_collibra(cfg: CollibraConfig, source: str = "collibra", client: CollibraClient | None = None) -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="collibra")
    cc = client or CollibraClient(cfg)

    type_ids = {k: cc.asset_type_id(v) for k, v in {"table": cfg.table_type, "column": cfg.column_type, "schema": cfg.schema_type, "term": cfg.term_type}.items()}
    missing = [k for k, v in type_ids.items() if not v]
    if "table" in missing:
        snap.warnings.append(f"asset type {cfg.table_type!r} not found; nothing imported")
        return snap
    for k in missing:
        snap.warnings.append(f"asset type for {k!r} not found; skipped")

    # relation-type roles by id, so we can classify relations without extra calls
    rt_roles: dict[str, tuple[str, str]] = {}
    for rt in cc.relation_types():
        rt_roles[rt["id"]] = ((rt.get("role") or "").lower(), (rt.get("coRole") or "").lower())

    def rel_role(rel: dict[str, Any]) -> tuple[str, str]:
        rtype = rel.get("type") or {}
        if rtype.get("id") in rt_roles:
            return rt_roles[rtype["id"]]
        return ((rtype.get("role") or "").lower(), (rtype.get("coRole") or "").lower())

    tables_raw = cc.assets_of_type(type_ids["table"])
    columns_raw = cc.assets_of_type(type_ids["column"]) if type_ids.get("column") else []
    schemas_raw = cc.assets_of_type(type_ids["schema"]) if type_ids.get("schema") else []
    terms_raw = cc.assets_of_type(type_ids["term"]) if type_ids.get("term") else []

    if cfg.domains:
        allowed = {d.lower() for d in cfg.domains}
        tables_raw = [a for a in tables_raw if ((a.get("domain") or {}).get("name") or "").lower() in allowed]

    schema_by_id = {a["id"]: _short_name(a) for a in schemas_raw}
    tables: dict[str, Table] = {}
    column_owner: dict[str, str] = {}  # column asset id -> table asset id
    column_name: dict[str, str] = {a["id"]: _short_name(a) for a in columns_raw}
    column_asset: dict[str, dict[str, Any]] = {a["id"]: a for a in columns_raw}

    for a in tables_raw:
        t = Table(name=_short_name(a), kind="table", source=source, properties={"collibra_id": a["id"], "domain": (a.get("domain") or {}).get("name", "")})
        if a.get("status"):
            t.tags.append(f"status={a['status'].get('name') if isinstance(a['status'], dict) else a['status']}")
        # attributes: description + tag-like
        for attr in cc.attributes(a["id"]):
            aname = ((attr.get("type") or {}).get("name") or "").strip()
            val = attr.get("value")
            if aname == cfg.description_attribute and val:
                t.description = str(val)
            elif aname in cfg.tag_attributes and val not in (None, "", False):
                t.tags.append(f"{aname}={val}")
        # relations: parent schema, child columns, table-table relations, terms
        for rel in cc.relations_to(a["id"]):
            role, corole = rel_role(rel)
            src = rel.get("source") or {}
            if role == cfg.contains_role.lower() and src.get("id") in schema_by_id:
                t.schema_name = schema_by_id[src["id"]]
        for rel in cc.relations_from(a["id"]):
            role, corole = rel_role(rel)
            tgt = rel.get("target") or {}
            if role == cfg.contains_role.lower() and tgt.get("id") in column_name:
                column_owner[tgt["id"]] = a["id"]
        tables[a["id"]] = t

    # columns: attach to owner table (also accept Column -> "is part of" -> Table direction)
    for cid in column_asset:
        owner = column_owner.get(cid)
        if owner is None:
            for rel in cc.relations_from(cid):
                role, corole = rel_role(rel)
                tgt = rel.get("target") or {}
                if tgt.get("id") in tables and (corole == cfg.contains_role.lower() or role in {"is part of", "belongs to"}):
                    owner = tgt["id"]
                    break
        if owner is None or owner not in tables:
            continue
        col = Column(name=column_name[cid], properties={"collibra_id": cid})
        for attr in cc.attributes(cid):
            aname = ((attr.get("type") or {}).get("name") or "").strip()
            val = attr.get("value")
            if aname == cfg.description_attribute and val:
                col.description = str(val)
            elif aname in {"Data Type", "Technical Data Type"} and val:
                col.data_type = str(val)
            elif aname in cfg.tag_attributes and val not in (None, "", False):
                col.tags.append(f"{aname}={val}")
        tables[owner].columns.append(col)
        column_owner[cid] = owner

    fk_roles = {r.lower() for r in cfg.fk_roles}
    tbl_roles = {r.lower() for r in cfg.table_relation_roles}
    # column -> column FK relations
    for cid, owner in column_owner.items():
        for rel in cc.relations_from(cid):
            role, _ = rel_role(rel)
            tgt = rel.get("target") or {}
            if role in fk_roles and tgt.get("id") in column_owner:
                to_owner = column_owner[tgt["id"]]
                snap.edges.append(
                    Edge(
                        kind="foreign_key",
                        from_table=tables[owner].fqn,
                        to_table=tables[to_owner].fqn,
                        from_columns=[column_name[cid]],
                        to_columns=[column_name[tgt["id"]]],
                        description=f"Collibra: {role}",
                        source=source,
                    )
                )
    # table -> table curated relations
    for tid, t in tables.items():
        for rel in cc.relations_from(tid):
            role, _ = rel_role(rel)
            tgt = rel.get("target") or {}
            if role in tbl_roles and tgt.get("id") in tables:
                snap.edges.append(Edge(kind="catalog_relation", from_table=t.fqn, to_table=tables[tgt["id"]].fqn, description=f"Collibra: {role}", confidence=0.8, source=source))

    # business terms
    term_roles = {r.lower() for r in cfg.term_roles}
    syn_roles = {r.lower() for r in cfg.synonym_relation_roles}
    term_name = {a["id"]: (a.get("displayName") or a.get("name") or "").strip() for a in terms_raw}
    for a in terms_raw:
        bt = BusinessTerm(name=term_name[a["id"]], source=source)
        for attr in cc.attributes(a["id"]):
            aname = ((attr.get("type") or {}).get("name") or "").strip()
            if aname == cfg.description_attribute and attr.get("value"):
                bt.description = str(attr["value"])
        for rel in cc.relations_from(a["id"]) + cc.relations_to(a["id"]):
            role, corole = rel_role(rel)
            other = rel.get("target") if (rel.get("source") or {}).get("id") == a["id"] else rel.get("source")
            oid = (other or {}).get("id")
            if role in term_roles or corole in term_roles:
                if oid in tables:
                    bt.targets.append(tables[oid].fqn)
                elif oid in column_owner:
                    bt.targets.append(f"{tables[column_owner[oid]].fqn}.{column_name[oid]}")
            if (role in syn_roles or corole in syn_roles) and oid in term_name:
                bt.synonyms.append(term_name[oid])
        if bt.name:
            snap.terms.append(bt)
    snap.tables = list(tables.values())
    return snap.stamp()


@register
class CollibraConnector:
    type_name: ClassVar[str] = "collibra"
    Config = CollibraConfig

    def __init__(self, name: str, config: CollibraConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        cc = CollibraClient(self.config)
        tid = cc.asset_type_id(self.config.table_type)
        if not tid:
            return f"connected, but asset type {self.config.table_type!r} not found"
        n = len(cc.assets_of_type(tid))
        return f"ok: {n} {self.config.table_type} assets"

    def introspect(self) -> SchemaSnapshot:
        return introspect_collibra(self.config, self.name)
