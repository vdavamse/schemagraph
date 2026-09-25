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
    """A Collibra instance (REST API 2.0) and the operating-model names to read from it."""

    host: str = Field(description="Collibra base URL, e.g. https://acme.collibra.com")
    username: str | None = Field(default=None, description="Basic-auth user (or use token)")
    password: str | None = Field(
        default=None,
        description="Basic-auth password (use ${COLLIBRA_PASSWORD})",
    )
    token: str | None = Field(
        default=None,
        description="Bearer/JWT token, alternative to username/password",
    )
    domains: list[str] | None = Field(
        default=None,
        description="Restrict to these domain names (default: all)",
    )
    # operating-model names
    table_type: str = "Table"
    column_type: str = "Column"
    schema_type: str = "Schema"
    database_type: str = "Database"
    term_type: str = "Business Term"
    contains_role: str = Field(
        default="contains",
        description="Relation role for Schema>Table and Table>Column hierarchy",
    )
    fk_roles: list[str] = Field(
        default_factory=lambda: ["references", "is reference to", "is foreign key of"],
        description="Column->Column relation roles that mean a foreign key",
    )
    table_relation_roles: list[str] = Field(
        default_factory=lambda: ["is related to", "relates to"],
        description="Table->Table relation roles to import as catalog relations",
    )
    term_roles: list[str] = Field(
        default_factory=lambda: ["represents", "is represented by", "is described by", "describes"],
        description="Business Term <-> Table/Column relation roles",
    )
    description_attribute: str = "Description"
    tag_attributes: list[str] = Field(
        default_factory=lambda: [
            "Data Classification",
            "Personal Identifiable Information",
            "Security Classification",
        ],
        description="Attribute types copied into tags as name=value",
    )
    synonym_relation_roles: list[str] = Field(
        default_factory=lambda: ["is synonym of", "has synonym"],
    )
    page_size: int = 500
    timeout: float = 60.0
    verify_ssl: bool = True


# Confidence of an edge imported from a curated table -> table relation.
CATALOG_RELATION_CONFIDENCE = 0.8
# Attribute types read as a column's data type.
DATA_TYPE_ATTRIBUTES: frozenset[str] = frozenset({"Data Type", "Technical Data Type"})
# Column -> table relation roles that mean "the column belongs to the table".
COLUMN_OF_TABLE_ROLES: frozenset[str] = frozenset({"is part of", "belongs to"})
# Attribute values that never become a tag.
_EMPTY_TAG_VALUES = (None, "", False)


class CollibraClient:
    """Thin authenticated client over the Collibra REST API 2.0 with offset paging."""

    def __init__(self, cfg: CollibraConfig, transport: httpx.BaseTransport | None = None):
        headers = {"Accept": "application/json"}
        if cfg.token:
            headers["Authorization"] = f"Bearer {cfg.token}"
        elif cfg.username and cfg.password:
            credentials = f"{cfg.username}:{cfg.password}".encode()
            headers["Authorization"] = "Basic " + base64.b64encode(credentials).decode()
        self.cfg = cfg
        self.http = httpx.Client(
            base_url=cfg.host.rstrip("/") + "/rest/2.0",
            headers=headers,
            timeout=cfg.timeout,
            verify=cfg.verify_ssl,
            transport=transport,
        )

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        """GET one endpoint (``None`` params dropped) and return the JSON body."""
        response = self.http.get(path, params={k: v for k, v in params.items() if v is not None})
        response.raise_for_status()
        return response.json()

    def paged(self, path: str, **params: Any):
        """Yield every ``results`` item of a paged endpoint, page by page."""
        offset = 0
        while True:
            data = self.get(path, limit=self.cfg.page_size, offset=offset, **params)
            results = data.get("results") or []
            yield from results
            if len(results) < self.cfg.page_size:
                break
            offset += self.cfg.page_size

    def asset_type_id(self, name: str) -> str | None:
        """Id of the asset type with exactly this name, or None."""
        data = self.get("/assetTypes", name=name, nameMatchMode="EXACT")
        results = data.get("results") or []
        return results[0]["id"] if results else None

    def assets_of_type(self, type_id: str) -> list[dict[str, Any]]:
        """Every asset of one type."""
        return list(self.paged("/assets", typeIds=type_id))

    def attributes(self, asset_id: str) -> list[dict[str, Any]]:
        """Every attribute of one asset."""
        return list(self.paged("/attributes", assetId=asset_id))

    def relations_from(self, asset_id: str) -> list[dict[str, Any]]:
        """Relations whose source is this asset."""
        return list(self.paged("/relations", sourceId=asset_id))

    def relations_to(self, asset_id: str) -> list[dict[str, Any]]:
        """Relations whose target is this asset."""
        return list(self.paged("/relations", targetId=asset_id))

    def relation_types(self) -> list[dict[str, Any]]:
        """Every relation type."""
        return list(self.paged("/relationTypes"))


class _RelationRoles:
    """Lowercased (role, coRole) of relations, by relation-type id when known.

    Fetches the relation types once so relations can be classified without extra calls.
    """

    def __init__(self, relation_types: list[dict[str, Any]]):
        self.by_type_id: dict[str, tuple[str, str]] = {}
        for relation_type in relation_types:
            self.by_type_id[relation_type["id"]] = (
                (relation_type.get("role") or "").lower(),
                (relation_type.get("coRole") or "").lower(),
            )

    def of(self, relation: dict[str, Any]) -> tuple[str, str]:
        """(role, coRole) of one relation, falling back to its inline type."""
        relation_type = relation.get("type") or {}
        if relation_type.get("id") in self.by_type_id:
            return self.by_type_id[relation_type["id"]]
        return (
            (relation_type.get("role") or "").lower(),
            (relation_type.get("coRole") or "").lower(),
        )


def _short_name(asset: dict[str, Any]) -> str:
    """Last segment of an asset's display name (after ``>`` then ``.``)."""
    name = asset.get("displayName") or asset.get("name") or ""
    for sep in (">", "."):
        if sep in name:
            name = name.split(sep)[-1]
    return name.strip()


def _asset_type_ids(client: CollibraClient, cfg: CollibraConfig) -> dict[str, str | None]:
    """Asset type id per role (table, column, schema, term), looked up in that order."""
    names = {
        "table": cfg.table_type,
        "column": cfg.column_type,
        "schema": cfg.schema_type,
        "term": cfg.term_type,
    }
    return {k: client.asset_type_id(v) for k, v in names.items()}


def _attributes(client: CollibraClient, asset_id: str) -> list[tuple[str, Any]]:
    """(stripped attribute type name, value) of every attribute of an asset."""
    pairs = []
    for attr in client.attributes(asset_id):
        attr_name = ((attr.get("type") or {}).get("name") or "").strip()
        pairs.append((attr_name, attr.get("value")))
    return pairs


def _in_domains(asset: dict[str, Any], allowed: set[str]) -> bool:
    """Whether an asset's domain name (lowercased) is in ``allowed``."""
    return ((asset.get("domain") or {}).get("name") or "").lower() in allowed


def _table_from_asset(
    asset: dict[str, Any],
    client: CollibraClient,
    cfg: CollibraConfig,
    source: str,
) -> Table:
    """Build a table from a Table asset: status tag, description and tag attributes."""
    table = Table(
        name=_short_name(asset),
        kind="table",
        source=source,
        properties={
            "collibra_id": asset["id"],
            "domain": (asset.get("domain") or {}).get("name", ""),
        },
    )
    if asset.get("status"):
        status = asset["status"]
        status_name = status.get("name") if isinstance(status, dict) else status
        table.tags.append(f"status={status_name}")
    # attributes: description + tag-like
    for attr_name, value in _attributes(client, asset["id"]):
        if attr_name == cfg.description_attribute and value:
            table.description = str(value)
        elif attr_name in cfg.tag_attributes and value not in _EMPTY_TAG_VALUES:
            table.tags.append(f"{attr_name}={value}")
    return table


def _build_tables(
    tables_raw: list[dict[str, Any]],
    client: CollibraClient,
    cfg: CollibraConfig,
    roles: _RelationRoles,
    schema_by_id: dict[str, str],
    column_name: dict[str, str],
    source: str,
) -> tuple[dict[str, Table], dict[str, str]]:
    """Build every table with its parent schema and find the columns it contains.

    Per table: attributes, relations to it (parent schema), relations from it (columns).

    Args:
        tables_raw: Table assets.
        client: Collibra client.
        cfg: Connector config (roles, attribute names).
        roles: Relation role classifier.
        schema_by_id: Schema short name by asset id.
        column_name: Column short name by asset id.
        source: Snapshot source name stamped on the tables.

    Returns:
        (tables by asset id, owning table asset id by column asset id).
    """
    contains_role = cfg.contains_role.lower()
    tables: dict[str, Table] = {}
    column_owner: dict[str, str] = {}  # column asset id -> table asset id
    for asset in tables_raw:
        table = _table_from_asset(asset, client, cfg, source)
        # relations: parent schema, child columns
        for rel in client.relations_to(asset["id"]):
            role, _corole = roles.of(rel)
            rel_source = rel.get("source") or {}
            if role == contains_role and rel_source.get("id") in schema_by_id:
                table.schema_name = schema_by_id[rel_source["id"]]
        for rel in client.relations_from(asset["id"]):
            role, _corole = roles.of(rel)
            rel_target = rel.get("target") or {}
            if role == contains_role and rel_target.get("id") in column_name:
                column_owner[rel_target["id"]] = asset["id"]
        tables[asset["id"]] = table
    return tables, column_owner


def _owner_from_column(
    column_id: str,
    client: CollibraClient,
    cfg: CollibraConfig,
    roles: _RelationRoles,
    tables: dict[str, Table],
) -> str | None:
    """Owning table of a column via a Column -> Table relation (``is part of`` direction)."""
    for rel in client.relations_from(column_id):
        role, corole = roles.of(rel)
        rel_target = rel.get("target") or {}
        is_column_of = corole == cfg.contains_role.lower() or role in COLUMN_OF_TABLE_ROLES
        if rel_target.get("id") in tables and is_column_of:
            return rel_target["id"]
    return None


def _attach_columns(
    column_asset: dict[str, dict[str, Any]],
    column_name: dict[str, str],
    column_owner: dict[str, str],
    tables: dict[str, Table],
    client: CollibraClient,
    cfg: CollibraConfig,
    roles: _RelationRoles,
) -> None:
    """Attach every column with a known owner to its table, in column-asset order.

    Mutates ``tables`` (columns appended) and ``column_owner`` (owners found from the
    column side added) in place.

    Args:
        column_asset: Column assets by id.
        column_name: Column short name by asset id.
        column_owner: Owning table asset id by column asset id.
        tables: Tables by asset id.
        client: Collibra client.
        cfg: Connector config.
        roles: Relation role classifier.
    """
    for column_id in column_asset:
        owner = column_owner.get(column_id)
        if owner is None:
            owner = _owner_from_column(column_id, client, cfg, roles, tables)
        if owner is None or owner not in tables:
            continue
        column = Column(name=column_name[column_id], properties={"collibra_id": column_id})
        for attr_name, value in _attributes(client, column_id):
            if attr_name == cfg.description_attribute and value:
                column.description = str(value)
            elif attr_name in DATA_TYPE_ATTRIBUTES and value:
                column.data_type = str(value)
            elif attr_name in cfg.tag_attributes and value not in _EMPTY_TAG_VALUES:
                column.tags.append(f"{attr_name}={value}")
        tables[owner].columns.append(column)
        column_owner[column_id] = owner


def _column_fk_edges(
    column_owner: dict[str, str],
    column_name: dict[str, str],
    tables: dict[str, Table],
    client: CollibraClient,
    cfg: CollibraConfig,
    roles: _RelationRoles,
    source: str,
) -> list[Edge]:
    """Foreign-key edges from column -> column relations with an FK role.

    Args:
        column_owner: Owning table asset id by column asset id.
        column_name: Column short name by asset id.
        tables: Tables by asset id.
        client: Collibra client.
        cfg: Connector config (``fk_roles``).
        roles: Relation role classifier.
        source: Snapshot source name stamped on the edges.

    Returns:
        The edges, per column in ``column_owner`` order.
    """
    fk_roles = {r.lower() for r in cfg.fk_roles}
    edges: list[Edge] = []
    for column_id, owner in column_owner.items():
        for rel in client.relations_from(column_id):
            role, _ = roles.of(rel)
            rel_target = rel.get("target") or {}
            if role in fk_roles and rel_target.get("id") in column_owner:
                to_owner = column_owner[rel_target["id"]]
                edges.append(
                    Edge(
                        kind="foreign_key",
                        from_table=tables[owner].fqn,
                        to_table=tables[to_owner].fqn,
                        from_columns=[column_name[column_id]],
                        to_columns=[column_name[rel_target["id"]]],
                        description=f"Collibra: {role}",
                        source=source,
                    )
                )
    return edges


def _table_relation_edges(
    tables: dict[str, Table],
    client: CollibraClient,
    cfg: CollibraConfig,
    roles: _RelationRoles,
    source: str,
) -> list[Edge]:
    """``catalog_relation`` edges from curated table -> table relations."""
    table_roles = {r.lower() for r in cfg.table_relation_roles}
    edges: list[Edge] = []
    for table_id, table in tables.items():
        for rel in client.relations_from(table_id):
            role, _ = roles.of(rel)
            rel_target = rel.get("target") or {}
            if role in table_roles and rel_target.get("id") in tables:
                edges.append(
                    Edge(
                        kind="catalog_relation",
                        from_table=table.fqn,
                        to_table=tables[rel_target["id"]].fqn,
                        description=f"Collibra: {role}",
                        confidence=CATALOG_RELATION_CONFIDENCE,
                        source=source,
                    )
                )
    return edges


def _business_terms(
    terms_raw: list[dict[str, Any]],
    tables: dict[str, Table],
    column_owner: dict[str, str],
    column_name: dict[str, str],
    client: CollibraClient,
    cfg: CollibraConfig,
    roles: _RelationRoles,
    source: str,
) -> list[BusinessTerm]:
    """Glossary terms with their table/column targets and synonyms.

    Per term: attributes, then relations from it followed by relations to it.

    Args:
        terms_raw: Business Term assets.
        tables: Tables by asset id.
        column_owner: Owning table asset id by column asset id.
        column_name: Column short name by asset id.
        client: Collibra client.
        cfg: Connector config (term and synonym roles).
        roles: Relation role classifier.
        source: Snapshot source name stamped on the terms.

    Returns:
        Every term with a non-empty name, in asset order.
    """
    term_roles = {r.lower() for r in cfg.term_roles}
    synonym_roles = {r.lower() for r in cfg.synonym_relation_roles}
    term_name = {
        a["id"]: (a.get("displayName") or a.get("name") or "").strip() for a in terms_raw
    }
    terms: list[BusinessTerm] = []
    for asset in terms_raw:
        term = BusinessTerm(name=term_name[asset["id"]], source=source)
        for attr_name, value in _attributes(client, asset["id"]):
            if attr_name == cfg.description_attribute and value:
                term.description = str(value)
        relations = client.relations_from(asset["id"]) + client.relations_to(asset["id"])
        for rel in relations:
            role, corole = roles.of(rel)
            is_outgoing = (rel.get("source") or {}).get("id") == asset["id"]
            other = rel.get("target") if is_outgoing else rel.get("source")
            other_id = (other or {}).get("id")
            if role in term_roles or corole in term_roles:
                if other_id in tables:
                    term.targets.append(tables[other_id].fqn)
                elif other_id in column_owner:
                    owner_fqn = tables[column_owner[other_id]].fqn
                    term.targets.append(f"{owner_fqn}.{column_name[other_id]}")
            if (role in synonym_roles or corole in synonym_roles) and other_id in term_name:
                term.synonyms.append(term_name[other_id])
        if term.name:
            terms.append(term)
    return terms


def introspect_collibra(
    cfg: CollibraConfig,
    source: str = "collibra",
    client: CollibraClient | None = None,
) -> SchemaSnapshot:
    """Read tables, columns, relations and business terms from Collibra.

    Without the Table asset type nothing is imported and the (unstamped) snapshot carries
    only a warning.

    Args:
        cfg: Connector config.
        source: Snapshot source name.
        client: Client to use (tests inject a mock transport); built from ``cfg`` if None.

    Returns:
        The snapshot.
    """
    snap = SchemaSnapshot(source=source, source_type="collibra")
    client = client or CollibraClient(cfg)

    type_ids = _asset_type_ids(client, cfg)
    missing = [k for k, v in type_ids.items() if not v]
    if "table" in missing:
        snap.warnings.append(f"asset type {cfg.table_type!r} not found; nothing imported")
        return snap
    for k in missing:
        snap.warnings.append(f"asset type for {k!r} not found; skipped")

    roles = _RelationRoles(client.relation_types())

    tables_raw = client.assets_of_type(type_ids["table"])
    columns_raw = client.assets_of_type(type_ids["column"]) if type_ids.get("column") else []
    schemas_raw = client.assets_of_type(type_ids["schema"]) if type_ids.get("schema") else []
    terms_raw = client.assets_of_type(type_ids["term"]) if type_ids.get("term") else []

    if cfg.domains:
        allowed = {d.lower() for d in cfg.domains}
        tables_raw = [a for a in tables_raw if _in_domains(a, allowed)]

    schema_by_id = {a["id"]: _short_name(a) for a in schemas_raw}
    column_name: dict[str, str] = {a["id"]: _short_name(a) for a in columns_raw}
    column_asset: dict[str, dict[str, Any]] = {a["id"]: a for a in columns_raw}

    tables, column_owner = _build_tables(
        tables_raw,
        client,
        cfg,
        roles,
        schema_by_id,
        column_name,
        source,
    )
    _attach_columns(column_asset, column_name, column_owner, tables, client, cfg, roles)
    snap.edges.extend(
        _column_fk_edges(column_owner, column_name, tables, client, cfg, roles, source)
    )
    snap.edges.extend(_table_relation_edges(tables, client, cfg, roles, source))
    snap.terms.extend(
        _business_terms(terms_raw, tables, column_owner, column_name, client, cfg, roles, source)
    )
    snap.tables = list(tables.values())
    return snap.stamp()


@register
class CollibraConnector:
    """Connector over a Collibra instance."""

    type_name: ClassVar[str] = "collibra"
    Config = CollibraConfig

    def __init__(self, name: str, config: CollibraConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        """Look up the Table asset type and count its assets."""
        client = CollibraClient(self.config)
        type_id = client.asset_type_id(self.config.table_type)
        if not type_id:
            return f"connected, but asset type {self.config.table_type!r} not found"
        count = len(client.assets_of_type(type_id))
        return f"ok: {count} {self.config.table_type} assets"

    def introspect(self) -> SchemaSnapshot:
        """Read the Collibra metadata into a snapshot."""
        return introspect_collibra(self.config, self.name)
