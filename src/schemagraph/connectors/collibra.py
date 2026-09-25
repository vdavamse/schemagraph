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
from collections.abc import Iterator
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
_DATA_TYPE_ATTRIBUTES: frozenset[str] = frozenset({"Data Type", "Technical Data Type"})
# Column -> table relation roles that mean "the column belongs to the table".
_COLUMN_OF_TABLE_ROLES: frozenset[str] = frozenset({"is part of", "belongs to"})
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

    def paged(self, path: str, **params: Any) -> Iterator[dict[str, Any]]:
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


def _in_domains(asset: dict[str, Any], allowed: set[str]) -> bool:
    """Whether an asset's domain name (lowercased) is in ``allowed``."""
    return ((asset.get("domain") or {}).get("name") or "").lower() in allowed


class _CollibraReader:
    """One Collibra introspection: the client, config and id maps shared by its phases.

    The phases run in a fixed order (asset types, assets, tables, columns, edges, terms)
    and each appends to ``snap`` or the id maps; the order of the HTTP requests and of
    every append is part of the snapshot and must not change.

    Attributes:
        cfg: Connector config.
        source: Snapshot source name stamped on tables, edges and terms.
        client: Collibra client.
        snap: The snapshot being filled.
        type_ids: Asset type id per role (table, column, schema, term).
        roles: Relation role classifier, set by ``read_assets``.
        tables_raw: Table assets (domain-filtered).
        terms_raw: Business Term assets.
        schema_by_id: Schema short name by asset id.
        column_name: Column short name by asset id.
        column_asset: Column assets by id.
        tables: Tables by asset id.
        column_owner: Owning table asset id by column asset id.
    """

    def __init__(self, cfg: CollibraConfig, source: str, client: CollibraClient):
        self.cfg = cfg
        self.source = source
        self.client = client
        self.snap = SchemaSnapshot(source=source, source_type="collibra")
        self.type_ids: dict[str, str | None] = {}
        self.roles: _RelationRoles | None = None
        self.tables_raw: list[dict[str, Any]] = []
        self.terms_raw: list[dict[str, Any]] = []
        self.schema_by_id: dict[str, str] = {}
        self.column_name: dict[str, str] = {}
        self.column_asset: dict[str, dict[str, Any]] = {}
        self.tables: dict[str, Table] = {}
        self.column_owner: dict[str, str] = {}

    def read_asset_types(self) -> bool:
        """Look up the asset type ids (table, column, schema, term, in that order).

        Adds a warning per missing type.

        Returns:
            False when the Table asset type is missing and nothing can be imported.
        """
        cfg = self.cfg
        names = {
            "table": cfg.table_type,
            "column": cfg.column_type,
            "schema": cfg.schema_type,
            "term": cfg.term_type,
        }
        self.type_ids = {k: self.client.asset_type_id(v) for k, v in names.items()}
        missing = [k for k, v in self.type_ids.items() if not v]
        if "table" in missing:
            self.snap.warnings.append(f"asset type {cfg.table_type!r} not found; nothing imported")
            return False
        for k in missing:
            self.snap.warnings.append(f"asset type for {k!r} not found; skipped")
        return True

    def read_assets(self) -> None:
        """Fetch the relation types and the assets of every found type; build the id maps."""
        client = self.client
        type_ids = self.type_ids
        self.roles = _RelationRoles(client.relation_types())

        tables_raw = client.assets_of_type(type_ids["table"])
        columns_raw = client.assets_of_type(type_ids["column"]) if type_ids.get("column") else []
        schemas_raw = client.assets_of_type(type_ids["schema"]) if type_ids.get("schema") else []
        self.terms_raw = client.assets_of_type(type_ids["term"]) if type_ids.get("term") else []

        if self.cfg.domains:
            allowed = {d.lower() for d in self.cfg.domains}
            tables_raw = [a for a in tables_raw if _in_domains(a, allowed)]
        self.tables_raw = tables_raw

        self.schema_by_id = {a["id"]: _short_name(a) for a in schemas_raw}
        self.column_name = {a["id"]: _short_name(a) for a in columns_raw}
        self.column_asset = {a["id"]: a for a in columns_raw}

    def _attributes(self, asset_id: str) -> list[tuple[str, Any]]:
        """(stripped attribute type name, value) of every attribute of an asset."""
        pairs = []
        for attr in self.client.attributes(asset_id):
            attr_name = ((attr.get("type") or {}).get("name") or "").strip()
            pairs.append((attr_name, attr.get("value")))
        return pairs

    def _table_from_asset(self, asset: dict[str, Any]) -> Table:
        """Build a table from a Table asset: status tag, description and tag attributes."""
        table = Table(
            name=_short_name(asset),
            kind="table",
            source=self.source,
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
        for attr_name, value in self._attributes(asset["id"]):
            if attr_name == self.cfg.description_attribute and value:
                table.description = str(value)
            elif attr_name in self.cfg.tag_attributes and value not in _EMPTY_TAG_VALUES:
                table.tags.append(f"{attr_name}={value}")
        return table

    def build_tables(self) -> None:
        """Build every table with its parent schema and find the columns it contains.

        Per table: attributes, relations to it (parent schema), relations from it (columns).
        Fills ``tables`` and ``column_owner``.
        """
        contains_role = self.cfg.contains_role.lower()
        for asset in self.tables_raw:
            table = self._table_from_asset(asset)
            # relations: parent schema, child columns
            for rel in self.client.relations_to(asset["id"]):
                role, _corole = self.roles.of(rel)
                rel_source = rel.get("source") or {}
                if role == contains_role and rel_source.get("id") in self.schema_by_id:
                    table.schema_name = self.schema_by_id[rel_source["id"]]
            for rel in self.client.relations_from(asset["id"]):
                role, _corole = self.roles.of(rel)
                rel_target = rel.get("target") or {}
                if role == contains_role and rel_target.get("id") in self.column_name:
                    self.column_owner[rel_target["id"]] = asset["id"]
            self.tables[asset["id"]] = table

    def _owner_from_column(self, column_id: str) -> str | None:
        """Owning table of a column via a Column -> Table relation (``is part of`` direction)."""
        for rel in self.client.relations_from(column_id):
            role, corole = self.roles.of(rel)
            rel_target = rel.get("target") or {}
            is_column_of = (
                corole == self.cfg.contains_role.lower() or role in _COLUMN_OF_TABLE_ROLES
            )
            if rel_target.get("id") in self.tables and is_column_of:
                return rel_target["id"]
        return None

    def attach_columns(self) -> None:
        """Attach every column with a known owner to its table, in column-asset order.

        Appends to the tables' columns and adds owners found from the column side to
        ``column_owner``.
        """
        cfg = self.cfg
        for column_id in self.column_asset:
            owner = self.column_owner.get(column_id)
            if owner is None:
                owner = self._owner_from_column(column_id)
            if owner is None or owner not in self.tables:
                continue
            column = Column(name=self.column_name[column_id], properties={"collibra_id": column_id})
            for attr_name, value in self._attributes(column_id):
                if attr_name == cfg.description_attribute and value:
                    column.description = str(value)
                elif attr_name in _DATA_TYPE_ATTRIBUTES and value:
                    column.data_type = str(value)
                elif attr_name in cfg.tag_attributes and value not in _EMPTY_TAG_VALUES:
                    column.tags.append(f"{attr_name}={value}")
            self.tables[owner].columns.append(column)
            self.column_owner[column_id] = owner

    def add_column_fk_edges(self) -> None:
        """Add foreign-key edges from column -> column relations with an FK role.

        Edges follow ``column_owner`` order.
        """
        fk_roles = {r.lower() for r in self.cfg.fk_roles}
        column_owner = self.column_owner
        for column_id, owner in column_owner.items():
            for rel in self.client.relations_from(column_id):
                role, _ = self.roles.of(rel)
                rel_target = rel.get("target") or {}
                if role in fk_roles and rel_target.get("id") in column_owner:
                    to_owner = column_owner[rel_target["id"]]
                    self.snap.edges.append(
                        Edge(
                            kind="foreign_key",
                            from_table=self.tables[owner].fqn,
                            to_table=self.tables[to_owner].fqn,
                            from_columns=[self.column_name[column_id]],
                            to_columns=[self.column_name[rel_target["id"]]],
                            description=f"Collibra: {role}",
                            source=self.source,
                        )
                    )

    def add_table_relation_edges(self) -> None:
        """Add ``catalog_relation`` edges from curated table -> table relations."""
        table_roles = {r.lower() for r in self.cfg.table_relation_roles}
        tables = self.tables
        for table_id, table in tables.items():
            for rel in self.client.relations_from(table_id):
                role, _ = self.roles.of(rel)
                rel_target = rel.get("target") or {}
                if role in table_roles and rel_target.get("id") in tables:
                    self.snap.edges.append(
                        Edge(
                            kind="catalog_relation",
                            from_table=table.fqn,
                            to_table=tables[rel_target["id"]].fqn,
                            description=f"Collibra: {role}",
                            confidence=CATALOG_RELATION_CONFIDENCE,
                            source=self.source,
                        )
                    )

    def add_business_terms(self) -> None:
        """Add glossary terms with their table/column targets and synonyms.

        Per term: attributes, then relations from it followed by relations to it. Terms
        with an empty name are dropped; the rest keep asset order.
        """
        cfg = self.cfg
        tables = self.tables
        term_roles = {r.lower() for r in cfg.term_roles}
        synonym_roles = {r.lower() for r in cfg.synonym_relation_roles}
        term_name = {
            a["id"]: (a.get("displayName") or a.get("name") or "").strip() for a in self.terms_raw
        }
        for asset in self.terms_raw:
            term = BusinessTerm(name=term_name[asset["id"]], source=self.source)
            for attr_name, value in self._attributes(asset["id"]):
                if attr_name == cfg.description_attribute and value:
                    term.description = str(value)
            relations = (
                self.client.relations_from(asset["id"]) + self.client.relations_to(asset["id"])
            )
            for rel in relations:
                role, corole = self.roles.of(rel)
                is_outgoing = (rel.get("source") or {}).get("id") == asset["id"]
                other = rel.get("target") if is_outgoing else rel.get("source")
                other_id = (other or {}).get("id")
                if role in term_roles or corole in term_roles:
                    if other_id in tables:
                        term.targets.append(tables[other_id].fqn)
                    elif other_id in self.column_owner:
                        owner_fqn = tables[self.column_owner[other_id]].fqn
                        term.targets.append(f"{owner_fqn}.{self.column_name[other_id]}")
                if (role in synonym_roles or corole in synonym_roles) and other_id in term_name:
                    term.synonyms.append(term_name[other_id])
            if term.name:
                self.snap.terms.append(term)

    def finish(self) -> SchemaSnapshot:
        """Store the tables (asset order) in the snapshot and return it stamped."""
        self.snap.tables = list(self.tables.values())
        return self.snap.stamp()


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
    reader = _CollibraReader(cfg, source, client or CollibraClient(cfg))
    if not reader.read_asset_types():
        return reader.snap
    reader.read_assets()
    reader.build_tables()
    reader.attach_columns()
    reader.add_column_fk_edges()
    reader.add_table_relation_edges()
    reader.add_business_terms()
    return reader.finish()


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
