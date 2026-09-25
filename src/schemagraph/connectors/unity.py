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
    """A Databricks workspace or OSS Unity Catalog server, optionally narrowed to some schemas."""

    host: str = Field(
        description=(
            "Workspace URL, e.g. https://adb-123.azuredatabricks.net"
            " or http://localhost:8080 for OSS"
        ),
    )
    token: str | None = Field(
        default=None,
        description="Personal access token (use ${DATABRICKS_TOKEN})",
    )
    catalogs: list[str] | None = Field(
        default=None,
        description="Restrict to these catalogs (default: all)",
    )
    schemas: list[str] | None = Field(
        default=None,
        description="Restrict to these schemas (default: all)",
    )
    include_views: bool = True
    lineage: bool = Field(default=False, description="Also query table lineage (Databricks only)")
    api_base: str = Field(default="/api/2.1/unity-catalog", description="API prefix")
    timeout: float = 30.0
    verify_ssl: bool = True


# Catalogs that hold Databricks internals, never user tables.
SKIPPED_CATALOGS: frozenset[str] = frozenset({"system", "__databricks_internal"})
# Schema present in every catalog that describes the catalog itself.
SKIPPED_SCHEMA = "information_schema"
# Page size of the tables endpoint.
TABLES_PAGE_SIZE = 50
# Table property prefix of Delta internals, left out of ``Table.properties``.
DELTA_PROPERTY_PREFIX = "delta."
# Catalogs listed by ``check()`` before eliding the rest.
CHECK_LISTED_CATALOGS = 5


class UnityClient:
    """Thin client over the Unity Catalog REST API with ``page_token`` paging."""

    def __init__(self, cfg: UnityConfig, transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {cfg.token}"} if cfg.token else {}
        self.cfg = cfg
        self.http = httpx.Client(
            base_url=cfg.host.rstrip("/"),
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

    def paged(self, path: str, key: str, **params: Any):
        """Yield every item under ``key`` of a paged endpoint, following ``next_page_token``."""
        token = None
        while True:
            data = self.get(path, page_token=token, **params)
            yield from data.get(key) or []
            token = data.get("next_page_token")
            if not token:
                break

    def catalogs(self) -> list[dict[str, Any]]:
        """Every catalog."""
        return list(self.paged(f"{self.cfg.api_base}/catalogs", "catalogs"))

    def schemas(self, catalog: str) -> list[dict[str, Any]]:
        """Every schema of a catalog."""
        return list(self.paged(f"{self.cfg.api_base}/schemas", "schemas", catalog_name=catalog))

    def tables(self, catalog: str, schema: str) -> list[dict[str, Any]]:
        """Every table of a schema, with columns and constraints."""
        return list(
            self.paged(
                f"{self.cfg.api_base}/tables",
                "tables",
                catalog_name=catalog,
                schema_name=schema,
                max_results=TABLES_PAGE_SIZE,
            )
        )

    def lineage(self, full_name: str) -> dict[str, Any]:
        """Table-level lineage of one table (Databricks lineage-tracking API)."""
        return self.get(
            "/api/2.0/lineage-tracking/table-lineage",
            table_name=full_name,
            include_entity_lineage="false",
        )


def _kind(table_type: str | None) -> str:
    """Table.kind for a Unity ``table_type``."""
    table_type_upper = (table_type or "").upper()
    if "VIEW" in table_type_upper:
        return "view"
    if table_type_upper == "EXTERNAL":
        return "external"
    return "table"


def _iter_tables(client: UnityClient, cfg: UnityConfig):
    """Yield (catalog, schema, raw table) for every table the config selects, lazily."""
    for catalog in client.catalogs():
        catalog_name = catalog.get("name")
        if not catalog_name or (cfg.catalogs and catalog_name not in cfg.catalogs):
            continue
        if catalog_name in SKIPPED_CATALOGS:
            continue
        for schema in client.schemas(catalog_name):
            schema_name = schema.get("name")
            if (
                not schema_name
                or schema_name == SKIPPED_SCHEMA
                or (cfg.schemas and schema_name not in cfg.schemas)
            ):
                continue
            for raw in client.tables(catalog_name, schema_name):
                yield catalog_name, schema_name, raw


def _table_from_unity(
    raw: dict[str, Any],
    catalog_name: str,
    schema_name: str,
    kind: str,
    source: str,
) -> Table:
    """Build a table (columns by ``position``) from a Unity table payload."""
    table = Table(
        name=raw["name"],
        schema=schema_name,
        catalog=catalog_name,
        kind=kind,
        description=raw.get("comment") or None,
        owner=raw.get("owner"),
        properties={
            k: str(v)
            for k, v in (raw.get("properties") or {}).items()
            if not k.startswith(DELTA_PROPERTY_PREFIX)
        },
        source=source,
    )
    for column in sorted(raw.get("columns") or [], key=lambda c: c.get("position", 0)):
        table.columns.append(
            Column(
                name=column["name"],
                data_type=column.get("type_text") or column.get("type_name"),
                description=column.get("comment") or None,
                nullable=column.get("nullable"),
            )
        )
    return table


def _constraint_edges(table: Table, raw: dict[str, Any], source: str) -> list[Edge]:
    """Apply ``table_constraints``: set the primary key, return the foreign-key edges.

    Mutates ``table`` in place (a later primary-key constraint replaces an earlier one).
    """
    edges: list[Edge] = []
    for cons in raw.get("table_constraints") or []:
        pk = cons.get("primary_key_constraint")
        fk = cons.get("foreign_key_constraint")
        if pk:
            table.primary_key = list(pk.get("child_columns") or [])
            for column in table.columns:
                if column.name in table.primary_key:
                    column.is_primary_key = True
        if fk:
            edges.append(
                Edge(
                    kind="foreign_key",
                    from_table=table.fqn,
                    to_table=fk.get("parent_table", ""),
                    from_columns=list(fk.get("child_columns") or []),
                    to_columns=list(fk.get("parent_columns") or []),
                    description=fk.get("name"),
                    source=source,
                )
            )
    return edges


def _lineage_edges(
    client: UnityClient,
    known: dict[str, Table],
    snap: SchemaSnapshot,
    source: str,
) -> list[Edge]:
    """Upstream -> table lineage edges; a failed lookup becomes a snapshot warning."""
    edges: list[Edge] = []
    for table in list(known.values()):
        try:
            lineage = client.lineage(table.fqn)
        except httpx.HTTPError as e:
            snap.warnings.append(f"lineage failed for {table.fqn}: {e}")
            continue
        for upstream in lineage.get("upstreams") or []:
            info = upstream.get("tableInfo") or {}
            name_parts = (info.get("catalog_name"), info.get("schema_name"), info.get("name"))
            upstream_fqn = ".".join(p for p in name_parts if p)
            if upstream_fqn:
                edges.append(
                    Edge(kind="lineage", from_table=upstream_fqn, to_table=table.fqn, source=source)
                )
    return edges


def introspect_unity(
    cfg: UnityConfig,
    source: str = "unity",
    client: UnityClient | None = None,
) -> SchemaSnapshot:
    """Read catalogs, schemas, tables, constraints and optional lineage from Unity Catalog.

    Foreign keys pointing at tables that were not read are dropped.

    Args:
        cfg: Connector config.
        source: Snapshot source name.
        client: Client to use (tests inject a mock transport); built from ``cfg`` if None.

    Returns:
        The stamped snapshot.
    """
    snap = SchemaSnapshot(source=source, source_type="unity")
    client = client or UnityClient(cfg)
    known: dict[str, Table] = {}  # lowercased fqn -> table
    for catalog_name, schema_name, raw in _iter_tables(client, cfg):
        kind = _kind(raw.get("table_type"))
        if kind == "view" and not cfg.include_views:
            continue
        table = _table_from_unity(raw, catalog_name, schema_name, kind, source)
        snap.edges.extend(_constraint_edges(table, raw, source))
        snap.tables.append(table)
        known[table.fqn.lower()] = table
    if cfg.lineage:
        snap.edges.extend(_lineage_edges(client, known, snap, source))
    # drop FK edges pointing at tables we did not see (keeps the graph honest)
    snap.edges = [e for e in snap.edges if e.kind != "foreign_key" or e.to_table.lower() in known]
    return snap.stamp()


@register
class UnityConnector:
    """Connector over Unity Catalog (Databricks or OSS)."""

    type_name: ClassVar[str] = "unity_catalog"
    Config = UnityConfig

    def __init__(self, name: str, config: UnityConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        """List the visible catalogs."""
        client = UnityClient(self.config)
        catalog_names = [c.get("name") for c in client.catalogs()]
        listed = ", ".join(catalog_names[:CHECK_LISTED_CATALOGS])
        more = "..." if len(catalog_names) > CHECK_LISTED_CATALOGS else ""
        return f"ok: {len(catalog_names)} catalogs ({listed}{more})"

    def introspect(self) -> SchemaSnapshot:
        """Read the catalog into a snapshot."""
        return introspect_unity(self.config, self.name)
