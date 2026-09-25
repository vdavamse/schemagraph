"""AWS Glue Data Catalog connector (boto3).

Glue has databases and tables with typed columns, comments, partition keys and
free-form ``Parameters`` - but no foreign keys. Relationship evidence for Glue tables
comes from other sources merged into the same graph (Collibra relations, dbt
lineage, DDL, join hints) or from name-based inference.

Optional: Lake Formation LF-tags per table are attached as ``tags`` when
``lf_tags=True`` (needs ``lakeformation:GetResourceLFTags``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from schemagraph.connectors.base import register
from schemagraph.model import Column, SchemaSnapshot, Table

if TYPE_CHECKING:
    import boto3


class GlueConfig(BaseModel):
    """An AWS Glue Data Catalog, optionally narrowed to some databases, with LF-tags."""

    region: str = Field(description="AWS region, e.g. eu-west-1")
    databases: list[str] | None = Field(
        default=None,
        description="Restrict to these Glue databases (default: all)",
    )
    catalog_id: str | None = Field(
        default=None,
        description="AWS account id of the catalog (default: caller's)",
    )
    profile: str | None = Field(
        default=None,
        description="AWS named profile (default: environment credentials)",
    )
    include_partition_keys: bool = True
    lf_tags: bool = Field(default=False, description="Attach Lake Formation LF-tags as table tags")
    endpoint_url: str | None = Field(
        default=None,
        description="Override endpoint (e.g. moto/localstack)",
    )


# Glue table ``Parameters`` copied into ``Table.properties``.
_KEPT_TABLE_PARAMETERS: frozenset[str] = frozenset(
    {"classification", "comment", "EXTERNAL", "has_encrypted_data", "typeOfData", "table_type"}
)
# Databases requested by ``check()``.
_CHECK_MAX_DATABASES = 50


def _session(cfg: GlueConfig) -> boto3.Session:
    """A boto3 session for the configured profile and region (boto3 imported lazily)."""
    import boto3

    return boto3.Session(profile_name=cfg.profile, region_name=cfg.region)


def _databases(glue_client: Any, catalog_kw: dict[str, Any]) -> list[dict[str, Any]]:
    """Every Glue database, all pages read up front."""
    databases: list[dict[str, Any]] = []
    for page in glue_client.get_paginator("get_databases").paginate(**catalog_kw):
        databases.extend(page.get("DatabaseList") or [])
    return databases


def _table_kind(table_type: str) -> str:
    """Table.kind for an uppercased Glue ``TableType``."""
    if "VIEW" in table_type:
        return "view"
    return "external" if table_type == "EXTERNAL_TABLE" else "table"


def _table_from_glue(
    raw: dict[str, Any],
    database_name: str,
    cfg: GlueConfig,
    source: str,
) -> Table:
    """Build a table from a Glue table payload: columns, then partition keys if configured."""
    storage = raw.get("StorageDescriptor") or {}
    kind = _table_kind((raw.get("TableType") or "").upper())
    params = {
        k: str(v)
        for k, v in (raw.get("Parameters") or {}).items()
        if k in _KEPT_TABLE_PARAMETERS
    }
    if storage.get("Location"):
        params["location"] = storage["Location"]
    table = Table(
        name=raw["Name"],
        schema=database_name,
        catalog=None,
        kind=kind,
        description=raw.get("Description") or (raw.get("Parameters") or {}).get("comment") or None,
        owner=raw.get("Owner"),
        properties=params,
        source=source,
    )
    for column in storage.get("Columns") or []:
        table.columns.append(
            Column(
                name=column["Name"],
                data_type=column.get("Type"),
                description=column.get("Comment") or None,
                properties={k: str(v) for k, v in (column.get("Parameters") or {}).items()},
            )
        )
    if cfg.include_partition_keys:
        for column in raw.get("PartitionKeys") or []:
            table.columns.append(
                Column(
                    name=column["Name"],
                    data_type=column.get("Type"),
                    description=column.get("Comment") or None,
                    tags=["partition_key"],
                )
            )
    return table


class _LakeFormation:
    """Lake Formation client of one introspection, created on first use and then reused.

    Creation is retried on the next ``client()`` call when it fails.
    """

    def __init__(
        self,
        cfg: GlueConfig,
        session: boto3.Session | None,
        lf_client: Any | None,
    ):
        self.cfg = cfg
        self.session = session
        self.lf_client = lf_client

    def client(self) -> Any:
        """The Lake Formation client, created from the session (or a new one) if not yet made."""
        if self.lf_client is None:
            self.lf_client = (self.session or _session(self.cfg)).client(
                "lakeformation",
                endpoint_url=self.cfg.endpoint_url,
            )
        return self.lf_client


def _attach_lf_tags(
    table: Table,
    database_name: str,
    catalog_kw: dict[str, Any],
    lake_formation: _LakeFormation,
    snap: SchemaSnapshot,
) -> None:
    """Attach a table's Lake Formation LF-tags to it and its columns.

    Any failure (permissions vary), including creating the client, becomes a snapshot
    warning; tags attached before the failure stay.

    Args:
        table: The table; mutated in place.
        database_name: Glue database of the table.
        catalog_kw: ``{"CatalogId": ...}`` or empty.
        lake_formation: Lazy Lake Formation client holder.
        snap: Snapshot that receives warnings.
    """
    try:
        response = lake_formation.client().get_resource_lf_tags(
            Resource={"Table": {"DatabaseName": database_name, "Name": table.name, **catalog_kw}},
            ShowAssignedLFTags=True,
        )
        for tag in response.get("LFTagOnDatabase", []) + response.get("LFTagsOnTable", []):
            for value in tag.get("TagValues") or []:
                table.tags.append(f"{tag.get('TagKey')}={value}")
        for column_tags in response.get("LFTagsOnColumns") or []:
            column = table.column(column_tags.get("Name", ""))
            if column:
                for tag in column_tags.get("LFTags") or []:
                    for value in tag.get("TagValues") or []:
                        column.tags.append(f"{tag.get('TagKey')}={value}")
    except Exception as e:  # permissions vary
        snap.warnings.append(f"LF-tags unavailable for {database_name}.{table.name}: {e}")


def introspect_glue(
    cfg: GlueConfig,
    source: str = "glue",
    glue_client: Any | None = None,
    lf_client: Any | None = None,
) -> SchemaSnapshot:
    """Read Glue databases and tables (and optionally LF-tags) into a stamped snapshot.

    Args:
        cfg: Connector config.
        source: Snapshot source name.
        glue_client: boto3 Glue client (tests inject a fake); created from a session if None.
        lf_client: Lake Formation client; created on first use if None.

    Returns:
        The stamped snapshot.
    """
    snap = SchemaSnapshot(source=source, source_type="glue")
    session = None
    if glue_client is None:
        session = _session(cfg)
        glue_client = session.client("glue", endpoint_url=cfg.endpoint_url)
    catalog_kw = {"CatalogId": cfg.catalog_id} if cfg.catalog_id else {}
    lake_formation = _LakeFormation(cfg, session, lf_client)

    for database in _databases(glue_client, catalog_kw):
        database_name = database["Name"]
        if cfg.databases and database_name not in cfg.databases:
            continue
        paginator = glue_client.get_paginator("get_tables")
        for page in paginator.paginate(DatabaseName=database_name, **catalog_kw):
            for raw in page.get("TableList") or []:
                table = _table_from_glue(raw, database_name, cfg, source)
                if cfg.lf_tags:
                    _attach_lf_tags(table, database_name, catalog_kw, lake_formation, snap)
                snap.tables.append(table)
    return snap.stamp()


@register
class GlueConnector:
    """Connector over the AWS Glue Data Catalog."""

    type_name: ClassVar[str] = "aws_glue"
    Config = GlueConfig

    def __init__(self, name: str, config: GlueConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        """Count the databases visible to the configured credentials."""
        client = _session(self.config).client("glue", endpoint_url=self.config.endpoint_url)
        catalog_kw = {"CatalogId": self.config.catalog_id} if self.config.catalog_id else {}
        response = client.get_databases(MaxResults=_CHECK_MAX_DATABASES, **catalog_kw)
        databases = response.get("DatabaseList") or []
        return f"ok: {len(databases)} databases visible"

    def introspect(self) -> SchemaSnapshot:
        """Read the Glue catalog into a snapshot."""
        return introspect_glue(self.config, self.name)
