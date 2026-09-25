"""AWS Glue Data Catalog connector (boto3).

Glue has databases and tables with typed columns, comments, partition keys and
free-form ``Parameters`` - but no foreign keys. Relationship evidence for Glue tables
comes from other sources merged into the same graph (Collibra relations, dbt
lineage, DDL, join hints) or from name-based inference.

Optional: Lake Formation LF-tags per table are attached as ``tags`` when
``lf_tags=True`` (needs ``lakeformation:GetResourceLFTags``).
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from schemagraph.connectors.base import register
from schemagraph.model import Column, SchemaSnapshot, Table


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
KEPT_TABLE_PARAMETERS: frozenset[str] = frozenset(
    {"classification", "comment", "EXTERNAL", "has_encrypted_data", "typeOfData", "table_type"}
)
# Databases requested by ``check()``.
CHECK_MAX_DATABASES = 50


def _session(cfg: GlueConfig):
    """A boto3 session for the configured profile and region (boto3 imported lazily)."""
    import boto3

    return boto3.Session(profile_name=cfg.profile, region_name=cfg.region)


def _databases(glue_client, catalog_kw: dict[str, Any]) -> list[dict[str, Any]]:
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
        if k in KEPT_TABLE_PARAMETERS
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


def _attach_lf_tags(
    table: Table,
    database_name: str,
    cfg: GlueConfig,
    catalog_kw: dict[str, Any],
    session,
    lf_client,
    snap: SchemaSnapshot,
):
    """Attach a table's Lake Formation LF-tags to it and its columns.

    Any failure (permissions vary) becomes a snapshot warning; tags attached before the
    failure stay.

    Args:
        table: The table; mutated in place.
        database_name: Glue database of the table.
        cfg: Connector config.
        catalog_kw: ``{"CatalogId": ...}`` or empty.
        session: boto3 session reused to create the Lake Formation client, or None.
        lf_client: Lake Formation client, or None to create one.
        snap: Snapshot that receives warnings.

    Returns:
        The Lake Formation client to reuse for the next table (None if it could not be made).
    """
    try:
        if lf_client is None:
            lf_client = (session or _session(cfg)).client(
                "lakeformation",
                endpoint_url=cfg.endpoint_url,
            )
        response = lf_client.get_resource_lf_tags(
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
    except Exception as e:  # noqa: BLE001 - permissions vary
        snap.warnings.append(f"LF-tags unavailable for {database_name}.{table.name}: {e}")
    return lf_client


def introspect_glue(
    cfg: GlueConfig,
    source: str = "glue",
    glue_client=None,
    lf_client=None,
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

    for database in _databases(glue_client, catalog_kw):
        database_name = database["Name"]
        if cfg.databases and database_name not in cfg.databases:
            continue
        paginator = glue_client.get_paginator("get_tables")
        for page in paginator.paginate(DatabaseName=database_name, **catalog_kw):
            for raw in page.get("TableList") or []:
                table = _table_from_glue(raw, database_name, cfg, source)
                if cfg.lf_tags:
                    lf_client = _attach_lf_tags(
                        table,
                        database_name,
                        cfg,
                        catalog_kw,
                        session,
                        lf_client,
                        snap,
                    )
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
        response = client.get_databases(MaxResults=CHECK_MAX_DATABASES, **catalog_kw)
        databases = response.get("DatabaseList") or []
        return f"ok: {len(databases)} databases visible"

    def introspect(self) -> SchemaSnapshot:
        """Read the Glue catalog into a snapshot."""
        return introspect_glue(self.config, self.name)
