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
    region: str = Field(description="AWS region, e.g. eu-west-1")
    databases: list[str] | None = Field(default=None, description="Restrict to these Glue databases (default: all)")
    catalog_id: str | None = Field(default=None, description="AWS account id of the catalog (default: caller's)")
    profile: str | None = Field(default=None, description="AWS named profile (default: environment credentials)")
    include_partition_keys: bool = True
    lf_tags: bool = Field(default=False, description="Attach Lake Formation LF-tags as table tags")
    endpoint_url: str | None = Field(default=None, description="Override endpoint (e.g. moto/localstack)")


def _session(cfg: GlueConfig):
    import boto3

    return boto3.Session(profile_name=cfg.profile, region_name=cfg.region)


def introspect_glue(cfg: GlueConfig, source: str = "glue", glue_client=None, lf_client=None) -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="glue")
    sess = None
    if glue_client is None:
        sess = _session(cfg)
        glue_client = sess.client("glue", endpoint_url=cfg.endpoint_url)
    catalog_kw = {"CatalogId": cfg.catalog_id} if cfg.catalog_id else {}

    dbs: list[dict[str, Any]] = []
    for page in glue_client.get_paginator("get_databases").paginate(**catalog_kw):
        dbs.extend(page.get("DatabaseList") or [])
    for db in dbs:
        dbname = db["Name"]
        if cfg.databases and dbname not in cfg.databases:
            continue
        for page in glue_client.get_paginator("get_tables").paginate(DatabaseName=dbname, **catalog_kw):
            for tbl in page.get("TableList") or []:
                sd = tbl.get("StorageDescriptor") or {}
                ttype = (tbl.get("TableType") or "").upper()
                kind = "view" if "VIEW" in ttype else ("external" if ttype == "EXTERNAL_TABLE" else "table")
                params = {k: str(v) for k, v in (tbl.get("Parameters") or {}).items() if k in {"classification", "comment", "EXTERNAL", "has_encrypted_data", "typeOfData", "table_type"}}
                if sd.get("Location"):
                    params["location"] = sd["Location"]
                t = Table(
                    name=tbl["Name"],
                    schema=dbname,
                    catalog=None,
                    kind=kind,
                    description=tbl.get("Description") or (tbl.get("Parameters") or {}).get("comment") or None,
                    owner=tbl.get("Owner"),
                    properties=params,
                    source=source,
                )
                for col in sd.get("Columns") or []:
                    t.columns.append(Column(name=col["Name"], data_type=col.get("Type"), description=col.get("Comment") or None, properties={k: str(v) for k, v in (col.get("Parameters") or {}).items()}))
                if cfg.include_partition_keys:
                    for col in tbl.get("PartitionKeys") or []:
                        t.columns.append(Column(name=col["Name"], data_type=col.get("Type"), description=col.get("Comment") or None, tags=["partition_key"]))
                if cfg.lf_tags:
                    try:
                        if lf_client is None:
                            lf_client = (sess or _session(cfg)).client("lakeformation", endpoint_url=cfg.endpoint_url)
                        resp = lf_client.get_resource_lf_tags(Resource={"Table": {"DatabaseName": dbname, "Name": tbl["Name"], **catalog_kw}}, ShowAssignedLFTags=True)
                        for tag in resp.get("LFTagOnDatabase", []) + resp.get("LFTagsOnTable", []):
                            for v in tag.get("TagValues") or []:
                                t.tags.append(f"{tag.get('TagKey')}={v}")
                        for ct in resp.get("LFTagsOnColumns") or []:
                            c = t.column(ct.get("Name", ""))
                            if c:
                                for tag in ct.get("LFTags") or []:
                                    for v in tag.get("TagValues") or []:
                                        c.tags.append(f"{tag.get('TagKey')}={v}")
                    except Exception as e:  # noqa: BLE001 - permissions vary
                        snap.warnings.append(f"LF-tags unavailable for {dbname}.{tbl['Name']}: {e}")
                snap.tables.append(t)
    return snap.stamp()


@register
class GlueConnector:
    type_name: ClassVar[str] = "aws_glue"
    Config = GlueConfig

    def __init__(self, name: str, config: GlueConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        client = _session(self.config).client("glue", endpoint_url=self.config.endpoint_url)
        kw = {"CatalogId": self.config.catalog_id} if self.config.catalog_id else {}
        dbs = client.get_databases(MaxResults=50, **kw).get("DatabaseList") or []
        return f"ok: {len(dbs)} databases visible"

    def introspect(self) -> SchemaSnapshot:
        return introspect_glue(self.config, self.name)
