"""Spider 2.0-Lite schema-file connector (benchmark use; not registered in the UI).

Reads ``spider2-lite/resource/databases/<dialect>/<db>/`` where each dataset folder
holds one ``<table>.json`` per table (``table_fullname``, ``column_names``,
``column_types``, ``nested_column_names``, ``description`` aligned with columns,
``sample_rows``) plus a ``DDL.csv`` whose statements may declare PK/FK (SQLite dbs do).

Date-sharded families such as GA4 ``events_20201101 … events_20210131`` collapse into
one table ``events_*`` (ReFoRCE-style prefix grouping) so the graph stays small; the
member names are kept in ``properties["members"]``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from schemagraph.connectors.ddl import DDLConfig, parse_ddl
from schemagraph.model import Column, SchemaSnapshot, Table

# date shards (YYYYMMDD, YYYYMM) and year shards (19xx/20xx); the stem may be empty ("_20230118")
_SHARD_RE = re.compile(r"^(?P<stem>.*?)_?(?P<suffix>\d{8}|\d{6}|(?:19|20)\d{2})$")
_DIGITS_RE = re.compile(r"\d+")


def family_signature(name: str) -> str:
    """``county_2018_1yr`` -> ``county_*_*yr``; ``2000_q1`` -> ``*_q*``; ``gsod2019`` -> ``gsod*``."""
    return _DIGITS_RE.sub("*", name.strip().lower())
DIALECT_FOR_PREFIX = {"bq": "bigquery", "ga": "bigquery", "sf": "snowflake", "local": "sqlite"}


class Spider2Config(BaseModel):
    root: str = Field(description="…/spider2-lite/resource/databases")
    dialect: str = Field(description="bigquery | snowflake | sqlite")
    db: str = Field(description="database folder name, e.g. austin, GITHUB_REPOS, Airlines")
    path: str | None = Field(default=None, description="explicit database folder; overrides root/dialect/db (Spider2-Snow keeps databases flat under resource/databases/<DB>)")
    sample_values: int = 5
    collapse_shards: bool = True
    collapse_families: bool = Field(default=True, description="Merge tables whose names differ only in digit runs and share >= family_jaccard of their columns")
    family_jaccard: float = 0.8
    use_nested_columns: bool = True


def canonical_table(name: str) -> str:
    """Collapse date-sharded table names: ``ds.events_20210101`` -> ``ds.events_*``."""
    parts = name.split(".")
    m = _SHARD_RE.match(parts[-1])
    if m:
        parts[-1] = m.group("stem").rstrip("_") + "_*"
    return ".".join(parts).lower()


def _split_fullname(full: str) -> tuple[str | None, str | None, str]:
    parts = full.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    return None, None, parts[0]


def _sample_values(rows: list[dict], col: str, limit: int) -> list[str]:
    out: list[str] = []
    for r in rows or []:
        v = r.get(col)
        if isinstance(v, str) and 0 < len(v) <= 80 and v not in out:
            out.append(v)
        if len(out) >= limit:
            break
    return out


def load_table_json(path: Path, cfg: Spider2Config, source: str) -> Table:
    d = json.loads(path.read_text(encoding="utf-8"))
    full = (d.get("table_fullname") or d.get("table_name") or path.stem).strip()
    catalog, schema, name = _split_fullname(full)
    name = name.strip()
    names = d.get("column_names") or []
    types = d.get("column_types") or []
    descs = d.get("description") or []
    if cfg.use_nested_columns and d.get("nested_column_names"):
        nn, nt = d["nested_column_names"], d.get("nested_column_types") or []
        seen = {n for n in names}
        extra = [(n, nt[i] if i < len(nt) else None) for i, n in enumerate(nn) if n not in seen]
    else:
        extra = []
    rows = d.get("sample_rows") or []
    cols: list[Column] = []
    for i, cname in enumerate(names):
        cols.append(
            Column(
                name=cname,
                data_type=types[i] if i < len(types) else None,
                description=(descs[i] or None) if i < len(descs) and isinstance(descs[i], str) else None,
                sample_values=_sample_values(rows, cname, cfg.sample_values),
            )
        )
    for cname, ctype in extra:
        cols.append(Column(name=cname, data_type=ctype, tags=["nested"]))
    table_desc = d.get("table_description") if isinstance(d.get("table_description"), str) else None
    return Table(name=name, schema=schema, catalog=catalog, columns=cols, description=table_desc, source=source, properties={"fullname": full})


def _cluster_name(names: list[str]) -> str:
    """Family name for one cluster: digit runs that vary across members become ``*``, digit runs
    shared by every member are kept, so two clusters with the same signature get distinct names
    (``imaging_level2_metadata_r*`` vs ``imaging_level4_metadata_r*``)."""
    parts = [re.split(r"(\d+)", n.strip().lower()) for n in names]
    if len({len(p) for p in parts}) != 1:
        return family_signature(names[0])
    out: list[str] = []
    for i, toks in enumerate(zip(*parts, strict=True)):
        if i % 2 == 1:
            out.append(toks[0] if len(set(toks)) == 1 else "*")
        elif len(set(toks)) == 1:
            out.append(toks[0])
        else:
            return family_signature(names[0])
    return "".join(out)


def _collapse_by_signature(tables: dict[str, Table], jaccard: float) -> dict[str, Table]:
    """Group tables (same catalog/schema) whose names share a digit-run signature and whose
    column sets overlap by >= ``jaccard``; keep one representative with a ``members`` list."""
    groups: dict[tuple[str | None, str | None, str], list[tuple[str, Table]]] = {}
    for key, t in tables.items():
        sig = family_signature(t.name)
        if "*" not in sig:
            continue
        groups.setdefault((t.catalog, t.schema_name, sig), []).append((key, t))
    out = dict(tables)
    for (_catalog, schema, _sig), members in groups.items():
        if len(members) < 2:
            continue
        # cluster by column-set similarity (greedy)
        clusters: list[list[tuple[str, Table]]] = []
        for key, t in sorted(members, key=lambda kt: kt[1].name):
            cols = {c.name.lower() for c in t.columns}
            placed = False
            for cl in clusters:
                ref = {c.name.lower() for c in cl[0][1].columns}
                inter = len(cols & ref)
                union = len(cols | ref) or 1
                if inter / union >= jaccard:
                    cl.append((key, t))
                    placed = True
                    break
            if not placed:
                clusters.append([(key, t)])
        for cl in clusters:
            if len(cl) < 2:
                continue
            rep = cl[0][1].model_copy(deep=True)
            member_fqns: list[str] = []
            for key, t in cl:
                member_fqns.extend(t.properties.get("members", t.fqn).split(","))
                for c in t.columns:
                    if rep.column(c.name) is None:
                        rep.columns.append(c)
                out.pop(key, None)
            rep.name = _cluster_name([t.name for _, t in cl])
            while rep.fqn.lower() in out:  # never overwrite another cluster (or table) with the same name
                rep.name += "_"
            stem = re.sub(r"[*_]+", " ", rep.name).strip()
            rep.properties = {"family": "true", "members": ",".join(dict.fromkeys(member_fqns)), "business_name": f"{schema or ''} {stem}".strip()}
            out[rep.fqn.lower()] = rep
    return out


def _resolver(tables: list[Table]):
    by_name = {t.fqn.lower(): t for t in tables}
    by_bare: dict[str, Table] = {}
    for t in tables:
        by_bare.setdefault(t.name.lower(), t)

    def resolve(n: str) -> Table | None:
        return by_name.get(n.lower()) or by_bare.get(n.split(".")[-1].lower())

    return resolve


def introspect_spider2(cfg: Spider2Config, source: str = "spider2") -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="spider2")
    base = Path(cfg.path) if cfg.path else Path(cfg.root) / cfg.dialect / cfg.db
    if not base.exists():
        snap.warnings.append(f"missing {base}")
        return snap
    json_files = sorted(p for p in base.rglob("*.json") if p.name != "DDL.json")
    families: dict[str, Table] = {}
    for jf in json_files:
        try:
            t = load_table_json(jf, cfg, source)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            snap.warnings.append(f"{jf.name}: {e}")
            continue
        key = canonical_table(t.fqn) if cfg.collapse_shards else t.fqn.lower()
        if key.endswith("_*") and cfg.collapse_shards:
            fam = families.get(key)
            if fam is None:
                fam = t.model_copy(deep=True)
                fam.name = key.split(".")[-1]
                stem = fam.name[:-2]
                fam.properties = {"family": "true", "members": t.fqn, "business_name": f"{t.schema_name or ''} {stem}".strip()}
                families[key] = fam
            else:
                fam.properties["members"] += "," + t.fqn
                for c in t.columns:
                    if fam.column(c.name) is None:
                        fam.columns.append(c)
        else:
            families[key] = t
    if cfg.collapse_families:
        families = _collapse_by_signature(families, cfg.family_jaccard)
    snap.tables = list(families.values())
    for t in snap.tables:
        if t.properties.get("family"):
            members = t.properties["members"].split(",")
            t.description = (t.description or "") + f" (date-sharded family of {len(members)} tables, {members[0].split('.')[-1]} … {members[-1].split('.')[-1]})"
            t.description = t.description.strip()

    # PK/FK from DDL.csv (SQLite dbs declare them; cloud DDL rarely does)
    for ddl_csv in base.rglob("DDL.csv"):
        try:
            with ddl_csv.open(encoding="utf-8", newline="") as fh:
                stmts = [row.get("DDL", "") for row in csv.DictReader(fh)]
        except (OSError, csv.Error) as e:
            snap.warnings.append(f"{ddl_csv}: {e}")
            continue
        ddl_text = ";\n".join(s.strip().rstrip(";") for s in stmts if s and s.strip())
        dialect = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}[cfg.dialect]
        parsed = parse_ddl(DDLConfig(ddl=ddl_text, dialect=dialect), source)
        resolve = _resolver(snap.tables)
        for pt in parsed.tables:
            t = resolve(pt.fqn)
            if t is None:
                continue
            for pk in pt.primary_key:
                if pk not in t.primary_key:
                    t.primary_key.append(pk)
                c = t.column(pk)
                if c:
                    c.is_primary_key = True
            for pc in pt.columns:
                c = t.column(pc.name)
                if c and not c.data_type:
                    c.data_type = pc.data_type
        for e in parsed.edges:
            a, b = resolve(e.from_table), resolve(e.to_table)
            if a and b:
                e.from_table, e.to_table = a.fqn, b.fqn
                snap.edges.append(e)
    return snap.stamp()
