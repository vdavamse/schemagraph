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


# Spider2 instance-id prefix -> sqlglot dialect of its database.
DIALECT_FOR_PREFIX = {"bq": "bigquery", "ga": "bigquery", "sf": "snowflake", "local": "sqlite"}
# Spider2Config.dialect -> sqlglot dialect used to parse ``DDL.csv``.
_DDL_DIALECTS = {"bigquery": "bigquery", "snowflake": "snowflake", "sqlite": "sqlite"}
# Sample values longer than this many characters are skipped.
MAX_SAMPLE_CHARS = 80


def family_signature(name: str) -> str:
    """Digit-run signature of a table name.

    ``county_2018_1yr`` -> ``county_*_*yr``; ``2000_q1`` -> ``*_q*``; ``gsod2019`` -> ``gsod*``.
    """
    return _DIGITS_RE.sub("*", name.strip().lower())


class Spider2Config(BaseModel):
    """One Spider 2.0 database folder of per-table JSON files plus an optional DDL.csv."""

    root: str = Field(description="…/spider2-lite/resource/databases")
    dialect: str = Field(description="bigquery | snowflake | sqlite")
    db: str = Field(description="database folder name, e.g. austin, GITHUB_REPOS, Airlines")
    path: str | None = Field(
        default=None,
        description=(
            "explicit database folder; overrides root/dialect/db"
            " (Spider2-Snow keeps databases flat under resource/databases/<DB>)"
        ),
    )
    sample_values: int = 5
    collapse_shards: bool = True
    collapse_families: bool = Field(
        default=True,
        description=(
            "Merge tables whose names differ only in digit runs"
            " and share >= family_jaccard of their columns"
        ),
    )
    family_jaccard: float = 0.8
    use_nested_columns: bool = True


def canonical_table(name: str) -> str:
    """Collapse date-sharded table names: ``ds.events_20210101`` -> ``ds.events_*``."""
    parts = name.split(".")
    match = _SHARD_RE.match(parts[-1])
    if match:
        parts[-1] = match.group("stem").rstrip("_") + "_*"
    return ".".join(parts).lower()


def _split_fullname(full: str) -> tuple[str | None, str | None, str]:
    """(catalog, schema, name) of a dotted name; everything before the schema is the catalog."""
    parts = full.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    return None, None, parts[0]


def _sample_values(rows: list[dict], col: str, limit: int) -> list[str]:
    """Up to ``limit`` distinct short, non-empty string values of a column in the sample rows."""
    out: list[str] = []
    for row in rows or []:
        value = row.get(col)
        if isinstance(value, str) and 0 < len(value) <= MAX_SAMPLE_CHARS and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def load_table_json(path: Path, cfg: Spider2Config, source: str) -> Table:
    """Build a table from one Spider2 ``<table>.json`` file.

    Columns follow ``column_names`` (with aligned types, descriptions and sample values),
    then the nested columns not already listed, tagged ``nested``.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    full = (doc.get("table_fullname") or doc.get("table_name") or path.stem).strip()
    catalog, schema, name = _split_fullname(full)
    name = name.strip()
    names = doc.get("column_names") or []
    types = doc.get("column_types") or []
    descriptions = doc.get("description") or []
    if cfg.use_nested_columns and doc.get("nested_column_names"):
        nested_names = doc["nested_column_names"]
        nested_types = doc.get("nested_column_types") or []
        seen = {n for n in names}
        extra = [
            (n, nested_types[i] if i < len(nested_types) else None)
            for i, n in enumerate(nested_names)
            if n not in seen
        ]
    else:
        extra = []
    rows = doc.get("sample_rows") or []
    columns: list[Column] = []
    for i, column_name in enumerate(names):
        has_description = i < len(descriptions) and isinstance(descriptions[i], str)
        columns.append(
            Column(
                name=column_name,
                data_type=types[i] if i < len(types) else None,
                description=(descriptions[i] or None) if has_description else None,
                sample_values=_sample_values(rows, column_name, cfg.sample_values),
            )
        )
    for column_name, column_type in extra:
        columns.append(Column(name=column_name, data_type=column_type, tags=["nested"]))
    raw_description = doc.get("table_description")
    table_description = raw_description if isinstance(raw_description, str) else None
    return Table(
        name=name,
        schema=schema,
        catalog=catalog,
        columns=columns,
        description=table_description,
        source=source,
        properties={"fullname": full},
    )


def _cluster_name(names: list[str]) -> str:
    """Family name for one cluster.

    Digit runs that vary across members become ``*``, digit runs shared by every member are
    kept, so two clusters with the same signature get distinct names
    (``imaging_level2_metadata_r*`` vs ``imaging_level4_metadata_r*``).
    """
    parts = [re.split(r"(\d+)", n.strip().lower()) for n in names]
    if len({len(p) for p in parts}) != 1:
        return family_signature(names[0])
    out: list[str] = []
    for i, tokens in enumerate(zip(*parts, strict=True)):
        if i % 2 == 1:
            out.append(tokens[0] if len(set(tokens)) == 1 else "*")
        elif len(set(tokens)) == 1:
            out.append(tokens[0])
        else:
            return family_signature(names[0])
    return "".join(out)


def _cluster_by_columns(
    members: list[tuple[str, Table]],
    jaccard: float,
) -> list[list[tuple[str, Table]]]:
    """Greedy clustering by column-set Jaccard against each cluster's first table.

    Members are visited sorted by table name (stable); a table joins the first cluster it
    matches, else starts a new one.
    """
    clusters: list[list[tuple[str, Table]]] = []
    for key, table in sorted(members, key=lambda kt: kt[1].name):
        columns = {c.name.lower() for c in table.columns}
        placed = False
        for cluster in clusters:
            reference = {c.name.lower() for c in cluster[0][1].columns}
            inter = len(columns & reference)
            union = len(columns | reference) or 1
            if inter / union >= jaccard:
                cluster.append((key, table))
                placed = True
                break
        if not placed:
            clusters.append([(key, table)])
    return clusters


def _merge_cluster(
    cluster: list[tuple[str, Table]],
    schema: str | None,
    out: dict[str, Table],
) -> None:
    """Replace a cluster's tables in ``out`` with one representative appended at the end.

    The representative is a deep copy of the first member with every member's missing
    columns, a :func:`_cluster_name` (suffixed with ``_`` until it collides with nothing
    in ``out``) and ``family``/``members``/``business_name`` properties. Mutates ``out``.
    """
    representative = cluster[0][1].model_copy(deep=True)
    member_fqns: list[str] = []
    for key, table in cluster:
        member_fqns.extend(table.properties.get("members", table.fqn).split(","))
        for column in table.columns:
            if representative.column(column.name) is None:
                representative.columns.append(column)
        out.pop(key, None)
    representative.name = _cluster_name([t.name for _, t in cluster])
    # never overwrite another cluster (or table) with the same name
    while representative.fqn.lower() in out:
        representative.name += "_"
    stem = re.sub(r"[*_]+", " ", representative.name).strip()
    representative.properties = {
        "family": "true",
        "members": ",".join(dict.fromkeys(member_fqns)),
        "business_name": f"{schema or ''} {stem}".strip(),
    }
    out[representative.fqn.lower()] = representative


def _collapse_by_signature(tables: dict[str, Table], jaccard: float) -> dict[str, Table]:
    """Collapse same-signature tables with overlapping columns into family tables.

    Groups tables (same catalog/schema) whose names share a digit-run signature and whose
    column sets overlap by >= ``jaccard``; keeps one representative with a ``members`` list.
    Untouched tables keep their position; representatives are appended in group order.
    """
    groups: dict[tuple[str | None, str | None, str], list[tuple[str, Table]]] = {}
    for key, table in tables.items():
        signature = family_signature(table.name)
        if "*" not in signature:
            continue
        groups.setdefault((table.catalog, table.schema_name, signature), []).append((key, table))
    out = dict(tables)
    for (_catalog, schema, _signature), members in groups.items():
        if len(members) < 2:
            continue
        for cluster in _cluster_by_columns(members, jaccard):
            if len(cluster) < 2:
                continue
            _merge_cluster(cluster, schema, out)
    return out


def _resolver(tables: list[Table]):
    """Resolve a table name by fqn, else by bare name (first table with that name wins)."""
    by_name = {t.fqn.lower(): t for t in tables}
    by_bare: dict[str, Table] = {}
    for table in tables:
        by_bare.setdefault(table.name.lower(), table)

    def resolve(name: str) -> Table | None:
        return by_name.get(name.lower()) or by_bare.get(name.split(".")[-1].lower())

    return resolve


def _add_shard(families: dict[str, Table], key: str, table: Table) -> None:
    """Fold a date-sharded table into its ``_*`` family table. Mutates ``families``."""
    family = families.get(key)
    if family is None:
        family = table.model_copy(deep=True)
        family.name = key.split(".")[-1]
        stem = family.name[:-2]
        family.properties = {
            "family": "true",
            "members": table.fqn,
            "business_name": f"{table.schema_name or ''} {stem}".strip(),
        }
        families[key] = family
    else:
        family.properties["members"] += "," + table.fqn
        for column in table.columns:
            if family.column(column.name) is None:
                family.columns.append(column)


def _load_families(
    base: Path,
    cfg: Spider2Config,
    source: str,
    snap: SchemaSnapshot,
) -> dict[str, Table]:
    """Load every table JSON (sorted by path), collapsing date shards when configured.

    Args:
        base: Database folder.
        cfg: Connector config.
        source: Snapshot source name stamped on the tables.
        snap: Snapshot that receives a warning per unreadable file.

    Returns:
        Tables by canonical (lowercased) key.
    """
    json_files = sorted(p for p in base.rglob("*.json") if p.name != "DDL.json")
    families: dict[str, Table] = {}
    for json_file in json_files:
        try:
            table = load_table_json(json_file, cfg, source)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            snap.warnings.append(f"{json_file.name}: {e}")
            continue
        key = canonical_table(table.fqn) if cfg.collapse_shards else table.fqn.lower()
        if key.endswith("_*") and cfg.collapse_shards:
            _add_shard(families, key, table)
        else:
            families[key] = table
    return families


def _describe_families(tables: list[Table]) -> None:
    """Append the member count and first/last member to each family's description."""
    for table in tables:
        if table.properties.get("family"):
            members = table.properties["members"].split(",")
            first = members[0].split(".")[-1]
            last = members[-1].split(".")[-1]
            table.description = (table.description or "") + (
                f" (date-sharded family of {len(members)} tables, {first} … {last})"
            )
            table.description = table.description.strip()


def _apply_ddl_csv(ddl_csv: Path, cfg: Spider2Config, source: str, snap: SchemaSnapshot) -> None:
    """Merge the keys, missing data types and foreign keys declared in one ``DDL.csv``.

    Mutates ``snap``: its tables gain primary keys and data types, resolved foreign keys are
    appended to its edges, an unreadable file becomes a warning.
    """
    try:
        with ddl_csv.open(encoding="utf-8", newline="") as fh:
            statements = [row.get("DDL", "") for row in csv.DictReader(fh)]
    except (OSError, csv.Error) as e:
        snap.warnings.append(f"{ddl_csv}: {e}")
        return
    ddl_text = ";\n".join(s.strip().rstrip(";") for s in statements if s and s.strip())
    dialect = _DDL_DIALECTS[cfg.dialect]
    parsed = parse_ddl(DDLConfig(ddl=ddl_text, dialect=dialect), source)
    resolve = _resolver(snap.tables)
    for parsed_table in parsed.tables:
        table = resolve(parsed_table.fqn)
        if table is None:
            continue
        for pk in parsed_table.primary_key:
            if pk not in table.primary_key:
                table.primary_key.append(pk)
            column = table.column(pk)
            if column:
                column.is_primary_key = True
        for parsed_column in parsed_table.columns:
            column = table.column(parsed_column.name)
            if column and not column.data_type:
                column.data_type = parsed_column.data_type
    for edge in parsed.edges:
        from_table, to_table = resolve(edge.from_table), resolve(edge.to_table)
        if from_table and to_table:
            edge.from_table, edge.to_table = from_table.fqn, to_table.fqn
            snap.edges.append(edge)


def introspect_spider2(cfg: Spider2Config, source: str = "spider2") -> SchemaSnapshot:
    """Read one Spider2 database folder into a stamped snapshot.

    A missing folder returns an unstamped snapshot with a warning. ``DDL.csv`` files are
    applied in filesystem order (unsorted).
    """
    snap = SchemaSnapshot(source=source, source_type="spider2")
    base = Path(cfg.path) if cfg.path else Path(cfg.root) / cfg.dialect / cfg.db
    if not base.exists():
        snap.warnings.append(f"missing {base}")
        return snap
    families = _load_families(base, cfg, source, snap)
    if cfg.collapse_families:
        families = _collapse_by_signature(families, cfg.family_jaccard)
    snap.tables = list(families.values())
    _describe_families(snap.tables)

    # PK/FK from DDL.csv (SQLite dbs declare them; cloud DDL rarely does)
    for ddl_csv in base.rglob("DDL.csv"):
        _apply_ddl_csv(ddl_csv, cfg, source, snap)
    return snap.stamp()
