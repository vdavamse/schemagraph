"""dbt connector: lineage DAG + relationships tests + descriptions from a dbt project.

Two input modes:

* ``manifest_path`` – a compiled ``target/manifest.json`` (preferred, exact lineage).
* ``project_dir``   – parse ``models/**/*.yml`` and ``models/**/*.sql`` directly,
  recovering ``ref()`` / ``source()`` calls by regex. No dbt install needed.

Edges produced:

* ``lineage``           – upstream -> downstream for every ref()/source().
* ``relationship_test`` – dbt ``relationships`` test (a foreign-key equivalent).
* ``foreign_key``       – dbt model ``constraints: [{type: foreign_key, ...}]``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, model_validator

from schemagraph.connectors.base import register
from schemagraph.model import BusinessTerm, Column, Edge, SchemaSnapshot, Table

# ``{{ ref('model') }}`` / ``{{ ref('package', 'model') }}`` in model SQL.
_REF_RE = re.compile(
    r"""\{\{\s*ref\(\s*['"]([^'"]+)['"]\s*"""
    r"""(?:,\s*['"]([^'"]+)['"]\s*)?\)\s*\}\}"""
)
# ``{{ source('source_name', 'table') }}`` in model SQL.
_SOURCE_RE = re.compile(
    r"""\{\{\s*source\(\s*['"]([^'"]+)['"]\s*,"""
    r"""\s*['"]([^'"]+)['"]\s*\)\s*\}\}"""
)
# A bare ``ref(...)`` / ``source(...)`` expression, e.g. a test's ``to:`` argument.
_REF_ARG_RE = re.compile(
    r"""(?:ref|source)\(\s*['"]([^'"]+)['"]"""
    r"""(?:\s*,\s*['"]([^'"]+)['"])?\s*\)"""
)


# Resource type of a manifest node -> Table.kind (anything else becomes "table").
_NODE_KINDS: dict[str, str] = {
    "model": "model",
    "source": "source",
    "seed": "seed",
    "snapshot": "snapshot",
}
# Manifest resource types that become tables.
_TABLE_RESOURCE_TYPES: frozenset[str] = frozenset({"model", "seed", "snapshot", "source"})
# Manifest resource types a ``ref()`` can point at.
_REF_RESOURCE_TYPES: frozenset[str] = frozenset({"model", "seed", "snapshot"})
# An ``accepted_values`` test contributes at most this many sample values.
MAX_ACCEPTED_VALUES = 20
# Cap on the column names guessed from a model's final SELECT.
MAX_GUESSED_COLUMNS = 200

# A relationship waiting for every table: (from table key, from column, to expr, to field).
_PendingRel = tuple[str, str, str, str]


class DbtConfig(BaseModel):
    """A dbt project: a compiled manifest.json, or a project directory parsed without dbt."""

    project_dir: str | None = Field(
        default=None,
        description="dbt project root (parsed without dbt)",
    )
    manifest_path: str | None = Field(default=None, description="Path to target/manifest.json")
    default_schema: str | None = Field(
        default=None,
        description="Schema for models without a config",
    )
    catalog: str | None = None

    @model_validator(mode="after")
    def _one_of(self):
        """Require at least one of ``project_dir`` and ``manifest_path``."""
        if not self.project_dir and not self.manifest_path:
            raise ValueError("either project_dir or manifest_path is required")
        return self


# ----------------------------------------------------------------------------- manifest mode


def _node_table(node: dict[str, Any], cfg: DbtConfig, source: str) -> Table:
    """Build a table from a manifest model/seed/snapshot/source node."""
    columns = []
    for column_name, spec in (node.get("columns") or {}).items():
        columns.append(
            Column(
                name=spec.get("name", column_name),
                data_type=spec.get("data_type"),
                description=spec.get("description") or None,
                tags=list(spec.get("tags") or []),
            )
        )
    schema = node.get("schema") or cfg.default_schema
    return Table(
        name=node.get("alias") or node.get("identifier") or node.get("name"),
        schema=schema,
        catalog=cfg.catalog,
        kind=_NODE_KINDS.get(node.get("resource_type"), "table"),
        description=node.get("description") or None,
        columns=columns,
        tags=list(node.get("tags") or []),
        properties={"unique_id": node.get("unique_id", ""), "dbt_name": node.get("name", "")},
        source=source,
    )


def _manifest_nodes(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """All manifest nodes by unique_id: ``nodes`` first, then ``sources``."""
    nodes: dict[str, dict[str, Any]] = {}
    nodes.update(data.get("nodes") or {})
    nodes.update(data.get("sources") or {})
    return nodes


def _manifest_tables(
    nodes: dict[str, dict[str, Any]],
    cfg: DbtConfig,
    source: str,
) -> dict[str, Table]:
    """Tables for every table-like node, by unique_id, in manifest order."""
    uid_to_table: dict[str, Table] = {}
    for uid, node in nodes.items():
        if node.get("resource_type") not in _TABLE_RESOURCE_TYPES:
            continue
        uid_to_table[uid] = _node_table(node, cfg, source)
    return uid_to_table


def _node_edges(
    uid: str,
    node: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    uid_to_table: dict[str, Table],
    source: str,
) -> list[Edge]:
    """Lineage, model-level FK constraints, then column-level FK constraints of one node.

    Args:
        uid: The node's unique_id (must be in ``uid_to_table``).
        node: The manifest node.
        nodes: Every manifest node, for resolving ``to:`` expressions.
        uid_to_table: Tables by unique_id.
        source: Snapshot source name stamped on the edges.

    Returns:
        The node's edges in that order.
    """
    table = uid_to_table[uid]
    edges: list[Edge] = []
    for dep in (node.get("depends_on") or {}).get("nodes") or []:
        if dep in uid_to_table:
            edges.append(
                Edge(
                    kind="lineage",
                    from_table=uid_to_table[dep].fqn,
                    to_table=table.fqn,
                    source=source,
                )
            )
    # model-level constraints (dbt >= 1.5)
    for cons in node.get("constraints") or []:
        if cons.get("type") == "foreign_key" and cons.get("to"):
            to_uid = _resolve_ref_expr(cons["to"], nodes)
            if to_uid in uid_to_table:
                edges.append(
                    Edge(
                        kind="foreign_key",
                        from_table=table.fqn,
                        to_table=uid_to_table[to_uid].fqn,
                        from_columns=list(cons.get("columns") or []),
                        to_columns=list(cons.get("to_columns") or []),
                        source=source,
                    )
                )
    for column_name, spec in (node.get("columns") or {}).items():
        for cons in spec.get("constraints") or []:
            if cons.get("type") == "foreign_key" and cons.get("to"):
                to_uid = _resolve_ref_expr(cons["to"], nodes)
                if to_uid in uid_to_table:
                    edges.append(
                        Edge(
                            kind="foreign_key",
                            from_table=table.fqn,
                            to_table=uid_to_table[to_uid].fqn,
                            from_columns=[column_name],
                            to_columns=list(cons.get("to_columns") or []),
                            source=source,
                        )
                    )
    return edges


def _apply_column_primary_keys(table: Table, node: dict[str, Any]) -> None:
    """Mark columns with a ``primary_key`` constraint. Mutates ``table`` in place."""
    for column_name, spec in (node.get("columns") or {}).items():
        for cons in spec.get("constraints") or []:
            if cons.get("type") == "primary_key":
                table.primary_key.append(column_name)
                column = table.column(column_name)
                if column:
                    column.is_primary_key = True


def _relationship_test_edges(
    data: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    uid_to_table: dict[str, Table],
    source: str,
) -> list[Edge]:
    """Edges from dbt ``relationships`` tests (a foreign-key equivalent)."""
    edges: list[Edge] = []
    for node in (data.get("nodes") or {}).values():
        if node.get("resource_type") != "test":
            continue
        meta = node.get("test_metadata") or {}
        if meta.get("name") != "relationships":
            continue
        kwargs = meta.get("kwargs") or {}
        attached = node.get("attached_node")
        deps = (node.get("depends_on") or {}).get("nodes") or []
        to_uid = _resolve_ref_expr(str(kwargs.get("to", "")), nodes)
        if attached not in uid_to_table:
            # fall back: the attached model is the dep that isn't the target
            others = [d for d in deps if d != to_uid and d in uid_to_table]
            attached = others[0] if others else None
        if attached and to_uid in uid_to_table:
            edges.append(
                Edge(
                    kind="relationship_test",
                    from_table=uid_to_table[attached].fqn,
                    to_table=uid_to_table[to_uid].fqn,
                    from_columns=[kwargs.get("column_name")] if kwargs.get("column_name") else [],
                    to_columns=[kwargs.get("field")] if kwargs.get("field") else [],
                    source=source,
                )
            )
    return edges


def _apply_unique_tests(data: dict[str, Any], uid_to_table: dict[str, Table]) -> None:
    """Treat ``unique`` tests as primary-key hints. Mutates the tables in place."""
    for node in (data.get("nodes") or {}).values():
        is_unique_test = (
            node.get("resource_type") == "test"
            and (node.get("test_metadata") or {}).get("name") == "unique"
        )
        if not is_unique_test:
            continue
        attached = node.get("attached_node")
        column_name = (node.get("test_metadata") or {}).get("kwargs", {}).get("column_name")
        if attached in uid_to_table and column_name:
            table = uid_to_table[attached]
            if column_name not in table.primary_key:
                table.primary_key.append(column_name)
            column = table.column(column_name)
            if column:
                column.is_primary_key = True


def _semantic_terms(
    data: dict[str, Any],
    uid_to_table: dict[str, Table],
    source: str,
) -> list[BusinessTerm]:
    """Glossary terms from semantic-model entities and measures (dbt semantic layer)."""
    terms: list[BusinessTerm] = []
    for semantic_model in (data.get("semantic_models") or {}).values():
        model_uid = (semantic_model.get("depends_on") or {}).get("nodes", [None])[0]
        if model_uid not in uid_to_table:
            continue
        model_fqn = uid_to_table[model_uid].fqn
        model_name = semantic_model.get("name")
        for entity in semantic_model.get("entities") or []:
            terms.append(
                BusinessTerm(
                    name=entity.get("name", ""),
                    description=f"{entity.get('type', '')} entity of semantic model {model_name}",
                    targets=[f"{model_fqn}.{entity.get('expr') or entity.get('name')}"],
                    source=source,
                )
            )
        for measure in semantic_model.get("measures") or []:
            terms.append(
                BusinessTerm(
                    name=measure.get("name", ""),
                    description=measure.get("description") or f"{measure.get('agg')} measure",
                    targets=[f"{model_fqn}.{measure.get('expr') or measure.get('name')}"],
                    source=source,
                )
            )
    return terms


def parse_manifest(cfg: DbtConfig, source: str) -> SchemaSnapshot:
    """Read a compiled ``manifest.json`` into a stamped snapshot (exact lineage)."""
    snap = SchemaSnapshot(source=source, source_type="dbt")
    data = json.loads(Path(cfg.manifest_path).read_text(encoding="utf-8"))
    nodes = _manifest_nodes(data)
    uid_to_table = _manifest_tables(nodes, cfg, source)
    snap.tables.extend(uid_to_table.values())

    for uid, node in nodes.items():
        if uid not in uid_to_table:
            continue
        snap.edges.extend(_node_edges(uid, node, nodes, uid_to_table, source))
        _apply_column_primary_keys(uid_to_table[uid], node)

    snap.edges.extend(_relationship_test_edges(data, nodes, uid_to_table, source))
    _apply_unique_tests(data, uid_to_table)
    snap.terms.extend(_semantic_terms(data, uid_to_table, source))
    return snap.stamp()


def _resolve_ref_expr(expr: str, nodes: dict[str, dict[str, Any]]) -> str | None:
    """Resolve "ref('x')" / "source('a','b')" / a unique_id to a node unique_id."""
    if expr in nodes:
        return expr
    match = _REF_ARG_RE.search(expr)
    if not match:
        return None
    a, b = match.group(1), match.group(2)
    if expr.strip().startswith("source"):
        for uid, node in nodes.items():
            is_source = node.get("resource_type") == "source"
            if is_source and node.get("source_name") == a and node.get("name") == b:
                return uid
        return None
    name = b or a
    for uid, node in nodes.items():
        if node.get("resource_type") in _REF_RESOURCE_TYPES and node.get("name") == name:
            return uid
    return None


# ----------------------------------------------------------------------------- project mode


def _discover_files(root: Path, model_paths: list[str]) -> tuple[list[Path], list[Path]]:
    """(yml files, sql files) under the project's model paths, unsorted."""
    yml_files: list[Path] = []
    sql_files: list[Path] = []
    for model_path in model_paths:
        base = root / model_path
        if not base.exists():
            continue
        yml_files += list(base.rglob("*.yml")) + list(base.rglob("*.yaml"))
        sql_files += list(base.rglob("*.sql"))
    return yml_files, sql_files


def _load_yaml(path: Path, snap: SchemaSnapshot) -> dict[str, Any] | None:
    """Load one YAML file; an invalid file becomes a snapshot warning and returns None."""
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        snap.warnings.append(f"{path}: {e}")
        return None


def _source_tables(
    doc: dict[str, Any],
    cfg: DbtConfig,
    source: str,
    tables_by_key: dict[str, Table],
    pending_rels: list[_PendingRel],
) -> None:
    """Add the ``sources:`` tables of one YAML doc.

    Args:
        doc: Parsed YAML document.
        cfg: dbt config (catalog).
        source: Snapshot source name stamped on the tables.
        tables_by_key: Tables by ``source:<src>.<name>`` key; mutated in place.
        pending_rels: Relationship tests / FK constraints are appended here.
    """
    for src in doc.get("sources") or []:
        source_schema = src.get("schema") or src.get("name")
        for spec in src.get("tables") or []:
            key = f"source:{src['name']}.{spec['name']}"
            table = Table(
                name=spec.get("identifier") or spec["name"],
                schema=source_schema,
                catalog=cfg.catalog,
                kind="source",
                description=spec.get("description") or src.get("description") or None,
                properties={"dbt_name": spec["name"], "source_name": src["name"]},
                source=source,
            )
            _apply_columns(table, spec.get("columns") or [], key, pending_rels)
            tables_by_key[key] = table


def _model_tables(
    doc: dict[str, Any],
    cfg: DbtConfig,
    default_schema: str,
    source: str,
    tables_by_key: dict[str, Table],
    pending_rels: list[_PendingRel],
) -> None:
    """Add or update the ``models:`` tables of one YAML doc.

    Args:
        doc: Parsed YAML document.
        cfg: dbt config (catalog).
        default_schema: Schema for models without a ``config.schema``.
        source: Snapshot source name stamped on new tables.
        tables_by_key: Tables by ``model:<name>`` key; mutated in place.
        pending_rels: Relationship tests / FK constraints are appended here.
    """
    for spec in doc.get("models") or []:
        key = f"model:{spec['name']}"
        table = tables_by_key.get(key) or Table(
            name=spec.get("alias") or spec["name"],
            schema=(spec.get("config") or {}).get("schema") or default_schema,
            catalog=cfg.catalog,
            kind="model",
            properties={"dbt_name": spec["name"]},
            source=source,
        )
        if spec.get("description"):
            table.description = spec["description"]
        _apply_columns(table, spec.get("columns") or [], key, pending_rels)
        tables_by_key[key] = table


def _apply_sql_model(
    sql_file: Path,
    root: Path,
    cfg: DbtConfig,
    default_schema: str,
    source: str,
    tables_by_key: dict[str, Table],
    refs_by_key: dict[str, list[str]],
) -> None:
    """Ensure a model exists for a ``.sql`` file and record its ``ref()``/``source()`` calls.

    Args:
        sql_file: The model's SQL file; its stem is the model name.
        root: Project root (``sql_path`` is stored relative to it).
        cfg: dbt config (catalog).
        default_schema: Schema for a model the YAML did not declare.
        source: Snapshot source name stamped on a new table.
        tables_by_key: Tables by key; mutated in place.
        refs_by_key: Upstream keys (refs, then sources) appended per model key.
    """
    name = sql_file.stem
    key = f"model:{name}"
    text = sql_file.read_text(encoding="utf-8", errors="replace")
    table = tables_by_key.get(key)
    if table is None:
        table = Table(
            name=name,
            schema=default_schema,
            catalog=cfg.catalog,
            kind="model",
            properties={"dbt_name": name},
            source=source,
        )
        tables_by_key[key] = table
    schema_match = re.search(r"""config\([^)]*schema\s*=\s*['"]([^'"]+)['"]""", text)
    if schema_match and not table.schema_name:
        table.schema_name = schema_match.group(1)
    table.properties["sql_path"] = str(sql_file.relative_to(root))
    refs = refs_by_key.setdefault(key, [])
    for ref in _REF_RE.finditer(text):
        dep_name = ref.group(2) or ref.group(1)
        refs.append(f"model:{dep_name}")
    for source_match in _SOURCE_RE.finditer(text):
        refs.append(f"source:{source_match.group(1)}.{source_match.group(2)}")
    # columns from SQL if the yml had none: best-effort final SELECT aliases
    if not table.columns:
        for column_name in _guess_select_columns(text):
            table.columns.append(Column(name=column_name))


def _lineage_edges(
    tables_by_key: dict[str, Table],
    refs_by_key: dict[str, list[str]],
    snap: SchemaSnapshot,
    source: str,
) -> list[Edge]:
    """Lineage edges upstream -> model; an unknown upstream becomes a snapshot warning."""
    edges: list[Edge] = []
    for key, table in tables_by_key.items():
        for ref in dict.fromkeys(refs_by_key.get(key, [])):
            upstream = tables_by_key.get(ref)
            if upstream is None:
                snap.warnings.append(f"{key} references unknown {ref}")
                continue
            edges.append(
                Edge(kind="lineage", from_table=upstream.fqn, to_table=table.fqn, source=source)
            )
    return edges


def _ref_key(to_expr: str) -> str | None:
    """Table key (``model:<name>`` / ``source:<src>.<name>``) of a ref/source expression."""
    match = _REF_ARG_RE.search(to_expr)
    if not match:
        return None
    if to_expr.strip().startswith("source"):
        return f"source:{match.group(1)}.{match.group(2)}"
    return f"model:{match.group(2) or match.group(1)}"


def _relationship_edges(
    pending_rels: list[_PendingRel],
    tables_by_key: dict[str, Table],
    source: str,
) -> list[Edge]:
    """Edges for the queued relationship tests / FK constraints whose endpoints exist."""
    edges: list[Edge] = []
    for from_key, from_col, to_expr, to_field in pending_rels:
        to_key = _ref_key(to_expr)
        if to_key is None:
            continue
        from_table, to_table = tables_by_key.get(from_key), tables_by_key.get(to_key)
        if from_table and to_table:
            edges.append(
                Edge(
                    kind="relationship_test",
                    from_table=from_table.fqn,
                    to_table=to_table.fqn,
                    from_columns=[from_col],
                    to_columns=[to_field] if to_field else [],
                    source=source,
                )
            )
    return edges


def parse_project(cfg: DbtConfig, source: str) -> SchemaSnapshot:
    """Parse a dbt project directory (YAML + SQL, no dbt install) into a stamped snapshot.

    YAML files are read first (sources, then models, per file in sorted order), then every
    model's SQL for ``ref()``/``source()`` lineage and a best-effort column list.
    """
    snap = SchemaSnapshot(source=source, source_type="dbt")
    root = Path(cfg.project_dir)
    project_file = root / "dbt_project.yml"
    project = (
        yaml.safe_load(project_file.read_text(encoding="utf-8")) if project_file.exists() else {}
    )
    model_paths = project.get("model-paths") or ["models"]
    default_schema = cfg.default_schema or _profile_schema(root, project) or "main"

    tables_by_key: dict[str, Table] = {}  # key = "model:<name>" or "source:<src>.<name>"
    pending_rels: list[_PendingRel] = []
    refs_by_key: dict[str, list[str]] = {}  # model key -> upstream keys from its SQL
    yml_files, sql_files = _discover_files(root, model_paths)

    # 1. sources.yml / schema.yml
    for yml_file in sorted(yml_files):
        doc = _load_yaml(yml_file, snap)
        if doc is None:
            continue
        _source_tables(doc, cfg, source, tables_by_key, pending_rels)
        _model_tables(doc, cfg, default_schema, source, tables_by_key, pending_rels)

    # 2. models/**/*.sql -> ensure model exists, extract lineage
    for sql_file in sorted(sql_files):
        _apply_sql_model(sql_file, root, cfg, default_schema, source, tables_by_key, refs_by_key)

    snap.edges.extend(_lineage_edges(tables_by_key, refs_by_key, snap, source))
    snap.edges.extend(_relationship_edges(pending_rels, tables_by_key, source))
    snap.tables = list(tables_by_key.values())
    return snap.stamp()


def _apply_column_tests(
    column: Column,
    spec: dict[str, Any],
    key: str,
    pending_rels: list[_PendingRel],
) -> set[str]:
    """Read a YAML column's tests: queue relationships, take accepted values as samples.

    Args:
        column: The column; ``accepted_values`` replaces its ``sample_values``.
        spec: The YAML column entry.
        key: Owning table key.
        pending_rels: ``relationships`` tests are appended here.

    Returns:
        The names of every test on the column.
    """
    tests = spec.get("tests") or spec.get("data_tests") or []
    names = set()
    for test in tests:
        if isinstance(test, str):
            names.add(test)
        elif isinstance(test, dict):
            for test_name, test_args in test.items():
                names.add(test_name)
                if test_name == "relationships" and isinstance(test_args, dict):
                    to_expr = str(test_args.get("to", ""))
                    pending_rels.append((key, spec["name"], to_expr, test_args.get("field", "")))
                if test_name == "accepted_values" and isinstance(test_args, dict):
                    values = [str(v) for v in (test_args.get("values") or [])]
                    column.sample_values = values[:MAX_ACCEPTED_VALUES]
    return names


def _apply_column_constraints(
    table: Table,
    column: Column,
    spec: dict[str, Any],
    key: str,
    pending_rels: list[_PendingRel],
) -> None:
    """Apply a YAML column's ``constraints:`` (primary key, queued foreign keys).

    Args:
        table: Owning table; a primary key is appended to its ``primary_key``.
        column: The column.
        spec: The YAML column entry.
        key: Owning table key.
        pending_rels: Foreign-key constraints are appended here.
    """
    for cons in spec.get("constraints") or []:
        if cons.get("type") == "primary_key":
            column.is_primary_key = True
            if spec["name"] not in table.primary_key:
                table.primary_key.append(spec["name"])
        if cons.get("type") == "foreign_key" and cons.get("to"):
            to_field = (cons.get("to_columns") or [""])[0]
            pending_rels.append((key, spec["name"], str(cons["to"]), to_field))


def _apply_columns(
    table: Table,
    columns: list[dict[str, Any]],
    key: str,
    pending_rels: list[_PendingRel],
) -> None:
    """Merge YAML column entries into a table, creating missing columns.

    Args:
        table: The table; mutated in place.
        columns: YAML column entries.
        key: The table's key.
        pending_rels: Relationship tests and FK constraints are appended here.
    """
    for spec in columns:
        column = table.column(spec["name"]) or Column(name=spec["name"])
        if column not in table.columns:
            table.columns.append(column)
        if spec.get("description"):
            column.description = spec["description"]
        if spec.get("data_type"):
            column.data_type = spec["data_type"]
        column.tags = list(spec.get("tags") or column.tags)
        names = _apply_column_tests(column, spec, key, pending_rels)
        if "unique" in names and "not_null" in names:
            column.is_primary_key = True
            if spec["name"] not in table.primary_key:
                table.primary_key.append(spec["name"])
        _apply_column_constraints(table, column, spec, key, pending_rels)


def _profile_schema(root: Path, project: dict[str, Any]) -> str | None:
    """Target schema of the project's profile in ``profiles.yml``, if one sits in the root."""
    profiles_file = root / "profiles.yml"
    if not profiles_file.exists():
        return None
    try:
        doc = yaml.safe_load(profiles_file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    profile_name = project.get("profile") or next(iter(doc), None)
    profile = doc.get(profile_name) or {}
    target = profile.get("target")
    output = (profile.get("outputs") or {}).get(target) or {}
    return output.get("schema")


def _split_top_level(segment: str) -> list[str]:
    """Split a SELECT list on commas outside parentheses."""
    depth = 0
    part = ""
    parts = []
    for ch in segment:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(part)
            part = ""
        else:
            part += ch
    parts.append(part)
    return parts


def _guess_select_columns(sql: str) -> list[str]:
    """Guess a model's columns from its SQL (very rough).

    Takes the last SELECT ... FROM block and pulls ``expr AS alias`` / bare identifiers.
    """
    body = re.sub(r"\{\{.*?\}\}", "", sql, flags=re.S)
    body = re.sub(r"--.*", "", body)
    idx = body.lower().rfind("select")
    if idx < 0:
        return []
    segment = body[idx + 6 :]
    from_match = re.search(r"\bfrom\b", segment, flags=re.I)
    if from_match:
        segment = segment[: from_match.start()]
    if "*" in segment and "," not in segment:
        return []
    out: list[str] = []
    for part in _split_top_level(segment):
        part = part.strip()
        if not part:
            continue
        alias_match = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", part, flags=re.I)
        if alias_match:
            out.append(alias_match.group(1))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", part):
            out.append(part.split(".")[-1])
    return out[:MAX_GUESSED_COLUMNS]


@register
class DbtConnector:
    """Connector over a dbt manifest (preferred) or project directory."""

    type_name: ClassVar[str] = "dbt"
    Config = DbtConfig

    def __init__(self, name: str, config: DbtConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        """Report whether the manifest or ``dbt_project.yml`` exists."""
        if self.config.manifest_path:
            p = Path(self.config.manifest_path)
            return "ok: manifest found" if p.exists() else f"missing manifest {p}"
        p = Path(self.config.project_dir or "")
        if (p / "dbt_project.yml").exists():
            return "ok: project found"
        return f"no dbt_project.yml in {p}"

    def introspect(self) -> SchemaSnapshot:
        """Parse the manifest if configured, else the project directory."""
        if self.config.manifest_path:
            return parse_manifest(self.config, self.name)
        return parse_project(self.config, self.name)
