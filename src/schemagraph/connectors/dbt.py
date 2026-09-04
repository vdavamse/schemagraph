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

_REF_RE = re.compile(r"""\{\{\s*ref\(\s*['"]([^'"]+)['"]\s*(?:,\s*['"]([^'"]+)['"]\s*)?\)\s*\}\}""")
_SOURCE_RE = re.compile(r"""\{\{\s*source\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)\s*\}\}""")
_REF_ARG_RE = re.compile(r"""(?:ref|source)\(\s*['"]([^'"]+)['"](?:\s*,\s*['"]([^'"]+)['"])?\s*\)""")


class DbtConfig(BaseModel):
    project_dir: str | None = Field(default=None, description="dbt project root (parsed without dbt)")
    manifest_path: str | None = Field(default=None, description="Path to target/manifest.json")
    default_schema: str | None = Field(default=None, description="Schema for models without a config")
    catalog: str | None = None

    @model_validator(mode="after")
    def _one_of(self):
        if not self.project_dir and not self.manifest_path:
            raise ValueError("either project_dir or manifest_path is required")
        return self


# ----------------------------------------------------------------------------- manifest mode


def _node_table(node: dict[str, Any], cfg: DbtConfig, source: str) -> Table:
    kind_map = {"model": "model", "source": "source", "seed": "seed", "snapshot": "snapshot"}
    cols = []
    for cname, c in (node.get("columns") or {}).items():
        cols.append(
            Column(
                name=c.get("name", cname),
                data_type=c.get("data_type"),
                description=c.get("description") or None,
                tags=list(c.get("tags") or []),
            )
        )
    schema = node.get("schema") or cfg.default_schema
    return Table(
        name=node.get("alias") or node.get("identifier") or node.get("name"),
        schema=schema,
        catalog=cfg.catalog,
        kind=kind_map.get(node.get("resource_type"), "table"),
        description=node.get("description") or None,
        columns=cols,
        tags=list(node.get("tags") or []),
        properties={"unique_id": node.get("unique_id", ""), "dbt_name": node.get("name", "")},
        source=source,
    )


def parse_manifest(cfg: DbtConfig, source: str) -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="dbt")
    data = json.loads(Path(cfg.manifest_path).read_text(encoding="utf-8"))
    nodes: dict[str, dict[str, Any]] = {}
    nodes.update(data.get("nodes") or {})
    nodes.update(data.get("sources") or {})
    uid_to_table: dict[str, Table] = {}
    for uid, node in nodes.items():
        if node.get("resource_type") not in {"model", "seed", "snapshot", "source"}:
            continue
        t = _node_table(node, cfg, source)
        uid_to_table[uid] = t
        snap.tables.append(t)

    # lineage
    for uid, node in nodes.items():
        if uid not in uid_to_table:
            continue
        for dep in (node.get("depends_on") or {}).get("nodes") or []:
            if dep in uid_to_table:
                snap.edges.append(
                    Edge(kind="lineage", from_table=uid_to_table[dep].fqn, to_table=uid_to_table[uid].fqn, source=source)
                )
        # model-level constraints (dbt >= 1.5)
        for cons in node.get("constraints") or []:
            if cons.get("type") == "foreign_key" and cons.get("to"):
                to_uid = _resolve_ref_expr(cons["to"], nodes)
                if to_uid in uid_to_table:
                    snap.edges.append(
                        Edge(
                            kind="foreign_key",
                            from_table=uid_to_table[uid].fqn,
                            to_table=uid_to_table[to_uid].fqn,
                            from_columns=list(cons.get("columns") or []),
                            to_columns=list(cons.get("to_columns") or []),
                            source=source,
                        )
                    )
        for cname, c in (node.get("columns") or {}).items():
            for cons in c.get("constraints") or []:
                if cons.get("type") == "foreign_key" and cons.get("to"):
                    to_uid = _resolve_ref_expr(cons["to"], nodes)
                    if to_uid in uid_to_table:
                        snap.edges.append(
                            Edge(
                                kind="foreign_key",
                                from_table=uid_to_table[uid].fqn,
                                to_table=uid_to_table[to_uid].fqn,
                                from_columns=[cname],
                                to_columns=list(cons.get("to_columns") or []),
                                source=source,
                            )
                        )
                if cons.get("type") == "primary_key":
                    uid_to_table[uid].primary_key.append(cname)
                    col = uid_to_table[uid].column(cname)
                    if col:
                        col.is_primary_key = True

    # relationships tests
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
            snap.edges.append(
                Edge(
                    kind="relationship_test",
                    from_table=uid_to_table[attached].fqn,
                    to_table=uid_to_table[to_uid].fqn,
                    from_columns=[kwargs.get("column_name")] if kwargs.get("column_name") else [],
                    to_columns=[kwargs.get("field")] if kwargs.get("field") else [],
                    source=source,
                )
            )
        # unique + not_null tests -> PK hint
    for node in (data.get("nodes") or {}).values():
        if node.get("resource_type") == "test" and (node.get("test_metadata") or {}).get("name") == "unique":
            attached = node.get("attached_node")
            col = (node.get("test_metadata") or {}).get("kwargs", {}).get("column_name")
            if attached in uid_to_table and col:
                t = uid_to_table[attached]
                if col not in t.primary_key:
                    t.primary_key.append(col)
                c = t.column(col)
                if c:
                    c.is_primary_key = True

    # semantic models / metrics as glossary terms (dbt semantic layer)
    for sm in (data.get("semantic_models") or {}).values():
        model_uid = (sm.get("depends_on") or {}).get("nodes", [None])[0]
        if model_uid in uid_to_table:
            for ent in sm.get("entities") or []:
                snap.terms.append(
                    BusinessTerm(
                        name=ent.get("name", ""),
                        description=f"{ent.get('type', '')} entity of semantic model {sm.get('name')}",
                        targets=[f"{uid_to_table[model_uid].fqn}.{ent.get('expr') or ent.get('name')}"],
                        source=source,
                    )
                )
            for m in sm.get("measures") or []:
                snap.terms.append(
                    BusinessTerm(
                        name=m.get("name", ""),
                        description=m.get("description") or f"{m.get('agg')} measure",
                        targets=[f"{uid_to_table[model_uid].fqn}.{m.get('expr') or m.get('name')}"],
                        source=source,
                    )
                )
    return snap.stamp()


def _resolve_ref_expr(expr: str, nodes: dict[str, dict[str, Any]]) -> str | None:
    """Resolve "ref('x')" / "source('a','b')" / a unique_id to a node unique_id."""
    if expr in nodes:
        return expr
    m = _REF_ARG_RE.search(expr)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if expr.strip().startswith("source"):
        for uid, n in nodes.items():
            if n.get("resource_type") == "source" and n.get("source_name") == a and n.get("name") == b:
                return uid
        return None
    name = b or a
    for uid, n in nodes.items():
        if n.get("resource_type") in {"model", "seed", "snapshot"} and n.get("name") == name:
            return uid
    return None


# ----------------------------------------------------------------------------- project mode


def parse_project(cfg: DbtConfig, source: str) -> SchemaSnapshot:
    snap = SchemaSnapshot(source=source, source_type="dbt")
    root = Path(cfg.project_dir)
    project = yaml.safe_load((root / "dbt_project.yml").read_text(encoding="utf-8")) if (root / "dbt_project.yml").exists() else {}
    model_paths = project.get("model-paths") or ["models"]
    default_schema = cfg.default_schema or _profile_schema(root, project) or "main"

    tables_by_key: dict[str, Table] = {}  # key = "model:<name>" or "source:<src>.<name>"
    pending_rels: list[tuple[str, str, str, str]] = []  # (from_key, from_col, to_expr, to_field)

    yml_files: list[Path] = []
    sql_files: list[Path] = []
    for mp in model_paths:
        base = root / mp
        if not base.exists():
            continue
        yml_files += list(base.rglob("*.yml")) + list(base.rglob("*.yaml"))
        sql_files += list(base.rglob("*.sql"))

    # 1. sources.yml / schema.yml
    for yf in sorted(yml_files):
        try:
            doc = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            snap.warnings.append(f"{yf}: {e}")
            continue
        for src in doc.get("sources") or []:
            sschema = src.get("schema") or src.get("name")
            for st in src.get("tables") or []:
                key = f"source:{src['name']}.{st['name']}"
                t = Table(
                    name=st.get("identifier") or st["name"],
                    schema=sschema,
                    catalog=cfg.catalog,
                    kind="source",
                    description=st.get("description") or src.get("description") or None,
                    properties={"dbt_name": st["name"], "source_name": src["name"]},
                    source=source,
                )
                _apply_columns(t, st.get("columns") or [], key, pending_rels)
                tables_by_key[key] = t
        for m in doc.get("models") or []:
            key = f"model:{m['name']}"
            t = tables_by_key.get(key) or Table(
                name=m.get("alias") or m["name"],
                schema=(m.get("config") or {}).get("schema") or default_schema,
                catalog=cfg.catalog,
                kind="model",
                properties={"dbt_name": m["name"]},
                source=source,
            )
            if m.get("description"):
                t.description = m["description"]
            _apply_columns(t, m.get("columns") or [], key, pending_rels)
            tables_by_key[key] = t

    # 2. models/**/*.sql -> ensure model exists, extract lineage
    for sf in sorted(sql_files):
        name = sf.stem
        key = f"model:{name}"
        text = sf.read_text(encoding="utf-8", errors="replace")
        t = tables_by_key.get(key)
        if t is None:
            t = Table(name=name, schema=default_schema, catalog=cfg.catalog, kind="model", properties={"dbt_name": name}, source=source)
            tables_by_key[key] = t
        m_schema = re.search(r"""config\([^)]*schema\s*=\s*['"]([^'"]+)['"]""", text)
        if m_schema and not t.schema_name:
            t.schema_name = m_schema.group(1)
        t.properties["sql_path"] = str(sf.relative_to(root))
        for ref in _REF_RE.finditer(text):
            dep_name = ref.group(2) or ref.group(1)
            t.properties.setdefault("refs", "")
            t.properties["refs"] = ",".join(filter(None, [t.properties["refs"], f"model:{dep_name}"]))
        for srcm in _SOURCE_RE.finditer(text):
            t.properties["refs"] = ",".join(filter(None, [t.properties.get("refs", ""), f"source:{srcm.group(1)}.{srcm.group(2)}"]))
        # columns from SQL if the yml had none: best-effort final SELECT aliases
        if not t.columns:
            for col in _guess_select_columns(text):
                t.columns.append(Column(name=col))

    for key, t in tables_by_key.items():
        refs = [r for r in (t.properties.pop("refs", "") or "").split(",") if r]
        for r in dict.fromkeys(refs):
            up = tables_by_key.get(r)
            if up is None:
                snap.warnings.append(f"{key} references unknown {r}")
                continue
            snap.edges.append(Edge(kind="lineage", from_table=up.fqn, to_table=t.fqn, source=source))

    for from_key, from_col, to_expr, to_field in pending_rels:
        m = _REF_ARG_RE.search(to_expr)
        if not m:
            continue
        to_key = f"source:{m.group(1)}.{m.group(2)}" if to_expr.strip().startswith("source") else f"model:{m.group(2) or m.group(1)}"
        ft, tt = tables_by_key.get(from_key), tables_by_key.get(to_key)
        if ft and tt:
            snap.edges.append(
                Edge(
                    kind="relationship_test",
                    from_table=ft.fqn,
                    to_table=tt.fqn,
                    from_columns=[from_col],
                    to_columns=[to_field] if to_field else [],
                    source=source,
                )
            )
    snap.tables = list(tables_by_key.values())
    return snap.stamp()


def _apply_columns(t: Table, cols: list[dict[str, Any]], key: str, pending_rels: list) -> None:
    for c in cols:
        col = t.column(c["name"]) or Column(name=c["name"])
        if col not in t.columns:
            t.columns.append(col)
        if c.get("description"):
            col.description = c["description"]
        if c.get("data_type"):
            col.data_type = c["data_type"]
        col.tags = list(c.get("tags") or col.tags)
        tests = c.get("tests") or c.get("data_tests") or []
        names = set()
        for test in tests:
            if isinstance(test, str):
                names.add(test)
            elif isinstance(test, dict):
                for tname, targs in test.items():
                    names.add(tname)
                    if tname == "relationships" and isinstance(targs, dict):
                        pending_rels.append((key, c["name"], str(targs.get("to", "")), targs.get("field", "")))
                    if tname == "accepted_values" and isinstance(targs, dict):
                        col.sample_values = [str(v) for v in (targs.get("values") or [])][:20]
        if "unique" in names and "not_null" in names:
            col.is_primary_key = True
            if c["name"] not in t.primary_key:
                t.primary_key.append(c["name"])
        for cons in c.get("constraints") or []:
            if cons.get("type") == "primary_key":
                col.is_primary_key = True
                if c["name"] not in t.primary_key:
                    t.primary_key.append(c["name"])
            if cons.get("type") == "foreign_key" and cons.get("to"):
                pending_rels.append((key, c["name"], str(cons["to"]), (cons.get("to_columns") or [""])[0]))


def _profile_schema(root: Path, project: dict[str, Any]) -> str | None:
    prof = root / "profiles.yml"
    if not prof.exists():
        return None
    try:
        doc = yaml.safe_load(prof.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    pname = project.get("profile") or next(iter(doc), None)
    p = doc.get(pname) or {}
    target = p.get("target")
    out = (p.get("outputs") or {}).get(target) or {}
    return out.get("schema")


def _guess_select_columns(sql: str) -> list[str]:
    """Very rough: take the last SELECT ... FROM block and pull `expr AS alias` / bare identifiers."""
    body = re.sub(r"\{\{.*?\}\}", "", sql, flags=re.S)
    body = re.sub(r"--.*", "", body)
    idx = body.lower().rfind("select")
    if idx < 0:
        return []
    seg = body[idx + 6 :]
    fidx = re.search(r"\bfrom\b", seg, flags=re.I)
    if fidx:
        seg = seg[: fidx.start()]
    if "*" in seg and "," not in seg:
        return []
    out: list[str] = []
    depth = 0
    part = ""
    parts = []
    for ch in seg:
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
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", p, flags=re.I)
        if m:
            out.append(m.group(1))
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", p):
            out.append(p.split(".")[-1])
    return out[:200]


@register
class DbtConnector:
    type_name: ClassVar[str] = "dbt"
    Config = DbtConfig

    def __init__(self, name: str, config: DbtConfig):
        self.name = name
        self.config = config

    def check(self) -> str:
        if self.config.manifest_path:
            p = Path(self.config.manifest_path)
            return "ok: manifest found" if p.exists() else f"missing manifest {p}"
        p = Path(self.config.project_dir or "")
        return "ok: project found" if (p / "dbt_project.yml").exists() else f"no dbt_project.yml in {p}"

    def introspect(self) -> SchemaSnapshot:
        if self.config.manifest_path:
            return parse_manifest(self.config, self.name)
        return parse_project(self.config, self.name)
