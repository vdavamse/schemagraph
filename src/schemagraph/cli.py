"""Command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from schemagraph.engine import Engine

app = typer.Typer(
    help="schemagraph: graph-native schema context engine for text-to-SQL.",
    no_args_is_help=True,
)
HomeOpt = Annotated[
    Path | None,
    typer.Option("--home", "-H", help="Data directory (default .schemagraph or $SCHEMAGRAPH_HOME)"),
]
OptOpt = Annotated[
    list[str] | None,
    typer.Option("--opt", help="extra LinkOptions as key=value (repeatable)"),
]


def _parse_opt(v: str):
    """Parse an ``--opt`` value: JSON when it parses, else the raw string.

    Numbers, booleans and lists come back typed; ``agg=top3`` gives the string ``"top3"``.
    """
    try:
        return json.loads(v)
    except ValueError:
        return v


def _parse_opts(opt: list[str] | None) -> dict:
    """Turn repeated ``--opt key=value`` flags into LinkOptions keyword arguments."""
    extra: dict = {}
    for kv in opt or []:
        k, v = kv.split("=", 1)
        extra[k] = _parse_opt(v)
    return extra


def _engine(home: Path | None) -> Engine:
    """Configure INFO logging and open the Engine under ``home``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return Engine(home)


@app.command()
def add_ddl(
    file: Annotated[Path, typer.Argument(help="SQL file with CREATE TABLE statements")],
    name: Annotated[str, typer.Option("--name", "-n")] = "ddl",
    dialect: Annotated[str | None, typer.Option(help="sqlglot dialect")] = None,
    schema: Annotated[
        str | None, typer.Option(help="default schema for unqualified tables")
    ] = None,
    home: HomeOpt = None,
):
    """Register a DDL file as a connection and build it."""
    eng = _engine(home)
    cfg = {"ddl": file.read_text(encoding="utf-8"), "dialect": dialect, "default_schema": schema}
    snap = eng.add_connection(name, "ddl", cfg)
    warnings = f", warnings: {snap.warnings}" if snap.warnings else ""
    typer.echo(f"{name}: {len(snap.tables)} tables, {len(snap.edges)} edges" + warnings)


@app.command()
def add_dbt(
    project_dir: Annotated[Path | None, typer.Option(help="dbt project root")] = None,
    manifest: Annotated[Path | None, typer.Option(help="target/manifest.json")] = None,
    name: Annotated[str, typer.Option("--name", "-n")] = "dbt",
    schema: Annotated[str | None, typer.Option(help="default schema")] = None,
    home: HomeOpt = None,
):
    """Register a dbt project (parsed without dbt) or a compiled manifest."""
    eng = _engine(home)
    cfg = {
        "project_dir": str(project_dir) if project_dir else None,
        "manifest_path": str(manifest) if manifest else None,
        "default_schema": schema,
    }
    snap = eng.add_connection(name, "dbt", cfg)
    counts = f"{len(snap.tables)} tables, {len(snap.edges)} edges, {len(snap.terms)} terms"
    warnings = f", warnings: {len(snap.warnings)}" if snap.warnings else ""
    typer.echo(f"{name}: {counts}" + warnings)


@app.command()
def add_duckdb(
    path: Path,
    name: Annotated[str, typer.Option("--name", "-n")] = "duckdb",
    home: HomeOpt = None,
):
    """Register a DuckDB database file."""
    eng = _engine(home)
    snap = eng.add_connection(name, "duckdb", {"path": str(path)})
    typer.echo(f"{name}: {len(snap.tables)} tables, {len(snap.edges)} edges")


@app.command()
def add(
    type_name: Annotated[
        str,
        typer.Argument(
            help="connector type: unity_catalog | aws_glue | collibra | ddl | dbt | duckdb"
        ),
    ],
    name: Annotated[str, typer.Argument()],
    config: Annotated[
        str,
        typer.Option(
            "--config", "-c", help="JSON config or @file.json; ${ENV} references allowed"
        ),
    ],
    no_build: bool = False,
    priority: Annotated[
        int | None,
        typer.Option(
            help=(
                "merge order: lower merges first and wins conflicting fields (default: by source "
                "type, user > collibra > dbt > unity_catalog > duckdb > ddl > aws_glue); omitted "
                "on re-register keeps the stored one"
            )
        ),
    ] = None,
    clear_priority: Annotated[
        bool, typer.Option(help="reset a stored priority to the source-type order")
    ] = False,
    home: HomeOpt = None,
):
    """Register any connector from a JSON config."""
    eng = _engine(home)
    if config.startswith("@"):
        cfg = json.loads(Path(config[1:]).read_text(encoding="utf-8"))
    else:
        cfg = json.loads(config)
    snap = eng.add_connection(
        name,
        type_name,
        cfg,
        build=not no_build,
        priority=priority,
        clear_priority=clear_priority,
    )
    if snap:
        typer.echo(
            f"{name}: {len(snap.tables)} tables, {len(snap.edges)} edges, {len(snap.terms)} terms"
        )
    else:
        typer.echo(f"{name}: registered (not built)")


@app.command()
def build(name: Annotated[str | None, typer.Argument()] = None, home: HomeOpt = None):
    """(Re)introspect one connection or all."""
    eng = _engine(home)
    res = eng.build(name)
    snaps = res if isinstance(res, list) else [res]
    for snap in snaps:
        typer.echo(
            f"{snap.source}: {len(snap.tables)} tables, {len(snap.edges)} edges, "
            f"{len(snap.terms)} terms"
        )
    typer.echo(json.dumps(eng.stats()))


@app.command()
def connections(home: HomeOpt = None):
    """List connections."""
    eng = _engine(home)
    for connection in eng.connections():
        typer.echo(
            f"{connection['name']:<20} {connection['type']:<14} "
            f"tables={connection['n_tables']:<5} edges={connection['n_edges']:<5} "
            f"built={connection['built_at'] or '-'}"
        )


@app.command()
def remove(name: str, home: HomeOpt = None):
    """Remove a connection and its snapshot."""
    _engine(home).remove_connection(name)
    typer.echo(f"removed {name}")


@app.command()
def link(
    question: str,
    max_tables: int = 20,
    columns: str = "relevant",
    llm: bool = typer.Option(False, "--llm", help="use Claude to pick anchor tables"),
    as_json: bool = typer.Option(False, "--json"),
    home: HomeOpt = None,
):
    """Link a question to a sub-schema and print annotated DDL."""
    eng = _engine(home)
    result = eng.link(question, max_tables=max_tables, columns=columns, use_llm=llm)
    typer.echo(result.model_dump_json(indent=2) if as_json else result.ddl)


@app.command()
def explain(question: str, home: HomeOpt = None):
    """Show activation seeds and PPR scores for a question."""
    typer.echo(json.dumps(_engine(home).explain(question), indent=2, default=str))


@app.command()
def path(a: str, b: str, home: HomeOpt = None):
    """Shortest join path(s) between two tables."""  # noqa: D402
    for join_path in _engine(home).join_path(a, b):
        typer.echo(" -> ".join(join_path))


@app.command()
def stats(home: HomeOpt = None):
    """Print graph statistics as JSON."""
    typer.echo(json.dumps(_engine(home).stats(), indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, home: HomeOpt = None):
    """Run the HTTP API + frontend."""
    import uvicorn

    from schemagraph.api.app import create_app

    uvicorn.run(create_app(_engine(home)), host=host, port=port)


@app.command()
def bench_spider1(
    tables_json: Annotated[Path, typer.Argument(help="Spider-format tables.json (declared FKs)")],
    questions_json: Annotated[
        Path, typer.Argument(help="questions [{db_id, question, query}]")
    ],
    max_tables: int = 4,
    anchor_k: int = 2,
    infer: bool = typer.Option(
        False, help="add name-based inferred edges on top of the declared FKs"
    ),
    limit: int | None = None,
    dialect: str = "sqlite",
    out: Path = Path("bench_results"),
    tag: str | None = typer.Option(None, help="output file tag"),
    opt: OptOpt = None,
):
    """Gold-table recall over a Spider-format dataset with real foreign keys (bridge tables, path union and pruning are measurable here)."""  # noqa: E501
    from schemagraph.bench.spider1 import format_table, run

    res = run(
        tables_json,
        questions_json,
        max_tables=max_tables,
        anchor_k=anchor_k,
        infer=infer,
        limit=limit,
        dialect=dialect,
        out_dir=out,
        tag=tag,
        **_parse_opts(opt),
    )
    typer.echo(format_table(res["summary"]))
    typer.echo(f"skipped (unknown db or no gold table): {res['summary']['config']['skipped']}")


@app.command()
def bench_spider2_lite(
    spider2_root: Annotated[Path, typer.Argument(help="path to the xlang-ai/Spider2 clone")],
    max_tables: int = 20,
    anchor_k: int = 6,
    docs: bool = typer.Option(
        True, help="append the task's external-knowledge document to the question"
    ),
    doc_chars: int = 4000,
    infer: bool = typer.Option(
        True, help="add name-based inferred edges (catalogs declare no FKs)"
    ),
    families: bool = typer.Option(
        True,
        help=(
            "collapse partition families (tables differing only in digit runs with the same "
            "columns)"
        ),
    ),
    dialect: Annotated[
        list[str] | None, typer.Option(help="bigquery | snowflake | sqlite (repeatable)")
    ] = None,
    limit: int | None = None,
    min_db_tables: int = typer.Option(
        0, help="only tasks whose database has at least this many tables"
    ),
    llm: bool = typer.Option(False, "--llm", help="use Claude to pick anchor tables"),
    out: Path = Path("bench_results"),
    tag: str | None = typer.Option(None, help="output file tag"),
    opt: OptOpt = None,
    suite: str = typer.Option(
        "lite",
        help="lite (547 tasks, 3 dialects) | snow (547 tasks, all Snowflake, larger schemas)",
    ),
    desc_weight: float = typer.Option(
        0.35, help="index weight of a token found only in a table/column description"
    ),
    only: Annotated[
        list[str] | None, typer.Option(help="only these instance ids (repeatable)")
    ] = None,
):
    """Gold-table recall of link_schema over the 547 Spider 2.0-Lite (or -Snow) tasks (no execution, no credentials)."""  # noqa: E501
    from schemagraph.bench.spider2_lite import format_table, run

    picker = None
    if llm:
        from schemagraph.llm.anchors import ClaudeAnchorPicker

        picker = ClaudeAnchorPicker()

    def progress(i: int, n: int, row) -> None:
        if i % 25 == 0 or i == n:
            typer.echo(
                f"  {i}/{n}  last={row.instance_id} strict={row.strict} recall={row.recall:.2f}",
                err=True,
            )

    res = run(
        spider2_root,
        max_tables=max_tables,
        anchor_k=anchor_k,
        use_docs=docs,
        doc_chars=doc_chars,
        infer=infer,
        dialects=set(dialect) if dialect else None,
        limit=limit,
        min_db_tables=min_db_tables,
        collapse_families=families,
        use_llm=llm,
        llm=picker,
        out_dir=out,
        progress=progress,
        tag=tag,
        suite=suite,
        desc_weight=desc_weight,
        only=set(only) if only else None,
        **_parse_opts(opt),
    )
    typer.echo(format_table(res["summary"]))
    typer.echo(f"\nresults written to {out}/")


@app.command()
def mcp(home: HomeOpt = None, transport: str = "stdio"):
    """Run the MCP server (stdio by default)."""
    from schemagraph.mcp.server import create_server

    logging.basicConfig(level=logging.WARNING)
    create_server(Engine(home)).run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    app()
