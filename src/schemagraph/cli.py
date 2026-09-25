"""Command-line interface."""

from __future__ import annotations

import json
import logging
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

from schemagraph.agent.results import Strategy
from schemagraph.engine import Engine

if TYPE_CHECKING:
    from schemagraph.agent.results import AgentConfig, AnswerResult

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
# Result rows and characters per cell ``ask`` prints.
ASK_PREVIEW_ROWS = 20
ASK_PREVIEW_CELL_CHARS = 40
# The judge study reports its progress every this many tasks.
JUDGE_PROGRESS_EVERY = 10
# Loggers that log every in-process MCP request at INFO.
_MCP_LOGGERS = ("httpx", "mcp")
# Packages of the agent extra that ``ask`` and ``bench-spider2-exec`` import (fastmcp comes with
# pydantic-ai-slim[mcp] and is the orchestrator's MCP client).
_AGENT_EXTRA_MODULES = ("pydantic_ai", "treequest", "fastmcp")


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
    engine = _engine(home)
    cfg = {"ddl": file.read_text(encoding="utf-8"), "dialect": dialect, "default_schema": schema}
    snap = engine.add_connection(name, "ddl", cfg)
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
    engine = _engine(home)
    cfg = {
        "project_dir": str(project_dir) if project_dir else None,
        "manifest_path": str(manifest) if manifest else None,
        "default_schema": schema,
    }
    snap = engine.add_connection(name, "dbt", cfg)
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
    engine = _engine(home)
    snap = engine.add_connection(name, "duckdb", {"path": str(path)})
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
    engine = _engine(home)
    if config.startswith("@"):
        cfg = json.loads(Path(config[1:]).read_text(encoding="utf-8"))
    else:
        cfg = json.loads(config)
    snap = engine.add_connection(
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
    engine = _engine(home)
    result = engine.build(name)
    snaps = result if isinstance(result, list) else [result]
    for snap in snaps:
        typer.echo(
            f"{snap.source}: {len(snap.tables)} tables, {len(snap.edges)} edges, "
            f"{len(snap.terms)} terms"
        )
    typer.echo(json.dumps(engine.stats()))


@app.command()
def connections(home: HomeOpt = None):
    """List connections."""
    engine = _engine(home)
    for connection in engine.connections():
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
    engine = _engine(home)
    result = engine.link(question, max_tables=max_tables, columns=columns, use_llm=llm)
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
    """Run the HTTP API + frontend, with the MCP server (streamable HTTP) at /mcp."""
    import uvicorn

    from schemagraph.api.app import create_app

    uvicorn.run(create_app(_engine(home), host=host), host=host, port=port)


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

    result = run(
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
    typer.echo(format_table(result["summary"]))
    typer.echo(f"skipped (unknown db or no gold table): {result['summary']['config']['skipped']}")


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

    result = run(
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
    typer.echo(format_table(result["summary"]))
    typer.echo(f"\nresults written to {out}/")


def _agent_config(strategy: Strategy, **settings: Any) -> AgentConfig:
    """Build the agent settings of ``ask`` and ``bench-spider2-exec``.

    Exits with the install hint, before anything imports them, when a package of the agent
    extra is missing.
    """
    from schemagraph.agent.results import AgentConfig

    missing = [module for module in _AGENT_EXTRA_MODULES if find_spec(module) is None]
    if missing:
        typer.echo(
            f"missing {', '.join(missing)}; install the agent extra: uv sync --extra agent",
            err=True,
        )
        raise typer.Exit(1)
    return AgentConfig(strategy=strategy, **settings)


def _quiet_mcp_logs() -> None:
    """Log the in-process MCP traffic only from WARNING up."""
    for name in _MCP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _answer_errors() -> tuple[type[Exception], ...]:
    """Errors ``ask`` reports as one line, without a traceback: bad input, not bugs.

    A missing model key is pydantic-ai's ``UserError``, raised before any node runs. Without
    pydantic-ai the tuple leaves it out, so evaluating it never raises.
    """
    from schemagraph.agent.execute import AgentError

    errors: tuple[type[Exception], ...] = (AgentError, ImportError, ValueError, FileNotFoundError)
    try:
        from pydantic_ai.exceptions import UserError
    except ImportError:
        return errors
    return (*errors, UserError)


def _print_answer(result: AnswerResult) -> None:
    """Print the chosen query, a result preview, the score and the usage per role."""
    from schemagraph.agent.prompts import preview

    typer.echo(result.sql or "-- no query")
    typer.echo(f"\n{preview(result.result, ASK_PREVIEW_ROWS, cell_chars=ASK_PREVIEW_CELL_CHARS)}")
    early = ", stopped early" if result.stopped_early else ""
    typer.echo(
        f"\nscore {result.score:.2f} ({result.chosen_by}), {result.nodes} nodes{early}, "
        f"{result.ms / 1000:.1f}s"
    )
    for role, usage in result.usage.by_role.items():
        errors = "" if usage.ok else " (errors)"
        typer.echo(
            f"  {role:9} {usage.model:24} calls={usage.calls} requests={usage.requests} "
            f"tokens={usage.input_tokens}/{usage.output_tokens} {usage.ms / 1000:.1f}s{errors}"
        )


StrategyOpt = Annotated[Strategy, typer.Option(help="abmcts | best_of_n | refine | single")]
SelectorOpt = Annotated[
    bool, typer.Option(help="final pairwise pick among the top candidates")
]


@app.command()
def ask(
    question: str,
    connection: Annotated[
        str | None,
        typer.Option(
            "--connection", "-c", help="duckdb connection to run on (default: the only one)"
        ),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option(help="run on this .duckdb/.sqlite file instead of the connection's"),
    ] = None,
    mcp_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "schemagraph MCP server (streamable HTTP) the agents read the schema from, e.g. "
                "http://127.0.0.1:8765/mcp (default: serve this home's schema for the call)"
            )
        ),
    ] = None,
    strategy: StrategyOpt = "abmcts",
    budget: Annotated[int, typer.Option(help="generator nodes")] = 16,
    batch: Annotated[int, typer.Option(help="nodes generated concurrently")] = 4,
    seed: int = 0,
    gen_model: Annotated[
        str | None,
        typer.Option(
            help="generator model (default $SCHEMAGRAPH_GEN_MODEL or alibaba:qwen3.8-max)"
        ),
    ] = None,
    judge_model: Annotated[
        str | None,
        typer.Option(
            help="judge/selector model (default $SCHEMAGRAPH_JUDGE_MODEL or typesafe:jev-1.13.0)"
        ),
    ] = None,
    judge: Annotated[bool, typer.Option(help="score candidates with the judge")] = True,
    selector: SelectorOpt = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="print the full AnswerResult as JSON")
    ] = False,
    home: HomeOpt = None,
):
    """Write, run (read-only) and pick SQL for a question (needs the agent extra and model keys)."""
    cfg = _agent_config(
        strategy,
        budget=budget,
        batch_size=batch,
        seed=seed,
        gen_model=gen_model,
        judge_model=judge_model,
        judge=judge,
        selector=selector,
        mcp_url=mcp_url,
    )
    engine = _engine(home)
    _quiet_mcp_logs()
    try:
        result = engine.answer(question, connection=connection, db=db, config=cfg)
    except _answer_errors() as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from None
    finally:
        engine.close()
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_answer(result)


def _judge_study_progress(what: str, done: int, total: int) -> None:
    """Report the judge study's progress every :data:`JUDGE_PROGRESS_EVERY` tasks."""
    if done % JUDGE_PROGRESS_EVERY == 0 or done == total:
        typer.echo(f"  {what} {done}/{total}", err=True)


def _task_progress(done: int, total: int, row: dict[str, Any]) -> None:
    """Report one finished execution-benchmark task."""
    typer.echo(
        f"  {done}/{total}  {row['instance_id']} ex={row.get('ex')} nodes={row.get('nodes')} "
        f"{row.get('error') or ''}",
        err=True,
    )


@app.command()
def bench_spider2_exec(
    spider2_root: Annotated[
        Path,
        typer.Argument(
            help="path to the xlang-ai/Spider2 clone (with the local SQLite databases unpacked)"
        ),
    ],
    strategy: StrategyOpt = "abmcts",
    budget: int = 16,
    batch: int = 4,
    seed: int = 0,
    gen_model: str | None = None,
    judge_model: str | None = None,
    selector: SelectorOpt = True,
    judge_only: Annotated[
        bool,
        typer.Option(
            "--judge-only",
            help=(
                "judge study: AUROC of each --compare-judge against execution match on one "
                "candidate pool"
            ),
        ),
    ] = False,
    compare_judge: Annotated[
        list[str] | None,
        typer.Option(help="judge models for --judge-only (repeatable; default Jev and Qwen)"),
    ] = None,
    pool: Annotated[
        Path | None, typer.Option(help="--judge-only: reuse this candidate pool (jsonl)")
    ] = None,
    pool_size: Annotated[
        int, typer.Option(help="--judge-only: candidates generated per task")
    ] = 8,
    docs: Annotated[bool, typer.Option(help="pass the task's external-knowledge document")] = True,
    concurrency: Annotated[
        int,
        typer.Option(
            help="tasks in parallel (>1 is faster but AB-MCTS runs are not reproducible)"
        ),
    ] = 1,
    limit: int | None = None,
    only: Annotated[
        list[str] | None, typer.Option(help="only these instance ids (repeatable)")
    ] = None,
    out: Path = Path("bench_results"),
    tag: Annotated[
        str | None,
        typer.Option(help="output file tag (default <strategy>_n<budget>, or judge)"),
    ] = None,
    resume: Annotated[bool, typer.Option(help="skip tasks already in the rows file")] = True,
):
    """Execution accuracy of the agent loop on the 135 Spider 2.0-Lite local (SQLite) tasks."""
    from schemagraph.bench import spider2_exec

    cfg = _agent_config(
        strategy,
        budget=budget,
        batch_size=batch,
        seed=seed,
        gen_model=gen_model,
        judge_model=judge_model,
        judge=True,
        selector=selector,
    )
    ids = set(only) if only else None
    _quiet_mcp_logs()
    if judge_only:
        from schemagraph.agent.models import DEFAULT_GEN_MODEL, DEFAULT_JUDGE_MODEL
        from schemagraph.bench import spider2_judge

        judges = compare_judge or [DEFAULT_JUDGE_MODEL, gen_model or DEFAULT_GEN_MODEL]
        report = spider2_judge.judge_only(
            spider2_root,
            judges=judges,
            cfg=cfg,
            pool_size=pool_size,
            pool=pool,
            limit=limit,
            only=ids,
            tag=tag or "judge",
            out_dir=out,
            use_docs=docs,
            progress=_judge_study_progress,
        )
        typer.echo(json.dumps(report, indent=2))
        return
    result = spider2_exec.run(
        spider2_root,
        cfg=cfg,
        limit=limit,
        only=ids,
        tag=tag,
        out_dir=out,
        seed=seed,
        use_docs=docs,
        resume=resume,
        concurrency=concurrency,
        progress=_task_progress,
    )
    typer.echo(spider2_exec.format_table(result["summary"]))
    typer.echo(f"\nresults written to {out}/")


TransportOpt = Annotated[
    Literal["stdio", "http", "streamable-http", "sse"],
    typer.Option(help="stdio, http (streamable HTTP at /mcp; alias streamable-http) or sse"),
]


@app.command()
def mcp(
    home: HomeOpt = None,
    transport: TransportOpt = "stdio",
    host: Annotated[
        str,
        typer.Option(help="Interface to bind with --transport http or sse"),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port with --transport http or sse")] = 8766,
):
    """Run the MCP server (stdio by default; --transport http serves http://HOST:PORT/mcp)."""
    from schemagraph.mcp.http import run_http
    from schemagraph.mcp.server import create_server

    logging.basicConfig(level=logging.WARNING)
    server = create_server(Engine(home), host=host)
    if transport in ("http", "streamable-http"):
        run_http(server, host=host, port=port)
    elif transport == "sse":
        server.settings.port = port
        server.run(transport="sse")
    else:
        server.run()


if __name__ == "__main__":  # pragma: no cover
    app()
