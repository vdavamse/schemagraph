"""The Engine: one object the CLI, API and MCP server all drive.

Owns the Store, (re)builds the in-memory SchemaGraph from stored snapshots, holds
the Linker, and runs connectors.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from schemagraph import connectors  # noqa: F401  (registers connector types)
from schemagraph.connectors.base import config_schema, connector_types, make_connector
from schemagraph.graph.build import SchemaGraph, build_graph
from schemagraph.linking.linker import Linker, LinkOptions
from schemagraph.model import BusinessTerm, Edge, LinkResult, SchemaSnapshot, Table
from schemagraph.store import Store, substitute_env

if TYPE_CHECKING:
    from schemagraph.agent.results import AgentConfig, AnswerResult

log = logging.getLogger("schemagraph")

DEFAULT_HOME = Path(os.environ.get("SCHEMAGRAPH_HOME", ".schemagraph"))


_AGENT_EXTRA_HINT = "schemagraph answers need the agent extra: uv sync --extra agent"

# Values of ``SCHEMAGRAPH_EMBED`` that turn embeddings on (anything else but ``auto`` is off).
_EMBED_TRUE_VALUES = {"1", "true", "yes", "on"}


def find_join_paths(schema_graph: SchemaGraph, a: str, b: str) -> list[list[str]]:
    """Shortest join path(s) between two tables of ``schema_graph``, as lists of table FQNs.

    Tables are looked up by FQN or unambiguous suffix. Falls back to lineage routes when no join
    route exists, so "how do these connect" still answers. :meth:`Engine.join_path` and the MCP
    server's :class:`~schemagraph.mcp.source.LinkerSource` both call this.

    Raises:
        KeyError: Either table is unknown.
    """
    from schemagraph.graph.pathfinding import union_of_shortest_paths

    table_a, table_b = schema_graph.find_table(a), schema_graph.find_table(b)
    if not table_a or not table_b:
        raise KeyError(a if not table_a else b)
    paths, _ = union_of_shortest_paths(schema_graph, [table_a.fqn], [table_b.fqn])
    if not paths:  # no join route: fall back to lineage
        paths, _ = union_of_shortest_paths(
            schema_graph,
            [table_a.fqn],
            [table_b.fqn],
            kinds=None,
        )
    return [[schema_graph.graph.nodes[node]["fqn"] for node in path] for path in paths]


class Engine:
    """The one object the CLI, API and MCP server drive.

    Owns the :class:`Store`, the merged :class:`SchemaGraph` rebuilt from every stored snapshot
    (``engine.graph`` is the SchemaGraph; ``engine.graph.graph`` is its NetworkX graph) and the
    :class:`Linker` over it. An ``RLock`` guards the graph and linker: :meth:`reload` swaps both
    under it and :meth:`link` / :meth:`explain` run under it; connector introspection runs
    outside it.
    """

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        llm: Any | None = "auto",
        embed: bool | str | None = None,
    ):
        """Open the store under ``home`` and build the graph and linker.

        Args:
            home: Data directory (default ``$SCHEMAGRAPH_HOME`` or ``.schemagraph``).
            llm: Anchor picker for ``use_llm``; ``"auto"`` builds a Claude picker when an
                Anthropic key is set, None disables it.
            embed: Seed paraphrases with the optional static embedding model
                (``LinkOptions.embed``; Spider2-Lite: +0.4 strict, +3 strict@7). ``None`` reads
                ``SCHEMAGRAPH_EMBED`` (``1`` / ``0`` / ``auto``, default ``auto``); ``"auto"``
                turns it on when the ``embed`` extra is installed. The model loads in
                :meth:`reload`, never inside a request; if it cannot load (not cached and
                offline), embeddings stay off with a warning. Set ``HF_HUB_OFFLINE=1`` on
                air-gapped hosts.
        """
        self.home = Path(home) if home else DEFAULT_HOME
        self.store = Store(self.home / "schemagraph.duckdb")
        self._lock = threading.RLock()
        self.graph: SchemaGraph = SchemaGraph()
        self.linker: Linker | None = None
        self._llm = llm
        self._embed_requested = self._resolve_embed(embed)
        # effective: requested and the model loaded at the last reload
        self._embed = False
        # a load failed: later reloads retry only from the local cache, never the Hub
        self._embed_failed = False
        self.reload()

    @staticmethod
    def _resolve_embed(embed: bool | str | None) -> bool:
        """Whether embeddings are requested (see ``embed`` in :meth:`__init__`)."""
        if embed is None:
            embed = os.environ.get("SCHEMAGRAPH_EMBED", "auto").strip().lower()
            if embed != "auto":
                embed = embed in _EMBED_TRUE_VALUES
        if embed != "auto":
            return bool(embed)
        try:
            import model2vec  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def has_embed(self) -> bool:
        """Whether embeddings are on (requested and the model loaded at the last reload)."""
        return self._embed

    # ----------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Close the store."""
        self.store.close()

    def reload(self) -> None:
        """Rebuild the graph and linker from every stored snapshot plus the user's curation."""
        with self._lock:
            snaps = self.store.snapshots()
            # priority 0: merged first, so curation wins conflicts
            snaps.append(self.store.user_snapshot())
            self.graph = build_graph(snaps)
            llm = self._resolve_llm()
            self.linker = Linker(self.graph, llm=llm)
            self._embed = (
                self._embed_requested
                and (not self._embed_failed or self._model_cached())
                and self._warm_embedder()
            )

    @staticmethod
    def _model_cached() -> bool:
        """Whether the embedding model can load without the network.

        True for a local directory or a model in the Hugging Face cache.
        """
        model = LinkOptions().embed_model
        if Path(model).is_dir():
            return True
        try:
            from huggingface_hub import try_to_load_from_cache

            return isinstance(try_to_load_from_cache(model, "model.safetensors"), str)
        except Exception:
            return False

    def _warm_embedder(self) -> bool:
        """Load the model and encode the catalog now; return whether it worked.

        The cost and any failure land at startup, not in a request. After a failure, reloads
        retry only once the model is in the local cache (a Hub timeout would otherwise be paid
        on every reload, under the lock).
        """
        assert self.linker is not None
        model = LinkOptions().embed_model
        t0 = time.perf_counter()
        try:
            self.linker.embedder(model)
        except Exception as e:
            log.warning(
                "embedding model %s unavailable, embeddings off until it is cached locally: %s",
                model,
                e,
            )
            self._embed_failed = True
            return False
        log.info(
            "embeddings on: %s, %d objects encoded in %.2fs",
            model,
            len(self.linker.embedder(model).nodes),
            time.perf_counter() - t0,
        )
        return True

    def _resolve_llm(self):
        """Return the anchor picker: the one given, or for ``"auto"`` Claude when a key is set."""
        if self._llm != "auto":
            return self._llm
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return None
        try:
            from schemagraph.llm.anchors import ClaudeAnchorPicker

            return ClaudeAnchorPicker()
        except Exception as e:  # pragma: no cover
            log.warning("LLM anchor picker unavailable: %s", e)
            return None

    @property
    def has_llm(self) -> bool:
        """Whether the linker has an LLM anchor picker."""
        return self.linker is not None and self.linker.llm is not None

    # ----------------------------------------------------------- connections
    def connector_types(self) -> list[str]:
        """Names of the registered connector types."""
        return connector_types()

    def connector_schema(self, type_name: str) -> dict[str, Any]:
        """JSON schema of one connector type's config."""
        return config_schema(type_name)

    def add_connection(
        self,
        name: str,
        type_name: str,
        config: dict[str, Any],
        *,
        build: bool = True,
        priority: int | None = None,
        clear_priority: bool = False,
    ) -> SchemaSnapshot | None:
        """Validate and register a connection, then optionally build it.

        Args:
            name: Connection name; it becomes the snapshot's ``source``.
            type_name: Registered connector type.
            config: Connector config (``${ENV}`` references allowed).
            build: Introspect it right away.
            priority: Merge order (lower merges first and wins conflicting fields); None keeps a
                stored priority, else merges by source type.
            clear_priority: Go back to the source-type order.

        Returns:
            The built snapshot, or None when ``build`` is false.
        """
        make_connector(type_name, name, substitute_env(config))  # validate config shape
        self.store.upsert_connection(
            name,
            type_name,
            config,
            priority=priority,
            clear_priority=clear_priority,
        )
        if build:
            return self.build(name)
        return None

    def remove_connection(self, name: str) -> None:
        """Delete a connection and its snapshot, then reload."""
        with self._lock:
            self.store.delete_connection(name)
        self.reload()

    def check_connection(self, name: str) -> str:
        """Run the connector's connectivity check and return its status message.

        Raises:
            KeyError: ``name`` is not a registered connection.
        """
        row = self.store.connection(name)
        if not row:
            raise KeyError(name)
        type_name, config = row
        return make_connector(type_name, name, substitute_env(config)).check()

    def build(self, name: str | None = None) -> SchemaSnapshot | list[SchemaSnapshot]:
        """Introspect one connection (or all) and persist the snapshot(s).

        Args:
            name: Connection to build; None builds every connection.

        Returns:
            The snapshot when ``name`` is given, else the list of all snapshots built.

        Raises:
            KeyError: A named connection is not registered.
        """
        names = [name] if name else [c["name"] for c in self.store.connections()]
        out: list[SchemaSnapshot] = []
        for connection_name in names:
            row = self.store.connection(connection_name)
            if not row:
                raise KeyError(connection_name)
            type_name, config = row
            connector = make_connector(type_name, connection_name, substitute_env(config))
            log.info("building %s (%s)", connection_name, type_name)
            snap = connector.introspect()
            snap.source = connection_name
            with self._lock:
                self.store.save_snapshot(snap)
            out.append(snap)
        self.reload()
        return out[0] if name else out

    def connections(self) -> list[dict[str, Any]]:
        """Summaries of every stored connection (configs redacted)."""
        return self.store.connections()

    # ----------------------------------------------------------- glossary / hints
    def upsert_term(self, term: BusinessTerm) -> None:
        """Insert or update a user glossary term, then reload."""
        self.store.upsert_term(term)
        self.reload()

    def delete_term(self, name: str) -> None:
        """Delete a user glossary term, then reload."""
        self.store.delete_term(name)
        self.reload()

    def add_join_hint(self, edge: Edge) -> int:
        """Store a user join hint, reload, and return its id."""
        hint_id = self.store.add_join_hint(edge)
        self.reload()
        return hint_id

    def delete_join_hint(self, hid: int) -> None:
        """Delete a user join hint, then reload."""
        self.store.delete_join_hint(hid)
        self.reload()

    # ----------------------------------------------------------- queries
    def _link_options(self, kw: dict[str, Any]) -> LinkOptions:
        """Build the options for a link: the engine's ``embed`` setting, overridden by ``kw``.

        ``use_llm`` is dropped when no anchor picker is configured. Call under the lock.
        """
        opts = LinkOptions(**{"embed": self._embed, **kw})
        if opts.use_llm and not self.has_llm:
            opts.use_llm = False
        return opts

    def link(self, question: str, **kw: Any) -> LinkResult:
        """Link a question to a sub-schema; ``kw`` are :class:`LinkOptions` fields."""
        assert self.linker is not None
        with self._lock:  # read the embed flag under the lock: a reload can turn it off
            return self.linker.link(question, self._link_options(kw))

    def explain(self, question: str, **kw: Any) -> dict[str, Any]:
        """Show the activation seeds and scores behind a link.

        Takes the same options as :meth:`link` (the engine's ``embed`` setting included), so it
        shows the seeds that ranked.
        """
        assert self.linker is not None
        with self._lock:
            return self.linker.explain(question, self._link_options(kw))

    def tables(self) -> list[Table]:
        """Every table in the graph, sorted by FQN."""
        return sorted(self.graph.tables.values(), key=lambda t: t.fqn)

    def table(self, fqn: str) -> Table | None:
        """Look a table up by FQN or unambiguous suffix."""
        return self.graph.find_table(fqn)

    def edges(self) -> list[Edge]:
        """Every relation edge in the graph."""
        return self.graph.all_edges()

    def terms(self) -> list[BusinessTerm]:
        """Every merged glossary term (the user glossary and the catalogs')."""
        return list(self.graph.terms.values())

    def join_path(self, a: str, b: str) -> list[list[str]]:
        """Shortest join path(s) between two tables, as lists of table FQNs.

        Falls back to lineage routes when no join route exists, so "how do these connect" still
        answers.

        Raises:
            KeyError: Either table is unknown.
        """
        return find_join_paths(self.graph, a, b)

    def stats(self) -> dict[str, Any]:
        """Graph counts plus the engine's ``llm``, ``embed`` and ``home`` settings."""
        stats = self.graph.stats()
        stats["llm"] = self.has_llm
        stats["embed"] = self.has_embed
        stats["home"] = str(self.home)
        return stats

    # ----------------------------------------------------------- answers (optional `agent` extra)
    def answer(
        self,
        question: str,
        *,
        connection: str | None = None,
        db: str | Path | None = None,
        config: AgentConfig | None = None,
        evidence: str | None = None,
    ) -> AnswerResult:
        """Write, run and pick SQL for ``question`` (:mod:`schemagraph.agent`).

        A sync wrapper of :meth:`answer_async`, which takes the same arguments; inside an event
        loop, await that instead.

        Raises:
            RuntimeError: Called inside a running event loop.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.answer_async(
                    question, connection=connection, db=db, config=config, evidence=evidence
                )
            )
        raise RuntimeError(
            "Engine.answer() called inside a running event loop; await Engine.answer_async()"
        )

    async def answer_async(
        self,
        question: str,
        *,
        connection: str | None = None,
        db: str | Path | None = None,
        config: AgentConfig | None = None,
        evidence: str | None = None,
    ) -> AnswerResult:
        """Write, run and pick SQL for ``question``, executing read-only.

        The agents read the schema over MCP: from ``config.mcp_url`` when it is set, else from
        this engine, served on an ephemeral localhost port for the call and scoped to the
        connection. The MCP tools link in a worker thread, so the engine lock is held only
        inside :meth:`link`, never across an ``await``.

        Args:
            question: The question.
            connection: The ``duckdb`` connection to run on (default: the only one). With
                ``db``, it only scopes the schema.
            db: A ``.duckdb``/``.sqlite`` file to run on instead of the connection's database.
            config: The agent settings (default ``AgentConfig()``).
            evidence: External knowledge for the question.

        Returns:
            The chosen query, its result and every candidate.

        Raises:
            ImportError: The ``agent`` extra is not installed.
            AgentError: The connection is unknown or cannot execute.
            ValueError: No connection was given and there is not exactly one ``duckdb`` one.
            FileNotFoundError: ``db`` does not exist.
        """
        try:
            from schemagraph.agent.answer import Answerer
            from schemagraph.agent.execute import executor_for_connection
            from schemagraph.agent.results import AgentConfig
            from schemagraph.agent.schema_client import SchemaClient
        except ImportError as error:  # pragma: no cover - depends on the installed extras
            raise ImportError(_AGENT_EXTRA_HINT) from error
        from schemagraph.mcp import create_server, serve_http_async

        cfg = config or AgentConfig()
        name = connection  # with db and no connection: run on the file, schema unscoped
        if connection is None and db is None:
            name = self._exec_connection()  # the only duckdb connection
        executor = executor_for_connection(self, name, db=db)
        try:
            async with AsyncExitStack() as stack:
                mcp_url = cfg.mcp_url or await stack.enter_async_context(
                    serve_http_async(create_server(self, connection=name))
                )
                answerer = Answerer(SchemaClient(mcp_url), executor, cfg)
                return await answerer.answer(question, evidence=evidence)
        finally:
            executor.close()

    def _exec_connection(self) -> str:
        """Return the only duckdb connection.

        Raises:
            ValueError: There is not exactly one.
        """
        names = [c["name"] for c in self.store.connections() if c["type"] == "duckdb"]
        if len(names) != 1:
            raise ValueError(
                f"pass a connection: {len(names)} duckdb connections "
                f"({', '.join(names) or 'none'}); only duckdb connections execute"
            )
        return names[0]
