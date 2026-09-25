"""Optional seed-side embedding activator (docs/ENTITIES.md, option 2).

The lexical activator only reaches objects that share a surface form with the question, so a
paraphrase ("turnover", "how many staff") seeds nothing. This activator embeds every table,
column and glossary term once (name words plus description) with a small static model
(model2vec: no torch; the model comes from the Hugging Face Hub on first use) and, at query
time, embeds the question's word n-grams and seeds the closest objects above a cosine
threshold. It adds ``(node, weight, reason)`` entries to the same
:class:`~schemagraph.linking.lexical.Activation`, so PPR, ranking, column selection and the
DDL reasons are unchanged. Opt in with ``LinkOptions(embed=True)``; needs the ``embed`` extra
(``uv sync --extra embed``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from schemagraph.graph.build import SchemaGraph
from schemagraph.linking.lexical import DEFAULT_MIN_NUMERIC_LEN, NGRAM_MAX, ngrams, tokenize

DEFAULT_MODEL = "minishlab/potion-base-8M"
MAX_PHRASES = 80  # question phrases embedded per query
# A text longer than this with a blank line is a question with an appended external document.
QUESTION_PART_MIN_CHARS = 400


def _spaced_words(text: str | None) -> str:
    """The text's tokens (stopwords kept) joined by single spaces; empty for None."""
    return " ".join(tokenize(text or "", keep_stop=True))


def question_part(text: str) -> str:
    """The question itself when an external document was appended after a blank line."""
    head, sep, _ = text.partition("\n\n")
    return head if sep and len(text) > QUESTION_PART_MIN_CHARS else text


def phrases(text: str) -> list[str]:
    """The question's tokens, then its space-joined n-grams, deduplicated and capped."""
    tokens = [
        token
        for token in tokenize(question_part(text))
        if not (token.isdigit() and len(token) < DEFAULT_MIN_NUMERIC_LEN)
    ]
    grams = [gram.replace("_", " ") for gram in ngrams(tokens, NGRAM_MAX)]
    out = list(dict.fromkeys(tokens + grams))
    return out[:MAX_PHRASES]


@lru_cache(maxsize=4)
def load_model(model_name: str):
    """Load the static model once per process, shared across Linker rebuilds.

    A failed load is not cached. The local cache is tried first (model2vec defaults to
    ``force_download=True``, which contacts the Hub, and waits out its timeout, on every load);
    the Hub only when the cache is missing or broken, e.g. a snapshot left without its weights
    by an interrupted download.

    Raises:
        ImportError: The ``embed`` extra is not installed.
    """
    try:
        from model2vec import StaticModel
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            "LinkOptions(embed=True) needs the 'embed' extra: uv sync --extra embed"
        ) from e
    try:
        return StaticModel.from_pretrained(model_name, force_download=False)
    except Exception:
        from model2vec.persistence.hf import maybe_get_cached_model_path

        if maybe_get_cached_model_path(model_name) is None:
            # nothing cached: the first attempt already asked the Hub, don't wait out its
            # timeout twice
            raise
        return StaticModel.from_pretrained(model_name, force_download=True)


def _object_text(schema_graph: SchemaGraph, attrs: dict) -> str | None:
    """The text embedded for a table, column or term node; None for any other node."""
    node_type = attrs.get("ntype")
    if node_type == "table":
        table = schema_graph.tables[attrs["fqn"].lower()]
        name = _spaced_words(table.name)
        business_name = _spaced_words(table.properties.get("business_name"))
        return f"{name} {business_name} {_spaced_words(table.description)}".strip()
    if node_type == "column":
        table = schema_graph.tables[attrs["fqn"].lower()]
        column = table.column(attrs["name"])
        if column is None:
            return None
        table_name = _spaced_words(table.name)
        name = _spaced_words(column.name)
        business_name = _spaced_words(column.properties.get("business_name"))
        description = _spaced_words(column.description)
        return f"{table_name} {name} {business_name} {description}".strip()
    if node_type == "term":
        term = schema_graph.terms[attrs["name"]]
        synonyms = " ".join(_spaced_words(s) for s in term.synonyms)
        return f"{_spaced_words(term.name)} {synonyms} {_spaced_words(term.description)}".strip()
    return None


class EmbeddingActivator:
    """Unit-normalised embeddings of every table, column and term, and the query side.

    Attributes:
        model_name: The model2vec model the vectors come from.
        model: The loaded static model.
        nodes: Node ids, aligned with the rows of ``vecs``.
        vecs: One unit-length row per node (shape ``(0, 1)`` when there is none).
    """

    def __init__(self, schema_graph: SchemaGraph, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = load_model(model_name)
        self.nodes: list[str] = []
        texts: list[str] = []
        for node, attrs in schema_graph.graph.nodes(data=True):
            text = _object_text(schema_graph, attrs)
            if text is None:
                continue
            texts.append(text)
            self.nodes.append(node)
        self.vecs = self._encode(texts) if texts else np.zeros((0, 1))

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts as unit-length rows (a zero vector stays zero)."""
        vectors = np.asarray(self.model.encode(texts), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def activate(
        self,
        question: str,
        *,
        threshold: float = 0.5,
        top_k: int = 5,
        weight: float = 0.8,
    ) -> list[tuple[str, float, str]]:
        """Seed the objects closest to each question phrase.

        Each phrase considers its ``top_k`` nearest objects; an object keeps its best cosine
        (the first phrase wins ties).

        Args:
            question: The natural-language question.
            threshold: Cosine floor for a seed.
            top_k: Objects considered per phrase.
            weight: Seed weight per unit of cosine.

        Returns:
            ``(node, weight * cosine, reason)`` per seeded object.
        """
        question_phrases = phrases(question)
        if not question_phrases or not self.nodes:
            return []
        sims = self._encode(question_phrases) @ self.vecs.T  # (phrases, objects)
        best: dict[str, tuple[float, str]] = {}
        k = min(top_k, sims.shape[1])
        for i, phrase in enumerate(question_phrases):
            row = sims[i]
            for j in np.argpartition(-row, k - 1)[:k]:
                similarity = float(row[j])
                if similarity < threshold:
                    continue
                node = self.nodes[j]
                if node not in best or similarity > best[node][0]:
                    best[node] = (similarity, phrase)
        return [
            (node, weight * similarity, f"embedding '{phrase}' ({similarity:.2f})")
            for node, (similarity, phrase) in best.items()
        ]
