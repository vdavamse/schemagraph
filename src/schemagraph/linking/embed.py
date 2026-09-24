"""Optional seed-side embedding activator (docs/ENTITIES.md, option 2).

The lexical activator only reaches objects that share a surface form with the question, so a
paraphrase ("turnover", "how many staff") seeds nothing. This activator embeds every table,
column and glossary term once (name words plus description) with a small static model
(model2vec: no torch; the model comes from the Hugging Face Hub on first use) and, at query time, embeds the question's word n-grams and
seeds the closest objects above a cosine threshold. It adds ``(node, weight, reason)`` entries
to the same :class:`~schemagraph.linking.lexical.Activation`, so PPR, ranking, column
selection and the DDL reasons are unchanged. Opt in with ``LinkOptions(embed=True)``; needs
the ``embed`` extra (``uv sync --extra embed``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from schemagraph.graph.build import SchemaGraph
from schemagraph.linking.lexical import ngrams, tokenize

DEFAULT_MODEL = "minishlab/potion-base-8M"
MAX_PHRASES = 80


def _words(text: str | None) -> str:
    return " ".join(tokenize(text or "", keep_stop=True))


def question_part(text: str) -> str:
    """The question itself when an external document was appended after a blank line."""
    head, sep, _ = text.partition("\n\n")
    return head if sep and len(text) > 400 else text


def phrases(text: str) -> list[str]:
    toks = [t for t in tokenize(question_part(text)) if not (t.isdigit() and len(t) < 4)]
    out = list(dict.fromkeys(toks + [g.replace("_", " ") for g in ngrams(toks, 3)]))
    return out[:MAX_PHRASES]


@lru_cache(maxsize=4)
def load_model(model_name: str):
    """The static model, loaded once per process and shared across Linker rebuilds (a failed load
    is not cached). The local cache first (model2vec defaults to ``force_download=True``, which
    contacts the Hub, and waits out its timeout, on every load); the Hub only when the cache is
    missing or broken, e.g. a snapshot left without its weights by an interrupted download."""
    try:
        from model2vec import StaticModel
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise ImportError("LinkOptions(embed=True) needs the 'embed' extra: uv sync --extra embed") from e
    try:
        return StaticModel.from_pretrained(model_name, force_download=False)
    except Exception:
        from model2vec.persistence.hf import maybe_get_cached_model_path

        if maybe_get_cached_model_path(model_name) is None:
            raise  # nothing cached: the first attempt already asked the Hub, don't wait out its timeout twice
        return StaticModel.from_pretrained(model_name, force_download=True)


class EmbeddingActivator:
    def __init__(self, schema_graph: SchemaGraph, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = load_model(model_name)
        self.nodes: list[str] = []
        texts: list[str] = []
        for n, d in schema_graph.graph.nodes(data=True):
            nt = d.get("ntype")
            if nt == "table":
                t = schema_graph.tables[d["fqn"].lower()]
                texts.append(f"{_words(t.name)} {_words(t.properties.get('business_name'))} {_words(t.description)}".strip())
            elif nt == "column":
                t = schema_graph.tables[d["fqn"].lower()]
                c = t.column(d["name"])
                if c is None:
                    continue
                texts.append(f"{_words(t.name)} {_words(c.name)} {_words(c.properties.get('business_name'))} {_words(c.description)}".strip())
            elif nt == "term":
                term = schema_graph.terms[d["name"]]
                texts.append(f"{_words(term.name)} {' '.join(_words(s) for s in term.synonyms)} {_words(term.description)}".strip())
            else:
                continue
            self.nodes.append(n)
        self.vecs = self._encode(texts) if texts else np.zeros((0, 1))

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self.model.encode(texts), dtype=float)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms

    def activate(self, question: str, *, threshold: float = 0.5, top_k: int = 5, weight: float = 0.8) -> list[tuple[str, float, str]]:
        """Seeds as ``(node, weight * cosine, reason)`` for the best objects of each question phrase."""
        ph = phrases(question)
        if not ph or not self.nodes:
            return []
        sims = self._encode(ph) @ self.vecs.T  # (phrases, objects)
        best: dict[str, tuple[float, str]] = {}
        k = min(top_k, sims.shape[1])
        for i, phrase in enumerate(ph):
            row = sims[i]
            for j in np.argpartition(-row, k - 1)[:k]:
                s = float(row[j])
                if s < threshold:
                    continue
                node = self.nodes[j]
                if node not in best or s > best[node][0]:
                    best[node] = (s, phrase)
        return [(node, weight * s, f"embedding '{phrase}' ({s:.2f})") for node, (s, phrase) in best.items()]
