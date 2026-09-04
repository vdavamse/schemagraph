"""Optional LLM pass: SchemaGraphSQL step 1 - pick source and destination tables.

One cheap Claude call over the *candidate* tables (already narrowed by PPR), asking
for the tables whose columns appear in filters/conditions (sources) and the tables
whose columns appear in the output (destinations). Everything after this call is
deterministic graph search. If no API key is configured the linker never calls this.

Uses the Anthropic SDK with a JSON-schema output format. Server-side refusal
fallbacks are enabled by default (``fallbacks="default"``) so a policy decline is
re-run on a fallback model inside the same request.
"""

from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field

from schemagraph.model import Table

DEFAULT_MODEL = os.environ.get("SCHEMAGRAPH_LLM_MODEL", "claude-opus-5")

SYSTEM = (
    "You are a database schema analyst. Given a natural-language question and a list of candidate "
    "tables, identify which tables the SQL answer would need. 'source' tables hold columns used in "
    "WHERE/JOIN conditions and filters; 'destination' tables hold columns that appear in the SELECT "
    "output or aggregations. A table may be both. Only use table names from the candidate list. "
    "Prefer fewer tables; intermediate join tables will be found automatically."
)


class AnchorPick(BaseModel):
    source_tables: list[str] = Field(description="Tables whose columns appear in filters or conditions")
    destination_tables: list[str] = Field(description="Tables whose columns appear in the output")
    rationale: str = Field(description="One sentence")


_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source_tables": {"type": "array", "items": {"type": "string"}},
        "destination_tables": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["source_tables", "destination_tables", "rationale"],
    "additionalProperties": False,
}


def _compact_schema(tables: list[Table], max_cols: int = 40) -> str:
    lines = []
    for t in tables:
        cols = ", ".join(f"{c.name}{' (' + c.description[:40] + ')' if c.description else ''}" for c in t.columns[:max_cols])
        extra = f" -- {t.description[:120]}" if t.description else ""
        lines.append(f"{t.fqn}: {cols}{extra}")
    return "\n".join(lines)


class ClaudeAnchorPicker:
    def __init__(self, model: str = DEFAULT_MODEL, *, fallbacks: bool = True, client=None):
        import anthropic

        self.model = model
        self.fallbacks = fallbacks
        self.client = client or anthropic.Anthropic()
        self.last_usage: dict | None = None

    def anchor_tables(self, question: str, candidates: list[Table]) -> tuple[list[str], list[str]]:
        if not candidates:
            return [], []
        user = f"Question: {question}\n\nCandidate tables (name: columns):\n{_compact_schema(candidates)}"
        kwargs: dict = dict(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}, "effort": "low"},
        )
        if self.fallbacks:
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        self.last_usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens, "model": resp.model}
        if resp.stop_reason == "refusal":
            return [], []
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        pick = AnchorPick.model_validate(json.loads(text))
        names = {t.fqn.lower(): t.fqn for t in candidates}

        def norm(xs: list[str]) -> list[str]:
            out = []
            for x in xs:
                lx = x.strip().lower()
                hit = names.get(lx) or next((v for k, v in names.items() if k.endswith("." + lx) or k.split(".")[-1] == lx), None)
                if hit and hit not in out:
                    out.append(hit)
            return out

        return norm(pick.source_tables), norm(pick.destination_tables)
