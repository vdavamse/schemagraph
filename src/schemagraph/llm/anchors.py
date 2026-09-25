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
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from schemagraph.model import Column, Table

if TYPE_CHECKING:
    import anthropic

DEFAULT_MODEL = os.environ.get("SCHEMAGRAPH_LLM_MODEL", "claude-opus-5")

SYSTEM = (
    "You are a database schema analyst. Given a natural-language question and a list of candidate "
    "tables, identify which tables the SQL answer would need. 'source' tables hold columns used in "
    "WHERE/JOIN conditions and filters; 'destination' tables hold columns that appear in the "
    "SELECT output or aggregations. A table may be both. Only use table names from the candidate "
    "list. Prefer fewer tables; intermediate join tables will be found automatically."
)


class AnchorPick(BaseModel):
    """The model's answer: source tables, destination tables and a one-sentence rationale."""

    source_tables: list[str] = Field(
        description="Tables whose columns appear in filters or conditions"
    )
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

# Beta flag for server-side refusal fallbacks (``fallbacks=True``).
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Response budget of the anchor call; the JSON answer is short.
MAX_TOKENS = 1024
# Column and table descriptions are cut to these lengths in the compact schema.
COLUMN_DESCRIPTION_CHARS = 40
TABLE_DESCRIPTION_CHARS = 120


def _column_entry(column: Column) -> str:
    """``name`` or ``name (short description)`` for the compact schema."""
    if column.description:
        return f"{column.name} ({column.description[:COLUMN_DESCRIPTION_CHARS]})"
    return column.name


def _compact_schema(tables: list[Table], max_cols: int = 40) -> str:
    """One line per table, ``fqn: col (desc), ... -- table description``, for the prompt."""
    lines = []
    for table in tables:
        columns = ", ".join(_column_entry(column) for column in table.columns[:max_cols])
        extra = (
            f" -- {table.description[:TABLE_DESCRIPTION_CHARS]}" if table.description else ""
        )
        lines.append(f"{table.fqn}: {columns}{extra}")
    return "\n".join(lines)


def _normalise_picks(names: dict[str, str], picks: list[str]) -> list[str]:
    """Map the model's table names onto candidate FQNs, dropping unknown names and duplicates.

    Args:
        names: Candidate FQNs keyed by their lowercased form.
        picks: Table names as the model wrote them.

    Returns:
        Candidate FQNs in pick order: an exact (case-insensitive) match, else the first
        candidate whose FQN ends with ``.<pick>`` or whose last part equals the pick.
    """
    out = []
    for pick in picks:
        lower_pick = pick.strip().lower()
        hit = names.get(lower_pick) or next(
            (
                v
                for k, v in names.items()
                if k.endswith("." + lower_pick) or k.split(".")[-1] == lower_pick
            ),
            None,
        )
        if hit and hit not in out:
            out.append(hit)
    return out


class ClaudeAnchorPicker:
    """Anchor picker backed by one Claude call (SchemaGraphSQL step 1).

    ``last_usage`` holds the token usage and model of the latest call.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        fallbacks: bool = True,
        client: anthropic.Anthropic | None = None,
    ):
        """Create the picker.

        Args:
            model: Claude model id.
            fallbacks: Use server-side refusal fallbacks (the beta endpoint).
            client: Anthropic client; a default ``anthropic.Anthropic()`` when None.
        """
        import anthropic

        self.model = model
        self.fallbacks = fallbacks
        self.client = client or anthropic.Anthropic()
        self.last_usage: dict | None = None

    def anchor_tables(self, question: str, candidates: list[Table]) -> tuple[list[str], list[str]]:
        """Ask Claude for the source and destination tables among ``candidates``.

        Args:
            question: The natural-language question.
            candidates: Tables already narrowed by the ranking.

        Returns:
            ``(source_fqns, destination_fqns)``, both candidate FQNs; empty on no candidates or
            a refusal.
        """
        if not candidates:
            return [], []
        user = (
            f"Question: {question}\n\n"
            f"Candidate tables (name: columns):\n{_compact_schema(candidates)}"
        )
        kwargs: dict = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA},
                "effort": "low",
            },
        )
        if self.fallbacks:
            resp = self.client.beta.messages.create(
                betas=[_FALLBACK_BETA],
                fallbacks="default",
                **kwargs,
            )
        else:
            resp = self.client.messages.create(**kwargs)
        self.last_usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "model": resp.model,
        }
        if resp.stop_reason == "refusal":
            return [], []
        text = next((block.text for block in resp.content if block.type == "text"), "{}")
        pick = AnchorPick.model_validate(json.loads(text))
        names = {t.fqn.lower(): t.fqn for t in candidates}
        return (
            _normalise_picks(names, pick.source_tables),
            _normalise_picks(names, pick.destination_tables),
        )
