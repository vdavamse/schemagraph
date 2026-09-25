"""Optional agentic layer for text-to-SQL: Qwen generator, Jev judge, TreeQuest AB-MCTS search."""

import os

# Pydantic AI prints a banner on first use; it would land in `ask --json` output.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
