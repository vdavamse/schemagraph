"""Which model plays which role, resolved once per answer.

Generator and critic default to Qwen on Alibaba DashScope (``ALIBABA_API_KEY`` or
``DASHSCOPE_API_KEY``); judge and selector default to TypeSafe Jev (``TYPESAFE_API_KEY``). Any
pydantic-ai model string works for each role through the ``SCHEMAGRAPH_*_MODEL`` variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from schemagraph.agent.results import AgentConfig

DEFAULT_GEN_MODEL = "alibaba:qwen3.8-max"
DEFAULT_JUDGE_MODEL = "typesafe:jev-1.13.0"
ENV_GEN = "SCHEMAGRAPH_GEN_MODEL"
ENV_JUDGE = "SCHEMAGRAPH_JUDGE_MODEL"
ENV_CRITIC = "SCHEMAGRAPH_CRITIC_MODEL"
ENV_ALIBABA_BASE_URL = "SCHEMAGRAPH_ALIBABA_BASE_URL"


def model_names(cfg: AgentConfig) -> dict[str, str]:
    """Name the model of each role: the config first, then the environment, then the default."""
    generator = cfg.gen_model or os.environ.get(ENV_GEN) or DEFAULT_GEN_MODEL
    judge = cfg.judge_model or os.environ.get(ENV_JUDGE) or DEFAULT_JUDGE_MODEL
    critic = cfg.critic_model or os.environ.get(ENV_CRITIC) or generator
    return {"generator": generator, "judge": judge, "selector": judge, "critic": critic}


def _alibaba_model(model: str) -> Any:
    """Build a DashScope Qwen model with forced tool choice and strict tool schemas off."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles import merge_profile
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.alibaba import AlibabaProvider

    base_url = os.environ.get(ENV_ALIBABA_BASE_URL)
    provider = AlibabaProvider(base_url=base_url) if base_url else AlibabaProvider()
    no_forced_tools = OpenAIModelProfile(
        openai_supports_tool_choice_required=False,
        openai_supports_strict_tool_definition=False,
    )
    profile = merge_profile(AlibabaProvider.model_profile(model), no_forced_tools)
    return OpenAIChatModel(model, provider=provider, profile=profile)


def resolve_model(name: Any) -> Any:
    """Return a pydantic-ai model for ``name``; a model object passes through.

    Qwen models on DashScope get ``tool_choice="auto"`` instead of ``"required"`` for structured
    output (pydantic-ai's Qwen profile sets that only for ``qwen-3-coder``; DashScope rejects forced
    tool choice for thinking models, pydantic-ai issue #1265) and no strict tool schemas. A
    text-only reply then triggers pydantic-ai's retry prompt for the output tool, bounded by
    ``output_retries``. Other strings (``"openai:..."``, ``"test"``, ...) are left for pydantic-ai
    to infer.
    """
    if not isinstance(name, str):
        return name
    provider, _, model = name.partition(":")
    if provider == "alibaba":
        return _alibaba_model(model)
    if provider == "typesafe":
        from pydantic_ai.models.typesafe import TypeSafeModel

        return TypeSafeModel(model)
    return name


def _roles_used(cfg: AgentConfig) -> dict[str, bool]:
    """Which roles ``cfg`` runs: no judge with the judge off, no critic without refinements."""
    return {
        "generator": True,
        "judge": cfg.judge,
        "selector": cfg.selector and cfg.strategy != "single",
        "critic": cfg.strategy in {"refine", "abmcts"},
    }


@dataclass
class AgentModels:
    """The resolved model of each role.

    Attributes:
        gen: Generator model.
        judge: Judge model, or None when the judge is off.
        selector: Pairwise selector model, or None when it does not run.
        critic: Critic model, or None when the strategy never refines.
        names: Model name by role, including roles that do not run.
    """

    gen: Any
    judge: Any
    selector: Any
    critic: Any
    names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(cls, cfg: AgentConfig) -> AgentModels:
        """Resolve only the roles ``cfg`` uses, once per model name.

        A missing API key therefore fails here, before any node runs, and only for a role that
        would run.
        """
        names = model_names(cfg)
        used = _roles_used(cfg)
        by_name: dict[str, Any] = {}

        def model_for(role: str) -> Any:
            if not used[role]:
                return None
            name = names[role]
            if name not in by_name:
                by_name[name] = resolve_model(name)
            return by_name[name]

        return cls(
            model_for("generator"),
            model_for("judge"),
            model_for("selector"),
            model_for("critic"),
            names,
        )
