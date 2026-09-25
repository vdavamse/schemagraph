"""Which model plays which role, resolved once per answer.

Generator and critic default to Qwen on Alibaba DashScope (``ALIBABA_API_KEY`` or
``DASHSCOPE_API_KEY``); judge and selector default to TypeSafe Jev (``TYPESAFE_API_KEY``). Any
pydantic-ai model string works for each role through the ``SCHEMAGRAPH_*_MODEL`` variables.
``openrouter:`` models reason at ``SCHEMAGRAPH_REASONING`` effort (default medium) and report
their billed cost; Jev goes through OpenRouter with ``TYPESAFE_BASE_URL=https://openrouter.ai/api``.
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
ENV_REASONING = "SCHEMAGRAPH_REASONING"
# Reasoning effort of ``openrouter:`` models; ``off`` asks the route not to reason.
REASONING_LEVELS = ("off", "low", "medium", "high")
DEFAULT_REASONING = "medium"


def model_names(cfg: AgentConfig) -> dict[str, str]:
    """Name the model of each role: the config first, then the environment, then the default."""
    generator = cfg.gen_model or os.environ.get(ENV_GEN) or DEFAULT_GEN_MODEL
    judge = cfg.judge_model or os.environ.get(ENV_JUDGE) or DEFAULT_JUDGE_MODEL
    critic = cfg.critic_model or os.environ.get(ENV_CRITIC) or generator
    return {"generator": generator, "judge": judge, "selector": judge, "critic": critic}


def reasoning_level() -> str:
    """Return the reasoning effort of ``openrouter:`` models, from the environment.

    Raises:
        ValueError: ``SCHEMAGRAPH_REASONING`` is not one of :data:`REASONING_LEVELS`.
    """
    level = os.environ.get(ENV_REASONING, DEFAULT_REASONING).strip().lower() or DEFAULT_REASONING
    if level not in REASONING_LEVELS:
        expected = ", ".join(REASONING_LEVELS)
        raise ValueError(f"{ENV_REASONING}={level!r}: expected one of {expected}")
    return level


def reasons(name: Any) -> bool:
    """Whether the model named ``name`` is asked to reason (an ``openrouter:`` model, not off)."""
    return isinstance(name, str) and name.startswith("openrouter:") and reasoning_level() != "off"


def _no_forced_tools() -> Any:
    """Profile override: ``tool_choice="auto"`` for structured output, no strict tool schemas.

    Qwen's thinking mode rejects a forced tool choice (pydantic-ai issue #1265), on DashScope
    directly and through a gateway; a text-only reply triggers pydantic-ai's output-tool retry.
    """
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    return OpenAIModelProfile(
        openai_supports_tool_choice_required=False,
        openai_supports_strict_tool_definition=False,
    )


def _alibaba_model(model: str) -> Any:
    """Build a DashScope Qwen model with forced tool choice and strict tool schemas off."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles import merge_profile
    from pydantic_ai.providers.alibaba import AlibabaProvider

    base_url = os.environ.get(ENV_ALIBABA_BASE_URL)
    provider = AlibabaProvider(base_url=base_url) if base_url else AlibabaProvider()
    profile = merge_profile(AlibabaProvider.model_profile(model), _no_forced_tools())
    return OpenAIChatModel(model, provider=provider, profile=profile)


def _openrouter_model(model: str) -> Any:
    """Build an OpenRouter model that reasons at :func:`reasoning_level` and reports its cost.

    Reasoning uses OpenRouter's unified ``reasoning`` field, which the gateway translates for the
    upstream (Qwen's thinking mode included); ``usage.include`` asks for the billed cost of every
    response, which the usage records sum. Forced tool choice is off while reasoning, as for
    DashScope.
    """
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.profiles import merge_profile
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    level = reasoning_level()
    settings = OpenRouterModelSettings(openrouter_usage={"include": True})
    if level == "off":
        settings["openrouter_reasoning"] = {"enabled": False}
        profile = None
    else:
        effort: Any = level
        settings["openrouter_reasoning"] = {"enabled": True, "effort": effort}
        profile = merge_profile(OpenRouterProvider.model_profile(model), _no_forced_tools())
    return OpenRouterModel(model, provider=OpenRouterProvider(), profile=profile, settings=settings)


def resolve_model(name: Any) -> Any:
    """Return a pydantic-ai model for ``name``; a model object passes through.

    Qwen models on DashScope get ``tool_choice="auto"`` instead of ``"required"`` for structured
    output (pydantic-ai's Qwen profile sets that only for ``qwen-3-coder``; DashScope rejects forced
    tool choice for thinking models, pydantic-ai issue #1265) and no strict tool schemas. A
    text-only reply then triggers pydantic-ai's retry prompt for the output tool, bounded by
    ``output_retries``. ``openrouter:`` models reason (:func:`_openrouter_model`). Other strings
    (``"openai:..."``, ``"test"``, ...) are left for pydantic-ai to infer.
    """
    if not isinstance(name, str):
        return name
    provider, _, model = name.partition(":")
    if provider == "alibaba":
        return _alibaba_model(model)
    if provider == "openrouter":
        return _openrouter_model(model)
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

    def reasoning(self, role: str) -> bool:
        """Whether ``role``'s model is asked to reason, and so needs reasoning headroom."""
        return reasons(self.names.get(role))
