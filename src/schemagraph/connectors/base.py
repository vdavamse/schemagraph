"""Connector protocol and registry."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from schemagraph.model import SchemaSnapshot


@runtime_checkable
class Connector(Protocol):
    """A connector introspects one source and returns a snapshot.

    Implementations are constructed with ``(name, config)`` where ``config`` is a
    plain dict validated against the connector's ``Config`` model.
    """

    type_name: ClassVar[str]
    name: str

    def introspect(self) -> SchemaSnapshot: ...

    def check(self) -> str:
        """Cheap connectivity check; returns a human-readable status line."""
        ...


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    _REGISTRY[cls.type_name] = cls
    return cls


def connector_types() -> list[str]:
    return sorted(_REGISTRY)


def config_schema(type_name: str) -> dict[str, Any]:
    cls = _REGISTRY[type_name]
    cfg: type[BaseModel] = cls.Config
    return cfg.model_json_schema()


def make_connector(type_name: str, name: str, config: dict[str, Any]):
    if type_name not in _REGISTRY:
        raise KeyError(f"unknown connector type {type_name!r}; known: {connector_types()}")
    cls = _REGISTRY[type_name]
    return cls(name, cls.Config.model_validate(config))
