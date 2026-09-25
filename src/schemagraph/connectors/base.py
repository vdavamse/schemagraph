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

    def introspect(self) -> SchemaSnapshot:
        """Read the source's metadata into a stamped snapshot."""
        ...

    def check(self) -> str:
        """Cheap connectivity check; returns a human-readable status line."""
        ...


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: add a connector class to the registry under its ``type_name``."""
    _REGISTRY[cls.type_name] = cls
    return cls


def connector_types() -> list[str]:
    """Registered connector type names, sorted."""
    return sorted(_REGISTRY)


def config_schema(type_name: str) -> dict[str, Any]:
    """JSON schema of a connector type's ``Config`` model (drives the UI form)."""
    cls = _REGISTRY[type_name]
    cfg: type[BaseModel] = cls.Config
    return cfg.model_json_schema()


def make_connector(type_name: str, name: str, config: dict[str, Any]):
    """Instantiate a registered connector with a validated config.

    Args:
        type_name: Registry name of the connector type.
        name: Connection name; becomes the snapshot's ``source``.
        config: Plain dict validated against the connector's ``Config`` model.

    Returns:
        The connector instance.

    Raises:
        KeyError: If ``type_name`` is not registered.
    """
    if type_name not in _REGISTRY:
        raise KeyError(f"unknown connector type {type_name!r}; known: {connector_types()}")
    cls = _REGISTRY[type_name]
    return cls(name, cls.Config.model_validate(config))
