"""Connector registry. Importing this package registers all built-in connectors."""

from schemagraph.connectors import collibra, dbt, ddl, duckdb_conn, glue, unity  # noqa: F401
from schemagraph.connectors.base import config_schema, connector_types, make_connector

__all__ = ["config_schema", "connector_types", "make_connector"]
