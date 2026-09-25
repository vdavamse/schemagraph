from __future__ import annotations

import os
import sqlite3
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the suite never touches the network; an uncached model skips its tests
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

try:  # the agent extra: any real model call in a test is an error, not a network request
    import pydantic_ai.models as _pydantic_ai_models

    _pydantic_ai_models.ALLOW_MODEL_REQUESTS = False
except ImportError:
    pass

import duckdb
import pytest

from schemagraph.connectors.ddl import DDLConfig, parse_ddl
from schemagraph.graph import build_graph
from schemagraph.model import SchemaSnapshot

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def embed_model():
    """The default embedding model name, if the ``embed`` extra is installed and the model is cached."""
    pytest.importorskip("model2vec")
    from schemagraph.linking.embed import DEFAULT_MODEL, load_model

    try:
        load_model(DEFAULT_MODEL)
    except Exception as e:  # not in the HF cache and HF_HUB_OFFLINE is set
        pytest.skip(f"embedding model not cached: {e}")
    return DEFAULT_MODEL

STORE_DDL = """
CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR, email VARCHAR, state VARCHAR, city VARCHAR);
CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customer(id),
  store_id INT,
  total_amount NUMERIC,
  order_date TIMESTAMP,
  status VARCHAR COMMENT 'current order status'
);
CREATE TABLE order_items (
  id INT PRIMARY KEY, order_id INT, product_id INT, quantity INT, line_total NUMERIC,
  FOREIGN KEY (order_id) REFERENCES orders(id),
  FOREIGN KEY (product_id) REFERENCES products(id)
);
CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR, category_id INT, price NUMERIC, FOREIGN KEY (category_id) REFERENCES product_category(id));
CREATE TABLE product_category (id INT PRIMARY KEY, name VARCHAR, segment VARCHAR);
CREATE TABLE shipment (id INT PRIMARY KEY, order_item_id INT REFERENCES order_items(id), shipped_date DATE, carrier VARCHAR);
CREATE TABLE audit_log (id INT PRIMARY KEY, actor VARCHAR, action VARCHAR);
"""

# the same store as a database with rows, for the execution layer (DuckDB wants referenced tables first)
STORE_DB_DDL = """
CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR, state VARCHAR);
CREATE TABLE product_category (id INT PRIMARY KEY, name VARCHAR);
CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR, category_id INT REFERENCES product_category(id), price DECIMAL(10, 2));
CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT REFERENCES customer(id), total_amount DECIMAL(10, 2), order_date DATE);
CREATE TABLE order_items (id INT PRIMARY KEY, order_id INT REFERENCES orders(id), product_id INT REFERENCES products(id), quantity INT);
"""
STORE_DB_ROWS = """
INSERT INTO customer VALUES (1, 'Ann', 'CA'), (2, 'Bob', 'NY'), (3, 'Cy', 'CA');
INSERT INTO product_category VALUES (1, 'Toys'), (2, 'Books');
INSERT INTO products VALUES (1, 'Car', 1, 10), (2, 'Novel', 2, 20);
INSERT INTO orders VALUES (1, 1, 30, '2024-01-01'), (2, 2, 20, '2024-02-01'), (3, 3, 10, '2024-03-01');
INSERT INTO order_items VALUES (1, 1, 1, 1), (2, 1, 2, 1), (3, 2, 2, 1), (4, 3, 1, 1);
"""


@pytest.fixture
def store_snapshot() -> SchemaSnapshot:
    snap = parse_ddl(DDLConfig(ddl=STORE_DDL, dialect="postgres", default_schema="public"), "store")
    snap.table("public.customer").column("state").sample_values = ["California", "Texas", "New York"]
    snap.table("public.shipment").column("carrier").sample_values = ["UPS", "FedEx", "DHL"]
    return snap


@pytest.fixture
def store_graph(store_snapshot):
    return build_graph([store_snapshot])


@pytest.fixture
def dbt_airbnb_dir() -> Path:
    return FIXTURES / "dbt_airbnb"


@pytest.fixture
def store_duckdb(tmp_path) -> Path:
    """A small store database (customer, orders, order_items, products, product_category) with FKs."""
    path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(path))
    con.execute(STORE_DB_DDL + STORE_DB_ROWS)
    con.close()
    return path


@pytest.fixture
def store_sqlite(tmp_path) -> Path:
    path = tmp_path / "store.sqlite"
    con = sqlite3.connect(path)
    con.executescript(STORE_DB_DDL.replace("DECIMAL(10, 2)", "REAL") + STORE_DB_ROWS)
    con.commit()
    con.close()
    return path
