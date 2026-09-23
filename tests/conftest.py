from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the suite never touches the network; an uncached model skips its tests

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
