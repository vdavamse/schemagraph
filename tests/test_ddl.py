from schemagraph.connectors.ddl import DDLConfig, parse_ddl


def test_parse_tables_and_fks(store_snapshot):
    fqns = {t.fqn for t in store_snapshot.tables}
    assert "public.orders" in fqns and "public.product_category" in fqns
    orders = store_snapshot.table("public.orders")
    assert orders.primary_key == ["id"]
    assert orders.column("status").description == "current order status"
    fks = {(e.from_table, e.to_table, tuple(e.from_columns), tuple(e.to_columns)) for e in store_snapshot.edges}
    assert ("public.orders", "public.customer", ("customer_id",), ("id",)) in fks  # inline REFERENCES
    assert ("public.order_items", "public.products", ("product_id",), ("id",)) in fks  # table-level FK, forward ref
    assert ("public.shipment", "public.order_items", ("order_item_id",), ("id",)) in fks
    assert all(e.kind == "foreign_key" for e in store_snapshot.edges)


def test_alter_table_fk_and_comment_on():
    ddl = """
    CREATE TABLE a (id INT PRIMARY KEY, b_id INT);
    CREATE TABLE b (id INT PRIMARY KEY);
    ALTER TABLE a ADD CONSTRAINT fk_ab FOREIGN KEY (b_id) REFERENCES b(id);
    COMMENT ON TABLE a IS 'the a table';
    COMMENT ON COLUMN a.b_id IS 'link to b';
    """
    snap = parse_ddl(DDLConfig(ddl=ddl, dialect="postgres"), "x")
    assert len(snap.edges) == 1 and snap.edges[0].from_table == "a" and snap.edges[0].to_table == "b"
    assert snap.table("a").description == "the a table"
    assert snap.table("a").column("b_id").description == "link to b"


def test_snowflake_qualified_names():
    ddl = "CREATE TABLE SALES.PUBLIC.ORDERS (ID NUMBER, AMOUNT NUMBER(10,2)); CREATE VIEW SALES.PUBLIC.V_ORDERS AS SELECT * FROM SALES.PUBLIC.ORDERS;"
    snap = parse_ddl(DDLConfig(ddl=ddl, dialect="snowflake"), "sf")
    assert snap.table("SALES.PUBLIC.ORDERS") is not None
    assert snap.table("SALES.PUBLIC.ORDERS").column("AMOUNT").data_type.upper().startswith("DECIMAL") or "NUMBER" in snap.table("SALES.PUBLIC.ORDERS").column("AMOUNT").data_type.upper()


def test_empty_ddl_warns():
    snap = parse_ddl(DDLConfig(ddl="SELECT 1"), "x")
    assert snap.warnings
