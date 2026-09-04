-- Example schema: paste into the UI or run `schemagraph add-ddl examples/store.sql --dialect postgres --schema public`
CREATE TABLE customer (
  id INT PRIMARY KEY,
  name VARCHAR,
  email VARCHAR,
  state VARCHAR COMMENT 'US state, e.g. California',
  city VARCHAR
);
COMMENT ON TABLE customer IS 'One row per customer account';

CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customer(id),
  store_id INT,
  total_amount NUMERIC COMMENT 'order revenue in USD',
  order_date TIMESTAMP,
  status VARCHAR COMMENT 'current order status: paid, refunded, pending'
);

CREATE TABLE product_category (id INT PRIMARY KEY, name VARCHAR, segment VARCHAR);

CREATE TABLE products (
  id INT PRIMARY KEY,
  name VARCHAR,
  category_id INT,
  price NUMERIC,
  FOREIGN KEY (category_id) REFERENCES product_category(id)
);
COMMENT ON TABLE products IS 'Product catalog: items by category and price';

CREATE TABLE order_items (
  id INT PRIMARY KEY,
  order_id INT,
  product_id INT,
  quantity INT,
  line_total NUMERIC,
  FOREIGN KEY (order_id) REFERENCES orders(id),
  FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE shipment (
  id INT PRIMARY KEY,
  order_item_id INT REFERENCES order_items(id),
  shipped_date DATE,
  carrier VARCHAR COMMENT 'UPS, FedEx, DHL'
);

CREATE TABLE audit_log (id INT PRIMARY KEY, actor VARCHAR, action VARCHAR, at TIMESTAMP);
