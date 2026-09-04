import { useState } from "react";
import { api } from "../api";

const EXAMPLE = `CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR, state VARCHAR);
CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customer(id),
  total_amount NUMERIC,
  order_date TIMESTAMP,
  status VARCHAR COMMENT 'current order status'
);`;

export default function DDLPage({ onChange }: { onChange: () => void }) {
  const [name, setName] = useState("ddl");
  const [dialect, setDialect] = useState("");
  const [schema, setSchema] = useState("");
  const [ddl, setDdl] = useState("");
  const [msg, setMsg] = useState<{ kind: "ok" | "err" | "warn"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true); setMsg(null);
    try {
      const r = await api.addDDL({ name, ddl, dialect: dialect || null, default_schema: schema || null });
      setMsg({ kind: r.warnings?.length ? "warn" : "ok", text: `${r.name}: ${r.tables} tables, ${r.edges} foreign keys${r.warnings?.length ? ` — ${r.warnings.join("; ")}` : ""}` });
      onChange();
    } catch (e) { setMsg({ kind: "err", text: String(e) }); } finally { setBusy(false); }
  }

  return (
    <div className="grid side">
      <div className="card">
        <h2>Paste DDL</h2>
        <p className="muted">Any dialect sqlglot understands. Inline <code>REFERENCES</code>, table-level <code>FOREIGN KEY</code>, <code>ALTER TABLE … ADD FOREIGN KEY</code>, <code>COMMENT</code> clauses and <code>COMMENT ON</code> statements become graph edges and descriptions.</p>
        <label>connection name</label>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
        <label>dialect (optional)</label>
        <select value={dialect} onChange={(e) => setDialect(e.target.value)}>
          <option value="">auto</option>
          {["postgres", "snowflake", "bigquery", "duckdb", "mysql", "tsql", "oracle", "spark", "databricks", "redshift", "trino", "hive", "sqlite"].map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <label>default schema for unqualified tables (optional)</label>
        <input type="text" value={schema} onChange={(e) => setSchema(e.target.value)} placeholder="public" />
        <div className="row" style={{ marginTop: 12 }}>
          <button className="ghost" onClick={() => setDdl(EXAMPLE)}>load example</button>
          <span style={{ flex: 1 }} />
          <button className="primary" disabled={!ddl.trim() || !name || busy} onClick={submit}>{busy ? "parsing…" : "Add & build"}</button>
        </div>
        {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
      </div>
      <div className="card">
        <textarea style={{ minHeight: "70vh" }} value={ddl} onChange={(e) => setDdl(e.target.value)} placeholder="CREATE TABLE …" spellCheck={false} />
      </div>
    </div>
  );
}
