import { useEffect, useState } from "react";
import { api, JoinHint, Term } from "../api";

const split = (s: string) => s.split(",").map((x) => x.trim()).filter(Boolean);

export default function GlossaryPage({ onChange }: { onChange: () => void }) {
  const [terms, setTerms] = useState<Term[]>([]);
  const [hints, setHints] = useState<JoinHint[]>([]);
  const [t, setT] = useState({ name: "", description: "", synonyms: "", targets: "" });
  const [h, setH] = useState({ from_table: "", to_table: "", from_columns: "", to_columns: "", description: "" });
  const [msg, setMsg] = useState<string | null>(null);

  const load = () => Promise.all([api.glossary(), api.joinHints()]).then(([g, j]) => { setTerms(g); setHints(j); });
  useEffect(() => { load().catch((e) => setMsg(String(e))); }, []);

  async function addTerm() {
    try {
      await api.upsertTerm({ name: t.name, description: t.description || undefined, synonyms: split(t.synonyms), targets: split(t.targets) });
      setT({ name: "", description: "", synonyms: "", targets: "" }); await load(); onChange();
    } catch (e) { setMsg(String(e)); }
  }
  async function addHint() {
    try {
      await api.addJoinHint({ from_table: h.from_table, to_table: h.to_table, from_columns: split(h.from_columns), to_columns: split(h.to_columns), description: h.description || undefined });
      setH({ from_table: "", to_table: "", from_columns: "", to_columns: "", description: "" }); await load(); onChange();
    } catch (e) { setMsg(String(e)); }
  }

  return (
    <div className="grid two">
      <div className="card">
        <h2>Business glossary</h2>
        <p className="muted">A term maps a business word to tables or <code>table.column</code> targets. Terms from Collibra and dbt semantic models appear here too; user terms override them.</p>
        <label>term *</label><input type="text" value={t.name} onChange={(e) => setT({ ...t, name: e.target.value })} placeholder="revenue" />
        <label>description</label><input type="text" value={t.description} onChange={(e) => setT({ ...t, description: e.target.value })} />
        <label>synonyms (comma-separated)</label><input type="text" value={t.synonyms} onChange={(e) => setT({ ...t, synonyms: e.target.value })} placeholder="turnover, sales" />
        <label>targets (comma-separated table or table.column)</label><input type="text" value={t.targets} onChange={(e) => setT({ ...t, targets: e.target.value })} placeholder="public.orders.total_amount" />
        <div className="row end" style={{ marginTop: 10 }}><button className="primary" disabled={!t.name} onClick={addTerm}>Save term</button></div>
        <h3>Terms ({terms.length})</h3>
        <table className="list">
          <thead><tr><th>term</th><th>synonyms</th><th>targets</th><th>source</th><th></th></tr></thead>
          <tbody>
            {terms.map((x) => (
              <tr key={x.name}>
                <td><b>{x.name}</b>{x.description && <div className="muted" style={{ fontSize: 11 }}>{x.description}</div>}</td>
                <td>{x.synonyms.join(", ")}</td>
                <td className="mono">{x.targets.join(", ")}</td>
                <td><span className="pill">{x.source}</span></td>
                <td>{x.source.includes("user") && <button className="danger" onClick={async () => { await api.deleteTerm(x.name); await load(); onChange(); }}>remove</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card">
        <h2>Join hints</h2>
        <p className="muted">Declare a join the catalogs do not know about (common in BigQuery, Snowflake, Glue where no FK is enforced). Hints are trusted like foreign keys.</p>
        <label>from table *</label><input type="text" value={h.from_table} onChange={(e) => setH({ ...h, from_table: e.target.value })} placeholder="lake.events" />
        <label>to table *</label><input type="text" value={h.to_table} onChange={(e) => setH({ ...h, to_table: e.target.value })} placeholder="sales.core.customers" />
        <label>from columns</label><input type="text" value={h.from_columns} onChange={(e) => setH({ ...h, from_columns: e.target.value })} placeholder="user_id" />
        <label>to columns</label><input type="text" value={h.to_columns} onChange={(e) => setH({ ...h, to_columns: e.target.value })} placeholder="id" />
        <label>description</label><input type="text" value={h.description} onChange={(e) => setH({ ...h, description: e.target.value })} />
        <div className="row end" style={{ marginTop: 10 }}><button className="primary" disabled={!h.from_table || !h.to_table} onClick={addHint}>Save hint</button></div>
        <h3>Hints ({hints.length})</h3>
        <table className="list">
          <thead><tr><th>from</th><th>to</th><th>on</th><th></th></tr></thead>
          <tbody>
            {hints.map((x) => (
              <tr key={x.id}>
                <td className="mono">{x.from_table}</td><td className="mono">{x.to_table}</td>
                <td className="mono">{x.from_columns.join(",")} = {x.to_columns.join(",")}</td>
                <td><button className="danger" onClick={async () => { await api.deleteJoinHint(x.id); await load(); onChange(); }}>remove</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        {msg && <div className="msg err">{msg}</div>}
      </div>
    </div>
  );
}
