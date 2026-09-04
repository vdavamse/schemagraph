import { useEffect, useState } from "react";
import { api, Connection, ConnectorType, JsonSchema } from "../api";

const HIDDEN_TYPES = new Set(["ddl"]); // DDL has its own page

function fieldType(prop: any, defs: Record<string, any> | undefined): { kind: "string" | "number" | "boolean" | "list" | "text"; secret: boolean; desc: string } {
  const variants = prop.anyOf ?? prop.oneOf ?? [prop];
  const nonNull = variants.find((v: any) => v.type !== "null") ?? variants[0];
  const t = nonNull.type;
  const secret = /token|password|secret/i.test(prop.title ?? "") || /token|password|secret/i.test(prop.description ?? "");
  const desc = prop.description ?? "";
  if (t === "boolean") return { kind: "boolean", secret, desc };
  if (t === "integer" || t === "number") return { kind: "number", secret, desc };
  if (t === "array") return { kind: "list", secret, desc };
  if (t === "string" && /statements|ddl/i.test(desc)) return { kind: "text", secret, desc };
  void defs;
  return { kind: "string", secret, desc };
}

function SchemaForm({ schema, value, onChange }: { schema: JsonSchema; value: Record<string, any>; onChange: (v: Record<string, any>) => void }) {
  const props = schema.properties ?? {};
  const required = new Set(schema.required ?? []);
  return (
    <>
      {Object.entries(props).map(([name, prop]) => {
        const ft = fieldType(prop, schema.$defs);
        const v = value[name];
        const label = (
          <label>
            {name}
            {required.has(name) ? " *" : ""} <span className="muted">— {ft.desc}</span>
          </label>
        );
        if (ft.kind === "boolean")
          return (
            <div key={name}>
              {label}
              <input type="checkbox" checked={!!(v ?? prop.default)} onChange={(e) => onChange({ ...value, [name]: e.target.checked })} />
            </div>
          );
        if (ft.kind === "list")
          return (
            <div key={name}>
              {label}
              <input type="text" placeholder="comma-separated" value={Array.isArray(v) ? v.join(", ") : (v ?? "")} onChange={(e) => onChange({ ...value, [name]: e.target.value })} />
            </div>
          );
        if (ft.kind === "text")
          return (
            <div key={name}>
              {label}
              <textarea value={v ?? ""} onChange={(e) => onChange({ ...value, [name]: e.target.value })} />
            </div>
          );
        return (
          <div key={name}>
            {label}
            <input
              type={ft.secret ? "password" : ft.kind === "number" ? "number" : "text"}
              placeholder={ft.secret ? "${ENV_VAR} references are allowed" : prop.default != null ? String(prop.default) : ""}
              value={v ?? ""}
              onChange={(e) => onChange({ ...value, [name]: ft.kind === "number" ? (e.target.value === "" ? "" : Number(e.target.value)) : e.target.value })}
            />
          </div>
        );
      })}
    </>
  );
}

function normalize(schema: JsonSchema, raw: Record<string, any>): Record<string, any> {
  const out: Record<string, any> = {};
  for (const [k, prop] of Object.entries(schema.properties ?? {})) {
    const ft = fieldType(prop, schema.$defs);
    let v = raw[k];
    if (v === "" || v === undefined) continue;
    if (ft.kind === "list" && typeof v === "string") v = v.split(",").map((s) => s.trim()).filter(Boolean);
    out[k] = v;
  }
  return out;
}

export default function ConnectionsPage({ onChange }: { onChange: () => void }) {
  const [types, setTypes] = useState<ConnectorType[]>([]);
  const [conns, setConns] = useState<Connection[]>([]);
  const [type, setType] = useState("unity_catalog");
  const [name, setName] = useState("");
  const [cfg, setCfg] = useState<Record<string, any>>({});
  const [build, setBuild] = useState(true);
  const [msg, setMsg] = useState<{ kind: "ok" | "err" | "warn"; text: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = () => Promise.all([api.connectorTypes(), api.connections()]).then(([t, c]) => { setTypes(t); setConns(c); });
  useEffect(() => { load().catch((e) => setMsg({ kind: "err", text: String(e) })); }, []);

  const schema = types.find((t) => t.type === type)?.schema ?? {};

  async function submit() {
    setBusy("add");
    setMsg(null);
    try {
      const res = await api.addConnection({ name, type, config: normalize(schema, cfg), build });
      setMsg({ kind: res.warnings?.length ? "warn" : "ok", text: `${res.name}: ${res.tables} tables, ${res.edges} edges${res.warnings?.length ? ` — ${res.warnings.slice(0, 3).join("; ")}` : ""}` });
      setName(""); setCfg({});
      await load(); onChange();
    } catch (e) {
      setMsg({ kind: "err", text: String(e) });
    } finally { setBusy(null); }
  }

  async function act(kind: "check" | "build" | "delete", n: string) {
    setBusy(`${kind}:${n}`); setMsg(null);
    try {
      if (kind === "check") setMsg({ kind: "ok", text: `${n}: ${(await api.checkConnection(n)).status}` });
      if (kind === "build") { const r = await api.buildConnection(n); setMsg({ kind: r.warnings?.length ? "warn" : "ok", text: `${n}: ${r.tables} tables, ${r.edges} edges, ${r.terms} terms` }); }
      if (kind === "delete") { if (!confirm(`Remove connection ${n} and its snapshot?`)) return; await api.deleteConnection(n); }
      await load(); onChange();
    } catch (e) { setMsg({ kind: "err", text: String(e) }); } finally { setBusy(null); }
  }

  return (
    <div className="grid side">
      <div className="card">
        <h2>Add a catalog connection</h2>
        <label>type</label>
        <select value={type} onChange={(e) => { setType(e.target.value); setCfg({}); }}>
          {types.filter((t) => !HIDDEN_TYPES.has(t.type)).map((t) => <option key={t.type} value={t.type}>{t.type}</option>)}
        </select>
        <label>name *</label>
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. prod-unity" />
        <SchemaForm schema={schema} value={cfg} onChange={setCfg} />
        <div className="row" style={{ marginTop: 12 }}>
          <label style={{ margin: 0 }}><input type="checkbox" checked={build} onChange={(e) => setBuild(e.target.checked)} />build now</label>
          <span style={{ flex: 1 }} />
          <button className="primary" disabled={!name || busy === "add"} onClick={submit}>{busy === "add" ? "working…" : "Add"}</button>
        </div>
        <p className="muted" style={{ fontSize: 12 }}>Secrets can be written as <code>${"{"}ENV_VAR{"}"}</code>; they are resolved when the connector runs and never returned by the API.</p>
        {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
      </div>
      <div className="card">
        <h2>Connections</h2>
        {conns.length === 0 && <p className="muted">No connections yet. Add one on the left, or paste DDL.</p>}
        {conns.length > 0 && (
          <table className="list">
            <thead><tr><th>name</th><th>type</th><th>tables</th><th>edges</th><th>terms</th><th>built</th><th></th></tr></thead>
            <tbody>
              {conns.map((c) => (
                <tr key={c.name}>
                  <td><b>{c.name}</b>{c.warnings.length > 0 && <div className="muted" style={{ fontSize: 11 }}>{c.warnings.length} warning(s): {c.warnings[0]}</div>}</td>
                  <td><span className="pill">{c.type}</span></td>
                  <td>{c.n_tables}</td><td>{c.n_edges}</td><td>{c.n_terms}</td>
                  <td className="muted">{c.built_at ? new Date(c.built_at).toLocaleString() : "—"}</td>
                  <td className="row end">
                    <button className="ghost" disabled={!!busy} onClick={() => act("check", c.name)}>check</button>
                    <button className="ghost" disabled={!!busy} onClick={() => act("build", c.name)}>{busy === `build:${c.name}` ? "…" : "rebuild"}</button>
                    <button className="danger" disabled={!!busy} onClick={() => act("delete", c.name)}>remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
