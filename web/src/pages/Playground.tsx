import { useState } from "react";
import { api, LinkResult } from "../api";

export default function PlaygroundPage({ llm }: { llm: boolean }) {
  const [q, setQ] = useState("");
  const [maxTables, setMaxTables] = useState(20);
  const [columns, setColumns] = useState("relevant");
  const [useLlm, setUseLlm] = useState(false);
  const [res, setRes] = useState<LinkResult | null>(null);
  const [explain, setExplain] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<"tables" | "ddl" | "explain">("tables");

  async function run() {
    setBusy(true); setErr(null); setExplain(null);
    try {
      setRes(await api.link({ question: q, max_tables: maxTables, columns, use_llm: useLlm }));
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }
  async function loadExplain() {
    setTab("explain");
    if (!explain) setExplain(await api.explain(q));
  }

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr" }}>
      <div className="card">
        <h2>Link a question to its sub-schema</h2>
        <div className="row">
          <input type="text" style={{ flex: 1 }} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && q && run()} placeholder="total revenue by product category for customers in California last quarter" />
          <button className="primary" disabled={!q || busy} onClick={run}>{busy ? "linking…" : "Link"}</button>
        </div>
        <div className="row" style={{ marginTop: 8, fontSize: 12 }}>
          <label style={{ margin: 0 }}>max tables <input type="number" style={{ width: 70, display: "inline-block" }} value={maxTables} onChange={(e) => setMaxTables(Number(e.target.value))} /></label>
          <label style={{ margin: 0 }}>columns <select style={{ width: 120, display: "inline-block" }} value={columns} onChange={(e) => setColumns(e.target.value)}><option value="relevant">relevant</option><option value="all">all</option></select></label>
          <label style={{ margin: 0 }} title={llm ? "" : "set ANTHROPIC_API_KEY on the backend to enable"}><input type="checkbox" disabled={!llm} checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />LLM anchor pass {llm ? "" : "(unavailable)"}</label>
        </div>
        {err && <div className="msg err">{err}</div>}
      </div>
      {res && (
        <div className="card">
          <div className="row">
            <span className="muted">
              {res.tables.length} tables · {res.join_paths.length} join paths · {res.stats.ms} ms · llm {res.stats.llm}
              {res.terms_matched.length > 0 && <> · matched: {res.terms_matched.map((t) => <span key={t} className="pill">{t}</span>)}</>}
            </span>
            <span style={{ flex: 1 }} />
            <button className={tab === "tables" ? "primary" : "ghost"} onClick={() => setTab("tables")}>tables & paths</button>
            <button className={tab === "ddl" ? "primary" : "ghost"} onClick={() => setTab("ddl")}>DDL for the LLM</button>
            <button className={tab === "explain" ? "primary" : "ghost"} onClick={loadExplain}>explain</button>
            <button className="ghost" onClick={() => navigator.clipboard.writeText(res.ddl)}>copy DDL</button>
          </div>
          {tab === "tables" && (
            <>
              <h3>Join paths (PathRAG-pruned, most reliable first)</h3>
              {res.join_paths.length === 0 && <p className="muted">No join path needed or anchors are not connected.</p>}
              {res.join_paths.map((p, i) => (
                <div key={i} style={{ marginBottom: 8 }}>
                  <div className="mono">{p.tables.join("  →  ")} <span className="score">reliability {p.reliability}</span></div>
                  {p.steps.map((s, j) => <div key={j} className="mono muted" style={{ paddingLeft: 12 }}><span className={`pill kind-${s.kind}`}>{s.kind}</span>{s.on} <span className="score">({s.source})</span></div>)}
                </div>
              ))}
              <h3>Tables</h3>
              {res.tables.map((t) => (
                <details key={t.fqn} open={t.is_anchor}>
                  <summary>
                    <b className="mono" style={{ color: "var(--text)" }}>{t.fqn}</b> {t.is_anchor && <span className="pill anchor">anchor</span>}<span className="pill">{t.kind}</span>
                    <span className="score">score {t.score}</span> {t.description && <span className="muted">— {t.description}</span>}
                  </summary>
                  <div className="cols" style={{ padding: "6px 0 10px 14px" }}>
                    {t.columns.map((c) => (
                      <div key={c.name} className="c"><span className="n">{c.name}</span><span className="muted">{c.data_type}</span>{c.reason && <span className="r">· {c.reason}</span>}</div>
                    ))}
                  </div>
                </details>
              ))}
            </>
          )}
          {tab === "ddl" && <pre className="ddl">{res.ddl}</pre>}
          {tab === "explain" && (explain ? <pre className="ddl">{JSON.stringify(explain, null, 2)}</pre> : <p className="muted">loading…</p>)}
        </div>
      )}
    </div>
  );
}
