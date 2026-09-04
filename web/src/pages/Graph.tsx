import { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";
import { api } from "../api";

const EDGE_COLORS: Record<string, string> = {
  foreign_key: "#178a4c",
  relationship_test: "#178a4c",
  join_hint: "#0b8f8f",
  lineage: "#5b3fc9",
  catalog_relation: "#b4560b",
  inferred: "#c0392b",
};
const KIND_COLORS: Record<string, string> = { table: "#2f6fed", view: "#7c9cf0", model: "#5b3fc9", source: "#8a8f9c", seed: "#8a8f9c", snapshot: "#8a8f9c", external: "#b4560b" };

export default function GraphPage() {
  const ref = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<any>(null);
  const [count, setCount] = useState<{ n: number; e: number } | null>(null);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cy: cytoscape.Core | null = null;
    api.graph().then((g) => {
      if (!ref.current) return;
      setCount({ n: g.nodes.length, e: g.edges.length });
      const ids = new Set(g.nodes.map((n) => n.id));
      cy = cytoscape({
        container: ref.current,
        elements: [
          ...g.nodes.map((n) => ({ data: { id: n.id, label: n.id.split(".").slice(-2).join("."), kind: n.kind, columns: n.columns } })),
          ...g.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e, i) => ({ data: { id: `e${i}`, source: e.source, target: e.target, kind: e.kind, on: e.on } })),
        ],
        style: [
          { selector: "node", style: { label: "data(label)", "font-size": 9, "text-valign": "bottom", "text-margin-y": 4, width: 22, height: 22, "background-color": (ele: any) => KIND_COLORS[ele.data("kind")] ?? "#2f6fed", color: "#1f2430" } },
          { selector: "node.hit", style: { "border-width": 3, "border-color": "#f59e0b" } },
          { selector: "node.dim", style: { opacity: 0.15 } },
          { selector: "edge", style: { width: 1.5, "line-color": (ele: any) => EDGE_COLORS[ele.data("kind")] ?? "#999", "curve-style": "bezier", "target-arrow-shape": (ele: any) => (ele.data("kind") === "lineage" ? "triangle" : "none"), "target-arrow-color": (ele: any) => EDGE_COLORS[ele.data("kind")] ?? "#999", opacity: 0.8 } },
          { selector: "edge.dim", style: { opacity: 0.08 } },
        ],
        layout: { name: g.nodes.length > 300 ? "grid" : "cose", animate: false, nodeRepulsion: () => 8000, idealEdgeLength: () => 80 } as any,
        wheelSensitivity: 0.2,
      });
      cy.on("tap", "node", async (evt) => {
        const id = evt.target.id();
        setSelected(await api.table(id));
      });
      (ref.current as any).__cy = cy;
    });
    return () => { cy?.destroy(); };
  }, []);

  useEffect(() => {
    const cy: cytoscape.Core | undefined = (ref.current as any)?.__cy;
    if (!cy) return;
    cy.elements().removeClass("hit dim");
    if (!filter.trim()) return;
    const f = filter.toLowerCase();
    const hits = cy.nodes().filter((n) => n.id().toLowerCase().includes(f));
    cy.elements().addClass("dim");
    hits.removeClass("dim").addClass("hit");
    hits.connectedEdges().removeClass("dim").connectedNodes().removeClass("dim");
  }, [filter]);

  return (
    <div className="grid side" style={{ gridTemplateColumns: "1fr 360px" }}>
      <div className="card">
        <div className="row">
          <h2 style={{ margin: 0 }}>Schema graph {count && <span className="muted" style={{ fontWeight: 400 }}>· {count.n} tables, {count.e} relations</span>}</h2>
          <span style={{ flex: 1 }} />
          <input type="text" style={{ width: 260 }} placeholder="highlight tables containing…" value={filter} onChange={(e) => setFilter(e.target.value)} />
        </div>
        <div className="legend">
          {Object.entries(EDGE_COLORS).map(([k, c]) => <span key={k} style={{ color: c }}>{k}</span>)}
        </div>
        <div id="cy" ref={ref} />
      </div>
      <div className="card">
        <h2>Table</h2>
        {!selected && <p className="muted">Click a node.</p>}
        {selected && (
          <>
            <div className="mono"><b>{selected.fqn}</b> <span className="pill">{selected.kind}</span></div>
            {selected.description && <p>{selected.description}</p>}
            <div className="muted" style={{ fontSize: 12 }}>source: {selected.source}{selected.row_count != null && ` · ${selected.row_count.toLocaleString()} rows`}{selected.owner && ` · owner ${selected.owner}`}</div>
            {selected.tags?.length > 0 && <div style={{ marginTop: 4 }}>{selected.tags.map((t: string) => <span key={t} className="pill">{t}</span>)}</div>}
            <h3>Columns ({selected.columns.length})</h3>
            <div style={{ maxHeight: 300, overflow: "auto" }}>
              {selected.columns.map((c: any) => (
                <div key={c.name} className="mono" style={{ fontSize: 12 }}>
                  {c.is_primary_key && "🔑 "}{c.name} <span className="muted">{c.data_type}</span>{c.description && <span className="muted"> — {c.description}</span>}
                  {c.sample_values?.length > 0 && <div className="muted" style={{ paddingLeft: 12, fontSize: 11 }}>{c.sample_values.slice(0, 5).join(", ")}</div>}
                </div>
              ))}
            </div>
            <h3>Relations ({selected.relations.length})</h3>
            {selected.relations.map((r: any, i: number) => (
              <div key={i} className="mono" style={{ fontSize: 12, marginBottom: 4 }}>
                <span className={`pill kind-${r.kind}`}>{r.kind}</span>{r.from_table} → {r.to_table}
                {r.from_columns.length > 0 && <div className="muted" style={{ paddingLeft: 12 }}>{r.from_columns.join(",")} = {r.to_columns.join(",")}</div>}
                <div className="muted" style={{ paddingLeft: 12, fontSize: 11 }}>{r.source}{r.description ? ` · ${r.description}` : ""}</div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
