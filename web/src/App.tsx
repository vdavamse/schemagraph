import { useEffect, useState } from "react";
import { api } from "./api";
import ConnectionsPage from "./pages/Connections";
import DDLPage from "./pages/DDL";
import PlaygroundPage from "./pages/Playground";
import GraphPage from "./pages/Graph";
import GlossaryPage from "./pages/Glossary";

type Page = "connections" | "ddl" | "playground" | "graph" | "glossary";

const PAGES: { id: Page; label: string }[] = [
  { id: "connections", label: "Connections" },
  { id: "ddl", label: "Paste DDL" },
  { id: "playground", label: "Link playground" },
  { id: "graph", label: "Graph" },
  { id: "glossary", label: "Glossary & hints" },
];

export default function App() {
  const [page, setPage] = useState<Page>(() => (location.hash.replace("#", "") as Page) || "connections");
  const [stats, setStats] = useState<Record<string, any> | null>(null);
  const [version, setVersion] = useState(0);
  const refresh = () => setVersion((v) => v + 1);

  useEffect(() => {
    api.health().then(setStats).catch(() => setStats(null));
  }, [version, page]);

  useEffect(() => {
    location.hash = page;
  }, [page]);

  return (
    <>
      <header className="top">
        <h1>schemagraph</h1>
        <nav>
          {PAGES.map((p) => (
            <button key={p.id} className={page === p.id ? "active" : ""} onClick={() => setPage(p.id)}>
              {p.label}
            </button>
          ))}
        </nav>
        <div className="stats">
          {stats
            ? `${stats.tables} tables · ${stats.columns} columns · ${stats.relations} relations · ${stats.terms} terms · ${stats.sources} sources · llm ${stats.llm ? "on" : "off"}`
            : "backend unreachable"}
        </div>
      </header>
      <main>
        {page === "connections" && <ConnectionsPage onChange={refresh} />}
        {page === "ddl" && <DDLPage onChange={refresh} />}
        {page === "playground" && <PlaygroundPage llm={!!stats?.llm} />}
        {page === "graph" && <GraphPage key={version} />}
        {page === "glossary" && <GlossaryPage onChange={refresh} />}
      </main>
    </>
  );
}
