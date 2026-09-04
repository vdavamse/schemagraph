export type JsonSchema = {
  properties?: Record<string, any>;
  required?: string[];
  $defs?: Record<string, any>;
};

export type ConnectorType = { type: string; schema: JsonSchema };

export type Connection = {
  name: string;
  type: string;
  config: Record<string, any>;
  built: boolean;
  n_tables: number;
  n_edges: number;
  n_terms: number;
  built_at: string | null;
  warnings: string[];
};

export type LinkedColumn = { name: string; data_type?: string | null; description?: string | null; score: number; reason?: string | null };
export type LinkedTable = { fqn: string; score: number; is_anchor: boolean; columns: LinkedColumn[]; description?: string | null; kind: string };
export type JoinStep = { from_table: string; to_table: string; kind: string; on: string; source: string };
export type JoinPath = { tables: string[]; steps: JoinStep[]; reliability: number };
export type LinkResult = {
  question: string;
  tables: LinkedTable[];
  join_paths: JoinPath[];
  anchors: string[];
  terms_matched: string[];
  glossary: Record<string, string[]>;
  ddl: string;
  stats: Record<string, any>;
};

export type TableSummary = { fqn: string; kind: string; columns: number; description?: string | null; source: string; row_count?: number | null; tags: string[] };
export type GraphExport = {
  nodes: { id: string; kind: string; columns: number; source: string }[];
  edges: { source: string; target: string; kind: string; on: string; provenance: string }[];
};
export type Term = { name: string; description?: string | null; synonyms: string[]; targets: string[]; source: string };
export type JoinHint = { id: number; from_table: string; to_table: string; from_columns: string[]; to_columns: string[]; description?: string | null };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = j.detail ?? JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  health: () => req<Record<string, any>>("/api/health"),
  connectorTypes: () => req<ConnectorType[]>("/api/connector-types"),
  connections: () => req<Connection[]>("/api/connections"),
  addConnection: (body: { name: string; type: string; config: Record<string, any>; build?: boolean }) =>
    req<any>("/api/connections", { method: "POST", body: JSON.stringify(body) }),
  addDDL: (body: { name: string; ddl: string; dialect?: string | null; default_schema?: string | null }) =>
    req<any>("/api/ddl", { method: "POST", body: JSON.stringify(body) }),
  deleteConnection: (name: string) => req<any>(`/api/connections/${encodeURIComponent(name)}`, { method: "DELETE" }),
  checkConnection: (name: string) => req<{ status: string }>(`/api/connections/${encodeURIComponent(name)}/check`, { method: "POST" }),
  buildConnection: (name: string) => req<any>(`/api/connections/${encodeURIComponent(name)}/build`, { method: "POST" }),
  tables: (q?: string) => req<TableSummary[]>(`/api/graph/tables${q ? `?q=${encodeURIComponent(q)}` : ""}`),
  table: (fqn: string) => req<any>(`/api/graph/tables/${encodeURIComponent(fqn)}`),
  graph: () => req<GraphExport>("/api/graph/export"),
  link: (body: { question: string; max_tables?: number; columns?: string; use_llm?: boolean; anchor_k?: number }) =>
    req<LinkResult>("/api/link", { method: "POST", body: JSON.stringify(body) }),
  explain: (question: string) => req<any>(`/api/explain?question=${encodeURIComponent(question)}`),
  glossary: () => req<Term[]>("/api/glossary"),
  upsertTerm: (t: { name: string; description?: string; synonyms: string[]; targets: string[] }) =>
    req<any>("/api/glossary", { method: "POST", body: JSON.stringify(t) }),
  deleteTerm: (name: string) => req<any>(`/api/glossary/${encodeURIComponent(name)}`, { method: "DELETE" }),
  joinHints: () => req<JoinHint[]>("/api/join-hints"),
  addJoinHint: (h: { from_table: string; to_table: string; from_columns: string[]; to_columns: string[]; description?: string }) =>
    req<any>("/api/join-hints", { method: "POST", body: JSON.stringify(h) }),
  deleteJoinHint: (id: number) => req<any>(`/api/join-hints/${id}`, { method: "DELETE" }),
};
