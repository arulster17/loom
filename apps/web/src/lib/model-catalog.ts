// The public model catalog from config/models.yaml, and pool-state labels. Pure, so
// it is unit tested; src/lib/models.ts does the file and Redis I/O.

export interface ModelInfo {
  id: string;
  displayName: string;
  maxContext: number;
  inputPerMtokMicros: number;
  cachedInputPerMtokMicros: number;
  outputPerMtokMicros: number;
  status: string; // registry status, e.g. "preview"
  startupNote: string | null;
}

export type PoolState = "ready" | "warming" | "cold" | "unavailable" | "unknown";

export interface PoolStatus {
  state: PoolState;
  retryAfterS?: number;
}

type Row = Record<string, unknown>;

/** YAML 1.1 ints like 77_000 (PyYAML reads them; js-yaml's YAML 1.2 keeps a string). */
function num(v: unknown): number {
  if (typeof v === "string" && /^\d[\d_]*$/.test(v)) return Number(v.replace(/_/g, ""));
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`models.yaml: not a number: ${String(v)}`);
  return n;
}

/** Cold starts the user should expect (measured: 70B ~7.5 min, budget 20). */
export function startupNote(id: string, params?: string): string | null {
  if (/70b/i.test(id) || /70b/i.test(params ?? "")) return "Can take up to ~20 min to start when it has been idle.";
  return null;
}

/** Servable models (priced and not disabled, as loom_platform.pricing.servable). */
export function catalogFromRegistry(doc: unknown): ModelInfo[] {
  const models = (doc as { models?: unknown })?.models;
  if (!Array.isArray(models)) throw new Error("models.yaml has no models list");
  const out: ModelInfo[] = [];
  for (const m of models as Row[]) {
    const pricing = m.pricing as Row | null | undefined;
    if (!pricing || m.status === "disabled") continue;
    const id = String(m.id);
    out.push({
      id,
      displayName: String(m.display_name ?? id),
      maxContext: num(m.max_context),
      inputPerMtokMicros: num(pricing.input_per_mtok),
      cachedInputPerMtokMicros: num(pricing.cached_input_per_mtok ?? pricing.input_per_mtok),
      outputPerMtokMicros: num(pricing.output_per_mtok),
      status: String(m.status ?? ""),
      startupNote: startupNote(id),
    });
  }
  return out;
}

/** A `loom:pool:<model>` hash (contracts §10). "ready" needs at least one replica. */
export function parsePoolHash(h: Record<string, string>): PoolStatus {
  const retry = Number(h.retry_after_s);
  const retryAfterS = Number.isFinite(retry) && retry > 0 ? retry : undefined;
  let replicas = 0;
  try {
    const list = JSON.parse(h.replicas ?? "[]");
    replicas = Array.isArray(list) ? list.length : 0;
  } catch {
    replicas = 0;
  }
  switch (h.state) {
    case "ready":
      return replicas > 0 ? { state: "ready" } : { state: "unavailable", retryAfterS };
    case "warming":
    case "cold":
    case "unavailable":
      return { state: h.state, retryAfterS };
    default:
      return { state: "unknown" };
  }
}

export function poolLabel(p: PoolStatus): { label: string; detail: string; tone: "ok" | "wait" | "bad" | "muted" } {
  const mins = p.retryAfterS ? Math.max(1, Math.round(p.retryAfterS / 60)) : null;
  switch (p.state) {
    case "ready":
      return { label: "Ready", detail: "Serving requests now.", tone: "ok" };
    case "warming":
      return {
        label: "Starting up",
        detail: mins ? `Should be ready in about ${mins} min.` : "Starting; try again shortly.",
        tone: "wait",
      };
    case "cold":
      return { label: "Asleep", detail: "Starts on the first request; that request gets a retry-later reply.", tone: "muted" };
    case "unavailable":
      return {
        label: "Temporarily unavailable",
        detail: mins ? `No GPU capacity right now. Try again in about ${mins} min.` : "No GPU capacity right now.",
        tone: "bad",
      };
    default:
      return { label: "Unknown", detail: "Status is not available right now.", tone: "muted" };
  }
}
