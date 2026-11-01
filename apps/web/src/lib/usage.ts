// Usage aggregation and CSV, pure (no database): the data layer fetches one row per
// (UTC day, model, key) and everything else is folded from those rows here.

import { microsToDecimalString } from "@/lib/money";

export interface UsageRow {
  day: string; // YYYY-MM-DD, UTC
  modelId: string;
  apiKeyId: string;
  keyName: string;
  keyPrefix: string;
  playground: boolean;
  requests: number;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  costMicros: bigint;
}

export interface UsageTotals {
  requests: number;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  costMicros: bigint;
}

export interface UsageGroup extends UsageTotals {
  key: string;
  label: string;
}

export function emptyTotals(): UsageTotals {
  return { requests: 0, promptTokens: 0, completionTokens: 0, cachedTokens: 0, costMicros: 0n };
}

function add(t: UsageTotals, r: UsageTotals): void {
  t.requests += r.requests;
  t.promptTokens += r.promptTokens;
  t.completionTokens += r.completionTokens;
  t.cachedTokens += r.cachedTokens;
  t.costMicros += r.costMicros;
}

export function totalsOf(rows: UsageRow[]): UsageTotals {
  const t = emptyTotals();
  for (const r of rows) add(t, r);
  return t;
}

export function keyLabel(r: Pick<UsageRow, "keyName" | "keyPrefix" | "playground">): string {
  if (r.playground) return "Playground";
  return r.keyName ? `${r.keyName} (${r.keyPrefix}…)` : `${r.keyPrefix}…`;
}

export function groupUsage(
  rows: UsageRow[],
  by: "day" | "model" | "key",
): UsageGroup[] {
  const groups = new Map<string, UsageGroup>();
  for (const r of rows) {
    const key = by === "day" ? r.day : by === "model" ? r.modelId : r.playground ? "playground" : r.apiKeyId;
    const label = by === "day" ? r.day : by === "model" ? r.modelId : keyLabel(r);
    let g = groups.get(key);
    if (!g) {
      g = { key, label, ...emptyTotals() };
      groups.set(key, g);
    }
    add(g, r);
  }
  const out = [...groups.values()];
  if (by === "day") out.sort((a, b) => (a.key < b.key ? 1 : a.key > b.key ? -1 : 0)); // newest first
  else out.sort((a, b) => (b.costMicros > a.costMicros ? 1 : b.costMicros < a.costMicros ? -1 : 0));
  return out;
}

export interface DayRange {
  from: string; // inclusive, YYYY-MM-DD (UTC)
  to: string; // inclusive, YYYY-MM-DD (UTC)
}

const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
const DAY_MS = 86_400_000;
export const MAX_RANGE_DAYS = 366;

function isoDay(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function parseDay(s: string | undefined): Date | null {
  if (!s || !DAY_RE.test(s)) return null;
  const d = new Date(`${s}T00:00:00Z`);
  return Number.isNaN(d.getTime()) || isoDay(d) !== s ? null : d;
}

/** A day range from query parameters; defaults to the last 30 days, clamped to a year. */
export function parseDayRange(
  params: { from?: string | string[]; to?: string | string[] },
  now: Date = new Date(),
): DayRange {
  const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v);
  const today = parseDay(isoDay(now))!;
  let to = parseDay(one(params.to)) ?? today;
  if (to > today) to = today;
  let from = parseDay(one(params.from)) ?? new Date(to.getTime() - 29 * DAY_MS);
  if (from > to) from = to;
  if ((to.getTime() - from.getTime()) / DAY_MS >= MAX_RANGE_DAYS) {
    from = new Date(to.getTime() - (MAX_RANGE_DAYS - 1) * DAY_MS);
  }
  return { from: isoDay(from), to: isoDay(to) };
}

/** [start, end) timestamps covering the inclusive day range. */
export function rangeBounds(r: DayRange): [Date, Date] {
  return [new Date(`${r.from}T00:00:00Z`), new Date(new Date(`${r.to}T00:00:00Z`).getTime() + DAY_MS)];
}

function csvCell(v: string | number): string {
  const s = String(v);
  // Quote anything with a delimiter, quote or newline; neutralize spreadsheet formulas.
  const safe = /^[=+\-@\t\r]/.test(s) && typeof v === "string" ? `'${s}` : s;
  return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

export const CSV_HEADER = [
  "date_utc",
  "model",
  "key_name",
  "key_prefix",
  "requests",
  "prompt_tokens",
  "completion_tokens",
  "cached_prompt_tokens",
  "cost_usd",
];

export function usageCsv(rows: UsageRow[]): string {
  const lines = [CSV_HEADER.join(",")];
  for (const r of rows) {
    lines.push(
      [
        r.day,
        r.modelId,
        r.playground ? "Playground" : r.keyName,
        r.keyPrefix,
        r.requests,
        r.promptTokens,
        r.completionTokens,
        r.cachedTokens,
        microsToDecimalString(r.costMicros),
      ]
        .map(csvCell)
        .join(","),
    );
  }
  return lines.join("\r\n") + "\r\n";
}
