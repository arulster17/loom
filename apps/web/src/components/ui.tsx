import type { ReactNode } from "react";

// Small shared building blocks. Plain Tailwind; no component library.

export const btn =
  "inline-flex items-center justify-center gap-2 rounded-md bg-zinc-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50";
export const btnSecondary =
  "inline-flex items-center justify-center gap-2 rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium text-zinc-800 hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50";
export const btnDanger =
  "inline-flex items-center justify-center gap-2 rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50";
export const input =
  "block w-full rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm shadow-xs focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500";
export const label = "block text-sm font-medium text-zinc-700";
export const th = "px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-zinc-500";
export const td = "px-3 py-2 text-sm text-zinc-800";
export const tdNum = "px-3 py-2 text-right text-sm tabular-nums text-zinc-800";
export const thNum = "px-3 py-2 text-right text-xs font-medium uppercase tracking-wide text-zinc-500";

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-6">
      <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
      {children ? <div className="mt-1 text-sm text-zinc-600">{children}</div> : null}
    </div>
  );
}

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-zinc-200 bg-white p-5 ${className}`}>
      {title ? <h2 className="mb-3 text-sm font-semibold text-zinc-900">{title}</h2> : null}
      {children}
    </section>
  );
}

const tones = {
  ok: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  wait: "bg-amber-50 text-amber-800 ring-amber-200",
  bad: "bg-red-50 text-red-800 ring-red-200",
  muted: "bg-zinc-100 text-zinc-700 ring-zinc-200",
} as const;

export function Badge({ tone = "muted", children }: { tone?: keyof typeof tones; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Notice({ tone = "muted", children }: { tone?: keyof typeof tones; children: ReactNode }) {
  return <div className={`rounded-md px-3 py-2 text-sm ring-1 ring-inset ${tones[tone]}`}>{children}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-sm text-zinc-500">{children}</p>;
}

export function formatDate(d: Date | null | undefined): string {
  if (!d) return "—";
  return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
}

export function formatInt(n: number): string {
  return n.toLocaleString("en-US");
}

export const statusTone = {
  active: "ok",
  pending_approval: "wait",
  suspended: "bad",
  rejected: "muted",
} as const;

export function statusLabel(s: string): string {
  return s === "pending_approval" ? "pending approval" : s;
}
