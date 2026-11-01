import { isOrgActive } from "@/lib/authz";
import { usageRows } from "@/lib/data/usage";
import { currentOrgContext } from "@/lib/session";
import { parseDayRange, usageCsv } from "@/lib/usage";

export const dynamic = "force-dynamic";

/** The caller's org usage as CSV, one row per (UTC day, model, key). */
export async function GET(req: Request) {
  const ctx = await currentOrgContext();
  if (!ctx) return new Response("Sign in first.\n", { status: 401 });
  if (!isOrgActive(ctx)) return new Response("This account is not active.\n", { status: 403 });
  const q = new URL(req.url).searchParams;
  const range = parseDayRange({ from: q.get("from") ?? undefined, to: q.get("to") ?? undefined });
  const csv = usageCsv(await usageRows(ctx, range));
  return new Response(csv, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="loom-usage-${range.from}-to-${range.to}.csv"`,
      "Cache-Control": "no-store",
    },
  });
}
