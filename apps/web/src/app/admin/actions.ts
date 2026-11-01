"use server";

import { revalidatePath } from "next/cache";

import { userMessage } from "@/lib/action-errors";
import { isPlatformAdmin } from "@/lib/authz";
import { adjustCredits, approveOrg, rejectOrg, suspendOrg, unsuspendOrg } from "@/lib/data/admin";
import { loadActor } from "@/lib/data/orgs";
import { formatUsd, parseUsdToMicros } from "@/lib/money";
import { sessionUserId } from "@/lib/session";

export type AdminState = { ok?: string; error?: string } | undefined;

// Every action re-checks the caller is an admin (here, and again inside the data layer's
// transaction), and the data layer writes the admin_audit_log row with the change.

async function adminActor() {
  const userId = await sessionUserId();
  const actor = userId ? await loadActor(userId) : null;
  return actor && isPlatformAdmin(actor) ? actor : null;
}

function promoMicros(): bigint {
  const raw = process.env.LOOM_SIGNUP_PROMO_MICROS;
  return raw && /^\d+$/.test(raw) ? BigInt(raw) : 5_000_000n;
}

function done(orgId: string, ok: string): AdminState {
  revalidatePath("/admin");
  revalidatePath(`/admin/orgs/${orgId}`);
  return { ok };
}

export async function orgStatusAction(_prev: AdminState, form: FormData): Promise<AdminState> {
  const actor = await adminActor();
  if (!actor) return { error: "Admins only." };
  const orgId = String(form.get("org_id") ?? "");
  const op = String(form.get("op") ?? "");
  const reason = String(form.get("reason") ?? "").trim().slice(0, 500);
  try {
    switch (op) {
      case "approve":
        await approveOrg(actor, orgId, promoMicros());
        return done(orgId, `Approved; ${formatUsd(promoMicros())} credit added.`);
      case "reject":
        await rejectOrg(actor, orgId, reason || null);
        return done(orgId, "Rejected.");
      case "suspend":
        if (!reason) return { error: "Give a reason for the suspension." };
        await suspendOrg(actor, orgId, reason);
        return done(orgId, "Suspended. Its API keys stop working within 5 seconds.");
      case "unsuspend":
        await unsuspendOrg(actor, orgId);
        return done(orgId, "Reactivated.");
      default:
        return { error: "Unknown action." };
    }
  } catch (e) {
    return { error: userMessage(e) };
  }
}

export async function adjustCreditsAction(_prev: AdminState, form: FormData): Promise<AdminState> {
  const actor = await adminActor();
  if (!actor) return { error: "Admins only." };
  const orgId = String(form.get("org_id") ?? "");
  try {
    const amount = parseUsdToMicros(String(form.get("amount") ?? ""));
    await adjustCredits(actor, orgId, amount, String(form.get("memo") ?? ""));
    return done(orgId, `${amount > 0n ? "Added" : "Removed"} ${formatUsd(amount > 0n ? amount : -amount)}.`);
  } catch (e) {
    return { error: userMessage(e) };
  }
}
