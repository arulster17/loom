"use server";

import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

import { userMessage } from "@/lib/action-errors";
import { AuthzError, canManageBilling } from "@/lib/authz";
import { saveAutoRecharge, validateTopup } from "@/lib/data/billing";
import { parseUsdToMicros } from "@/lib/money";
import { createCheckout, ServiceError } from "@/lib/services";
import { currentOrgContext } from "@/lib/session";

export type FormState = { ok?: string; error?: string } | undefined;

async function appBaseUrl(): Promise<string> {
  if (process.env.AUTH_URL) return new URL(process.env.AUTH_URL).origin;
  const h = await headers();
  const host = h.get("x-forwarded-host") ?? h.get("host") ?? "localhost:3000";
  const proto = h.get("x-forwarded-proto") ?? "http";
  return `${proto}://${host}`;
}

export async function checkoutAction(_prev: FormState, form: FormData): Promise<FormState> {
  const ctx = await currentOrgContext();
  if (!ctx) return { error: "Sign in again." };
  let url: string;
  try {
    if (!canManageBilling(ctx)) throw new AuthzError("Only org owners and admins can buy credit.");
    const amountMicros = parseUsdToMicros(String(form.get("amount") ?? ""));
    validateTopup(amountMicros);
    const base = await appBaseUrl();
    url = await createCheckout({
      orgId: ctx.orgId,
      amountMicros,
      customerEmail: ctx.email,
      successUrl: `${base}/dashboard/billing?checkout=success`,
      cancelUrl: `${base}/dashboard/billing?checkout=cancelled`,
    });
  } catch (e) {
    if (e instanceof ServiceError) {
      console.error("checkout failed:", e.message);
      return { error: "Payments are not available right now. Try again in a few minutes." };
    }
    return { error: userMessage(e) };
  }
  redirect(url);
}

export async function autoRechargeAction(_prev: FormState, form: FormData): Promise<FormState> {
  const ctx = await currentOrgContext();
  if (!ctx) return { error: "Sign in again." };
  try {
    await saveAutoRecharge(ctx, {
      enabled: form.get("enabled") === "on",
      thresholdMicros: parseUsdToMicros(String(form.get("threshold") ?? "")),
      amountMicros: parseUsdToMicros(String(form.get("amount") ?? "")),
    });
  } catch (e) {
    return { error: userMessage(e) };
  }
  revalidatePath("/dashboard/billing");
  return { ok: "Saved." };
}
