"use server";

import { revalidatePath } from "next/cache";

import { userMessage } from "@/lib/action-errors";
import { createKey, revokeKey } from "@/lib/data/keys";
import { parseUsdToMicros } from "@/lib/money";
import { currentOrgContext } from "@/lib/session";

export type CreateKeyState =
  | { ok: true; plaintext: string; prefix: string; name: string }
  | { ok: false; error: string }
  | undefined;

export async function createKeyAction(_prev: CreateKeyState, form: FormData): Promise<CreateKeyState> {
  const ctx = await currentOrgContext();
  if (!ctx) return { ok: false, error: "Sign in again." };
  const name = String(form.get("name") ?? "").trim();
  const limitRaw = String(form.get("spend_limit") ?? "").trim();
  try {
    const spendLimitMicros = limitRaw ? parseUsdToMicros(limitRaw) : null;
    if (spendLimitMicros !== null && spendLimitMicros <= 0n) {
      return { ok: false, error: "A spend limit must be more than $0, or left empty for none." };
    }
    const key = await createKey(ctx, { name, spendLimitMicros });
    revalidatePath("/dashboard/keys");
    return { ok: true, plaintext: key.plaintext, prefix: key.prefix, name };
  } catch (e) {
    return { ok: false, error: userMessage(e) };
  }
}

export type RevokeState = { error?: string } | undefined;

export async function revokeKeyAction(_prev: RevokeState, form: FormData): Promise<RevokeState> {
  const ctx = await currentOrgContext();
  if (!ctx) return { error: "Sign in again." };
  try {
    await revokeKey(ctx, String(form.get("key_id") ?? ""));
  } catch (e) {
    return { error: userMessage(e) };
  }
  revalidatePath("/dashboard/keys");
  return undefined;
}
