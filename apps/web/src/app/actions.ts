"use server";

import { AuthError } from "next-auth";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { z } from "zod";

import { signIn, signOut } from "@/auth";
import { revokePlaygroundKeys } from "@/lib/data/playground";
import { redisCounterStore } from "@/lib/redis";
import { sessionUserId } from "@/lib/session";
import { checkSigninLimit, clientIp, signinLimits } from "@/lib/signin-limit";

export type SignInState = { error?: string } | undefined;

const emailSchema = z.email().max(254);

export async function signInWithEmail(_prev: SignInState, form: FormData): Promise<SignInState> {
  const email = emailSchema.safeParse(String(form.get("email") ?? "").trim());
  if (!email.success) return { error: "Enter a valid email address." };
  // Per email and per client IP (src/lib/signin-limit.ts); fails open if Redis is down.
  const limit = await checkSigninLimit(redisCounterStore, {
    email: email.data,
    ip: clientIp(await headers()),
    secret: process.env.AUTH_SECRET ?? "loom-signin-limit",
    limits: signinLimits(),
  });
  if (!limit.allowed) return { error: "Too many sign-in emails. Try again in a few minutes." };
  try {
    await signIn("nodemailer", { email: email.data, redirectTo: "/dashboard" });
  } catch (e) {
    // signIn ends with a redirect to the check-email page, which is thrown; let it through.
    if (e instanceof AuthError) return { error: "We could not send the sign-in email. Try again in a minute." };
    throw e;
  }
  return undefined;
}

export async function signInWithGoogle(): Promise<void> {
  await signIn("google", { redirectTo: "/dashboard" });
}

/** Signing out also revokes the user's playground keys. */
export async function signOutAction(): Promise<void> {
  const userId = await sessionUserId();
  if (userId) await revokePlaygroundKeys(userId);
  await signOut({ redirect: false });
  redirect("/signin");
}
