import "server-only";

// `@auth/core` is next-auth's own core (next-auth pins one exact version and re-exports
// these classes as `AuthError` etc.). Imported from there, not from "next-auth", because
// next-auth's entry needs the Next.js bundler and this module is part of the Auth.js config.
import { AccessDenied } from "@auth/core/errors";

import { redisCounterStore } from "@/lib/redis";
import { checkSigninLimit, clientIp, signinLimits } from "@/lib/signin-limit";

// The magic-link limit (src/lib/signin-limit.ts), checked inside the Nodemailer provider's
// send (src/auth.config.ts), so every path on which Auth.js sends a sign-in email is
// limited: the app's sign-in form (signInWithEmail) and a direct
// POST /api/auth/signin/nodemailer alike, each send counted once.

/**
 * Over the limit: no email is sent. An AuthError, so next-auth's server-side `signIn`
 * rethrows it to the sign-in form's action, which shows "Too many sign-in emails"; on
 * Auth.js's own endpoint it is a client-safe "AccessDenied" redirect to the sign-in page.
 */
export class SigninLimitExceeded extends AccessDenied {}

/** Counts this send per email and per client IP (the right-most X-Forwarded-For entry);
 * throws SigninLimitExceeded over either limit. Fails open if Redis is down. */
export async function enforceSigninLimit(email: string, headers: Pick<Headers, "get">): Promise<void> {
  const verdict = await checkSigninLimit(redisCounterStore, {
    email,
    ip: clientIp(headers),
    secret: process.env.AUTH_SECRET ?? "loom-signin-limit",
    limits: signinLimits(),
  });
  if (!verdict.allowed) throw new SigninLimitExceeded(`too many sign-in emails (per ${verdict.reason})`);
}
