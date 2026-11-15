// Rate limit for magic-link emails (signInWithEmail in src/app/actions.ts): fixed
// windows per email address and per client IP, counted in Redis. Pure apart from the
// injected counter store, so it is unit tested; src/lib/redis.ts has the Redis store.

import { createHmac } from "node:crypto";

type Env = Record<string, string | undefined>;

export interface SigninLimits {
  emailLimit: number; // sends per email address per window
  ipLimit: number; // sends per client IP per window
  windowS: number;
}

export const DEFAULT_SIGNIN_LIMITS: SigninLimits = { emailLimit: 5, ipLimit: 20, windowS: 900 };

function positiveInt(raw: string | undefined, fallback: number): number {
  return raw && /^\d+$/.test(raw.trim()) && Number(raw) > 0 ? Number(raw) : fallback;
}

/** LOOM_SIGNIN_EMAIL_LIMIT, LOOM_SIGNIN_IP_LIMIT, LOOM_SIGNIN_WINDOW_S (positive integers;
 * anything else means the default). */
export function signinLimits(env: Env = process.env): SigninLimits {
  return {
    emailLimit: positiveInt(env.LOOM_SIGNIN_EMAIL_LIMIT, DEFAULT_SIGNIN_LIMITS.emailLimit),
    ipLimit: positiveInt(env.LOOM_SIGNIN_IP_LIMIT, DEFAULT_SIGNIN_LIMITS.ipLimit),
    windowS: positiveInt(env.LOOM_SIGNIN_WINDOW_S, DEFAULT_SIGNIN_LIMITS.windowS),
  };
}

export function normaliseEmail(email: string): string {
  return email.trim().toLowerCase();
}

/**
 * The client IP for rate limiting: the RIGHT-most `X-Forwarded-For` entry, which is the
 * one our own proxy or load balancer appended (entries to its left come from the client
 * and can be anything). Without a proxy, Next.js sets the header to the socket address.
 * Else `X-Real-IP`, else one shared "unknown" bucket.
 */
export function clientIp(headers: Pick<Headers, "get">): string {
  const xff = headers.get("x-forwarded-for");
  const last = xff?.split(",").map((s) => s.trim()).filter(Boolean).at(-1);
  if (last) return last.toLowerCase();
  const real = headers.get("x-real-ip")?.trim();
  return real ? real.toLowerCase() : "unknown";
}

/** Redis key names. Emails and IPs are HMAC'd, never stored in key names in plaintext. */
export function signinLimitKeys(
  email: string,
  ip: string,
  windowS: number,
  nowMs: number,
  secret: string,
): { email: string; ip: string } {
  const window = Math.floor(nowMs / 1000 / windowS);
  const h = (kind: string, v: string) => createHmac("sha256", secret).update(`${kind}:${v}`).digest("hex");
  return {
    email: `loom:web:signin:email:${h("email", normaliseEmail(email))}:${window}`,
    ip: `loom:web:signin:ip:${h("ip", ip)}:${window}`,
  };
}

/** Counts one event under each key, returning the counts. Each key lives `ttlS`. */
export interface CounterStore {
  incr(keys: string[], ttlS: number): Promise<number[]>;
}

export interface SigninLimitInput {
  email: string;
  ip: string;
  nowMs?: number;
  secret: string;
  limits?: SigninLimits;
  timeoutMs?: number;
  warn?: (msg: string) => void;
}

export type SigninVerdict = { allowed: true; failedOpen?: boolean } | { allowed: false; reason: "email" | "ip" };

function timeout(ms: number): Promise<never> {
  return new Promise((_, reject) => {
    const t = setTimeout(() => reject(new Error(`no answer in ${ms} ms`)), ms);
    t.unref?.();
  });
}

/**
 * Counts this send and says whether it is within both limits. If the store fails or
 * does not answer within `timeoutMs`, the send is allowed (fail open, with a warning):
 * a Redis outage must not lock everyone out of signing in.
 */
export async function checkSigninLimit(store: CounterStore, input: SigninLimitInput): Promise<SigninVerdict> {
  const limits = input.limits ?? DEFAULT_SIGNIN_LIMITS;
  const keys = signinLimitKeys(input.email, input.ip, limits.windowS, input.nowMs ?? Date.now(), input.secret);
  let counts: number[];
  try {
    counts = await Promise.race([store.incr([keys.email, keys.ip], limits.windowS), timeout(input.timeoutMs ?? 1500)]);
  } catch (e) {
    (input.warn ?? console.warn)(
      `sign-in rate limit unavailable, allowing the email: ${e instanceof Error ? e.message : "error"}`,
    );
    return { allowed: true, failedOpen: true };
  }
  const [emailCount, ipCount] = counts;
  if (emailCount > limits.emailLimit) return { allowed: false, reason: "email" };
  if (ipCount > limits.ipLimit) return { allowed: false, reason: "ip" };
  return { allowed: true };
}
