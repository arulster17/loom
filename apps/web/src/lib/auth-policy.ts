// Sign-in policy, kept free of Auth.js imports so it is unit-tested directly.

type Env = Record<string, string | undefined>;

/**
 * LOOM_GOOGLE_EMAIL_LINKING (default on): whether a Google sign-in may attach to an
 * existing user with the same email (Auth.js `allowDangerousEmailAccountLinking`), e.g. a
 * user who first signed in by magic link. Safe only because `allowSignIn` refuses Google
 * profiles whose email Google has not verified.
 */
export function googleEmailLinking(env: Env = process.env): boolean {
  const v = (env.LOOM_GOOGLE_EMAIL_LINKING ?? "").trim().toLowerCase();
  if (v === "" || v === "true" || v === "1") return true;
  if (v === "false" || v === "0") return false;
  throw new Error(`LOOM_GOOGLE_EMAIL_LINKING must be true or false, not "${env.LOOM_GOOGLE_EMAIL_LINKING}".`);
}

/** Auth.js `signIn` callback: a Google sign-in needs `email_verified === true`. */
export function allowSignIn({
  account,
  profile,
}: {
  account?: { provider?: string } | null;
  profile?: { email_verified?: unknown } | null;
}): boolean {
  if (account?.provider === "google") return profile?.email_verified === true;
  return true;
}
