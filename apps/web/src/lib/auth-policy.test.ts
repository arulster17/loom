import { describe, expect, it } from "vitest";

import { authConfig } from "@/auth.config";
import { allowSignIn, googleEmailLinking } from "@/lib/auth-policy";

const GOOGLE_ENV = { AUTH_GOOGLE_ID: "id", AUTH_GOOGLE_SECRET: "secret" };

function googleProvider(env: Record<string, string>) {
  const p = authConfig(env).providers.find((x) => (typeof x === "function" ? x() : x).id === "google");
  const resolved = typeof p === "function" ? p() : p;
  return resolved as { options?: { allowDangerousEmailAccountLinking?: boolean } } | undefined;
}

describe("Google sign-in requires a verified email", () => {
  const signIn = (account: object | null, profile?: object) =>
    authConfig(GOOGLE_ENV).callbacks!.signIn!({ user: { id: "u" }, account, profile } as never);

  it("refuses a Google profile whose email is not verified", async () => {
    expect(await signIn({ provider: "google", type: "oidc" }, { email: "a@x.com", email_verified: false })).toBe(false);
    expect(await signIn({ provider: "google", type: "oidc" }, { email: "a@x.com" })).toBe(false);
    expect(await signIn({ provider: "google", type: "oidc" }, { email: "a@x.com", email_verified: "true" })).toBe(false);
  });

  it("accepts a verified Google profile and other providers", async () => {
    expect(await signIn({ provider: "google", type: "oidc" }, { email: "a@x.com", email_verified: true })).toBe(true);
    expect(await signIn({ provider: "nodemailer", type: "email" })).toBe(true);
  });

  it("pure policy", () => {
    expect(allowSignIn({ account: { provider: "google" }, profile: { email_verified: true } })).toBe(true);
    expect(allowSignIn({ account: { provider: "google" }, profile: { email_verified: false } })).toBe(false);
    expect(allowSignIn({ account: { provider: "google" }, profile: undefined })).toBe(false);
    expect(allowSignIn({ account: { provider: "nodemailer" } })).toBe(true);
    expect(allowSignIn({ account: null })).toBe(true);
  });
});

describe("LOOM_GOOGLE_EMAIL_LINKING", () => {
  it("defaults to linking (the current behaviour)", () => {
    expect(googleEmailLinking({})).toBe(true);
    expect(googleEmailLinking({ LOOM_GOOGLE_EMAIL_LINKING: "" })).toBe(true);
    expect(googleProvider(GOOGLE_ENV)?.options?.allowDangerousEmailAccountLinking).toBe(true);
  });

  it("can turn linking off", () => {
    expect(googleEmailLinking({ LOOM_GOOGLE_EMAIL_LINKING: "false" })).toBe(false);
    expect(googleEmailLinking({ LOOM_GOOGLE_EMAIL_LINKING: "0" })).toBe(false);
    expect(googleEmailLinking({ LOOM_GOOGLE_EMAIL_LINKING: "true" })).toBe(true);
    expect(
      googleProvider({ ...GOOGLE_ENV, LOOM_GOOGLE_EMAIL_LINKING: "false" })?.options?.allowDangerousEmailAccountLinking,
    ).toBe(false);
  });

  it("refuses an unrecognised value", () => {
    expect(() => googleEmailLinking({ LOOM_GOOGLE_EMAIL_LINKING: "maybe" })).toThrow(/LOOM_GOOGLE_EMAIL_LINKING/);
  });

  it("has no Google provider without its credentials", () => {
    expect(googleProvider({})).toBeUndefined();
  });
});
