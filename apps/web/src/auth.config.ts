import PostgresAdapter from "@auth/pg-adapter";
import type { NextAuthConfig } from "next-auth";
import type { Provider } from "next-auth/providers";
import Google from "next-auth/providers/google";
import Nodemailer from "next-auth/providers/nodemailer";

import { allowSignIn, googleEmailLinking } from "@/lib/auth-policy";
import type { UserRole } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { ensureUserOrg, syncBootstrapAdmin } from "@/lib/data/orgs";

// The Auth.js configuration, apart from `NextAuth(...)` itself (src/auth.ts), so tests can
// build it without the Next.js runtime.

type Env = Record<string, string | undefined>;

export function googleEnabled(env: Env = process.env): boolean {
  return !!(env.AUTH_GOOGLE_ID && env.AUTH_GOOGLE_SECRET);
}

function providers(env: Env): Provider[] {
  const list: Provider[] = [
    // Magic link. Local dev: EMAIL_SERVER=smtp://localhost:1025 (Mailpit).
    Nodemailer({
      server: env.EMAIL_SERVER ?? "smtp://localhost:1025",
      from: env.EMAIL_FROM ?? "Loom <no-reply@loom.local>",
      maxAge: 15 * 60,
    }),
  ];
  if (googleEnabled(env)) {
    // With LOOM_GOOGLE_EMAIL_LINKING (default on) a Google sign-in may attach to an
    // existing magic-link user with the same email; the signIn callback below refuses
    // every Google profile whose email Google has not verified.
    list.push(
      Google({
        clientId: env.AUTH_GOOGLE_ID,
        clientSecret: env.AUTH_GOOGLE_SECRET,
        allowDangerousEmailAccountLinking: googleEmailLinking(env),
      }),
    );
  }
  return list;
}

export function authConfig(env: Env = process.env): NextAuthConfig {
  return {
    adapter: PostgresAdapter(getPool()),
    providers: providers(env),
    session: { strategy: "database", maxAge: 30 * 24 * 3600, updateAge: 24 * 3600 },
    trustHost: true,
    pages: { signIn: "/signin", verifyRequest: "/signin/check-email", error: "/signin" },
    callbacks: {
      signIn({ account, profile }) {
        return allowSignIn({ account, profile });
      },
      session({ session, user }) {
        // Only the id is trusted from here; role and org are re-read per request
        // (src/lib/session.ts). Role is exposed for display.
        const role: UserRole = (user as { role?: string }).role === "admin" ? "admin" : "user";
        return { ...session, user: { ...session.user, id: user.id, email: user.email, role } };
      },
    },
    events: {
      // First sign-in: the user's org, pending approval (contracts §4).
      async createUser({ user }) {
        if (user.id) await ensureUserOrg(user.id);
      },
      async signIn({ user }) {
        if (!user.id) return;
        await ensureUserOrg(user.id); // also repairs a user whose createUser event failed
        await syncBootstrapAdmin(user.id);
      },
    },
  };
}
