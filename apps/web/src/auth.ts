import PostgresAdapter from "@auth/pg-adapter";
import NextAuth from "next-auth";
import type { Provider } from "next-auth/providers";
import Google from "next-auth/providers/google";
import Nodemailer from "next-auth/providers/nodemailer";

import type { UserRole } from "@/lib/authz";
import { getPool } from "@/lib/db";
import { ensureUserOrg, syncBootstrapAdmin } from "@/lib/data/orgs";

declare module "next-auth" {
  interface Session {
    user: { id: string; email: string; role: UserRole; name?: string | null; image?: string | null };
  }
}

export function googleEnabled(): boolean {
  return !!(process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET);
}

function providers(): Provider[] {
  const list: Provider[] = [
    // Magic link. Local dev: EMAIL_SERVER=smtp://localhost:1025 (Mailpit).
    Nodemailer({
      server: process.env.EMAIL_SERVER ?? "smtp://localhost:1025",
      from: process.env.EMAIL_FROM ?? "Loom <no-reply@loom.local>",
      maxAge: 15 * 60,
    }),
  ];
  if (googleEnabled()) {
    // Google verifies email addresses, so a Google sign-in may attach to an existing
    // magic-link user with the same email.
    list.push(Google({ allowDangerousEmailAccountLinking: true }));
  }
  return list;
}

export const { handlers, auth, signIn, signOut } = NextAuth(() => ({
  adapter: PostgresAdapter(getPool()),
  providers: providers(),
  session: { strategy: "database", maxAge: 30 * 24 * 3600, updateAge: 24 * 3600 },
  trustHost: true,
  pages: { signIn: "/signin", verifyRequest: "/signin/check-email", error: "/signin" },
  callbacks: {
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
}));
