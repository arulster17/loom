import NextAuth from "next-auth";

import { authConfig } from "@/auth.config";
import type { UserRole } from "@/lib/authz";

export { googleEnabled } from "@/auth.config";

declare module "next-auth" {
  interface Session {
    user: { id: string; email: string; role: UserRole; name?: string | null; image?: string | null };
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth(() => authConfig());
