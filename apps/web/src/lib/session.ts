import "server-only";

import { notFound, redirect } from "next/navigation";

import { auth } from "@/auth";
import { isOrgActive, isPlatformAdmin, type Actor, type OrgContext } from "@/lib/authz";
import { ensureUserOrg, loadActor, loadOrgContext } from "@/lib/data/orgs";

// Every page, server action and route handler gets its caller from here. The session
// only identifies the user; role, org and org status are read fresh from Postgres on
// every request, so suspensions and role changes apply immediately.

export async function sessionUserId(): Promise<string | null> {
  const session = await auth();
  return session?.user?.id ?? null;
}

/** The caller's org context, or null when signed out. */
export async function currentOrgContext(): Promise<OrgContext | null> {
  const userId = await sessionUserId();
  if (!userId) return null;
  let ctx = await loadOrgContext(userId);
  if (!ctx) {
    await ensureUserOrg(userId);
    ctx = await loadOrgContext(userId);
  }
  return ctx;
}

export async function requireOrg(): Promise<OrgContext> {
  const ctx = await currentOrgContext();
  if (!ctx) redirect("/signin");
  return ctx;
}

/** Dashboard pages: signed in, and the org is active; otherwise the status page. */
export async function requireActiveOrg(): Promise<OrgContext> {
  const ctx = await requireOrg();
  if (!isOrgActive(ctx)) redirect("/pending");
  return ctx;
}

/** Admin pages: a platform admin, else a plain 404 (admin pages are not advertised). */
export async function requireAdmin(): Promise<Actor> {
  const userId = await sessionUserId();
  if (!userId) redirect("/signin");
  const actor = await loadActor(userId);
  if (!actor || !isPlatformAdmin(actor)) notFound();
  return actor;
}
