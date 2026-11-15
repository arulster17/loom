import Link from "next/link";

import { NavLinks } from "@/components/nav";
import { SignOutButton } from "@/components/sign-out";
import { requireActiveOrg } from "@/lib/session";

const LINKS = [
  { href: "/dashboard", label: "Overview" },
  { href: "/dashboard/playground", label: "Playground" },
  { href: "/dashboard/keys", label: "API keys" },
  { href: "/dashboard/usage", label: "Usage" },
  { href: "/dashboard/billing", label: "Billing" },
  { href: "/dashboard/models", label: "Models" },
];

export default async function DashboardLayout({ children }: LayoutProps<"/dashboard">) {
  const ctx = await requireActiveOrg();
  return (
    <div className="min-h-screen">
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-3">
          <div className="flex items-center gap-6">
            <Link href="/dashboard" className="font-semibold tracking-tight">
              Loom
            </Link>
            <NavLinks links={LINKS} />
          </div>
          <div className="flex items-center gap-4 text-sm text-zinc-600">
            {ctx.userRole === "admin" ? (
              <Link href="/admin" className="hover:text-zinc-900">
                Admin
              </Link>
            ) : null}
            <span title={ctx.orgName}>{ctx.email}</span>
            <SignOutButton />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
