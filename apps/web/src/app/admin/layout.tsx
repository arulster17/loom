import Link from "next/link";

import { SignOutButton } from "@/components/sign-out";
import { requireAdmin } from "@/lib/session";

export default async function AdminLayout({ children }: LayoutProps<"/admin">) {
  const actor = await requireAdmin();
  return (
    <div className="min-h-screen">
      <header className="border-b border-zinc-200 bg-zinc-900 text-zinc-100">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-6 text-sm">
            <Link href="/admin" className="font-semibold">
              Loom admin
            </Link>
            <Link href="/dashboard" className="text-zinc-300 hover:text-white">
              Dashboard
            </Link>
          </div>
          <div className="flex items-center gap-4 text-sm text-zinc-300">
            <span>{actor.email}</span>
            <SignOutButton className="text-sm text-zinc-300 hover:text-white" />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
