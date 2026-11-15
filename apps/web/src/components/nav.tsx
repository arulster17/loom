"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLinks({ links }: { links: { href: string; label: string }[] }) {
  const path = usePathname();
  return (
    <nav className="flex flex-wrap gap-1">
      {links.map((l) => {
        const active = l.href === "/dashboard" ? path === l.href : path.startsWith(l.href);
        return (
          <Link
            key={l.href}
            href={l.href}
            className={`rounded-md px-2.5 py-1.5 text-sm ${active ? "bg-zinc-900 text-white" : "text-zinc-700 hover:bg-zinc-200"}`}
          >
            {l.label}
          </Link>
        );
      })}
    </nav>
  );
}
