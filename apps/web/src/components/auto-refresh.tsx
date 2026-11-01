"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

/** Re-renders the page every `everyMs` for up to `forMs` (e.g. until a webhook lands). */
export function AutoRefresh({ everyMs = 3000, forMs = 60_000 }: { everyMs?: number; forMs?: number }) {
  const router = useRouter();
  useEffect(() => {
    const started = Date.now();
    const t = setInterval(() => {
      if (Date.now() - started > forMs) clearInterval(t);
      else router.refresh();
    }, everyMs);
    return () => clearInterval(t);
  }, [router, everyMs, forMs]);
  return null;
}
