import type { Metadata } from "next";
import { connection } from "next/server";

import { brandName } from "@/lib/brand";

import "./globals.css";

// The title carries the product name (LOOM_BRAND_NAME, M3 spec §6.11), read for each
// request; without it a page shows only its own title.
export async function generateMetadata(): Promise<Metadata> {
  await connection();
  const brand = brandName();
  return {
    title: brand ? { default: brand, template: `%s · ${brand}` } : { default: "Open-model inference", template: "%s" },
    description: "Open-model inference with published benchmarks and per-request costs.",
  };
}

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // Every page renders per request, so its scripts carry that request's CSP nonce
  // (src/proxy.ts); a page prerendered at build time would have none and not run.
  await connection();
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full font-sans">{children}</body>
    </html>
  );
}
