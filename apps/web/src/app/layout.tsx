import type { Metadata } from "next";
import { connection } from "next/server";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Loom", template: "%s · Loom" },
  description: "Open-model inference with published benchmarks and per-request costs.",
};

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
