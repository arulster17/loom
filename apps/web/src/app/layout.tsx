import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Loom", template: "%s · Loom" },
  description: "Open-model inference with published benchmarks and per-request costs.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full font-sans">{children}</body>
    </html>
  );
}
