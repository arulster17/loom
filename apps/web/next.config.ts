import path from "node:path";
import type { NextConfig } from "next";

import { STATIC_SECURITY_HEADERS } from "./src/lib/security-headers";

const nextConfig: NextConfig = {
  // Self-contained server for apps/web/Dockerfile (.next/standalone/server.js).
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname),
  // Node-only libraries stay runtime requires instead of being bundled.
  serverExternalPackages: ["pg", "nodemailer", "ioredis"],
  poweredByHeader: false,
  // Constant security headers on every response, static files included. The
  // Content-Security-Policy needs a per-request nonce: src/proxy.ts sets it.
  async headers() {
    return [{ source: "/:path*", headers: STATIC_SECURITY_HEADERS }];
  },
  turbopack: {
    root: path.join(__dirname),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
