import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server for apps/web/Dockerfile (.next/standalone/server.js).
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname),
  // Node-only libraries stay runtime requires instead of being bundled.
  serverExternalPackages: ["pg", "nodemailer", "ioredis"],
  poweredByHeader: false,
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
