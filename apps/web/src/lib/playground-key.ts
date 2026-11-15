// Deterministic secret half of a playground key (design in src/lib/data/playground.ts).
//
// secret = 40 base62 characters drawn from HMAC-SHA256(server secret, "<prefix>:<n>"),
// n = 0, 1, ..., keeping bytes < 248 (= 4 * 62) and mapping them mod 62, so every
// character is uniform. Only a server holding the secret can rebuild a key from its
// stored prefix; the database holds only the prefix and the SHA-256, as for every key.

import { createHmac } from "node:crypto";

import { BASE62, SECRET_LEN } from "@/lib/apikeys";

export function playgroundServerSecret(env: NodeJS.ProcessEnv = process.env): Buffer {
  const explicit = env.LOOM_PLAYGROUND_KEY_SECRET;
  if (explicit) return Buffer.from(explicit, "utf8");
  if (env.AUTH_SECRET) {
    // Domain-separated from Auth.js's own use of AUTH_SECRET.
    return createHmac("sha256", env.AUTH_SECRET).update("loom-playground-key-v1").digest();
  }
  throw new Error("Set AUTH_SECRET (or LOOM_PLAYGROUND_KEY_SECRET) to use the playground.");
}

export function derivePlaygroundSecret(serverSecret: Buffer, prefix: string): string {
  let out = "";
  for (let n = 0; out.length < SECRET_LEN; n++) {
    const block = createHmac("sha256", serverSecret).update(`${prefix}:${n}`).digest();
    for (const byte of block) {
      if (byte < 248 && out.length < SECRET_LEN) out += BASE62[byte % 62];
    }
  }
  return out;
}
