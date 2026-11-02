// API keys, byte-for-byte the format of services/platform_common/src/loom_platform/apikeys.py:
// `loom_sk_<12 base62>_<40 base62>` (61 chars). Stored: the first 20 characters
// (`prefix`) and the SHA-256 of the full key (`key_hash`, 32 bytes). The plaintext is
// returned once at creation and never stored or logged.

import { createHash, randomInt } from "node:crypto";

export const KEY_TAG = "loom_sk_";
export const LOOKUP_LEN = 12;
export const SECRET_LEN = 40;
export const PREFIX_LEN = KEY_TAG.length + LOOKUP_LEN; // 20
export const KEY_LEN = PREFIX_LEN + 1 + SECRET_LEN; // 61

export const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const KEY_RE = new RegExp(`^${KEY_TAG}[0-9A-Za-z]{${LOOKUP_LEN}}_[0-9A-Za-z]{${SECRET_LEN}}$`);
const PREFIX_RE = new RegExp(`^${KEY_TAG}[0-9A-Za-z]{${LOOKUP_LEN}}$`);

export class InvalidApiKey extends Error {}

export interface NewApiKey {
  plaintext: string; // show once; never store or log
  prefix: string;
  keyHash: Buffer; // 32 bytes, api_keys.key_hash
}

export function randomBase62(n: number): string {
  let out = "";
  for (let i = 0; i < n; i++) out += BASE62[randomInt(62)]; // unbiased, CSPRNG
  return out;
}

export function isWellFormed(key: string): boolean {
  return KEY_RE.test(key);
}

export function isPrefix(prefix: string): boolean {
  return PREFIX_RE.test(prefix);
}

/** SHA-256 of the full key. Throws InvalidApiKey for anything not shaped like one. */
export function hashApiKey(key: string): Buffer {
  if (!isWellFormed(key)) throw new InvalidApiKey("malformed API key");
  return createHash("sha256").update(key, "ascii").digest();
}

export function keyPrefix(key: string): string {
  if (!isWellFormed(key)) throw new InvalidApiKey("malformed API key");
  return key.slice(0, PREFIX_LEN);
}

/** Assembles a key from its prefix and secret part, with its hash. */
export function assembleApiKey(prefix: string, secret: string): NewApiKey {
  const plaintext = `${prefix}_${secret}`;
  return { plaintext, prefix, keyHash: hashApiKey(plaintext) };
}

export function generateApiKey(): NewApiKey {
  return assembleApiKey(KEY_TAG + randomBase62(LOOKUP_LEN), randomBase62(SECRET_LEN));
}
