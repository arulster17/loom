import { describe, expect, it } from "vitest";

import { assembleApiKey, generateApiKey, hashApiKey, InvalidApiKey, isWellFormed, KEY_LEN, keyPrefix } from "./apikeys";
import { derivePlaygroundSecret, playgroundServerSecret } from "./playground-key";

// Computed with services/platform_common's loom_platform.apikeys.hash_api_key
// (uv run python -c 'from loom_platform.apikeys import hash_api_key; ...').
const VECTORS = [
  {
    key: "loom_sk_AbCdEf012345_0123456789abcdefghijABCDEFGHIJklmnopqrst",
    prefix: "loom_sk_AbCdEf012345",
    sha256: "a5dda345bc601d877c72605db0eecd52bb0bd96f0896b337f3d47c1415150816",
  },
  {
    key: "loom_sk_000000000000_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
    prefix: "loom_sk_000000000000",
    sha256: "c0729af68978f682c7b3518d12d1c2974efa16d6081d6c9c83798bbba0a3be18",
  },
];

describe("API key hashing matches the Python implementation", () => {
  it.each(VECTORS)("$prefix", ({ key, prefix, sha256 }) => {
    expect(hashApiKey(key).toString("hex")).toBe(sha256);
    expect(keyPrefix(key)).toBe(prefix);
  });
});

describe("API key format", () => {
  it("generates 61-character base62 keys with a 20-character prefix", () => {
    const seen = new Set<string>();
    for (let i = 0; i < 200; i++) {
      const k = generateApiKey();
      expect(k.plaintext).toHaveLength(KEY_LEN);
      expect(isWellFormed(k.plaintext)).toBe(true);
      expect(k.prefix).toBe(k.plaintext.slice(0, 20));
      expect(k.keyHash).toHaveLength(32);
      seen.add(k.plaintext);
    }
    expect(seen.size).toBe(200);
  });

  it("refuses malformed keys before hashing", () => {
    for (const bad of [
      "",
      "loom_sk_short_x",
      "sk-0123456789",
      "loom_sk_AbCdEf012345_0123456789abcdefghijABCDEFGHIJklmnopqrs!",
      "loom_sk_AbCdEf012345_0123456789abcdefghijABCDEFGHIJklmnopqrst ",
      "LOOM_SK_AbCdEf012345_0123456789abcdefghijABCDEFGHIJklmnopqrst",
    ]) {
      expect(isWellFormed(bad)).toBe(false);
      expect(() => hashApiKey(bad)).toThrow(InvalidApiKey);
    }
  });
});

describe("playground key derivation", () => {
  const secret = Buffer.from("test-secret");

  it("is deterministic per prefix and yields a valid key", () => {
    const prefix = "loom_sk_AbCdEf012345";
    const s1 = derivePlaygroundSecret(secret, prefix);
    expect(s1).toBe(derivePlaygroundSecret(secret, prefix));
    expect(s1).toMatch(/^[0-9A-Za-z]{40}$/);
    expect(isWellFormed(assembleApiKey(prefix, s1).plaintext)).toBe(true);
  });

  it("differs per prefix and per server secret", () => {
    const a = derivePlaygroundSecret(secret, "loom_sk_AAAAAAAAAAAA");
    expect(derivePlaygroundSecret(secret, "loom_sk_AAAAAAAAAAAB")).not.toBe(a);
    expect(derivePlaygroundSecret(Buffer.from("other"), "loom_sk_AAAAAAAAAAAA")).not.toBe(a);
  });

  it("needs a configured secret, and separates it from AUTH_SECRET", () => {
    expect(() => playgroundServerSecret({} as NodeJS.ProcessEnv)).toThrow();
    const derived = playgroundServerSecret({ AUTH_SECRET: "abc" } as unknown as NodeJS.ProcessEnv);
    expect(derived.toString("utf8")).not.toBe("abc");
    expect(playgroundServerSecret({ LOOM_PLAYGROUND_KEY_SECRET: "x", AUTH_SECRET: "abc" } as unknown as NodeJS.ProcessEnv).toString()).toBe("x");
  });
});
