import { describe, expect, it, vi } from "vitest";

import { assertEnvironment, checkEnvironment, DEV_AUTH_SECRET, DEV_INTERNAL_TOKEN, loomEnv } from "@/lib/env-check";
import { register } from "@/instrumentation";

const PROD_OK = {
  LOOM_ENV: "prod",
  AUTH_SECRET: "a-real-random-secret",
  AUTH_URL: "https://app.loom.example",
  LOOM_BILLING_INTERNAL_TOKEN: "a-real-token",
};

describe("LOOM_ENV", () => {
  it("is prod when unset, dev or prod when set, anything else is refused", () => {
    expect(loomEnv({})).toBe("prod");
    expect(loomEnv({ LOOM_ENV: "dev" })).toBe("dev");
    expect(loomEnv({ LOOM_ENV: "prod" })).toBe("prod");
    expect(loomEnv({ LOOM_ENV: "development" })).toBeNull();
    expect(checkEnvironment({ ...PROD_OK, LOOM_ENV: "staging" })).toEqual([expect.stringMatching(/LOOM_ENV/)]);
  });
});

describe("checkEnvironment", () => {
  it("lets dev start on the compose defaults, or with nothing set", () => {
    expect(checkEnvironment({ LOOM_ENV: "dev" })).toEqual([]);
    expect(
      checkEnvironment({ LOOM_ENV: "dev", AUTH_SECRET: DEV_AUTH_SECRET, LOOM_INTERNAL_TOKEN: DEV_INTERNAL_TOKEN }),
    ).toEqual([]);
  });

  it("lets prod start with real values", () => {
    expect(checkEnvironment(PROD_OK)).toEqual([]);
    const { LOOM_BILLING_INTERNAL_TOKEN: _, ...rest } = PROD_OK;
    void _;
    expect(checkEnvironment({ ...rest, LOOM_INTERNAL_TOKEN: "compose-name" })).toEqual([]);
  });

  it("applies the prod checks when LOOM_ENV is unset", () => {
    const { LOOM_ENV: _, ...unset } = PROD_OK;
    void _;
    expect(checkEnvironment(unset)).toEqual([]);
    expect(checkEnvironment({})).toHaveLength(3);
  });

  it.each([
    ["AUTH_SECRET missing", { AUTH_SECRET: undefined }, /AUTH_SECRET is not set/],
    ["AUTH_SECRET dev default", { AUTH_SECRET: DEV_AUTH_SECRET }, /AUTH_SECRET is the docker-compose dev default/],
    ["AUTH_URL missing", { AUTH_URL: undefined }, /AUTH_URL is not set/],
    ["internal token missing", { LOOM_BILLING_INTERNAL_TOKEN: undefined }, /INTERNAL_TOKEN.*not set/],
    ["internal token dev default", { LOOM_BILLING_INTERNAL_TOKEN: DEV_INTERNAL_TOKEN }, /internal token is the docker-compose dev default/],
    [
      "internal token dev default under the compose name",
      { LOOM_BILLING_INTERNAL_TOKEN: undefined, LOOM_INTERNAL_TOKEN: DEV_INTERNAL_TOKEN },
      /internal token is the docker-compose dev default/,
    ],
  ])("refuses prod with %s", (_name, change, message) => {
    const problems = checkEnvironment({ ...PROD_OK, ...change });
    expect(problems).toEqual([expect.stringMatching(message)]);
    expect(() => assertEnvironment({ ...PROD_OK, ...change })).toThrow(message);
  });

  it("does not require AUTH_TRUST_HOST", () => {
    expect(checkEnvironment({ ...PROD_OK, AUTH_TRUST_HOST: undefined })).toEqual([]);
  });
});

describe("instrumentation register()", () => {
  it("exits the Node.js server outside dev with a dev default, starts it in dev", async () => {
    const saved = { ...process.env };
    const exit = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      Object.assign(process.env, PROD_OK, { NEXT_RUNTIME: "nodejs", AUTH_SECRET: DEV_AUTH_SECRET });
      await register();
      expect(exit).toHaveBeenCalledWith(1);
      expect(logged).toHaveBeenCalledWith(expect.stringMatching(/AUTH_SECRET/));
      exit.mockClear();
      process.env.LOOM_ENV = "dev";
      await register();
      expect(exit).not.toHaveBeenCalled();
    } finally {
      process.env = saved;
      exit.mockRestore();
      logged.mockRestore();
    }
  });
});
