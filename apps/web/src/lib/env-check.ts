// Production guard (contracts §2, LOOM_ENV). docker-compose.yml sets LOOM_ENV=dev; unset
// means prod, so a deployment that forgets it gets the strict checks, never the dev
// defaults. Run once at server start by src/instrumentation.ts.

type Env = Record<string, string | undefined>;

export type LoomEnv = "dev" | "prod";

/** Dev values from docker-compose.yml: fine locally, refused anywhere else. */
export const DEV_AUTH_SECRET = "loom-dev-auth-secret-not-for-production";
export const DEV_INTERNAL_TOKEN = "loom-dev-internal-token";

/** LOOM_ENV, or null when it is set to something other than dev or prod. */
export function loomEnv(env: Env = process.env): LoomEnv | null {
  const v = (env.LOOM_ENV ?? "").trim();
  if (v === "") return "prod";
  return v === "dev" || v === "prod" ? v : null;
}

/** What stops this server from starting; empty when it may start. */
export function checkEnvironment(env: Env = process.env): string[] {
  const mode = loomEnv(env);
  if (mode === null) return [`LOOM_ENV must be "dev" or "prod", not "${env.LOOM_ENV}".`];
  if (mode === "dev") return [];
  const problems: string[] = [];
  const secret = env.AUTH_SECRET ?? "";
  if (!secret) problems.push("AUTH_SECRET is not set.");
  else if (secret === DEV_AUTH_SECRET) problems.push("AUTH_SECRET is the docker-compose dev default.");
  // Without AUTH_URL, Auth.js (trustHost: true) builds its URLs from the request's Host
  // header; with it, every request's origin is rewritten to AUTH_URL.
  if (!env.AUTH_URL) problems.push("AUTH_URL is not set (the public URL of this app).");
  const token = env.LOOM_BILLING_INTERNAL_TOKEN || env.LOOM_INTERNAL_TOKEN || "";
  if (!token) problems.push("LOOM_BILLING_INTERNAL_TOKEN (or LOOM_INTERNAL_TOKEN) is not set.");
  else if (token === DEV_INTERNAL_TOKEN) problems.push("The billing internal token is the docker-compose dev default.");
  return problems;
}

/** Throws when this server must not start. */
export function assertEnvironment(env: Env = process.env): void {
  const problems = checkEnvironment(env);
  if (problems.length > 0) {
    throw new Error(
      `Refusing to start with LOOM_ENV=${env.LOOM_ENV || "prod (unset)"}:\n- ${problems.join("\n- ")}\n` +
        "Set LOOM_ENV=dev only for local development.",
    );
  }
}

/** Exits the process when this server must not start. (A throw from the instrumentation
 * hook alone leaves Next.js running and answering 500 to every request.) */
export function enforceEnvironment(env: Env = process.env): void {
  try {
    assertEnvironment(env);
  } catch (e) {
    console.error((e as Error).message);
    process.exit(1);
  }
}
