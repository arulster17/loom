// Runs once when a Next.js server starts (not during `next build`): outside LOOM_ENV=dev
// the server refuses to start on a missing secret or a docker-compose dev default
// (src/lib/env-check.ts). Settings it ignores outside dev (LOOM_ADMIN_EMAILS) are logged as
// warnings.

export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  const { enforceEnvironment, environmentWarnings } = await import("@/lib/env-check");
  enforceEnvironment(process.env);
  for (const w of environmentWarnings(process.env)) console.warn(w);
}
