import { AuthzError } from "@/lib/authz";

/** A message safe to show the user; unexpected errors are logged (no request bodies)
 * and shown generically. */
export function userMessage(e: unknown): string {
  if (e instanceof AuthzError || e instanceof RangeError) return e.message;
  console.error("action failed:", e instanceof Error ? `${e.name}: ${e.message}` : "unknown error");
  return "Something went wrong. Try again.";
}
