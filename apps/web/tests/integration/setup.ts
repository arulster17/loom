import { afterAll, inject } from "vitest";

import { closePool } from "@/lib/db";

const url = inject("databaseUrl");
if (url) process.env.DATABASE_URL = url;
process.env.AUTH_SECRET ??= "integration-test-secret";
process.env.LOOM_ADMIN_EMAILS = "boss@example.com";

afterAll(async () => {
  await closePool();
});
