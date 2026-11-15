// Integration tests run against the real local Postgres (docker compose up -d postgres),
// in a throwaway database migrated with the platform's own runner (`uv run loom-migrate`),
// dropped at the end. Like the Python fixtures: LOOM_TEST_DATABASE_URL picks the server;
// when it is unset and Postgres is down the tests skip, when it is set they fail.

import { execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import path from "node:path";

import { Client } from "pg";
import type { TestProject } from "vitest/node";

declare module "vitest" {
  export interface ProvidedContext {
    databaseUrl: string | null;
  }
}

const REPO_ROOT = path.resolve(__dirname, "../../../..");

export default async function setup(project: TestProject) {
  const explicit = process.env.LOOM_TEST_DATABASE_URL;
  const adminUrl = explicit ?? "postgresql://loom:loom@localhost:5432/loom";
  const admin = new Client({ connectionString: adminUrl, connectionTimeoutMillis: 2000 });
  try {
    await admin.connect();
  } catch (e) {
    if (explicit) throw e;
    console.warn(`Postgres not reachable at ${adminUrl}; integration tests skip.`);
    project.provide("databaseUrl", null);
    return;
  }
  const name = `loom_web_test_${randomBytes(6).toString("hex")}`;
  await admin.query(`CREATE DATABASE ${name}`);
  const url = new URL(adminUrl);
  url.pathname = `/${name}`;
  const dbUrl = url.toString();
  try {
    execFileSync("uv", ["run", "loom-migrate", "--database-url", dbUrl], { cwd: REPO_ROOT, stdio: "pipe" });
  } catch (e) {
    await admin.query(`DROP DATABASE IF EXISTS ${name} WITH (FORCE)`);
    await admin.end();
    const err = e as { stderr?: Buffer };
    throw new Error(`loom-migrate failed: ${err.stderr?.toString() ?? e}`);
  }
  project.provide("databaseUrl", dbUrl);

  return async () => {
    await admin.query(`DROP DATABASE IF EXISTS ${name} WITH (FORCE)`);
    await admin.end();
  };
}
