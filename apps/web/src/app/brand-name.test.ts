// The product name is a setting (docs/phase1-m3-spec.md §6.11, A35, A36): LOOM_BRAND_NAME,
// read on the server at request time (one image serves any name). With it set, the root
// layout's title template, the sign-in heading and the dashboard and admin headers show
// it; unset, they show no name (each page's own title, a neutral header). The magic-link
// sender has no default name: it comes from EMAIL_FROM. The scan for any other `Loom` in
// user-facing text is services/devstack/tests/test_devstack_m3_brand.py.

import { createElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("next/link", async () => {
  const { createElement: h } = await import("react");
  return {
    default: ({ href, children }: { href: string; children?: ReactNode }) => h("a", { href }, children),
  };
});
vi.mock("next/navigation", () => ({ redirect: vi.fn(), usePathname: () => "/dashboard" }));
vi.mock("next/server", () => ({ connection: vi.fn(async () => undefined) }));
vi.mock("@/lib/session", () => ({
  requireActiveOrg: vi.fn(async () => ({
    userId: "u1",
    orgId: "o1",
    orgName: "Example org",
    email: "user@example.test",
    userRole: "member",
    orgRole: "owner",
  })),
  requireAdmin: vi.fn(async () => ({ userId: "a1", email: "admin@example.test" })),
}));
vi.mock("@/components/nav", () => ({ NavLinks: () => null }));
vi.mock("@/components/sign-out", () => ({ SignOutButton: () => null }));
vi.mock("@/auth", () => ({ auth: vi.fn(async () => null), googleEnabled: () => false }));
vi.mock("@/app/actions", () => ({ signInWithGoogle: vi.fn() }));
vi.mock("@/app/signin/email-form", () => ({ EmailForm: () => null }));

const NAME = /(?<![A-Za-z0-9_-])Loom(?![A-Za-z0-9_])/;

type Title = { default?: string; template?: string; absolute?: string };
type Component = (props: Record<string, unknown>) => Promise<ReactNode> | ReactNode;

let saved: string | undefined;

beforeEach(() => {
  saved = process.env.LOOM_BRAND_NAME;
});

afterEach(() => {
  if (saved === undefined) delete process.env.LOOM_BRAND_NAME;
  else process.env.LOOM_BRAND_NAME = saved;
});

function setBrand(name: string | undefined) {
  if (name === undefined) delete process.env.LOOM_BRAND_NAME;
  else process.env.LOOM_BRAND_NAME = name;
}

async function rootTitle(): Promise<Title> {
  const mod = (await import("@/app/layout")) as unknown as {
    generateMetadata?: (props: unknown, parent: unknown) => unknown;
    metadata?: { title?: unknown };
  };
  const meta = (
    mod.generateMetadata
      ? await mod.generateMetadata({ params: Promise.resolve({}) }, Promise.resolve({}))
      : mod.metadata
  ) as { title?: unknown } | undefined;
  const title = meta?.title;
  if (typeof title === "string") return { default: title };
  return (title ?? {}) as Title;
}

function applied(t: Title, page: string): string {
  return t.template ? t.template.replace("%s", page) : page;
}

async function render(component: Component, props: Record<string, unknown>): Promise<string> {
  const node = await component(props);
  return renderToStaticMarkup(createElement("div", null, node));
}

function text(html: string): string {
  return html
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
}

function part(html: string, tag: string): string {
  const m = html.match(new RegExp(`<${tag}[\\s>][\\s\\S]*?</${tag}>`));
  expect(m, `no <${tag}> rendered`).not.toBeNull();
  return text(m![0]);
}

async function dashboardHeader(): Promise<string> {
  const { default: DashboardLayout } = await import("@/app/dashboard/layout");
  return part(await render(DashboardLayout as unknown as Component, { children: null }), "header");
}

async function adminHeader(): Promise<string> {
  const { default: AdminLayout } = await import("@/app/admin/layout");
  return part(await render(AdminLayout as unknown as Component, { children: null }), "header");
}

async function signInHeading(): Promise<string> {
  const { default: SignInPage } = await import("@/app/signin/page");
  const html = await render(SignInPage as unknown as Component, { searchParams: Promise.resolve({}) });
  const m = html.match(/<(h1|h2)[\s>][\s\S]*?<\/\1>/);
  expect(m, "the sign-in page has no heading").not.toBeNull();
  return text(m![0]);
}

describe("root layout title", () => {
  it("names the product in the template and the default when LOOM_BRAND_NAME is set", async () => {
    setBrand("Acme Test");
    const t = await rootTitle();
    expect(t.template ?? "").toContain("%s");
    expect(applied(t, "Sign in")).toContain("Sign in");
    expect(applied(t, "Sign in")).toContain("Acme Test");
    expect(t.default ?? t.absolute ?? "").toContain("Acme Test");
  });

  it("reads the name at request time, not when the module loads", async () => {
    setBrand("First Name");
    expect(applied(await rootTitle(), "Usage")).toContain("First Name");
    setBrand("Second Name");
    const t = await rootTitle();
    expect(applied(t, "Usage")).toContain("Second Name");
    expect(applied(t, "Usage")).not.toContain("First Name");
  });

  it("shows only the page's own title when LOOM_BRAND_NAME is unset", async () => {
    setBrand(undefined);
    const t = await rootTitle();
    expect([undefined, "%s"]).toContain(t.template);
    expect(applied(t, "Sign in")).toBe("Sign in");
    for (const s of [t.default, t.template, t.absolute]) expect(s ?? "").not.toMatch(NAME);
  });
});

describe("sign-in heading", () => {
  it("shows the name when set", async () => {
    setBrand("Acme Test");
    expect(await signInHeading()).toContain("Acme Test");
  });

  it("shows no name when unset", async () => {
    setBrand(undefined);
    const heading = await signInHeading();
    expect(heading).not.toMatch(NAME);
    expect(heading.length).toBeGreaterThan(0);
  });
});

describe("app headers", () => {
  it("dashboard and admin headers show the name when set", async () => {
    setBrand("Acme Test");
    expect(await dashboardHeader()).toContain("Acme Test");
    expect(await adminHeader()).toContain("Acme Test");
  });

  it("follow a changed name without a new import", async () => {
    setBrand("First Name");
    await dashboardHeader();
    setBrand("Second Name");
    expect(await dashboardHeader()).toContain("Second Name");
    expect(await adminHeader()).toContain("Second Name");
  });

  it("show no name when unset", async () => {
    setBrand(undefined);
    const dash = await dashboardHeader();
    const admin = await adminHeader();
    expect(dash).not.toMatch(NAME);
    expect(admin).not.toMatch(NAME);
    expect(dash).toContain("user@example.test");
    expect(admin).toContain("admin@example.test");
  });
});

describe("magic-link sender", () => {
  type Mail = { id?: string; from?: string; options?: { from?: string } };

  async function sender(env: Record<string, string>): Promise<string | undefined> {
    const { authConfig } = await import("@/auth.config");
    const providers = authConfig(env).providers.map((p) => (typeof p === "function" ? p() : p)) as Mail[];
    const mail = providers.find((p) => p.id === "nodemailer");
    expect(mail, "no nodemailer provider").toBeDefined();
    return mail!.options?.from ?? mail!.from;
  }

  it("has no default name: without EMAIL_FROM the sender names no one", async () => {
    const from = (await sender({})) ?? "";
    expect(from).not.toMatch(NAME);
    expect(from).not.toContain("<");
  });

  it("is EMAIL_FROM when set", async () => {
    const from = await sender({ EMAIL_FROM: "Acme Test <no-reply@example.test>" });
    expect(from).toBe("Acme Test <no-reply@example.test>");
  });
});
