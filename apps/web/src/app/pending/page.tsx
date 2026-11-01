import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";

import { SignOutButton } from "@/components/sign-out";
import { requireOrg } from "@/lib/session";

export const metadata: Metadata = { title: "Account status" };

const COPY = {
  pending_approval: {
    title: "Your account is awaiting approval",
    body: [
      "Thanks for signing up. A person on the Loom team reviews every new account, usually within one working day.",
      "Once approved, your account gets $5 of free credit and you can create API keys and use the playground. We will not email you about it yet, so check back here.",
    ],
  },
  rejected: {
    title: "Your signup was not approved",
    body: ["If you think this is a mistake, reply to your sign-in email or contact the Loom team."],
  },
  suspended: {
    title: "Your account is suspended",
    body: [
      "API keys for this account are refused while it is suspended. Your balance and usage history are kept.",
      "Contact the Loom team to find out more.",
    ],
  },
} as const;

export default async function PendingPage() {
  const ctx = await requireOrg();
  if (ctx.orgStatus === "active") redirect("/dashboard");
  const copy = COPY[ctx.orgStatus];

  return (
    <main className="mx-auto flex min-h-screen max-w-lg flex-col justify-center px-6">
      <p className="text-sm text-zinc-500">Signed in as {ctx.email}</p>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">{copy.title}</h1>
      {copy.body.map((p) => (
        <p key={p} className="mt-3 text-sm text-zinc-700">
          {p}
        </p>
      ))}
      <div className="mt-8 flex items-center gap-4">
        <Link href="/pending" className="text-sm text-zinc-700 underline">
          Check again
        </Link>
        {ctx.userRole === "admin" ? (
          <Link href="/admin" className="text-sm text-zinc-700 underline">
            Admin
          </Link>
        ) : null}
        <SignOutButton />
      </div>
    </main>
  );
}
