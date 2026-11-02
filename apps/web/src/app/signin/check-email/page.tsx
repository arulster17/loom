import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "Check your email" };

export default function CheckEmailPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-sm flex-col justify-center px-6">
      <h1 className="text-2xl font-semibold tracking-tight">Check your email</h1>
      <p className="mt-2 text-sm text-zinc-600">
        We sent you a sign-in link. It works once and expires in 15 minutes. You can close this tab.
      </p>
      <Link href="/signin" className="mt-6 text-sm text-zinc-700 underline">
        Use a different email
      </Link>
    </main>
  );
}
