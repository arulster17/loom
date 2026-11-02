import type { Metadata } from "next";
import { redirect } from "next/navigation";

import { signInWithGoogle } from "@/app/actions";
import { auth, googleEnabled } from "@/auth";
import { btnSecondary } from "@/components/ui";

import { EmailForm } from "./email-form";

export const metadata: Metadata = { title: "Sign in" };

const ERRORS: Record<string, string> = {
  Verification: "That sign-in link has expired or was already used. Ask for a new one.",
  OAuthAccountNotLinked: "That email already signs in another way. Use the email link.",
  AccessDenied: "Sign-in was refused.",
  Configuration: "Sign-in is not configured correctly. Tell the Loom team.",
};

export default async function SignInPage(props: PageProps<"/signin">) {
  if ((await auth())?.user) redirect("/dashboard");
  const { error } = await props.searchParams;
  const message = typeof error === "string" ? (ERRORS[error] ?? "Sign-in failed. Try again.") : null;

  return (
    <main className="mx-auto flex min-h-screen max-w-sm flex-col justify-center px-6">
      <h1 className="text-2xl font-semibold tracking-tight">Loom</h1>
      <p className="mt-1 text-sm text-zinc-600">
        Sign in or create an account. New accounts are reviewed by a person before they can make API calls; approved
        accounts start with $5 of free credit.
      </p>
      <div className="mt-6 space-y-4 rounded-lg border border-zinc-200 bg-white p-5">
        {message ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-800">{message}</p> : null}
        <EmailForm />
        {googleEnabled() ? (
          <>
            <div className="text-center text-xs text-zinc-500">or</div>
            <form action={signInWithGoogle}>
              <button type="submit" className={`${btnSecondary} w-full`}>
                Continue with Google
              </button>
            </form>
          </>
        ) : null}
      </div>
    </main>
  );
}
