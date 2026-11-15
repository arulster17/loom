"use client";

import { useActionState } from "react";

import { signInWithEmail } from "@/app/actions";
import { btn, input, label } from "@/components/ui";

export function EmailForm() {
  const [state, action, pending] = useActionState(signInWithEmail, undefined);
  return (
    <form action={action} className="space-y-3">
      <div>
        <label htmlFor="email" className={label}>
          Email
        </label>
        <input id="email" name="email" type="email" required autoComplete="email" placeholder="you@company.com" className={`${input} mt-1`} />
      </div>
      {state?.error ? <p className="text-sm text-red-700">{state.error}</p> : null}
      <button type="submit" disabled={pending} className={`${btn} w-full`}>
        {pending ? "Sending…" : "Email me a sign-in link"}
      </button>
    </form>
  );
}
