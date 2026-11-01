"use client";

import { useActionState, useState } from "react";

import { btn, btnDanger, btnSecondary, input, label } from "@/components/ui";

import { createKeyAction, revokeKeyAction } from "./actions";

export function CreateKeyForm() {
  const [state, action, pending] = useActionState(createKeyAction, undefined);
  const [copied, setCopied] = useState(false);

  return (
    <div className="space-y-4">
      <form action={action} className="grid gap-3 sm:grid-cols-[1fr_12rem_auto] sm:items-end">
        <div>
          <label htmlFor="name" className={label}>
            Name
          </label>
          <input id="name" name="name" maxLength={100} placeholder="e.g. production backend" className={`${input} mt-1`} />
        </div>
        <div>
          <label htmlFor="spend_limit" className={label}>
            Spend limit (USD, optional)
          </label>
          <input id="spend_limit" name="spend_limit" inputMode="decimal" placeholder="no limit" className={`${input} mt-1`} />
        </div>
        <button type="submit" disabled={pending} className={btn}>
          {pending ? "Creating…" : "Create key"}
        </button>
      </form>
      {state && !state.ok ? <p className="text-sm text-red-700">{state.error}</p> : null}
      {state?.ok ? (
        <div className="rounded-md border border-amber-300 bg-amber-50 p-4">
          <p className="text-sm font-medium text-amber-900">
            Copy your new key{state.name ? ` “${state.name}”` : ""} now. It will not be shown again.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <code className="break-all rounded bg-white px-2 py-1 font-mono text-xs ring-1 ring-amber-200" data-testid="new-key">
              {state.plaintext}
            </code>
            <button
              type="button"
              className={btnSecondary}
              onClick={() => {
                void navigator.clipboard.writeText(state.plaintext).then(() => setCopied(true));
              }}
            >
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function RevokeKeyButton({ keyId, keyLabel }: { keyId: string; keyLabel: string }) {
  const [state, action, pending] = useActionState(revokeKeyAction, undefined);
  return (
    <form
      action={action}
      onSubmit={(e) => {
        if (!confirm(`Revoke ${keyLabel}? Requests with it will fail within a few seconds. This cannot be undone.`)) {
          e.preventDefault();
        }
      }}
    >
      <input type="hidden" name="key_id" value={keyId} />
      <button type="submit" disabled={pending} className={btnDanger}>
        {pending ? "Revoking…" : "Revoke"}
      </button>
      {state?.error ? <p className="mt-1 text-xs text-red-700">{state.error}</p> : null}
    </form>
  );
}
