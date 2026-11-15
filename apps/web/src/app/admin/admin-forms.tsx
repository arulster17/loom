"use client";

import { useActionState } from "react";

import { btn, btnDanger, btnSecondary, input, label } from "@/components/ui";

import { adjustCreditsAction, orgStatusAction } from "./actions";

function Result({ state }: { state: { ok?: string; error?: string } | undefined }) {
  if (state?.error) return <p className="mt-1 text-xs text-red-700">{state.error}</p>;
  if (state?.ok) return <p className="mt-1 text-xs text-emerald-700">{state.ok}</p>;
  return null;
}

export function ApproveRejectButtons({ orgId }: { orgId: string }) {
  const [state, action, pending] = useActionState(orgStatusAction, undefined);
  return (
    <form action={action}>
      <input type="hidden" name="org_id" value={orgId} />
      <div className="flex justify-end gap-2">
        <button type="submit" name="op" value="approve" disabled={pending} className={btn}>
          Approve
        </button>
        <button
          type="submit"
          name="op"
          value="reject"
          disabled={pending}
          className={btnDanger}
          onClick={(e) => {
            if (!confirm("Reject this signup?")) e.preventDefault();
          }}
        >
          Reject
        </button>
      </div>
      <Result state={state} />
    </form>
  );
}

export function SuspendForm({ orgId, status }: { orgId: string; status: string }) {
  const [state, action, pending] = useActionState(orgStatusAction, undefined);
  if (status === "suspended") {
    return (
      <form action={action}>
        <input type="hidden" name="org_id" value={orgId} />
        <button type="submit" name="op" value="unsuspend" disabled={pending} className={btnSecondary}>
          Reactivate org
        </button>
        <Result state={state} />
      </form>
    );
  }
  if (status !== "active") return <p className="text-sm text-zinc-500">Only active orgs can be suspended.</p>;
  return (
    <form
      action={action}
      className="space-y-2"
      onSubmit={(e) => {
        if (!confirm("Suspend this org? All its API keys stop working within 5 seconds.")) e.preventDefault();
      }}
    >
      <input type="hidden" name="org_id" value={orgId} />
      <div>
        <label htmlFor="reason" className={label}>
          Reason (kept in the audit log)
        </label>
        <input id="reason" name="reason" required maxLength={500} className={`${input} mt-1`} />
      </div>
      <button type="submit" name="op" value="suspend" disabled={pending} className={btnDanger}>
        Suspend now
      </button>
      <Result state={state} />
    </form>
  );
}

export function AdjustCreditsForm({ orgId }: { orgId: string }) {
  const [state, action, pending] = useActionState(adjustCreditsAction, undefined);
  return (
    <form action={action} className="space-y-2">
      <input type="hidden" name="org_id" value={orgId} />
      <div className="grid gap-2 sm:grid-cols-[10rem_1fr]">
        <div>
          <label htmlFor="amount" className={label}>
            Amount (USD)
          </label>
          <input id="amount" name="amount" required placeholder="10 or -2.50" className={`${input} mt-1`} />
        </div>
        <div>
          <label htmlFor="memo" className={label}>
            Note (shown to the org)
          </label>
          <input id="memo" name="memo" required maxLength={500} className={`${input} mt-1`} />
        </div>
      </div>
      <p className="text-xs text-zinc-500">Positive adds credit (a grant); negative removes it (a correction).</p>
      <button type="submit" disabled={pending} className={btn}>
        Post adjustment
      </button>
      <Result state={state} />
    </form>
  );
}
