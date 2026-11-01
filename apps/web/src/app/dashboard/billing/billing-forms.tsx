"use client";

import { useActionState, useState } from "react";

import { btn, btnSecondary, input, label } from "@/components/ui";

import { autoRechargeAction, checkoutAction } from "./actions";

const PRESETS = ["10", "25", "50", "100"];

export function AddCreditForm({ disabled }: { disabled: boolean }) {
  const [state, action, pending] = useActionState(checkoutAction, undefined);
  const [amount, setAmount] = useState("25");
  return (
    <form action={action} className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => setAmount(p)}
            className={amount === p ? btn : btnSecondary}
            disabled={disabled}
          >
            ${p}
          </button>
        ))}
      </div>
      <div className="flex items-end gap-3">
        <div>
          <label htmlFor="amount" className={label}>
            Amount (USD, $5 to $1,000)
          </label>
          <input
            id="amount"
            name="amount"
            inputMode="decimal"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            className={`${input} mt-1 w-40`}
            disabled={disabled}
          />
        </div>
        <button type="submit" disabled={pending || disabled} className={btn}>
          {pending ? "Opening checkout…" : "Add credit"}
        </button>
      </div>
      {state?.error ? <p className="text-sm text-red-700">{state.error}</p> : null}
    </form>
  );
}

export function AutoRechargeForm({
  enabled,
  threshold,
  amount,
  disabled,
}: {
  enabled: boolean;
  threshold: string;
  amount: string;
  disabled: boolean;
}) {
  const [state, action, pending] = useActionState(autoRechargeAction, undefined);
  return (
    <form action={action} className="space-y-3">
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" name="enabled" defaultChecked={enabled} disabled={disabled} />
        Recharge automatically when my balance runs low
      </label>
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label htmlFor="threshold" className={label}>
            When balance falls below (USD)
          </label>
          <input id="threshold" name="threshold" inputMode="decimal" defaultValue={threshold} className={`${input} mt-1 w-40`} disabled={disabled} />
        </div>
        <div>
          <label htmlFor="ar_amount" className={label}>
            Add (USD)
          </label>
          <input id="ar_amount" name="amount" inputMode="decimal" defaultValue={amount} className={`${input} mt-1 w-40`} disabled={disabled} />
        </div>
        <button type="submit" disabled={pending || disabled} className={btnSecondary}>
          {pending ? "Saving…" : "Save"}
        </button>
      </div>
      {state?.error ? <p className="text-sm text-red-700">{state.error}</p> : null}
      {state?.ok ? <p className="text-sm text-emerald-700">{state.ok}</p> : null}
    </form>
  );
}
