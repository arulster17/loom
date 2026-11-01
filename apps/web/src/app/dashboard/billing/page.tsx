import type { Metadata } from "next";

import { Badge, Card, Empty, Notice, PageHeader, formatDate, td, tdNum, th, thNum } from "@/components/ui";
import { canManageBilling } from "@/lib/authz";
import { creditHistory, getAutoRecharge, getBalance } from "@/lib/data/billing";
import { formatUsd, microsToDecimalString } from "@/lib/money";
import { requireActiveOrg } from "@/lib/session";

import { AddCreditForm, AutoRechargeForm } from "./billing-forms";

export const metadata: Metadata = { title: "Billing" };

const KIND_LABEL = {
  purchase: "Credit purchase",
  promo: "Free credit",
  refund: "Refund",
  adjustment: "Adjustment by Loom",
} as const;

export default async function BillingPage(props: PageProps<"/dashboard/billing">) {
  const ctx = await requireActiveOrg();
  const { checkout } = await props.searchParams;
  const [balance, history, auto] = await Promise.all([getBalance(ctx), creditHistory(ctx), getAutoRecharge(ctx)]);
  const manage = canManageBilling(ctx);

  return (
    <>
      <PageHeader title="Billing">
        Loom is prepaid: you add credit, and each request is charged from it at the listed per-token price.
      </PageHeader>
      {checkout === "success" ? (
        <div className="mb-6">
          <Notice tone="ok">Payment received. Your credit appears here once Stripe confirms it, usually within seconds.</Notice>
        </div>
      ) : checkout === "cancelled" ? (
        <div className="mb-6">
          <Notice>Checkout cancelled. You were not charged.</Notice>
        </div>
      ) : null}
      <div className="grid gap-4 md:grid-cols-[16rem_1fr]">
        <Card title="Balance">
          <p className="text-3xl font-semibold tabular-nums" data-testid="balance">
            {formatUsd(balance.balanceMicros)}
          </p>
          <p className="mt-2 text-xs text-zinc-500">
            {formatUsd(balance.creditMicros)} added, {formatUsd(balance.usageMicros, 4)} used.
          </p>
          {balance.balanceMicros <= 0n ? (
            <p className="mt-2 text-sm text-red-700">Requests are refused until you add credit.</p>
          ) : null}
        </Card>
        <Card title="Add credit">
          {manage ? null : <p className="mb-3 text-sm text-zinc-600">Only org owners and admins can buy credit.</p>}
          <AddCreditForm disabled={!manage} />
          <p className="mt-3 text-xs text-zinc-500">You pay on a Stripe checkout page; Loom never sees your card.</p>
        </Card>
      </div>
      <Card title="Auto-recharge" className="mt-6">
        <AutoRechargeForm
          enabled={auto.enabled}
          threshold={microsToDecimalString(auto.thresholdMicros).replace(/\.?0+$/, "")}
          amount={microsToDecimalString(auto.amountMicros).replace(/\.?0+$/, "")}
          disabled={!manage}
        />
      </Card>
      <Card title="Receipts and credit history" className="mt-6">
        {history.length === 0 ? (
          <Empty>Nothing yet.</Empty>
        ) : (
          <div className="-mx-5 overflow-x-auto">
            <table className="min-w-full divide-y divide-zinc-200">
              <thead>
                <tr>
                  <th className={th}>Date</th>
                  <th className={th}>What</th>
                  <th className={th}>Note</th>
                  <th className={th}>Reference</th>
                  <th className={thNum}>Amount</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {history.map((h) => (
                  <tr key={h.id}>
                    <td className={td}>{formatDate(h.createdAt)}</td>
                    <td className={td}>
                      <Badge tone={h.amountMicros > 0n ? "ok" : "muted"}>{KIND_LABEL[h.kind]}</Badge>
                    </td>
                    <td className={td}>{h.memo ?? ""}</td>
                    <td className={`${td} font-mono text-xs`}>{h.externalRef ?? ""}</td>
                    <td className={tdNum}>{formatUsd(h.amountMicros)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
