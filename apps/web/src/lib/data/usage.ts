import "server-only";

import type { OrgContext } from "@/lib/authz";
import { getPool, type Db } from "@/lib/db";
import { PLAYGROUND_SCOPE } from "@/lib/data/keys";
import { rangeBounds, type DayRange, type UsageRow } from "@/lib/usage";

/**
 * The org's usage in the range, one row per (UTC day, model, key). Only counts and
 * costs: Loom stores no prompts or completions.
 */
export async function usageRows(ctx: OrgContext, range: DayRange, db: Db = getPool()): Promise<UsageRow[]> {
  const [start, end] = rangeBounds(range);
  const r = await db.query<{
    day: string;
    model_id: string;
    api_key_id: string;
    key_name: string;
    key_prefix: string;
    playground: boolean;
    requests: string;
    prompt_tokens: string;
    completion_tokens: string;
    cached_tokens: string;
    cost_micros: string;
  }>(
    `SELECT to_char(e.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day,
            e.model_id, e.api_key_id, k.name AS key_name, k.prefix AS key_prefix,
            ($4 = ANY (k.scopes)) AS playground,
            count(*) AS requests,
            sum(e.prompt_tokens) AS prompt_tokens,
            sum(e.completion_tokens) AS completion_tokens,
            sum(e.cached_prompt_tokens) AS cached_tokens,
            sum(e.cost_micros) AS cost_micros
     FROM usage_events e
     JOIN api_keys k ON k.id = e.api_key_id AND k.org_id = e.org_id
     WHERE e.org_id = $1 AND e.created_at >= $2 AND e.created_at < $3
     GROUP BY 1, 2, 3, 4, 5, 6
     ORDER BY 1 DESC, 2, 4`,
    [ctx.orgId, start, end, PLAYGROUND_SCOPE],
  );
  return r.rows.map((x) => ({
    day: x.day,
    modelId: x.model_id,
    apiKeyId: x.api_key_id,
    keyName: x.key_name,
    keyPrefix: x.key_prefix,
    playground: x.playground,
    requests: Number(x.requests),
    promptTokens: Number(x.prompt_tokens),
    completionTokens: Number(x.completion_tokens),
    cachedTokens: Number(x.cached_tokens),
    costMicros: BigInt(x.cost_micros),
  }));
}
