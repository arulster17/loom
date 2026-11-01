-- Metered requests, one row per request that passed credit reservation, written by the
-- billing worker from the Redis usage stream (docs/phase1-contracts.md).
--
-- Partitioned by month on created_at. A partitioned table's primary key must contain
-- the partition key, so it is (id, created_at): the gateway fixes created_at (request
-- receive time) once per request and every redelivery carries the same pair, so
-- INSERT ... ON CONFLICT (id, created_at) DO NOTHING de-duplicates exactly.
--
-- There is no default partition: a missing month fails the insert loudly (the event
-- stays pending in the stream) rather than landing somewhere a later partition cannot be
-- created over. The billing worker calls loom_ensure_usage_partitions at start and daily.

CREATE TABLE usage_events (
    id                    uuid NOT NULL,
    org_id                uuid NOT NULL REFERENCES orgs (id),
    api_key_id            uuid NOT NULL REFERENCES api_keys (id),
    model_id              text NOT NULL,
    -- '<input>:<cached input>:<output>' micro-dollars per 1M tokens (loom_platform.pricing).
    price_version         text NOT NULL CHECK (price_version ~ '^[0-9]+:[0-9]+:[0-9]+$'),
    cloud                 text NOT NULL,
    region                text,
    prompt_tokens         int NOT NULL CHECK (prompt_tokens >= 0),
    completion_tokens     int NOT NULL CHECK (completion_tokens >= 0),
    cached_prompt_tokens  int NOT NULL DEFAULT 0
                          CHECK (cached_prompt_tokens >= 0 AND cached_prompt_tokens <= prompt_tokens),
    latency_ms            int NOT NULL CHECK (latency_ms >= 0),
    ttft_ms               int CHECK (ttft_ms >= 0),
    status                text NOT NULL CHECK (status IN ('ok', 'aborted', 'error')),
    cost_micros           bigint NOT NULL CHECK (cost_micros >= 0),
    -- The per-org per-minute usage transaction that debited this event; NULL until then
    -- (and forever for zero-cost events, which need no posting).
    ledger_transaction_id uuid REFERENCES ledger_transactions (id),
    created_at            timestamptz NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE INDEX usage_events_org_time ON usage_events (org_id, created_at);
CREATE INDEX usage_events_key_time ON usage_events (api_key_id, created_at);
CREATE INDEX usage_events_unposted ON usage_events (org_id)
    WHERE ledger_transaction_id IS NULL AND cost_micros > 0;

-- Creates the monthly partitions covering p_months months from p_from's month (UTC).
-- Idempotent; returns how many it created.
CREATE FUNCTION loom_ensure_usage_partitions(p_from date, p_months int) RETURNS int
LANGUAGE plpgsql AS $$
DECLARE
    v_start   date := date_trunc('month', p_from)::date;
    v_lo      date;
    v_hi      date;
    v_name    text;
    v_created int := 0;
BEGIN
    FOR i IN 0 .. p_months - 1 LOOP
        v_lo := (v_start + make_interval(months => i))::date;
        v_hi := (v_start + make_interval(months => i + 1))::date;
        v_name := 'usage_events_' || to_char(v_lo, 'YYYY_MM');
        IF to_regclass(v_name) IS NULL THEN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF usage_events FOR VALUES FROM (%L) TO (%L)',
                v_name, v_lo::text || ' 00:00:00+00', v_hi::text || ' 00:00:00+00');
            v_created := v_created + 1;
        END IF;
    END LOOP;
    RETURN v_created;
END $$;

-- Last month through 12 months ahead, so a fresh database works for a year even before
-- the billing worker first runs.
SELECT loom_ensure_usage_partitions(
    ((now() AT TIME ZONE 'UTC') - interval '1 month')::date, 14);

-- All metered usage of an org, posted to the ledger or not.
CREATE FUNCTION loom_org_usage_total(p_org_id uuid) RETURNS bigint
LANGUAGE sql STABLE AS $$
    SELECT coalesce(sum(cost_micros), 0)::bigint FROM usage_events WHERE org_id = p_org_id
$$;

CREATE FUNCTION loom_key_usage_total(p_api_key_id uuid) RETURNS bigint
LANGUAGE sql STABLE AS $$
    SELECT coalesce(sum(cost_micros), 0)::bigint FROM usage_events
    WHERE api_key_id = p_api_key_id
$$;
