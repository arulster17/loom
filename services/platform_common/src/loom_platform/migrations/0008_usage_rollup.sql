-- Usage totals without re-summing every usage event ever (docs/phase1-contracts.md §3).
--
-- loom_org_usage_total / loom_key_usage_total summed all of an org's or key's
-- usage_events. They run every minute per loaded org (billing reconcile), on every
-- gateway balance load, and on the web app's balance, keys and admin pages, so their
-- cost grew with all history. Now:
--
--   usage total = posted usage (a rollup row, kept by trigger)
--               + unposted usage (events not yet in a ledger batch: a few minutes'
--                 worth, read through a partial index)
--
-- An event counts in the rollup exactly while it is linked to a ledger transaction
-- (`ledger_transaction_id IS NOT NULL`). The per-minute batch links events in the same
-- transaction that posts them, and the trigger adds them to the rollup in that same
-- transaction, so any snapshot sees an event either in the rollup or among the unposted
-- ones, never both or neither. Results equal the old full sums (tested on randomized
-- data). Dropping or truncating a partition bypasses row triggers: the rollup then keeps
-- the history the old sums would have lost (usage totals are all-time).

CREATE TABLE usage_posted_org (
    org_id        uuid PRIMARY KEY REFERENCES orgs (id),
    posted_micros bigint NOT NULL DEFAULT 0
);

CREATE TABLE usage_posted_key (
    api_key_id    uuid PRIMARY KEY REFERENCES api_keys (id),
    posted_micros bigint NOT NULL DEFAULT 0
);

-- Unposted usage per key (the per-org one exists: usage_events_unposted).
CREATE INDEX usage_events_unposted_key ON usage_events (api_key_id)
    WHERE ledger_transaction_id IS NULL AND cost_micros > 0;

CREATE FUNCTION loom_usage_posted_bump(p_org uuid, p_key uuid, p_micros bigint) RETURNS void
LANGUAGE sql AS $$
    INSERT INTO usage_posted_org AS t (org_id, posted_micros) VALUES (p_org, p_micros)
    ON CONFLICT (org_id) DO UPDATE SET posted_micros = t.posted_micros + EXCLUDED.posted_micros;
    INSERT INTO usage_posted_key AS t (api_key_id, posted_micros) VALUES (p_key, p_micros)
    ON CONFLICT (api_key_id) DO UPDATE SET posted_micros = t.posted_micros + EXCLUDED.posted_micros;
$$;

CREATE FUNCTION loom_usage_events_rollup() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP IN ('UPDATE', 'DELETE') AND OLD.ledger_transaction_id IS NOT NULL
       AND OLD.cost_micros <> 0 THEN
        PERFORM loom_usage_posted_bump(OLD.org_id, OLD.api_key_id, -OLD.cost_micros);
    END IF;
    IF TG_OP IN ('UPDATE', 'INSERT') AND NEW.ledger_transaction_id IS NOT NULL
       AND NEW.cost_micros <> 0 THEN
        PERFORM loom_usage_posted_bump(NEW.org_id, NEW.api_key_id, NEW.cost_micros);
    END IF;
    RETURN NULL;
END $$;

CREATE TRIGGER usage_events_rollup
    AFTER INSERT OR UPDATE OR DELETE ON usage_events
    FOR EACH ROW EXECUTE FUNCTION loom_usage_events_rollup();

-- Backfill, after the trigger exists: creating it locked usage_events against writes
-- until this transaction commits, so nothing is counted twice or missed.
INSERT INTO usage_posted_org (org_id, posted_micros)
SELECT org_id, sum(cost_micros) FROM usage_events
WHERE ledger_transaction_id IS NOT NULL GROUP BY org_id;

INSERT INTO usage_posted_key (api_key_id, posted_micros)
SELECT api_key_id, sum(cost_micros) FROM usage_events
WHERE ledger_transaction_id IS NOT NULL GROUP BY api_key_id;

CREATE OR REPLACE FUNCTION loom_org_usage_total(p_org_id uuid) RETURNS bigint
LANGUAGE sql STABLE AS $$
    SELECT coalesce((SELECT posted_micros FROM usage_posted_org WHERE org_id = p_org_id), 0)
         + coalesce((SELECT sum(cost_micros) FROM usage_events
                     WHERE org_id = p_org_id
                       AND ledger_transaction_id IS NULL AND cost_micros > 0), 0)
$$;

CREATE OR REPLACE FUNCTION loom_key_usage_total(p_api_key_id uuid) RETURNS bigint
LANGUAGE sql STABLE AS $$
    SELECT coalesce((SELECT posted_micros FROM usage_posted_key
                     WHERE api_key_id = p_api_key_id), 0)
         + coalesce((SELECT sum(cost_micros) FROM usage_events
                     WHERE api_key_id = p_api_key_id
                       AND ledger_transaction_id IS NULL AND cost_micros > 0), 0)
$$;
