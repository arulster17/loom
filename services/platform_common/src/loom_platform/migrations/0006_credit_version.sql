-- Version of an org's credit total, so a stale reader can never overwrite a newer total
-- in Redis (set_credit.lua compares versions).
--
-- The version is the number of the org's non-usage ledger transactions. The ledger is
-- append-only and every snapshot sees a prefix of the commit order, so of two snapshots
-- the one with the larger count is the newer, and equal counts mean the same credit
-- total. Read it in the same statement as loom_org_credit_total (one snapshot).
CREATE FUNCTION loom_org_credit_version(p_org_id uuid) RETURNS bigint
LANGUAGE sql STABLE AS $$
    SELECT count(*)::bigint
    FROM ledger_transactions
    WHERE org_id = p_org_id AND kind <> 'usage'
$$;
