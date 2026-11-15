-- Double-entry ledger (docs/PLAN.md, accepted 2026-10-08).
--
-- Every amount is integer micro-dollars. An account's balance is the sum of its
-- entries; for an org's 'org_credit' account a positive balance is credit the org can
-- spend. Every transaction's entries sum to zero (deferred constraint trigger, checked at
-- commit). ledger_transactions and ledger_entries are append-only: corrections are new
-- transactions, never edits.
--
-- Postings by kind (org_credit / counter-account):
--   purchase    +X org_credit   -X stripe_clearing
--   usage       -X org_credit   +X revenue          (one per org per minute, billing worker)
--   promo       +X org_credit   -X promo_expense    (signup credit, on approval)
--   refund      -X org_credit   +X stripe_clearing  (money back to the card)
--   adjustment  +/-X org_credit against promo_expense (grant) or revenue (correction)

CREATE TABLE ledger_accounts (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     uuid REFERENCES orgs (id),
    kind       text NOT NULL
               CHECK (kind IN ('org_credit', 'revenue', 'stripe_clearing', 'promo_expense')),
    created_at timestamptz NOT NULL DEFAULT now(),
    -- org_credit accounts belong to one org; the others are platform-wide singletons.
    CHECK ((kind = 'org_credit') = (org_id IS NOT NULL))
);
CREATE UNIQUE INDEX ledger_accounts_org_kind ON ledger_accounts (org_id, kind)
    WHERE org_id IS NOT NULL;
CREATE UNIQUE INDEX ledger_accounts_system_kind ON ledger_accounts (kind)
    WHERE org_id IS NULL;

-- Fixed ids for the platform accounts (loom_platform.ledger.SYSTEM_ACCOUNTS).
INSERT INTO ledger_accounts (id, org_id, kind) VALUES
    ('00000000-0000-0000-0000-000000000001', NULL, 'revenue'),
    ('00000000-0000-0000-0000-000000000002', NULL, 'stripe_clearing'),
    ('00000000-0000-0000-0000-000000000003', NULL, 'promo_expense');

-- Every org gets its credit account when it is created.
CREATE FUNCTION loom_create_org_credit_account() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO ledger_accounts (org_id, kind) VALUES (NEW.id, 'org_credit');
    RETURN NULL;
END $$;

CREATE TRIGGER orgs_create_credit_account
    AFTER INSERT ON orgs FOR EACH ROW EXECUTE FUNCTION loom_create_org_credit_account();

CREATE TABLE ledger_transactions (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind            text NOT NULL
                    CHECK (kind IN ('purchase', 'usage', 'refund', 'promo', 'adjustment')),
    -- Posting the same key twice is a no-op (loom_post_transaction). Formats are listed
    -- in docs/phase1-contracts.md, e.g. 'usage:<org_id>:<minute>', 'promo:signup:<org_id>',
    -- 'stripe:<event id>'.
    idempotency_key text NOT NULL UNIQUE,
    org_id          uuid REFERENCES orgs (id),
    external_ref    text,
    memo            text,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ledger_transactions_org ON ledger_transactions (org_id, created_at);

CREATE TABLE ledger_entries (
    id             bigserial PRIMARY KEY,
    transaction_id uuid NOT NULL REFERENCES ledger_transactions (id),
    account_id     uuid NOT NULL REFERENCES ledger_accounts (id),
    amount_micros  bigint NOT NULL CHECK (amount_micros <> 0)
);
CREATE INDEX ledger_entries_transaction ON ledger_entries (transaction_id);
CREATE INDEX ledger_entries_account ON ledger_entries (account_id);

-- Zero-sum: checked at commit for every transaction that got entries.
CREATE FUNCTION loom_ledger_check_balanced() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    v_sum   bigint;
    v_count int;
BEGIN
    SELECT coalesce(sum(amount_micros), 0), count(*) INTO v_sum, v_count
    FROM ledger_entries WHERE transaction_id = NEW.transaction_id;
    IF v_sum <> 0 THEN
        RAISE EXCEPTION 'ledger transaction % is unbalanced: entries sum to %',
            NEW.transaction_id, v_sum USING ERRCODE = 'check_violation';
    END IF;
    IF v_count < 2 THEN
        RAISE EXCEPTION 'ledger transaction % has % entries; at least 2 are required',
            NEW.transaction_id, v_count USING ERRCODE = 'check_violation';
    END IF;
    RETURN NULL;
END $$;

CREATE CONSTRAINT TRIGGER ledger_entries_balanced
    AFTER INSERT ON ledger_entries DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION loom_ledger_check_balanced();

-- A transaction row with no entries is refused at commit too.
CREATE FUNCTION loom_ledger_check_has_entries() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF (SELECT count(*) FROM ledger_entries WHERE transaction_id = NEW.id) < 2 THEN
        RAISE EXCEPTION 'ledger transaction % has fewer than 2 entries', NEW.id
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN NULL;
END $$;

CREATE CONSTRAINT TRIGGER ledger_transactions_have_entries
    AFTER INSERT ON ledger_transactions DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION loom_ledger_check_has_entries();

-- Append-only. The triggers bind every role including the owner; only a superuser can
-- get round them (by disabling triggers), which is a recorded admin action.
CREATE FUNCTION loom_forbid_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only: % is not allowed', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'insufficient_privilege';
END $$;

CREATE TRIGGER ledger_entries_append_only
    BEFORE UPDATE OR DELETE ON ledger_entries
    FOR EACH ROW EXECUTE FUNCTION loom_forbid_change();
CREATE TRIGGER ledger_entries_no_truncate
    BEFORE TRUNCATE ON ledger_entries
    FOR EACH STATEMENT EXECUTE FUNCTION loom_forbid_change();
CREATE TRIGGER ledger_transactions_append_only
    BEFORE UPDATE OR DELETE ON ledger_transactions
    FOR EACH ROW EXECUTE FUNCTION loom_forbid_change();
CREATE TRIGGER ledger_transactions_no_truncate
    BEFORE TRUNCATE ON ledger_transactions
    FOR EACH STATEMENT EXECUTE FUNCTION loom_forbid_change();
REVOKE UPDATE, DELETE, TRUNCATE ON ledger_entries, ledger_transactions FROM PUBLIC;

-- The one way to post: balanced, idempotent, org-scoped.
--
-- Inserts the transaction and its entries, or, when idempotency_key was posted before
-- with the same kind, org and entries, returns that transaction with created = false.
-- The same key with different content raises unique_violation. Every account must be
-- a platform account or the given org's own credit account, so a posting can never
-- touch another tenant's balance.
CREATE FUNCTION loom_post_transaction(
    p_kind            text,
    p_idempotency_key text,
    p_org_id          uuid,
    p_external_ref    text,
    p_memo            text,
    p_account_ids     uuid[],
    p_amounts         bigint[],
    OUT transaction_id uuid,
    OUT created        boolean)
LANGUAGE plpgsql AS $$
DECLARE
    v_existing ledger_transactions%ROWTYPE;
    v_want     text[];
    v_have     text[];
BEGIN
    IF p_account_ids IS NULL OR p_amounts IS NULL
       OR cardinality(p_account_ids) <> cardinality(p_amounts)
       OR cardinality(p_amounts) < 2 THEN
        RAISE EXCEPTION 'ledger: need at least 2 entries, one amount per account'
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF (SELECT sum(a) FROM unnest(p_amounts) AS a) <> 0 THEN
        RAISE EXCEPTION 'ledger: entries sum to %, not 0',
            (SELECT sum(a) FROM unnest(p_amounts) AS a) USING ERRCODE = 'check_violation';
    END IF;
    IF EXISTS (
        SELECT 1 FROM unnest(p_account_ids) AS u(account_id)
        LEFT JOIN ledger_accounts la ON la.id = u.account_id
        WHERE la.id IS NULL OR (la.org_id IS NOT NULL AND la.org_id IS DISTINCT FROM p_org_id)
    ) THEN
        RAISE EXCEPTION 'ledger: unknown account, or an org account outside org %', p_org_id
            USING ERRCODE = 'foreign_key_violation';
    END IF;

    INSERT INTO ledger_transactions (kind, idempotency_key, org_id, external_ref, memo)
    VALUES (p_kind, p_idempotency_key, p_org_id, p_external_ref, p_memo)
    ON CONFLICT (idempotency_key) DO NOTHING
    RETURNING id INTO transaction_id;

    IF transaction_id IS NULL THEN
        SELECT * INTO v_existing FROM ledger_transactions
        WHERE idempotency_key = p_idempotency_key;
        SELECT array_agg(a::text || ':' || m::text ORDER BY a::text || ':' || m::text)
            INTO v_want FROM unnest(p_account_ids, p_amounts) AS u(a, m);
        SELECT array_agg(account_id::text || ':' || amount_micros::text
                         ORDER BY account_id::text || ':' || amount_micros::text)
            INTO v_have FROM ledger_entries WHERE ledger_entries.transaction_id = v_existing.id;
        IF v_existing.kind <> p_kind OR v_existing.org_id IS DISTINCT FROM p_org_id
           OR v_want IS DISTINCT FROM v_have THEN
            RAISE EXCEPTION 'ledger: idempotency key % was already posted with other content',
                p_idempotency_key USING ERRCODE = 'unique_violation';
        END IF;
        transaction_id := v_existing.id;
        created := false;
        RETURN;
    END IF;

    INSERT INTO ledger_entries (transaction_id, account_id, amount_micros)
    SELECT loom_post_transaction.transaction_id, u.a, u.m
    FROM unnest(p_account_ids, p_amounts) AS u(a, m);
    created := true;
END $$;

-- Balances. The Redis balance cache (docs/phase1-contracts.md) uses the last two:
-- available = credit_total - usage_total, where usage_total counts every metered
-- request whether or not its per-minute batch has been posted yet.

CREATE FUNCTION loom_org_credit_account(p_org_id uuid) RETURNS uuid
LANGUAGE sql STABLE AS $$
    SELECT id FROM ledger_accounts WHERE org_id = p_org_id AND kind = 'org_credit'
$$;

-- Posted balance of the org's credit account (excludes usage not yet batched).
CREATE FUNCTION loom_org_balance(p_org_id uuid) RETURNS bigint
LANGUAGE sql STABLE AS $$
    SELECT coalesce(sum(e.amount_micros), 0)::bigint
    FROM ledger_entries e
    WHERE e.account_id = loom_org_credit_account(p_org_id)
$$;

-- Everything credited to the org except usage debits: purchases, promos, refunds,
-- adjustments.
CREATE FUNCTION loom_org_credit_total(p_org_id uuid) RETURNS bigint
LANGUAGE sql STABLE AS $$
    SELECT coalesce(sum(e.amount_micros), 0)::bigint
    FROM ledger_entries e
    JOIN ledger_transactions t ON t.id = e.transaction_id
    WHERE e.account_id = loom_org_credit_account(p_org_id) AND t.kind <> 'usage'
$$;

-- The billing worker refreshes the Redis credit total when this fires.
CREATE FUNCTION loom_notify_credit_changed() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('loom_credit_changed',
                      json_build_object('org_id', NEW.org_id, 'kind', NEW.kind)::text);
    RETURN NULL;
END $$;

CREATE TRIGGER ledger_transactions_notify_credit
    AFTER INSERT ON ledger_transactions FOR EACH ROW
    WHEN (NEW.kind <> 'usage' AND NEW.org_id IS NOT NULL)
    EXECUTE FUNCTION loom_notify_credit_changed();
