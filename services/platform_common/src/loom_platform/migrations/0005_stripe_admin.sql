-- Stripe webhook events, the admin audit log, and signup approval.

-- One row per Stripe event id: the webhook handler inserts with ON CONFLICT DO NOTHING
-- and posts its ledger credit with idempotency key 'stripe:<event id>'.
CREATE TABLE stripe_events (
    id           text PRIMARY KEY,
    type         text NOT NULL,
    payload      jsonb NOT NULL,
    received_at  timestamptz NOT NULL DEFAULT now(),
    processed_at timestamptz
);

-- Append-only record of every admin action. actor_user_id has no foreign key on
-- purpose: deleting a user must not need to rewrite the log.
CREATE TABLE admin_audit_log (
    id            bigserial PRIMARY KEY,
    actor_user_id uuid NOT NULL,
    action        text NOT NULL,
    target_type   text NOT NULL,
    target_id     text NOT NULL,
    details       jsonb NOT NULL DEFAULT '{}',
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX admin_audit_log_target ON admin_audit_log (target_type, target_id);

CREATE TRIGGER admin_audit_log_append_only
    BEFORE UPDATE OR DELETE ON admin_audit_log
    FOR EACH ROW EXECUTE FUNCTION loom_forbid_change();
CREATE TRIGGER admin_audit_log_no_truncate
    BEFORE TRUNCATE ON admin_audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION loom_forbid_change();
REVOKE UPDATE, DELETE, TRUNCATE ON admin_audit_log FROM PUBLIC;

-- Approve a pending signup: the org becomes active, the signup credit is posted, and
-- the action is audited, all in the caller's transaction. Called by the web app's admin
-- page (and loom_platform.ledger.approve_org), so the money logic lives in one place.
--
-- The credit is a 'promo' transaction with idempotency key 'promo:signup:<org id>', so
-- an org can never receive it twice, even if it is approved, suspended and approved
-- again some other way. p_promo_micros = 0 approves without a credit.
CREATE FUNCTION loom_approve_org(
    p_org_id        uuid,
    p_admin_user_id uuid,
    p_promo_micros  bigint DEFAULT 5000000,
    OUT promo_transaction_id uuid)
LANGUAGE plpgsql AS $$
DECLARE
    v_status text;
    v_role   text;
BEGIN
    SELECT role INTO v_role FROM users WHERE id = p_admin_user_id;
    IF v_role IS DISTINCT FROM 'admin' THEN
        RAISE EXCEPTION 'user % is not an admin', p_admin_user_id
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    SELECT status INTO v_status FROM orgs WHERE id = p_org_id FOR UPDATE;
    IF v_status IS NULL THEN
        RAISE EXCEPTION 'org % does not exist', p_org_id USING ERRCODE = 'no_data_found';
    END IF;
    IF v_status <> 'pending_approval' THEN
        RAISE EXCEPTION 'org % is %, not pending_approval', p_org_id, v_status
            USING ERRCODE = 'object_not_in_prerequisite_state';
    END IF;
    IF p_promo_micros < 0 THEN
        RAISE EXCEPTION 'promo amount must be >= 0' USING ERRCODE = 'invalid_parameter_value';
    END IF;

    UPDATE orgs SET status = 'active', approved_at = now(), approved_by = p_admin_user_id
    WHERE id = p_org_id;

    IF p_promo_micros > 0 THEN
        SELECT t.transaction_id INTO promo_transaction_id
        FROM loom_post_transaction(
            'promo', 'promo:signup:' || p_org_id::text, p_org_id, NULL, 'signup credit',
            ARRAY[loom_org_credit_account(p_org_id),
                  '00000000-0000-0000-0000-000000000003'::uuid],
            ARRAY[p_promo_micros, -p_promo_micros]) AS t;
    END IF;

    INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id, details)
    VALUES (p_admin_user_id, 'org.approve', 'org', p_org_id::text,
            jsonb_build_object('promo_micros', p_promo_micros,
                               'promo_transaction_id', promo_transaction_id));
END $$;
