-- The runtime role (docs/phase1-contracts.md §3, plan R8).
--
-- Two Postgres roles: the owner, which runs loom-migrate and owns every object, and a
-- runtime login (`loom_app`, LOOM_APP_DATABASE_USER) that every service uses. The
-- runtime login gets its privileges only through a NOLOGIN group role
-- (`loom_runtime`, LOOM_APP_DATABASE_GROUP) that it is a member of, so migrations grant
-- to one name. Being neither owner nor superuser, the runtime login cannot disable,
-- drop or replace the ledger's triggers, and the append-only tables give it only
-- SELECT and INSERT: append-only and zero-sum hold for it whatever the triggers' state.
--
-- Role names are configurable, so the group's name is not written here: loom-migrate
-- creates the group role before applying any migration and passes its name in the
-- session setting `loom.runtime_role`. Grants therefore go through dynamic SQL:
--
--     DO $$ BEGIN
--         EXECUTE format('GRANT SELECT, INSERT ON new_table TO %I',
--                        current_setting('loom.runtime_role'));
--     END $$;
--
-- **Every later migration that adds a table must grant on it the same way** (a test
-- checks that every table has an explicit grant to the group; partitions and
-- schema_migrations excepted: services reach partitions through their parent). New
-- sequences need USAGE, new functions EXECUTE.

DO $$
DECLARE
    r text := current_setting('loom.runtime_role', true);
BEGIN
    IF r IS NULL OR r = '' THEN
        RAISE EXCEPTION 'loom.runtime_role is not set: apply migrations with loom-migrate'
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
        RAISE EXCEPTION 'runtime group role % does not exist', r
            USING ERRCODE = 'undefined_object';
    END IF;

    -- Nobody but the owner creates objects in the schema (PostgreSQL 15+ already has
    -- this default; older clusters and upgraded ones do not).
    REVOKE CREATE ON SCHEMA public FROM PUBLIC;
    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', r);

    -- Append-only (triggers forbid UPDATE/DELETE/TRUNCATE for everyone; the runtime
    -- role is not even granted them).
    EXECUTE format('GRANT SELECT, INSERT ON ledger_entries, ledger_transactions, '
                   'admin_audit_log TO %I', r);
    -- Accounts are created by the orgs insert trigger and never changed.
    EXECUTE format('GRANT SELECT, INSERT ON ledger_accounts TO %I', r);

    -- Auth.js (@auth/pg-adapter) creates, updates and deletes all four.
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON users, accounts, sessions, '
                   'verification_token TO %I', r);
    -- Web app (signup, approval, limits, keys, auto-recharge), gateway (last use of a
    -- key), billing (usage linking, Stripe bookkeeping, the usage rollup trigger's
    -- upserts), pool manager (pod records).
    EXECUTE format('GRANT SELECT, INSERT, UPDATE ON orgs, api_keys, auto_recharge, '
                   'usage_events, usage_posted_org, usage_posted_key, stripe_events, '
                   'stripe_payments, serving_pods TO %I', r);
    EXECUTE format('GRANT SELECT, INSERT ON org_members, serving_spend_alerts TO %I', r);

    EXECUTE format('GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO %I', r);
    EXECUTE format('GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO %I', r);
END $$;

-- The one thing services create at run time: the monthly usage_events partitions
-- (billing). The function runs with its owner's rights and a fixed search_path, so the
-- runtime role can call it without being able to create anything itself; the
-- partitions it makes belong to the owner.
ALTER FUNCTION loom_ensure_usage_partitions(date, int)
    SECURITY DEFINER SET search_path = public, pg_temp;
REVOKE EXECUTE ON FUNCTION loom_ensure_usage_partitions(date, int) FROM PUBLIC;
