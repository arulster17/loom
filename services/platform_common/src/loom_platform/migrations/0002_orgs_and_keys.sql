-- Organizations, membership and API keys.
--
-- Signups need an admin's approval (user decision 2026-10-31): a new org starts in
-- 'pending_approval', its keys are refused by the gateway, and the $5 signup credit is
-- posted only when an admin approves it (loom_approve_org, migration 0005).

CREATE TABLE orgs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name              text NOT NULL,
    status            text NOT NULL DEFAULT 'pending_approval'
                      CHECK (status IN ('pending_approval', 'active', 'suspended', 'rejected')),
    approved_at       timestamptz,
    approved_by       uuid REFERENCES users (id) ON DELETE SET NULL,
    suspended_at      timestamptz,
    suspended_reason  text,
    retention_days    int NOT NULL DEFAULT 0 CHECK (retention_days >= 0),
    -- Per-org limit overrides; NULL means the platform default (LOOM_DEFAULT_ORG_*).
    rpm_limit         int CHECK (rpm_limit > 0),
    tpm_limit         int CHECK (tpm_limit > 0),
    concurrency_limit int CHECK (concurrency_limit > 0),
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE org_members (
    org_id     uuid NOT NULL REFERENCES orgs (id) ON DELETE CASCADE,
    user_id    uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    role       text NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, user_id)
);
CREATE INDEX org_members_user_id ON org_members (user_id);

CREATE TABLE api_keys (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id             uuid NOT NULL REFERENCES orgs (id),
    created_by         uuid REFERENCES users (id) ON DELETE SET NULL,
    name               text NOT NULL DEFAULT '',
    -- 'loom_sk_' + 12 base62 characters: the displayable, non-secret part of the key.
    prefix             text NOT NULL UNIQUE CHECK (prefix ~ '^loom_sk_[0-9A-Za-z]{12}$'),
    -- SHA-256 of the full key; the plaintext is shown once and never stored.
    key_hash           bytea NOT NULL UNIQUE CHECK (octet_length(key_hash) = 32),
    scopes             text[] NOT NULL DEFAULT '{}',
    -- Lifetime spend cap for this key, micro-dollars; NULL = no cap.
    spend_limit_micros bigint CHECK (spend_limit_micros >= 0),
    -- Per-key limit overrides; NULL means the platform default (LOOM_DEFAULT_KEY_*).
    rpm_limit          int CHECK (rpm_limit > 0),
    tpm_limit          int CHECK (tpm_limit > 0),
    concurrency_limit  int CHECK (concurrency_limit > 0),
    created_at         timestamptz NOT NULL DEFAULT now(),
    last_used_at       timestamptz,
    revoked_at         timestamptz
);
CREATE INDEX api_keys_org_id ON api_keys (org_id);

-- Auth-cache invalidation. The gateway caches key records in Redis for at most
-- LOOM_AUTH_CACHE_TTL_S (4 s), which alone bounds revocation and suspension delay; a
-- listener on this channel drops the cached record at once (docs/phase1-contracts.md).
-- NOTIFY is delivered on commit, so the listener never sees an uncommitted change.
CREATE FUNCTION loom_notify_api_key_changed() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('loom_auth_changed', json_build_object(
        'kind', 'api_key',
        'org_id', NEW.org_id,
        'key_id', NEW.id,
        'key_hash', encode(NEW.key_hash, 'hex'))::text);
    RETURN NULL;
END $$;

CREATE TRIGGER api_keys_notify_changed
    AFTER UPDATE OF revoked_at, scopes, spend_limit_micros, rpm_limit, tpm_limit,
                    concurrency_limit
    ON api_keys FOR EACH ROW EXECUTE FUNCTION loom_notify_api_key_changed();

CREATE FUNCTION loom_notify_org_changed() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('loom_auth_changed', json_build_object(
        'kind', 'org',
        'org_id', NEW.id,
        'status', NEW.status)::text);
    RETURN NULL;
END $$;

CREATE TRIGGER orgs_notify_changed
    AFTER UPDATE OF status, rpm_limit, tpm_limit, concurrency_limit
    ON orgs FOR EACH ROW EXECUTE FUNCTION loom_notify_org_changed();

CREATE TABLE auto_recharge (
    org_id           uuid PRIMARY KEY REFERENCES orgs (id) ON DELETE CASCADE,
    enabled          boolean NOT NULL DEFAULT false,
    threshold_micros bigint NOT NULL CHECK (threshold_micros >= 0),
    amount_micros    bigint NOT NULL CHECK (amount_micros > 0)
);
