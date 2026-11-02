-- Auth.js Postgres adapter tables (@auth/pg-adapter), plus Loom's admin role.
--
-- Column names and quoting are exactly what the adapter's queries use ("userId",
-- "providerAccountId", "sessionToken", "emailVerified"). Ids are uuid instead of the
-- adapter docs' SERIAL: the adapter never supplies an id on insert (it relies on the
-- column default and RETURNING) and passes ids back as strings, so uuid works, and every
-- other platform table can reference users with uuid.

CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE users (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name            text,
    email           citext UNIQUE,
    "emailVerified" timestamptz,
    image           text,
    -- Loom: 'admin' may approve signups, adjust credits, suspend orgs, disable models.
    -- Granting or removing it is recorded in admin_audit_log.
    role            text NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE accounts (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    "userId"            uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    type                text NOT NULL,
    provider            text NOT NULL,
    "providerAccountId" text NOT NULL,
    refresh_token       text,
    access_token        text,
    expires_at          bigint,
    id_token            text,
    scope               text,
    session_state       text,
    token_type          text,
    UNIQUE (provider, "providerAccountId")
);
CREATE INDEX accounts_user_id ON accounts ("userId");

CREATE TABLE sessions (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    "userId"       uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    expires        timestamptz NOT NULL,
    "sessionToken" text NOT NULL UNIQUE
);
CREATE INDEX sessions_user_id ON sessions ("userId");

CREATE TABLE verification_token (
    identifier text NOT NULL,
    expires    timestamptz NOT NULL,
    token      text NOT NULL,
    PRIMARY KEY (identifier, token)
);
