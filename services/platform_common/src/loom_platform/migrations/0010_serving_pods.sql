-- Serving pods and the spend that accrues on them (docs/phase1-m2-spec.md §5). Written by
-- the pool manager (loom-poolmgr) only.
--
-- A row is written before RunPod is called, so an ambiguous create can be found and
-- cleaned up and a restarted manager can adopt what it owns. pod_id is null until the
-- create answers. Secrets are never stored here: no token, no key, no certificate.
CREATE TABLE serving_pods (
    id                 uuid PRIMARY KEY,
    model_id           text NOT NULL,
    pod_name           text UNIQUE NOT NULL,
    pod_id             text UNIQUE,
    -- creating | bootstrapping | ready | draining | failed | terminated
    state              text NOT NULL,
    gpu_type_id        text,
    gpu_count          int,
    data_center        text,
    -- RunPod costPerHr plus disk, from the API.
    cost_per_hr_micros bigint,
    public_ip          text,
    proxy_port         int,
    ssh_port           int,
    cert_not_after     timestamptz,
    lease_until        timestamptz,
    -- Cold-start stage offsets (pod_created ... first_token), named as the bench names them.
    stages             jsonb NOT NULL DEFAULT '{}',
    failure            text,
    -- manager | watchdog | reaper | admin | runpod
    terminated_by      text,
    created_at         timestamptz NOT NULL,
    ready_at           timestamptz,
    drain_started_at   timestamptz,
    terminated_at      timestamptz
);

-- The live pods of a model (max_replicas counts every row not terminated).
CREATE INDEX serving_pods_live ON serving_pods (model_id) WHERE state <> 'terminated';
-- Month-to-date spend sums this month's rows.
CREATE INDEX serving_pods_created_at ON serving_pods (created_at);

-- Spend alerts sent: once per (UTC month, fraction of the monthly cap).
CREATE TABLE serving_spend_alerts (
    month    date,
    fraction numeric,
    sent_at  timestamptz,
    PRIMARY KEY (month, fraction)
);
