# Runbook: the pod CA (rollover, expiry, the gateway's certificate)

Spec: [phase1-m3-spec.md](phase1-m3-spec.md) §7 (alarm P3, and G3 for the gateway's
certificate). Contracts: [phase1-contracts.md](phase1-contracts.md) §2, §9.

## What there is

- **The pod CA** (`LOOM_POOLMGR_CA_CERT` / `_KEY` in the manager, secret `pod-ca`): signs
  every pod's server certificate (SAN = the pod id), the manager's client certificate and
  the gateway's client certificate. `loom-deploy secrets init` makes it with 2 years of
  validity. The manager exports `loom_poolmgr_ca_expiry_seconds` and refuses to start
  when the CA has less than `server_cert_renew_before_s` (2 days) left.
- **The trust bundle** (`LOOM_POD_CA_CERT` in the gateway): the gateway can trust several
  CA certificates, but in M3 `loom-deploy` renders it from `pod-ca` alone (one CA). A pod
  trusts, for its clients, only the CA that signed its own certificate. Old + new bundles
  (`pod-ca-next`) are M4's.
- **The gateway's client certificate**: issued by the manager from the CA in use, 7 days,
  into `LOOM_POOLMGR_GATEWAY_CERT_DIR/loom-gateway.pem` (key and certificate in one
  file), at start unless the file holds a current one and then every 24 h. The gateway
  checks that file and the CA bundle every `LOOM_GATEWAY_CERT_CHECK_S` (60 s) and uses a
  changed set for new upstream connections; streams already open finish on theirs.
  It exports `loom_gateway_client_cert_expiry_seconds`.

## Alarms

- **P3** `loom_poolmgr_ca_expiry_seconds` below 30 days. Do the rollover below now: it
  takes an afternoon, the refusal at 2 days takes serving down.
  (A manager that refuses to start publishes nothing, so the alarm treats missing data as
  breaching: a P3 in ALARM with the manager down means look at `docker logs "$PM"`
  (`$PM` as in step 3) first.)
- **G3** `loom_gateway_client_cert_expiry_seconds` below 2 days: the rotation is broken.
  Check, in this order: the manager is running (H1); its log has no `issuing the gateway's
  client certificate failed` (disk full, the directory not writable); the file in the
  shared volume (`gateway-cert`) is recent and mode 0400, owned by the gateway's uid
  (10001); the gateway's log has no `failed to load; keeping the old TLS context` error.
  Restarting the manager issues a new certificate at once if the file is missing, broken,
  from another CA or has under 6 days left.

## Rollover to a new CA (not zero downtime in M3)

**The rollover is not zero-downtime in M3** (accepted for M3, spec §7). A pod trusts, for
its clients, only the CA that signed its own certificate. Once the manager has switched to
the new CA and re-keyed a pod, that pod refuses the gateway's old-CA client certificate
until the manager has issued one from the new CA (at its start) and the gateway's next
check has loaded it. And after the deploy that switches `pod-ca` the gateway trusts only
the new CA, so it refuses every pod still serving a certificate from the old one.
**Expect requests to fail from that deploy until the manager has re-keyed every pod and
the gateway has loaded its new client certificate: minutes, longer with more pods**, not
one `LOOM_GATEWAY_CERT_CHECK_S` window. Roll over at low traffic; scaling the pool down
first shortens the outage. The real fix is M4-1 (the pods' client-trust bundle holding the
old and the new CA, `pod-ca-next`). Acceptance B20 records the outage's length, the number
of pods and the failed requests.

M3 has no `pod-ca-next` secret, and `loom-deploy` renders `LOOM_POD_CA_CERT` from `pod-ca`
alone, so there is no old + new bundle: the steps that put the new CA beside the old one
in the bundle (make `pod-ca-next`, deploy the gateway with the bundle old + new, drop the
old CA from the bundle) are M4's, together with the pods' client-trust bundle (spec §7,
"M4 prerequisite: zero-downtime CA rollover"). The box runs the manager and the gateway in
one compose project, so the switch is a deploy (`loom-deploy apply --release <sha>`,
gateway first: P4).

1. **Make the new CA** on the operator's machine (nothing here needs the box, Postgres or
   Redis):

   ```bash
   uv run loom-poolmgr new-ca --out ./pod-ca-new
   ```

   It writes `ca.crt`, `ca.key` and the manager's client certificate from it
   (`loom-poolmgr.crt`, `loom-poolmgr.key`, valid as long as the CA), keys mode 0600, and
   refuses to overwrite an existing CA. Check `openssl x509 -in pod-ca-new/ca.crt -noout
   -dates`.
2. **Switch `pod-ca` and `poolmgr-client` to the new values**: put them into Secrets
   Manager as new versions of `loom/platform/pod-ca` and `loom/platform/poolmgr-client`.
   Each value is a JSON object `{"crt": ..., "key": ...}` (the certificate and key PEMs
   as strings). Build it with `jq` and use the same `put-secret-value` pattern as
   `aws-setup.md` step 8, from `./pod-ca-new` (`ca.crt`, `ca.key` for `pod-ca`;
   `loom-poolmgr.crt`, `loom-poolmgr.key` for `poolmgr-client`).

   First note each secret's current version id (`aws secretsmanager list-secret-version-ids
   --region us-east-1 --secret-id <id>`: the one with stage `AWSCURRENT`). It is the way
   back, and the rollback below needs it:

   ```bash
   jq -n --rawfile crt pod-ca-new/ca.crt --rawfile key pod-ca-new/ca.key '{crt: $crt, key: $key}' \
     | aws secretsmanager put-secret-value --region us-east-1 \
       --secret-id loom/platform/pod-ca --secret-string file:///dev/stdin
   jq -n --rawfile crt pod-ca-new/loom-poolmgr.crt --rawfile key pod-ca-new/loom-poolmgr.key '{crt: $crt, key: $key}' \
     | aws secretsmanager put-secret-value --region us-east-1 \
       --secret-id loom/platform/poolmgr-client --secret-string file:///dev/stdin
   ```

   Then delete the local directory (the key never stays on a laptop).
3. **Deploy** (`loom-deploy apply --release <sha>`; the gateway first, then the manager:
   P4). The outage begins here: requests can fail until the manager has re-keyed every pod
   and the gateway has loaded its new client certificate (see above). On start the manager
   issues the gateway's client certificate from the new CA (the file was from the old CA,
   so it is replaced, whatever its age) and begins to issue pod certificates from the new
   CA. Recreating the manager makes it adopt the running pods, and adoption already re-keys
   each of them from the new CA. So `rekey` is a safety net, not a required step: it
   re-issues every ready pod's certificate again. After the recreate it is redundant and
   harmless (a pod whose adoption fails is terminated and replaced with a certificate from
   the new CA), and step 4 is the gate:

   ```bash
   # On the box. The compose project is $LOOM_DEPLOY_PROJECT (default loom); the manager's
   # container is found by its compose labels, so no release sha or compose file is needed.
   PM=$(docker ps -q -f "label=com.docker.compose.project=${LOOM_DEPLOY_PROJECT:-loom}" \
        -f label=com.docker.compose.service=poolmgr)
   docker exec "$PM" loom-poolmgr rekey                   # every ready pod
   docker exec "$PM" loom-poolmgr rekey --model qwen3-8b  # or one model
   ```

   The command queues the re-key; the manager applies it on its next tick (seconds). Pods
   keep their tokens and stay in routing. Pods that are not ready get a certificate from
   the new CA when they are created.
4. **Check that every pod's certificate chains to the new CA** (this is the gate, not the
   `rekey`):

   ```bash
   docker exec "$PM" loom-poolmgr status     # $PM as in step 3
   ```

   Under each ready pod a line shows `certificate: issued by <CA name>, expires ...
   (chains to the manager's CA)`. A pod whose line says `does NOT chain to the manager's
   CA` still has an old certificate: run `rekey` again, and if it persists look at the
   manager's log for `admin rekey of pod ... failed`. A pod whose line says `certificate:
   not read (...)` blocks the next step too: its certificate could not be checked. Do not
   go on until no pod says either.
5. **The gateway's client certificate** is from the new CA already (step 3). Check
   `loom_gateway_client_cert_expiry_seconds` (about 7 days) on the gateway's metrics, and
   that the file in the `gateway-cert` volume was written after the switch; the gateway's
   log says `reloaded the client certificate and pod CA bundle`, and requests succeed
   again. Confirm `loom_poolmgr_ca_expiry_seconds` shows the new CA's validity (P3
   returns to OK within its next 1-hour period). The old CA's secret versions need no
   cleanup: Secrets Manager has no per-version delete. After the put the old version keeps
   the `AWSPREVIOUS` label (the way back, still readable by the instance role) until the
   next rollover's put; only then does it lose its label, and Secrets Manager eventually
   removes it.

If a step fails, going back means moving the `AWSCURRENT` stage of each of the two
secrets back to the version id you noted before the put, a deploy and a `rekey`. Roll
back only a secret whose put succeeded, i.e. one whose `AWSCURRENT` is no longer the noted
id: if the `pod-ca` put succeeded and the `poolmgr-client` put failed, `poolmgr-client`'s
`AWSPREVIOUS` is an even older certificate, or none:

```bash
aws secretsmanager update-secret-version-stage --region us-east-1 \
  --secret-id loom/platform/pod-ca --version-stage AWSCURRENT \
  --move-to-version-id <the version id you noted before the put> \
  --remove-from-version-id <the current (new) version id>
```

(the same for `loom/platform/poolmgr-client`). Do not try to delete the new version. The
same outage applies in that direction, because the pods again trust only the CA that
signed them.

## With the mock backend (the M3 box)

The whole runbook runs on the box with the mock pods (`provider: local`,
`LOOM_POOLMGR_MOCK_BACKEND=true`): the local pods' proxies take their certificates the
same way (written as files instead of over SSH), `rekey` reaches them, and `status` reads
their certificates over mTLS from inside the manager's container. Run it once with a
load going (acceptance B20): requests may fail only during the outage described above
(none before the deploy that switches the CA, none once every pod is re-keyed and the
gateway has loaded its new client certificate). Record the outage's length, the number of
pods and the failed requests.

With the mock backend the manager's local pods die when the manager is recreated, so no
old pod is ever re-keyed there. The mock run's outage comes from the manager recreate plus
the pods' cold start (new pods get certificates from the new CA when they are created),
not from re-keying.

## Notes

- `loom-poolmgr issue-ca` stays dev-only (it also writes the gateway's certificate with a
  30-day lifetime for compose); `new-ca` is the production command.
- The manager's own client certificate (secret `poolmgr-client`) is not rotated by the
  manager: `new-ca` makes it as long-lived as the CA. A rollover replaces it with the CA.
- A gateway client certificate or CA file that cannot be read, or that is not a valid PEM,
  never replaces the working one: the gateway logs an error and keeps its context. Fix the
  file and the next check (at most a minute) picks it up.
