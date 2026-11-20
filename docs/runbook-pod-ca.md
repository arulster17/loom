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
check has loaded it. **Expect requests to fail for up to about `LOOM_GATEWAY_CERT_CHECK_S`
(60 s) after the CA switch**: do the rollover in a quiet period.

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
   Manager as new versions of `pod-ca` (`crt`, `key`) and `poolmgr-client` (`crt`, `key`)
   with the `printf %s ... | aws secretsmanager put-secret-value --secret-string
   file:///dev/stdin` pattern (the previous version is the way back), then delete the
   local directory (the key never stays on a laptop).
3. **Deploy** (the gateway first, then the manager: P4). From here requests can fail until
   the gateway has loaded its new client certificate (see above). On start the manager
   issues the gateway's client certificate from the new CA (the file was from the old CA,
   so it is replaced, whatever its age) and begins to issue pod certificates from the new
   CA. Then re-issue every pod's certificate now instead of waiting for each to near
   expiry:

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
4. **Check that every pod's certificate chains to the new CA**:

   ```bash
   docker exec "$PM" loom-poolmgr status     # $PM as in step 3
   ```

   Under each ready pod a line shows `certificate: issued by <CA name>, expires ...
   (chains to the manager's CA)`. A pod whose line says `does NOT chain to the manager's
   CA` still has an old certificate: run `rekey` again, and if it persists look at the
   manager's log for `admin rekey of pod ... failed`. Do not go on until none says it.
5. **The gateway's client certificate** is from the new CA already (step 3). Check
   `loom_gateway_client_cert_expiry_seconds` (about 7 days) on the gateway's metrics, and
   that the file in the `gateway-cert` volume was written after the switch; the gateway's
   log says `reloaded the client certificate and pod CA bundle`, and requests succeed
   again. Confirm `loom_poolmgr_ca_expiry_seconds` shows the new CA's validity (P3
   returns to OK within its next 1-hour period). Once everything serves, delete the old
   CA's secret versions.

If a step fails, going back is the previous secret versions of `pod-ca` and
`poolmgr-client`, a deploy and a `rekey`; the same window applies in that direction,
because the pods again trust only the CA that signed them.

## With the mock backend (the M3 box)

The whole runbook runs on the box with the mock pods (`provider: local`,
`LOOM_POOLMGR_MOCK_BACKEND=true`): the local pods' proxies take their certificates the
same way (written as files instead of over SSH), `rekey` reaches them, and `status` reads
their certificates over mTLS from inside the manager's container. Run it once with a
load going (acceptance B20): requests may fail only in the window after the CA switch
described above (none before the switch, none once every pod is re-keyed and the gateway
has loaded its new client certificate). Record the window's length and the failed
requests.

## Notes

- `loom-poolmgr issue-ca` stays dev-only (it also writes the gateway's certificate with a
  30-day lifetime for compose); `new-ca` is the production command.
- The manager's own client certificate (secret `poolmgr-client`) is not rotated by the
  manager: `new-ca` makes it as long-lived as the CA. A rollover replaces it with the CA.
- A gateway client certificate or CA file that cannot be read, or that is not a valid PEM,
  never replaces the working one: the gateway logs an error and keeps its context. Fix the
  file and the next check (at most a minute) picks it up.
