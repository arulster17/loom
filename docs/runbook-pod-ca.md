# Runbook: the pod CA (rollover, expiry, the gateway's certificate)

Spec: [phase1-m3-spec.md](phase1-m3-spec.md) §7 (alarm P3, and G3 for the gateway's
certificate). Contracts: [phase1-contracts.md](phase1-contracts.md) §2, §9.

## What there is

- **The pod CA** (`LOOM_POOLMGR_CA_CERT` / `_KEY` in the manager, secret `pod-ca`): signs
  every pod's server certificate (SAN = the pod id), the manager's client certificate and
  the gateway's client certificate. `loom-deploy secrets init` makes it with 2 years of
  validity. The manager exports `loom_poolmgr_ca_expiry_seconds` and refuses to start
  when the CA has less than `server_cert_renew_before_s` (2 days) left.
- **The trust bundle** (`LOOM_POD_CA_CERT` in the gateway and the manager's pods): may hold
  several CA certificates; the gateway trusts a pod whose certificate any of them signed.
- **The gateway's client certificate**: issued by the manager from the CA in use, 7 days,
  into `LOOM_POOLMGR_GATEWAY_CERT_DIR/gateway-client.pem` (key and certificate in one
  file), at start unless the file holds a current one and then every 24 h. The gateway
  checks that file and the CA bundle every `LOOM_GATEWAY_CERT_CHECK_S` (60 s) and uses a
  changed set for new upstream connections; streams already open finish on theirs.
  It exports `loom_gateway_client_cert_expiry_seconds`.

## Alarms

- **P3** `loom_poolmgr_ca_expiry_seconds` below 30 days. Do the rollover below now: it
  takes an afternoon, the refusal at 2 days takes serving down.
  (A manager that refuses to start publishes nothing, so the alarm treats missing data as
  breaching: a P3 in ALARM with the manager down means look at `docker compose logs
  poolmgr` first.)
- **G3** `loom_gateway_client_cert_expiry_seconds` below 2 days: the rotation is broken.
  Check, in this order: the manager is running (H1); its log has no `issuing the gateway's
  client certificate failed` (disk full, the directory not writable); the file in the
  shared volume (`gateway-cert`) is recent and mode 0400, owned by the gateway's uid
  (10001); the gateway's log has no `failed to load; keeping the old TLS context` error.
  Restarting the manager issues a new certificate at once if the file is missing, broken,
  from another CA or has under 6 days left.

## Rollover to a new CA (zero downtime)

The order matters: the gateway must trust the new CA before any pod presents a
certificate from it, and the manager must switch before the old CA is dropped. The box
runs the manager and the gateway in one compose project, so every step is a deploy
(`loom-deploy apply --release <sha>`, gateway first: P4).

1. **Make the new CA** on the operator's machine (nothing here needs the box, Postgres or
   Redis):

   ```bash
   uv run loom-poolmgr new-ca --out ./pod-ca-next
   ```

   It writes `ca.crt`, `ca.key` and the manager's client certificate from it
   (`loom-poolmgr.crt`, `loom-poolmgr.key`, valid as long as the CA), keys mode 0600, and
   refuses to overwrite an existing CA. Check `openssl x509 -in pod-ca-next/ca.crt -noout
   -dates`. Put the values into Secrets Manager as `pod-ca-next` (`crt`, `key`) and
   `poolmgr-client-next` (`crt`, `key`) with the `printf %s ... | aws secretsmanager
   put-secret-value --secret-string file:///dev/stdin` pattern, then delete the local
   directory (the key never stays on a laptop).
2. **Deploy the gateway with the bundle old + new**: the file `LOOM_POD_CA_CERT` names
   holds the old CA certificate followed by the new one. The gateway picks the new bundle
   up within a minute without a restart. Check: the gateway's log says `reloaded the
   client certificate and pod CA bundle`; requests still succeed.
3. **Switch the manager to the new CA**: swap the `pod-ca` and `poolmgr-client` secrets
   for the new values (the old ones stay in `pod-ca-next`'s history; keep them until step
   6), deploy (the gateway first, then the manager). On start the manager issues the
   gateway's client certificate from the new CA (the file was from the old CA, so it is
   replaced, whatever its age) and begins to issue pod certificates from the new CA.
   Then re-issue every pod's certificate now instead of waiting for each to near expiry:

   ```bash
   docker compose exec poolmgr loom-poolmgr rekey            # every ready pod
   docker compose exec poolmgr loom-poolmgr rekey --model qwen3-8b   # or one model
   ```

   The command queues the re-key; the manager applies it on its next tick (seconds). Pods
   keep their tokens and stay in routing. Pods that are not ready get a certificate from
   the new CA when they are created.
4. **Check that every pod's certificate chains to the new CA**:

   ```bash
   docker compose exec poolmgr loom-poolmgr status
   ```

   Under each ready pod a line shows `certificate: issued by <CA name>, expires ...
   (chains to the manager's CA)`. A pod whose line says `does NOT chain to the manager's
   CA` still has an old certificate: run `rekey` again, and if it persists look at the
   manager's log for `admin rekey of pod ... failed`. Do not go on until none says it.
5. **The gateway's client certificate** is from the new CA already (step 3). Check
   `loom_gateway_client_cert_expiry_seconds` (about 7 days) on the gateway's metrics, and
   that the file in the `gateway-cert` volume was written after the switch.
6. **Drop the old CA from the bundle and deploy**: `LOOM_POD_CA_CERT` holds only the new
   CA certificate. The gateway picks it up within a minute; the old CA's certificates are
   no longer trusted anywhere. Confirm `loom_poolmgr_ca_expiry_seconds` shows the new CA's
   validity (P3 returns to OK within its next 1-hour period) and delete the old CA's
   secret versions.

If a step fails, going back is the reverse of the one before: the previous secret
version and a deploy. Until step 6 the gateway trusts both CAs, so either direction
keeps serving.

## With the mock backend (the M3 box)

The whole runbook runs on the box with the mock pods (`provider: local`,
`LOOM_POOLMGR_MOCK_BACKEND=true`): the local pods' proxies take their certificates the
same way (written as files instead of over SSH), `rekey` reaches them, and `status` reads
their certificates over mTLS from inside the manager's container. Run it once with a
load going (acceptance B20: no failed request while the CA changes).

## Notes

- `loom-poolmgr issue-ca` stays dev-only (it also writes the gateway's certificate with a
  30-day lifetime for compose); `new-ca` is the production command.
- The manager's own client certificate (secret `poolmgr-client`) is not rotated by the
  manager: `new-ca` makes it as long-lived as the CA. A rollover replaces it with the CA.
- A gateway client certificate or CA file that cannot be read, or that is not a valid PEM,
  never replaces the working one: the gateway logs an error and keeps its context. Fix the
  file and the next check (at most a minute) picks it up.
