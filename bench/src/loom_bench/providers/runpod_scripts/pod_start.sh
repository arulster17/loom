# requires: TTL_EPOCH STAGE_FILE LOOM_ROOT SSH_DIR SSHD_CONFIG CTL_ROOT JOB_UID JOB_USER JOB_HOME AUTHORIZE_ACCOUNT_KEY GRAPHQL_URL
# The pod's start command (`bash -c`; RunPod's PID 1 runs it). Anyone holding the
# account key can read it through the API, so it carries no secrets and no URLs
# beyond RunPod's own. Order matters:
#   1. the TTL watchdog, before anything that can fail or hang;
#   2. sshd (key-only, root only) and the unprivileged job user;
#   3. sleep forever: if this exits, RunPod restarts the container.
# The TTL is an absolute epoch, so a container restart does not extend it, and the
# watchdog terminates (never stops) the pod with the pod-scoped key RunPod injects.
set +euo pipefail

mkdir -p "$LOOM_ROOT" "$(dirname "$STAGE_FILE")"
chmod 755 "$LOOM_ROOT"
stage image_pulled >/dev/null

# Reads RUNPOD_API_KEY and RUNPOD_POD_ID from its own environment: neither reaches argv.
TERMINATE_PY='import json, os, sys, urllib.request
key, pod = os.environ.get("RUNPOD_API_KEY", ""), os.environ.get("RUNPOD_POD_ID", "")
if not key or not pod:
    sys.exit("terminate: RUNPOD_API_KEY or RUNPOD_POD_ID is missing")
body = {"query": "mutation T($podId: String!) { podTerminate(input: {podId: $podId}) }",
        "variables": {"podId": pod}}
req = urllib.request.Request(sys.argv[1], data=json.dumps(body).encode(), method="POST",
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
             "User-Agent": "loom-pod"})
with urllib.request.urlopen(req, timeout=30) as resp:
    out = resp.read().decode()
if "\"errors\"" in out:
    sys.exit("terminate: " + out[:300])
print("terminate requested")'

loom_terminate() {
  while :; do
    echo "loom-terminate $1 $(date +%s)"
    python3 -c "$TERMINATE_PY" "$GRAPHQL_URL"
    sleep 30
  done >>"$LOOM_ROOT/terminate.log" 2>&1
}

(
  remaining=$((TTL_EPOCH - $(date +%s)))
  [ "$remaining" -gt 0 ] && sleep "$remaining"
  loom_terminate ttl
) &

loom_setup() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >"$LOOM_ROOT/apt.log" 2>&1 \
    && apt-get install -y -qq --no-install-recommends openssh-server curl ca-certificates \
      >>"$LOOM_ROOT/apt.log" 2>&1 || return 1
  mkdir -p /run/sshd "$SSH_DIR" "$CTL_ROOT" || return 1
  chmod 700 "$SSH_DIR" "$CTL_ROOT" || return 1
  {
    printf '%s\n' "${LOOM_SSH_PUBKEY:-}"
    [ "$AUTHORIZE_ACCOUNT_KEY" = 1 ] && printf '%s\n' "${PUBLIC_KEY:-}"
  } | grep -E '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-[a-z0-9-]+) [A-Za-z0-9+/=]+' \
    >"$SSH_DIR/authorized_keys" || return 1
  chmod 600 "$SSH_DIR/authorized_keys" || return 1
  ssh-keygen -A >/dev/null 2>&1 || return 1
  cat >"$SSHD_CONFIG" <<EOF || return 1
Port 22
PermitRootLogin prohibit-password
AllowUsers root
AuthorizedKeysFile $SSH_DIR/authorized_keys
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
PermitUserEnvironment no
AllowAgentForwarding no
AllowTcpForwarding no
AllowStreamLocalForwarding no
X11Forwarding no
PermitTunnel no
EOF
  id -u "$JOB_USER" >/dev/null 2>&1 \
    || useradd --uid "$JOB_UID" --user-group --home-dir "$JOB_HOME" --no-create-home \
      --shell /usr/sbin/nologin "$JOB_USER" || return 1
  install -d -o "$JOB_UID" -g "$JOB_UID" -m 700 "$JOB_HOME" || return 1
  # sshd starts with an empty environment, so neither it nor any session inherits
  # the secrets in this one.
  env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /usr/sbin/sshd -f "$SSHD_CONFIG" \
    -E "$LOOM_ROOT/sshd.log" || return 1
  stage sshd_ready >/dev/null
}

if ! loom_setup; then
  echo "loom-error pod setup failed; terminating" >>"$LOOM_ROOT/setup.log"
  loom_terminate setup_failed &
fi

while :; do
  sleep infinity
done
