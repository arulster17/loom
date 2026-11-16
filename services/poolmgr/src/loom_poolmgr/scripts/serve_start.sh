# requires: LEASE_FILE LEASE_POLL_S STAGE_FILE LOOM_ROOT SSH_DIR SSHD_CONFIG GRAPHQL_URL
# A serving pod's start command (`bash -c`; RunPod's PID 1 runs it), based on the bench's
# pod_start. Anyone holding the account key can read it through the API, so it carries no
# secrets and no URLs beyond RunPod's own. Order matters:
#   1. the lease watchdog, before anything that can fail or hang;
#   2. sshd (key-only, root only, no forwarding), authorizing only LOOM_SSH_PUBKEY;
#   3. sleep forever: if this exits, RunPod restarts the container.
#
# The lease is max(LOOM_LEASE_EPOCH, the epoch in LEASE_FILE), re-read every
# LEASE_POLL_S; the manager moves it forward with serve_lease over SSH. There is no
# absolute TTL: a pod lives as long as its lease is renewed. LOOM_LEASE_EPOCH is
# absolute, so a container restart does not extend it. A missing or non-numeric
# LOOM_LEASE_EPOCH terminates at once (fail closed); an unreadable lease file is ignored.
# The watchdog terminates (never stops: a stopped pod keeps billing for its disk) with
# the pod-scoped key RunPod injects, retrying every 30 s. Every command is found through
# PATH.
set +euo pipefail

mkdir -p "$LOOM_ROOT" "$(dirname "$STAGE_FILE")" 2>/dev/null
chmod 755 "$LOOM_ROOT" 2>/dev/null

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
  local log="$LOOM_ROOT/terminate.log"
  { : >>"$log"; } 2>/dev/null || log=/dev/null
  while :; do
    echo "loom-terminate $1 $(date +%s)"
    python3 -c "$TERMINATE_PY" "$GRAPHQL_URL"
    sleep 30
  done >>"$log" 2>&1
}

# A decimal epoch from $1 (surrounding whitespace allowed) on stdout; status 1 if $1 is
# anything else.
epoch_of() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  [[ "$v" =~ ^[0-9]{1,18}$ ]] || return 1
  printf '%s' "$((10#$v))"
}

# The watchdog. Each poll re-reads the lease file before it reads the clock.
(
  if ! env_epoch="$(epoch_of "${LOOM_LEASE_EPOCH:-}")"; then
    loom_terminate bad_lease_epoch
  fi
  while :; do
    lease="$env_epoch"
    if file_epoch="$(epoch_of "$(cat "$LEASE_FILE" 2>/dev/null)")" \
      && [ "$file_epoch" -gt "$lease" ]; then
      lease="$file_epoch"
    fi
    now="$(date +%s)"
    if [[ "$now" =~ ^[0-9]+$ ]] && [ "$now" -ge "$lease" ]; then
      break
    fi
    sleep "$LEASE_POLL_S"
  done
  loom_terminate lease_expired
) &

# One public key, nothing else on the line: no authorized_keys options, no second key.
PUBKEY_RE='^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521)) [A-Za-z0-9+/]+={0,3}( [^[:cntrl:]]*)?$'

loom_setup() {
  local key="${LOOM_SSH_PUBKEY:-}" sshd_bin
  [[ "$key" =~ $PUBKEY_RE ]] || {
    echo "loom-error LOOM_SSH_PUBKEY is empty or not one public key"
    return 1
  }
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >"$LOOM_ROOT/apt.log" 2>&1 \
    && apt-get install -y -qq --no-install-recommends openssh-server curl ca-certificates \
      >>"$LOOM_ROOT/apt.log" 2>&1 || return 1
  mkdir -p /run/sshd "$SSH_DIR" || return 1
  chmod 700 "$SSH_DIR" || return 1
  printf '%s\n' "$key" >"$SSH_DIR/authorized_keys" || return 1
  chmod 600 "$SSH_DIR/authorized_keys" || return 1
  ssh-keygen -A >/dev/null 2>&1 || return 1
  cat >"$SSHD_CONFIG" <<EOF || return 1
Port 22
PermitRootLogin prohibit-password
AllowUsers root
AuthorizedKeysFile $SSH_DIR/authorized_keys
AuthorizedKeysCommand none
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
PermitUserEnvironment no
DisableForwarding yes
AllowAgentForwarding no
AllowTcpForwarding no
AllowStreamLocalForwarding no
X11Forwarding no
PermitTunnel no
EOF
  # An absolute path (sshd re-executes itself); PATH first, then where Debian puts it.
  sshd_bin="$(PATH="$PATH:/usr/sbin:/sbin" command -v sshd)" || return 1
  # sshd starts with an empty environment, so neither it nor any session inherits the
  # pod-scoped key in this one.
  env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin "$sshd_bin" -f "$SSHD_CONFIG" \
    -E "$LOOM_ROOT/sshd.log" || return 1
  stage sshd_ready >/dev/null
}

if ! loom_setup >>"$LOOM_ROOT/setup.log" 2>&1; then
  echo "loom-error pod setup failed; terminating" >>"$LOOM_ROOT/setup.log"
  loom_terminate setup_failed &
fi

while :; do
  sleep infinity
done
