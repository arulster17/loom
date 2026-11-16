# requires: PROXY_DIR PROXY_VENV PROXY_PORT UPSTREAM BUNDLE_SHA256 PROXY_UID ENGINE_UID
# Deliver the pod proxy's secrets and start (or reload) the proxy. Runs as root over SSH
# as `bash -s`, never detached: the secrets follow this script on the same stdin, so
# they are never in the pod env, on a command line, or in a file outside PROXY_DIR.
#
# Stdin layout (loom_poolmgr.bootstrap.bootstrap_stdin): this script, then a line
# `loom-payload-v1`, then base64 of a tar archive holding server.crt, server.key,
# ca.crt, token, allow (one CIDR per line) and proxy-bundle (the pinned loom-podproxy
# bundle). Everything runs from the last line, so bash has read the whole script before
# the payload is read.
#
# Idempotent, so the manager re-keys a pod (adoption, certificate renewal) by running it
# again with a new payload: the files are replaced, and a running proxy from the same
# bundle gets SIGHUP (it reloads them for new connections and keeps open ones). Otherwise
# the bundle (checked against BUNDLE_SHA256) is installed into PROXY_VENV and the proxy
# is started as PROXY_UID. Prints `loom-proxy started <pid>` or `loom-proxy reloaded
# <pid>`.

PAYLOAD_FILES=(server.crt server.key ca.crt token allow proxy-bundle)
PROXY_FILES=(server.crt server.key ca.crt token allow)

ensure_user() {
  local name="$1" uid="$2" existing
  if existing="$(id -u "$name" 2>/dev/null)"; then
    [ "$existing" = "$uid" ] || fail "user $name has uid $existing, not $uid"
  else
    useradd --uid "$uid" --user-group --home-dir /nonexistent --no-create-home \
      --shell /usr/sbin/nologin "$name" </dev/null || fail "cannot create user $name"
  fi
}

receive_payload() {
  local dir="$1" marker f
  IFS= read -r marker || fail "no payload on stdin"
  [ "$marker" = "loom-payload-v1" ] || fail "stdin does not carry a loom-payload-v1 payload"
  base64 -d | tar -x -f - -C "$dir" --no-same-owner --no-same-permissions \
    || fail "cannot unpack the payload"
  for f in "${PAYLOAD_FILES[@]}"; do
    [ -s "$dir/$f" ] || fail "the payload has no $f"
  done
}

# The pid of the proxy started from PROXY_VENV, if it is running.
proxy_pid() {
  local pid
  pid="$(cat "$PROXY_VENV.pid" 2>/dev/null)" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q loom-podproxy || return 1
  printf '%s' "$pid"
}

stop_proxy() {
  local pid i
  pid="$(proxy_pid)" || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for i in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done
  kill -KILL "$pid" 2>/dev/null || true
}

# Install the bundle unless PROXY_VENV already holds this one; sets `installed` to 1 when
# it installed it now.
installed=0
install_bundle() {
  local bundle="$1" wheels wheel
  [ "$(cat "$PROXY_VENV/.loom-bundle-sha256" 2>/dev/null)" != "$BUNDLE_SHA256" ] || return 0
  stop_proxy
  wheels="$(dirname "$bundle")/wheels"
  mkdir -p "$wheels"
  tar -x --force-local -f "$bundle" -C "$wheels" --no-same-owner \
    || fail "cannot unpack the proxy bundle"
  wheel="$(find "$wheels" -name 'loom_podproxy-*.whl' | head -n 1)"
  [ -n "$wheel" ] || fail "the proxy bundle holds no loom_podproxy wheel"
  rm -rf "$PROXY_VENV"
  mkdir -p "$(dirname "$PROXY_VENV")"
  python3 -m venv --without-pip "$PROXY_VENV" || fail "cannot create $PROXY_VENV"
  python3 -m pip --python "$PROXY_VENV/bin/python" install --quiet --no-index \
    --find-links "$(dirname "$wheel")" loom-podproxy </dev/null >&2 \
    || fail "cannot install the proxy bundle"
  chmod -R a+rX "$PROXY_VENV"
  # Written last: a half-done install is redone next time.
  printf '%s\n' "$BUNDLE_SHA256" >"$PROXY_VENV/.loom-bundle-sha256"
  installed=1
}

incoming=""
loom_bootstrap() {
  local f pid
  umask 077
  [[ "$BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail "BUNDLE_SHA256 is not a sha256"
  [[ "$PROXY_UID" =~ ^[0-9]+$ && "$ENGINE_UID" =~ ^[0-9]+$ ]] || fail "bad uid"
  [ "$PROXY_UID" != "$ENGINE_UID" ] || fail "PROXY_UID and ENGINE_UID must differ"

  # Root-only staging next to PROXY_DIR (not /tmp), removed on exit.
  mkdir -p "$(dirname "$PROXY_DIR")"
  incoming="$(mktemp -d "$(dirname "$PROXY_DIR")/.incoming.XXXXXX")"
  trap '[ -z "$incoming" ] || rm -rf -- "$incoming"' EXIT
  receive_payload "$incoming"
  printf '%s  %s\n' "$BUNDLE_SHA256" "$incoming/proxy-bundle" | sha256sum -c --status \
    || fail "the proxy bundle does not match BUNDLE_SHA256"

  ensure_user loomproxy "$PROXY_UID"
  ensure_user loomengine "$ENGINE_UID"

  install -d -m 0700 -o "$PROXY_UID" -g "$PROXY_UID" "$PROXY_DIR"
  for f in "${PROXY_FILES[@]}"; do
    install -m 0600 -o "$PROXY_UID" -g "$PROXY_UID" "$incoming/$f" "$PROXY_DIR/.$f.new"
    mv -f "$PROXY_DIR/.$f.new" "$PROXY_DIR/$f"
  done

  install_bundle "$incoming/proxy-bundle"
  if [ "$installed" = 0 ] && pid="$(proxy_pid)"; then
    kill -HUP "$pid" || fail "cannot signal the proxy ($pid)"
    echo "loom-proxy reloaded $pid"
    return 0
  fi

  stop_proxy
  # Its own session, an empty environment, and the proxy's own uid: it can read
  # PROXY_DIR, nothing in it can read PID 1's environment.
  setsid setpriv --reuid="$PROXY_UID" --regid="$PROXY_UID" --clear-groups --no-new-privs \
    env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME=/nonexistent \
    "$PROXY_VENV/bin/loom-podproxy" --host 0.0.0.0 --port "$PROXY_PORT" \
    --upstream "http://$UPSTREAM" --cert "$PROXY_DIR/server.crt" \
    --key "$PROXY_DIR/server.key" --client-ca "$PROXY_DIR/ca.crt" \
    --token-file "$PROXY_DIR/token" --allow-ip-file "$PROXY_DIR/allow" \
    >>"$PROXY_VENV.log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$PROXY_VENV.pid"
  sleep 2
  kill -0 "$pid" 2>/dev/null || fail_log "the proxy exited at start" "$PROXY_VENV.log"
  echo "loom-proxy started $pid"
}

loom_bootstrap; exit $?
