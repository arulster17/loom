# requires: WARM FETCH_WEIGHTS CACHED_WEIGHTS WEIGHTS_DIR PORT SERVED_MODEL READY_TIMEOUT_S ENGINE_STALL_S LOG_DIR STAGE_FILE ENGINE_ENV ENGINE_CMD ENGINE_PIDFILE PROC1_ENVIRON SCAN_DIRS ENGINE_UID PROXY_DIR
# The bench's start_engine for a serving pod: download the weights at their pinned
# revisions, start the engine bound to loopback as the unprivileged ENGINE_UID (made by
# serve_bootstrap), wait until it serves one token, and prove that uid can read neither
# PID 1's environment nor the proxy's secrets in PROXY_DIR. Runs as root over SSH,
# detached (`run_detached`); it carries no secrets. The served checkpoints are ungated,
# so the download uses no Hugging Face token, and the engine runs offline against it.
# FETCH_WEIGHTS / CACHED_WEIGHTS: flat repo revision pairs, as in the bench.

DOWNLOAD_PY='import sys
from huggingface_hub import snapshot_download
a = sys.argv[1:]
assert len(a) % 2 == 0, "repo/revision pairs"
for repo, rev in zip(a[::2], a[1::2]):
    snapshot_download(repo, revision=rev, ignore_patterns=["original/*", "*.pth", "*.gguf"])'

# Environment variable names that never reach the engine.
SECRET_NAME_RE='TOKEN|SECRET|PASSWORD|CREDENTIAL|(^|_)KEY($|_)'

mkdir -p "$LOG_DIR" "$WEIGHTS_DIR"
# The engine's own writable home (compile and kernel caches), next to its pid file.
ENGINE_HOME="$(dirname "$ENGINE_PIDFILE")/engine-home"
install -d -m 0700 -o "$ENGINE_UID" -g "$ENGINE_UID" "$ENGINE_HOME"

# The value of NAME in PID 1's environment, on stdout (captured, never echoed).
proc1_value() {
  local kv
  while IFS= read -r -d '' kv; do
    if [ "${kv%%=*}" = "$1" ]; then
      printf '%s' "${kv#*=}"
      return 0
    fi
  done <"$PROC1_ENVIRON"
  return 0
}

# Run "$@" as the engine uid with the engine's environment (set below).
engine_exec() {
  setpriv --reuid="$ENGINE_UID" --regid="$ENGINE_UID" --clear-groups --no-new-privs \
    env -i "${engine_env[@]}" "$@"
}

# The image's own Python, found on PID 1's PATH (this SSH session has sshd's default
# PATH, which may miss a venv the image puts on PATH).
image_python="$(PATH="$(proc1_value PATH)" && command -v python3)" || true
[ -n "$image_python" ] || fail "python3 is not on the image's PATH"

gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
sysinfo gpus "$gpus"
gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c .)"
sysinfo gpu_count "$gpu_count"
sysinfo driver_version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
sysinfo cuda_version "$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n 1)"
sysinfo kernel "$(uname -r)"
dc="$(proc1_value RUNPOD_DC_ID)"
[ -z "$dc" ] || sysinfo data_center "$dc"

if [ "$WARM" = 0 ]; then
  grep -q '^sshd_ready ' "$STAGE_FILE" || fail "pod start did not finish: lease watchdog state unknown"
  sed 's/^/loom-stage /' "$STAGE_FILE"
fi
[ $((${#FETCH_WEIGHTS[@]} + ${#CACHED_WEIGHTS[@]})) -gt 0 ] || fail "the launch names no weights"
if [ "${#CACHED_WEIGHTS[@]}" -gt 0 ]; then
  HF_HOME="$WEIGHTS_DIR" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$image_python" -c "$DOWNLOAD_PY" "${CACHED_WEIGHTS[@]}" >>"$LOG_DIR/weights.log" 2>&1 \
    || fail_log "weights for ${CACHED_WEIGHTS[*]} are not cached" "$LOG_DIR/weights.log"
fi
if [ "${#FETCH_WEIGHTS[@]}" -gt 0 ]; then
  HF_HOME="$WEIGHTS_DIR" HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$image_python" -c "$DOWNLOAD_PY" "${FETCH_WEIGHTS[@]}" >>"$LOG_DIR/weights.log" 2>&1 \
    || fail_log "weight download failed" "$LOG_DIR/weights.log"
fi
# Readable by the engine uid, whatever umask the download ran with.
chmod -R a+rX "$WEIGHTS_DIR"
stage weights_ready

# PID 1's environment (CUDA, PATH, LD_LIBRARY_PATH, ...) minus secrets and Loom's own
# variables, then the engine's home, offline HF settings and the launch's (validated) env.
engine_env=()
while IFS= read -r -d '' kv; do
  name="${kv%%=*}"
  case "$name" in
    HOME | HF_TOKEN | HUGGING_FACE_HUB_TOKEN | PUBLIC_KEY | RUNPOD_* | LOOM_* | AWS_*) continue ;;
  esac
  if printf '%s' "$name" | grep -Eq "$SECRET_NAME_RE"; then
    continue
  fi
  engine_env+=("$kv")
done <"$PROC1_ENVIRON"
engine_env+=("HOME=$ENGINE_HOME" "HF_HOME=$WEIGHTS_DIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
engine_env+=(${ENGINE_ENV[@]+"${ENGINE_ENV[@]}"})

if [ -f "$ENGINE_PIDFILE" ] && kill -0 "$(cat "$ENGINE_PIDFILE")" 2>/dev/null; then
  fail "an engine is already running (pid $(cat "$ENGINE_PIDFILE"))"
fi
# Its own session and process group, so the whole tree can be signalled.
setsid setpriv --reuid="$ENGINE_UID" --regid="$ENGINE_UID" --clear-groups --no-new-privs \
  env -i "${engine_env[@]}" "${ENGINE_CMD[@]}" >>"$LOG_DIR/engine.log" 2>&1 </dev/null &
engine_pid=$!
echo "$engine_pid" >"$ENGINE_PIDFILE"
stage engine_started

engine_fail() {
  tail -n 100 "$LOG_DIR/engine.log" >&2 || true
  fail "$1"
}

# Not healthy by the deadline, or no new engine output for ENGINE_STALL_S, fails the
# start: a hang fails fast instead of billing until the ready timeout.
deadline=$(($(date +%s) + READY_TIMEOUT_S))
log_size=-1
last_output=$(date +%s)
until curl -fs -o /dev/null "http://127.0.0.1:$PORT/health"; do
  kill -0 "$engine_pid" 2>/dev/null || engine_fail "engine process exited"
  now=$(date +%s)
  [ "$now" -lt "$deadline" ] || engine_fail "engine not healthy after ${READY_TIMEOUT_S}s"
  size=$(wc -c <"$LOG_DIR/engine.log" 2>/dev/null | tr -d ' ')
  if [ "${size:-0}" != "$log_size" ]; then
    log_size="${size:-0}"
    last_output=$now
  elif [ $((now - last_output)) -ge "$ENGINE_STALL_S" ]; then
    engine_fail "engine stalled: no output for ${ENGINE_STALL_S}s before it was healthy"
  fi
  sleep 2
done
stage engine_healthy

curl -fsS -o /dev/null -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL\",\"prompt\":\"Hello\",\"max_tokens\":1}" \
  "http://127.0.0.1:$PORT/v1/completions" || engine_fail "first completion failed"
stage first_token

# Engine isolation, checked on every engine start:
#   1. the engine uid cannot read PID 1's environment;
#   2. it can neither list PROXY_DIR nor read any file in it (certificate key, token);
#   3. its environment carries no secret names (names only are printed);
#   4. no secret value is written anywhere it could read (weights excluded).
if engine_exec cat "$PROC1_ENVIRON" >/dev/null 2>&1; then
  fail "the engine uid can read $PROC1_ENVIRON"
fi
if engine_exec ls "$PROXY_DIR" >/dev/null 2>&1; then
  fail "the engine uid can list $PROXY_DIR"
fi
for f in "$PROXY_DIR"/*; do
  [ -e "$f" ] || continue
  if engine_exec cat "$f" >/dev/null 2>&1; then
    fail "the engine uid can read $f"
  fi
done
leaked="$(engine_exec env | cut -d= -f1 | grep -E "^(HF_TOKEN|HUGGING_FACE_|RUNPOD_|PUBLIC_KEY|AWS_)|$SECRET_NAME_RE" || true)"
[ -z "$leaked" ] || fail "the engine environment carries secret names: $(echo "$leaked" | paste -sd, -)"
scan=()
for d in "${SCAN_DIRS[@]}"; do
  [ -d "$d" ] && scan+=("$d")
done
rp_value="$(proc1_value RUNPOD_API_KEY)"
if [ "${#scan[@]}" -gt 0 ] && [ "${#rp_value}" -ge 8 ]; then
  # The pattern arrives through a pipe (process substitution), never a file or argv.
  if grep -rlqF --exclude-dir="$(basename "$WEIGHTS_DIR")" -f <(printf '%s\n' "$rp_value") \
    "${scan[@]}" 2>/dev/null; then
    rp_value=""
    fail "a secret value is stored under ${scan[*]}"
  fi
fi
rp_value=""
sysinfo engine_isolation ok
