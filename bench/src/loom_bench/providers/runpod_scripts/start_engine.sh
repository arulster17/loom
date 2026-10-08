# requires: WARM MODEL_REPO MODEL_REVISION WEIGHTS_DIR PORT SERVED_MODEL READY_TIMEOUT_S ENGINE_STALL_S LOG_DIR STAGE_FILE ENGINE_ENV ENGINE_CMD ENGINE_PIDFILE PROC1_ENVIRON JOB_UID JOB_HOME SCAN_DIRS
# Download weights at the pinned revision (cold only), start the engine as a
# process in this container (RunPod has no Docker in the pod) bound to loopback,
# wait until it serves one token, and prove the job user is isolated from secrets.
# Runs as root over SSH. The engine runs as root too (accepted for now), with
# PID 1's environment minus every secret, and offline against the downloaded cache.

DOWNLOAD_PY='import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=["original/*", "*.pth", "*.gguf"])'

# Environment variable names that never reach the engine.
SECRET_NAME_RE='TOKEN|SECRET|PASSWORD|CREDENTIAL|(^|_)KEY($|_)'

mkdir -p "$LOG_DIR" "$WEIGHTS_DIR"

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

# The image's own Python, found on PID 1's PATH. This SSH session has sshd's default
# PATH, which misses a venv the image puts on PATH (SGLang's is /opt/sglang/bin), and
# only the image's Python has huggingface_hub.
image_python="$(PATH="$(proc1_value PATH)" && command -v python3)" || true
[ -n "$image_python" ] || fail "python3 is not on the image's PATH"

# The host's facts come first, so a start that fails or hangs still reports them
# (the provider records them in engine_start_failed).
gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
sysinfo gpus "$gpus"
gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c .)"
sysinfo gpu_count "$gpu_count"
sysinfo driver_version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
sysinfo cuda_version "$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n 1)"
sysinfo kernel "$(uname -r)"
# GPU-to-GPU links from `nvidia-smi topo -m`, one row per GPU: GPU0:X,SYS;GPU1:SYS,X.
# SYS between GPUs means they sit on different CPU sockets (see the NCCL P2P note in
# docs/runbook-runpod.md).
topo="$(nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | awk -v n="$gpu_count" '
  $1 ~ /^GPU[0-9]+$/ {
    row = $1 ":"; self = 0
    for (i = 2; i <= n + 1; i++) { row = row (i > 2 ? "," : "") $i; if ($i == "X") self = 1 }
    if (self) out = out (out ? ";" : "") row  # the header row has no X diagonal
  }
  END { print out }')"
[ -z "$topo" ] || sysinfo gpu_topology "$topo"
# RunPod injects the datacenter; the create response does not always carry machine.dataCenterId.
dc="$(proc1_value RUNPOD_DC_ID)"
[ -z "$dc" ] || sysinfo data_center "$dc"

if [ "$WARM" = 0 ]; then
  grep -q '^sshd_ready ' "$STAGE_FILE" || fail "pod start did not finish: TTL watchdog state unknown"
  sed 's/^/loom-stage /' "$STAGE_FILE"
  # The token lives only in this shell variable and the download child's
  # environment; it is never exported, written to disk, logged, or given to the engine.
  tok="$(proc1_value HF_TOKEN)"
  case "$tok" in
    *'{{'*) fail "the RunPod secret for HF_TOKEN was not substituted" ;;
  esac
  if ! HF_TOKEN="$tok" HF_HOME="$WEIGHTS_DIR" HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$image_python" -c "$DOWNLOAD_PY" "$MODEL_REPO" "$MODEL_REVISION" >>"$LOG_DIR/weights.log" 2>&1; then
    tok=""
    fail_log "weight download failed" "$LOG_DIR/weights.log"
  fi
  tok=""
else
  HF_HOME="$WEIGHTS_DIR" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$image_python" -c "$DOWNLOAD_PY" "$MODEL_REPO" "$MODEL_REVISION" >>"$LOG_DIR/weights.log" 2>&1 \
    || fail_log "weights for $MODEL_REPO@$MODEL_REVISION are not cached" "$LOG_DIR/weights.log"
fi
stage weights_ready

# PID 1's environment (CUDA, PATH, LD_LIBRARY_PATH, ...) minus secrets and Loom's
# own variables, then offline HF settings and the launch's (validated) env.
engine_env=()
while IFS= read -r -d '' kv; do
  name="${kv%%=*}"
  case "$name" in
    HF_TOKEN | HUGGING_FACE_HUB_TOKEN | PUBLIC_KEY | RUNPOD_* | LOOM_* | AWS_*) continue ;;
  esac
  if printf '%s' "$name" | grep -Eq "$SECRET_NAME_RE"; then
    continue
  fi
  engine_env+=("$kv")
done <"$PROC1_ENVIRON"
engine_env+=("HF_HOME=$WEIGHTS_DIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
engine_env+=(${ENGINE_ENV[@]+"${ENGINE_ENV[@]}"})

if [ -f "$ENGINE_PIDFILE" ] && kill -0 "$(cat "$ENGINE_PIDFILE")" 2>/dev/null; then
  fail "an engine is already running (pid $(cat "$ENGINE_PIDFILE"))"
fi
# Its own session and process group, so stop_engine can signal the whole tree.
setsid env -i "${engine_env[@]}" "${ENGINE_CMD[@]}" >>"$LOG_DIR/engine.log" 2>&1 </dev/null &
engine_pid=$!
echo "$engine_pid" >"$ENGINE_PIDFILE"
stage engine_started

engine_fail() {
  tail -n 100 "$LOG_DIR/engine.log" >&2 || true
  fail "$1"
}

# Not healthy by the deadline, or no new engine output for ENGINE_STALL_S, fails the
# start: a hang (874110b6 sat silent after NCCL init) fails fast instead of billing
# until the ready timeout.
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

# Job isolation, checked on every engine start and again before every job:
#   1. the job user cannot read PID 1's environment;
#   2. the environment a job gets carries no secret names (names only are printed);
#   3. no secret value is written anywhere a job could read (weights excluded).
assert_job_isolated
leaked="$(job_exec env | cut -d= -f1 | grep -E "^(HF_|HUGGING_FACE_|RUNPOD_|PUBLIC_KEY|AWS_)|$SECRET_NAME_RE" || true)"
[ -z "$leaked" ] || fail "job environment carries secret names: $(echo "$leaked" | paste -sd, -)"
scan=()
for d in "${SCAN_DIRS[@]}"; do
  [ -d "$d" ] && scan+=("$d")
done
hf_value="$(proc1_value HF_TOKEN)"
rp_value="$(proc1_value RUNPOD_API_KEY)"
if [ "${#scan[@]}" -gt 0 ] && { [ "${#hf_value}" -ge 8 ] || [ "${#rp_value}" -ge 8 ]; }; then
  # Patterns arrive through a pipe (process substitution), never a file or argv.
  if grep -rlqF --exclude-dir="$(basename "$WEIGHTS_DIR")" \
    -f <(for v in "$hf_value" "$rp_value"; do [ "${#v}" -ge 8 ] && printf '%s\n' "$v"; done) \
    "${scan[@]}" 2>/dev/null; then
    hf_value="" rp_value=""
    fail "a secret value is stored under ${scan[*]}"
  fi
fi
hf_value="" rp_value=""
sysinfo job_isolation ok

