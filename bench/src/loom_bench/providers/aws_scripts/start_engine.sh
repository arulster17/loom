# requires: WARM REGION HF_SECRET_ID IMAGE MODEL_REPO MODEL_REVISION WEIGHTS_DIR CONTAINER PORT SERVED_MODEL READY_TIMEOUT_S LOG_DIR STAGE_FILE ENGINE_ENV ENGINE_CMD
# Pull the engine image, download weights at the pinned revision (cold only),
# start the engine container and wait until it serves one token.

DOWNLOAD_PY='import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=["original/*", "*.pth", "*.gguf"])'

mkdir -p "$LOG_DIR" "$WEIGHTS_DIR"

if [ "$WARM" = 0 ]; then
  cloud-init status --wait >/dev/null 2>&1 || true
  grep -q '^user_data_done ' "$STAGE_FILE" || fail "user-data did not finish: TTL backstop not armed"
  sed 's/^/loom-stage /' "$STAGE_FILE"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker pull --quiet "$IMAGE" >>"$LOG_DIR/pull.log" 2>&1 || fail_log "image pull failed" "$LOG_DIR/pull.log"
stage image_pulled

if [ "$WARM" = 0 ]; then
  # The token lives only in this shell's environment and the short-lived download
  # container; it is never written to disk, logged, or given to the engine.
  HF_TOKEN="$(aws secretsmanager get-secret-value --region "$REGION" --secret-id "$HF_SECRET_ID" \
    --query SecretString --output text)" || fail "could not read the Hugging Face token secret"
  export HF_TOKEN
  if ! docker run --rm --env HF_TOKEN --env HF_HUB_DISABLE_PROGRESS_BARS=1 \
    --volume "$WEIGHTS_DIR:/root/.cache/huggingface" --entrypoint python3 "$IMAGE" \
    -c "$DOWNLOAD_PY" "$MODEL_REPO" "$MODEL_REVISION" >>"$LOG_DIR/weights.log" 2>&1; then
    unset HF_TOKEN
    fail_log "weight download failed" "$LOG_DIR/weights.log"
  fi
  unset HF_TOKEN
fi
stage weights_ready

for kv in "${ENGINE_ENV[@]}"; do
  export "$kv"
done
"${ENGINE_CMD[@]}" >>"$LOG_DIR/engine-run.log" 2>&1 || fail_log "docker run failed" "$LOG_DIR/engine-run.log"
stage engine_started

engine_fail() {
  docker logs --tail 100 "$CONTAINER" >&2 2>&1 || true
  fail "$1"
}

deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
until curl -fs -o /dev/null "http://127.0.0.1:$PORT/health"; do
  [ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ] \
    || engine_fail "engine container exited"
  [ "$(date +%s)" -lt "$deadline" ] || engine_fail "engine not healthy after ${READY_TIMEOUT_S}s"
  sleep 2
done
stage engine_healthy

curl -fsS -o /dev/null -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL\",\"prompt\":\"Hello\",\"max_tokens\":1}" \
  "http://127.0.0.1:$PORT/v1/completions" || engine_fail "first completion failed"
stage first_token

sysinfo gpus "$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
sysinfo driver_version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
sysinfo cuda_version "$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n 1)"
sysinfo image_digest "$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
sysinfo docker "$(docker version --format '{{.Server.Version}}')"
sysinfo kernel "$(uname -r)"
