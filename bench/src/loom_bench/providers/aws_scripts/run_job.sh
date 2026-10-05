# requires: WORK_DIR ENV_ROOT CLIENT_IMAGE CLIENT_UID JOB_URL WHEEL_URL WHEEL_NAME WHEEL_SHA256 PIP_EXTRAS BENCH_CMD RESULT_URL GPU_CSV_URL SAMPLE_GPU MODEL_CACHE_DIR MODEL_CACHE_MOUNT MODEL_REVISION
# Run one LoadJob (`bench job run`) or EvalJob (`bench quality job`) on the GPU
# host: measured latency has no WAN hop, and the engine is only reachable on the
# host's loopback. Inputs and outputs move through presigned URLs. The client
# container runs as CLIENT_UID (denied the metadata service by user-data, so no
# instance-role credentials), gets no secrets or AWS settings in its environment,
# mounts its virtualenv read-only and only WORK_DIR writable, and is removed after.
# With MODEL_CACHE_DIR set it also mounts, read-only at MODEL_CACHE_MOUNT, the
# model's HF cache folder the engine start downloaded: the job loads the tokenizer
# from its MODEL_REVISION snapshot, so a gated model needs no token here.

rm -rf "$WORK_DIR"
install -d -o "$CLIENT_UID" -g "$CLIENT_UID" "$WORK_DIR"
curl -fsS --retry 3 -o "$WORK_DIR/job.json" "$JOB_URL" || fail "could not fetch job.json"
curl -fsS --retry 3 -o "$WORK_DIR/$WHEEL_NAME" "$WHEEL_URL" || fail "could not fetch the bench wheel"
echo "$WHEEL_SHA256  $WORK_DIR/$WHEEL_NAME" | sha256sum --check --quiet || fail "wheel checksum mismatch"
chown -R "$CLIENT_UID:$CLIENT_UID" "$WORK_DIR"

MODEL_MOUNT=""
if [ -n "$MODEL_CACHE_DIR" ]; then
  [ -f "$MODEL_CACHE_DIR/snapshots/$MODEL_REVISION/tokenizer.json" ] \
    || fail "no tokenizer.json in snapshot $MODEL_REVISION under $MODEL_CACHE_DIR: was this model's engine started on this host?"
  MODEL_MOUNT="$MODEL_CACHE_DIR:$MODEL_CACHE_MOUNT:ro"
fi

# One virtualenv per wheel and extras, reused by every job on this host.
ENV_DIR="$ENV_ROOT/$WHEEL_SHA256${PIP_EXTRAS:+-$PIP_EXTRAS}"
if [ ! -x "$ENV_DIR/bin/bench" ]; then
  rm -rf "$ENV_DIR"
  install -d -o "$CLIENT_UID" -g "$CLIENT_UID" "$ENV_DIR"
  docker run --rm --user "$CLIENT_UID:$CLIENT_UID" --env HOME=/tmp \
    --volume "$ENV_DIR:/env" --volume "$WORK_DIR:/work:ro" "$CLIENT_IMAGE" \
    sh -c 'python -m venv /env && /env/bin/pip install --quiet --no-cache-dir "/work/$1${2:+[$2]}"' \
    sh "$WHEEL_NAME" "$PIP_EXTRAS" \
    >>"$WORK_DIR/install.log" 2>&1 || fail_log "bench wheel install failed" "$WORK_DIR/install.log"
fi

SMI_PID=""
trap '[ -z "$SMI_PID" ] || kill "$SMI_PID" 2>/dev/null || true' EXIT
if [ "$SAMPLE_GPU" = 1 ]; then
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits -lms 1000 >"$WORK_DIR/gpu.csv" 2>>"$WORK_DIR/gpu.err" &
  SMI_PID=$!
fi

rc=0
docker run --rm --network host --user "$CLIENT_UID:$CLIENT_UID" --env HOME=/tmp \
  --volume "$ENV_DIR:/env:ro" --volume "$WORK_DIR:/work" ${MODEL_MOUNT:+--volume "$MODEL_MOUNT"} "$CLIENT_IMAGE" \
  /env/bin/bench "${BENCH_CMD[@]}" --in /work/job.json --out /work/result.json \
  >>"$WORK_DIR/client.log" 2>&1 || rc=$?

if [ -n "$SMI_PID" ]; then
  kill "$SMI_PID" 2>/dev/null || true
  wait "$SMI_PID" 2>/dev/null || true
  SMI_PID=""
fi
[ "$rc" -eq 0 ] || fail_log "bench ${BENCH_CMD[*]} exited $rc" "$WORK_DIR/client.log"

curl -fsS --retry 3 -T "$WORK_DIR/result.json" "$RESULT_URL" || fail "result upload failed"
if [ "$SAMPLE_GPU" = 1 ]; then
  curl -fsS --retry 3 -T "$WORK_DIR/gpu.csv" "$GPU_CSV_URL" || fail "GPU sample upload failed"
fi
echo "loom-job-done"
