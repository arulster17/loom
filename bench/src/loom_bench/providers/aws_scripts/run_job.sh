# requires: WORK_DIR ENV_ROOT CLIENT_IMAGE CLIENT_UID JOB_URL WHEEL_URL WHEEL_NAME WHEEL_SHA256 REQS_URL REQS_SHA256 BENCH_CMD RESULT_URL GPU_CSV_URL SAMPLE_GPU MODEL_CACHE_DIR MODEL_CACHE_MOUNT MODEL_FOLDER MODEL_REVISION DATA_URL DATA_SHA256 DATA_PATH DATA_DIR DATA_MOUNT
# Run one LoadJob (`bench job run`) or EvalJob (`bench quality job`) on the GPU
# host: measured latency has no WAN hop, and the engine is only reachable on the
# host's loopback. Inputs and outputs move through presigned URLs. The client
# container runs as CLIENT_UID (denied the metadata service by user-data, so no
# instance-role credentials), gets no secrets or AWS settings in its environment,
# mounts its virtualenv read-only and only WORK_DIR writable, and is removed after.
# With MODEL_CACHE_DIR set (the HF hub cache the engine start downloaded into) it also
# mounts that cache read-only at MODEL_CACHE_MOUNT: the job loads the tokenizer from
# MODEL_FOLDER's MODEL_REVISION snapshot, so a gated model needs no token here. The
# whole hub cache, not the model's folder: huggingface_hub may keep a file's content in
# a hub-level store (snapshots/<rev>/tokenizer.json -> ../../blobs/<etag> ->
# ../../blobs/<xx>/<sha256>), which a folder-only mount leaves dangling. Every snapshot
# file must resolve inside the mount, or the job fails here instead of in the container.
# With DATA_URL set (a workload's pinned public dataset), the first job on the host
# downloads it to DATA_PATH (under DATA_DIR), checks DATA_SHA256 and leaves it
# root-owned and world-readable; later jobs reuse it. DATA_DIR is mounted read-only at
# DATA_MOUNT, where the job's workload path points.
# Dependencies come only from requirements.txt, exported from uv.lock with every
# package hash-pinned (pip --require-hashes); the wheel is installed --no-deps and
# `pip check` confirms the two agree. CLIENT_IMAGE is pinned by digest.

rm -rf "$WORK_DIR"
install -d -o "$CLIENT_UID" -g "$CLIENT_UID" "$WORK_DIR"
curl -fsS --retry 3 -o "$WORK_DIR/job.json" "$JOB_URL" || fail "could not fetch job.json"
curl -fsS --retry 3 -o "$WORK_DIR/$WHEEL_NAME" "$WHEEL_URL" || fail "could not fetch the bench wheel"
echo "$WHEEL_SHA256  $WORK_DIR/$WHEEL_NAME" | sha256sum --check --quiet || fail "wheel checksum mismatch"
curl -fsS --retry 3 -o "$WORK_DIR/requirements.txt" "$REQS_URL" || fail "could not fetch requirements.txt"
echo "$REQS_SHA256  $WORK_DIR/requirements.txt" | sha256sum --check --quiet || fail "requirements checksum mismatch"
chown -R "$CLIENT_UID:$CLIENT_UID" "$WORK_DIR"

MODEL_MOUNT=""
if [ -n "$MODEL_CACHE_DIR" ]; then
  snapshot="$MODEL_CACHE_DIR/$MODEL_FOLDER/snapshots/$MODEL_REVISION"
  [ -f "$snapshot/tokenizer.json" ] \
    || fail "no tokenizer.json in snapshot $MODEL_REVISION under $MODEL_CACHE_DIR/$MODEL_FOLDER: was this model's engine started on this host?"
  hub="$(readlink -f "$MODEL_CACHE_DIR")"
  for f in "$snapshot"/*; do
    real="$(readlink -f "$f" || true)"
    [ -n "$real" ] && [ -e "$real" ] || fail "snapshot file $f is a dangling link"
    case "$real" in
      "$hub"/*) ;;
      *) fail "snapshot file $f resolves outside $MODEL_CACHE_DIR ($real): the client container would not see it" ;;
    esac
  done
  MODEL_MOUNT="$MODEL_CACHE_DIR:$MODEL_CACHE_MOUNT:ro"
fi

DATA_MOUNT_ARG=""
if [ -n "$DATA_URL" ]; then
  case "$DATA_PATH" in "$DATA_DIR"/*) ;; *) fail "dataset path $DATA_PATH is not under $DATA_DIR" ;; esac
  if [ ! -f "$DATA_PATH" ]; then
    data_dir="$(dirname "$DATA_PATH")"
    mkdir -p "$data_dir"
    chmod 755 "$DATA_DIR" "$data_dir"
    curl -fsS -L --proto =https --proto-redir =https --retry 3 -o "$DATA_PATH.part" "$DATA_URL" \
      || { rm -f "$DATA_PATH.part"; fail "could not fetch the workload dataset"; }
    echo "$DATA_SHA256  $DATA_PATH.part" | sha256sum --check --quiet \
      || { rm -f "$DATA_PATH.part"; fail "workload dataset checksum mismatch"; }
    chmod 644 "$DATA_PATH.part"
    mv "$DATA_PATH.part" "$DATA_PATH"
  fi
  DATA_MOUNT_ARG="$DATA_DIR:$DATA_MOUNT:ro"
fi

# One virtualenv per wheel and requirements (so per extra), reused by every job on this host.
ENV_DIR="$ENV_ROOT/$WHEEL_SHA256-$REQS_SHA256"
if [ ! -x "$ENV_DIR/bin/bench" ]; then
  rm -rf "$ENV_DIR"
  install -d -o "$CLIENT_UID" -g "$CLIENT_UID" "$ENV_DIR"
  INSTALL='python -m venv /env'
  INSTALL+=' && /env/bin/pip install --quiet --no-cache-dir --require-hashes --no-deps -r /work/requirements.txt'
  INSTALL+=' && /env/bin/pip install --quiet --no-cache-dir --no-deps "/work/$1" && /env/bin/pip check'
  docker run --rm --user "$CLIENT_UID:$CLIENT_UID" --env HOME=/tmp \
    --volume "$ENV_DIR:/env" --volume "$WORK_DIR:/work:ro" "$CLIENT_IMAGE" \
    sh -c "$INSTALL" sh "$WHEEL_NAME" \
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
  --volume "$ENV_DIR:/env:ro" --volume "$WORK_DIR:/work" ${MODEL_MOUNT:+--volume "$MODEL_MOUNT"} \
  ${DATA_MOUNT_ARG:+--volume "$DATA_MOUNT_ARG"} "$CLIENT_IMAGE" \
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
