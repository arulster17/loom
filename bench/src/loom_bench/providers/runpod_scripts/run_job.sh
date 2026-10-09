# requires: WORK_DIR ENV_ROOT PYTHON_DIR PYTHON_URL PYTHON_SHA256 JOB_URL WHEEL_URL WHEEL_NAME WHEEL_SHA256 REQS_URL REQS_SHA256 BENCH_CMD RESULT_URL GPU_CSV_URL SAMPLE_GPU MODEL_CACHE_DIR MODEL_REVISION DATA_URL DATA_SHA256 DATA_PATH PROC1_ENVIRON JOB_UID JOB_HOME PROC_ROOT
# Run one LoadJob (`bench job run`) or EvalJob (`bench quality job`) in the pod:
# measured latency has no WAN hop and the engine listens only on loopback. Runs as
# root over SSH; the job itself runs as JOB_UID through `job_exec` (no secrets in
# its environment, PID 1's environ unreadable, no new privileges). Inputs and
# outputs move through presigned URLs handled only here, as root, and never in a
# command line. The client Python is a pinned python-build-standalone build, not
# the engine image's; the virtualenv is built as the job user (no package code
# runs as root) and then made root-owned, so jobs cannot modify it.
# Dependencies come only from requirements.txt, exported from uv.lock with every
# package hash-pinned (pip --require-hashes); the wheel is installed --no-deps and
# `pip check` confirms the two agree. With MODEL_CACHE_DIR set, the job loads the
# tokenizer from the MODEL_REVISION snapshot the engine start downloaded, so a gated
# model needs no token here. With DATA_URL set (a workload's pinned public dataset),
# the first job in the pod downloads it to DATA_PATH, checks DATA_SHA256 and leaves it
# root-owned and world-readable; later jobs reuse it.

assert_job_isolated
echo "loom-sys job_isolation ok"

# Kill every process the job user still runs, so nothing outlives the job.
kill_job_processes() {
  local d
  for d in "$PROC_ROOT"/[0-9]*; do
    [ -d "$d" ] || continue
    if [ "$(ls -nd "$d" 2>/dev/null | awk '{print $3}')" = "$JOB_UID" ]; then
      kill -KILL "${d##*/}" 2>/dev/null || true
    fi
  done
}

rm -rf "$WORK_DIR"
install -d -o "$JOB_UID" -g "$JOB_UID" -m 700 "$WORK_DIR"
fetch "$JOB_URL" "$WORK_DIR/job.json" || fail "could not fetch job.json"
fetch "$WHEEL_URL" "$WORK_DIR/$WHEEL_NAME" || fail "could not fetch the bench wheel"
echo "$WHEEL_SHA256  $WORK_DIR/$WHEEL_NAME" | sha256sum --check --quiet || fail "wheel checksum mismatch"
fetch "$REQS_URL" "$WORK_DIR/requirements.txt" || fail "could not fetch requirements.txt"
echo "$REQS_SHA256  $WORK_DIR/requirements.txt" | sha256sum --check --quiet || fail "requirements checksum mismatch"
chown -R "$JOB_UID:$JOB_UID" "$WORK_DIR"

if [ -n "$MODEL_CACHE_DIR" ]; then
  [ -f "$MODEL_CACHE_DIR/snapshots/$MODEL_REVISION/tokenizer.json" ] \
    || fail "no tokenizer.json in snapshot $MODEL_REVISION under $MODEL_CACHE_DIR: was this model's engine started in this pod?"
fi

if [ -n "$DATA_URL" ] && [ ! -f "$DATA_PATH" ]; then
  data_dir="$(dirname "$DATA_PATH")"
  mkdir -p "$data_dir"
  chmod 755 "$(dirname "$data_dir")" "$data_dir"
  fetch "$DATA_URL" "$DATA_PATH.part" || { rm -f "$DATA_PATH.part"; fail "could not fetch the workload dataset"; }
  echo "$DATA_SHA256  $DATA_PATH.part" | sha256sum --check --quiet \
    || { rm -f "$DATA_PATH.part"; fail "workload dataset checksum mismatch"; }
  chmod 644 "$DATA_PATH.part"
  mv "$DATA_PATH.part" "$DATA_PATH"
fi

if [ ! -x "$PYTHON_DIR/bin/python3" ]; then
  tarball="$WORK_DIR/python.tar.gz"
  fetch "$PYTHON_URL" "$tarball" || fail "could not fetch the client Python"
  echo "$PYTHON_SHA256  $tarball" | sha256sum --check --quiet || fail "client Python checksum mismatch"
  rm -rf "$PYTHON_DIR"
  mkdir -p "$PYTHON_DIR"
  tar -xzf "$tarball" -C "$PYTHON_DIR" --strip-components=1 || fail "could not unpack the client Python"
  rm -f "$tarball"
  chown -R 0:0 "$PYTHON_DIR"
  chmod -R go-w "$PYTHON_DIR"
fi

# One virtualenv per wheel and requirements (so per extra), reused by every job in this pod.
ENV_DIR="$ENV_ROOT/$WHEEL_SHA256-$REQS_SHA256"
if [ ! -x "$ENV_DIR/bin/bench" ]; then
  rm -rf "$ENV_DIR"
  mkdir -p "$ENV_ROOT"
  install -d -o "$JOB_UID" -g "$JOB_UID" "$ENV_DIR"
  {
    job_exec "$PYTHON_DIR/bin/python3" -m venv "$ENV_DIR" \
      && job_exec "$ENV_DIR/bin/pip" install --quiet --no-cache-dir --require-hashes --no-deps \
        -r "$WORK_DIR/requirements.txt" \
      && job_exec "$ENV_DIR/bin/pip" install --quiet --no-cache-dir --no-deps "$WORK_DIR/$WHEEL_NAME" \
      && job_exec "$ENV_DIR/bin/pip" check
  } >>"$WORK_DIR/install.log" 2>&1 || {
    rm -rf "$ENV_DIR"
    fail_log "bench wheel install failed" "$WORK_DIR/install.log"
  }
  kill_job_processes
  chown -R 0:0 "$ENV_DIR"
  chmod -R go-w "$ENV_DIR"
fi

SMI_PID=""
trap '[ -z "$SMI_PID" ] || kill "$SMI_PID" 2>/dev/null || true' EXIT
if [ "$SAMPLE_GPU" = 1 ]; then
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits -lms 1000 >"$WORK_DIR/gpu.csv" 2>>"$WORK_DIR/gpu.err" &
  SMI_PID=$!
fi

rc=0
(cd "$WORK_DIR" && job_exec "$ENV_DIR/bin/bench" "${BENCH_CMD[@]}" \
  --in "$WORK_DIR/job.json" --out "$WORK_DIR/result.json") >>"$WORK_DIR/client.log" 2>&1 || rc=$?
kill_job_processes

if [ -n "$SMI_PID" ]; then
  kill "$SMI_PID" 2>/dev/null || true
  wait "$SMI_PID" 2>/dev/null || true
  SMI_PID=""
fi
[ "$rc" -eq 0 ] || fail_log "bench ${BENCH_CMD[*]} exited $rc" "$WORK_DIR/client.log"

upload "$WORK_DIR/result.json" "$RESULT_URL" || fail "result upload failed"
if [ "$SAMPLE_GPU" = 1 ]; then
  upload "$WORK_DIR/gpu.csv" "$GPU_CSV_URL" || fail "GPU sample upload failed"
fi
echo "loom-job-done"
