# requires: TRACE_DIR MAX_BYTES UPLOAD_URL
# Archive the trace files the engine's profiler wrote into TRACE_DIR during one profile
# window (loom_bench.profile), upload the archive through a presigned URL and delete the
# files, so the next window's collection holds only its own. The engine container writes
# them as root through a bind mount of TRACE_DIR at the same path
# (engines.docker_run_argv); this runs as root over SSM. Windows run one at a time on a
# host, and /stop_profile has returned (the engine flushed every worker's trace) before
# this runs. The client container (CLIENT_UID) never mounts TRACE_DIR.

[ -d "$TRACE_DIR" ] || fail "no trace directory $TRACE_DIR: was the engine started with --profiler-config torch_profiler_dir=$TRACE_DIR?"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
(cd "$TRACE_DIR" && find . -maxdepth 3 -type f -print0) >"$work/files"
[ -s "$work/files" ] || fail "the profiler wrote no files under $TRACE_DIR"
sysinfo profile_files "$(tr -cd '\0' <"$work/files" | wc -c | tr -d ' ')"
tar -C "$TRACE_DIR" --null -T "$work/files" -czf "$work/trace.tar.gz" || fail "could not archive the trace"
size="$(wc -c <"$work/trace.tar.gz" | tr -d " ")"
sysinfo profile_bytes "$size"
[ "$size" -le "$MAX_BYTES" ] || {
  (cd "$TRACE_DIR" && xargs -0 rm -f <"$work/files")
  fail "trace archive is $size bytes, over the $MAX_BYTES-byte cap: lower profile.max_iterations"
}
curl -fsS --retry 3 -T "$work/trace.tar.gz" "$UPLOAD_URL" || fail "trace upload failed"
(cd "$TRACE_DIR" && xargs -0 rm -f <"$work/files")
echo "loom-profile-done"
