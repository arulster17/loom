# Helpers shared by every Loom RunPod pod script. Never enable `set -x`: these
# scripts run as root in a container whose PID 1 environment holds HF_TOKEN and
# the pod-scoped RUNPOD_API_KEY.

stage() {
  local t
  t="$(date +%s.%N)"
  echo "$1 $t" >>"$STAGE_FILE"
  echo "loom-stage $1 $t"
}

sysinfo() {
  echo "loom-sys $1 $2"
}

fail() {
  echo "loom-error $1" >&2
  exit 1
}

fail_log() {
  tail -n 50 "$2" >&2 || true
  fail "$1"
}

# Run "$@" as the unprivileged job user with a clean, allowlisted environment: no
# HF_TOKEN, RUNPOD_*, PUBLIC_KEY or AWS_* ever reaches a job or eval process.
# --no-new-privs stops setuid escalation; the job uid cannot read PID 1's
# /proc/<pid>/environ (only its owner, root, may).
job_exec() {
  setpriv --reuid="$JOB_UID" --regid="$JOB_UID" --clear-groups --no-new-privs \
    env -i HOME="$JOB_HOME" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 "$@"
}

# Fail unless the job user is denied PID 1's environment (where the secrets live).
assert_job_isolated() {
  if job_exec cat "$PROC1_ENVIRON" >/dev/null 2>&1; then
    fail "job user can read $PROC1_ENVIRON"
  fi
}

# Presigned URLs go to curl in a config on stdin, never in argv: /proc/<pid>/cmdline
# is world-readable, and the job user must not see them.
fetch() {
  printf 'url = "%s"\n' "$1" | curl -fsS --retry 3 -K - -o "$2"
}

upload() {
  printf 'url = "%s"\nupload-file = "%s"\n' "$2" "$1" | curl -fsS --retry 3 -K - -o /dev/null
}
