# Helpers shared by every Loom serving pod script (copied from the bench's generic ones).
# Never trace these scripts: they run as root in a container whose PID 1 environment
# holds the pod-scoped RunPod key, and the bootstrap's stdin carries the proxy secrets.

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
