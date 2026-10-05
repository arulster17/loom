# Helpers shared by every Loom host script. Never enable `set -x`: the engine
# start script holds a Hugging Face token in its environment while downloading.

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
