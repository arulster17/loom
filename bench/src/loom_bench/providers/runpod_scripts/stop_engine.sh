# requires: ENGINE_PIDFILE STOP_TIMEOUT_S
# Stop the engine's whole process group (TERM, then KILL after STOP_TIMEOUT_S) and
# wait until no process holds the GPUs, so a warm restart gets all their memory.

if [ -f "$ENGINE_PIDFILE" ]; then
  pid="$(cat "$ENGINE_PIDFILE")"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  deadline=$(($(date +%s) + STOP_TIMEOUT_S))
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  rm -f "$ENGINE_PIDFILE"
fi

deadline=$(($(date +%s) + STOP_TIMEOUT_S))
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "GPU memory still held ${STOP_TIMEOUT_S}s after stop"
  sleep 2
done
echo "loom-stopped"
