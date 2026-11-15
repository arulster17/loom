#!/usr/bin/env bash
# run.sh without Docker, for a machine that already IS the TensorRT-LLM container
# (a RunPod pod started from the same pinned image). Same steps, same three runs:
# plain tools, strict tools (stock parser), strict tools (parser with the space
# restored). Between the last two only the patch changes.
#
#   ./run_pod.sh                               # Qwen/Qwen3-8B BF16
#   MODEL=nvidia/Qwen3-8B-FP8 REVISION=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 ./run_pod.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
REVISION="${REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
PORT="${PORT:-8000}"
OUT="${OUT:-$HERE/out}"
SERVER_PID=""

mkdir -p "$OUT"
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -9 -f trtllm-serve 2>/dev/null || true
  SERVER_PID=""
  sleep 3
}
trap stop_server EXIT

echo "== versions"
python3 -c 'import tensorrt_llm, xgrammar, transformers; print("tensorrt_llm", tensorrt_llm.__version__, "| xgrammar", xgrammar.__version__, "| transformers", transformers.__version__)' 2>&1 | tail -1 | tee "$OUT/versions.txt"
echo "image: ${IMAGE_REF:-unknown (set IMAGE_REF)}" | tee -a "$OUT/versions.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$OUT/versions.txt"

echo "== weights ($MODEL @ $REVISION)"
SNAPSHOT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', revision='$REVISION'))" | tail -1)"
echo "$SNAPSHOT"

echo "== template vs forced prefix (no GPU work)"
python3 "$HERE/template_check.py" "$SNAPSHOT" 2>&1 | { grep -v -i warn || true; } | tee "$OUT/template_check.txt"

serve() {  # $1 = stock | patched
  stop_server
  trtllm-serve "$SNAPSHOT" --backend pytorch --tool_parser qwen3 \
    --max_seq_len 8192 --served_model_name "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --extra_llm_api_options "$HERE/options.yaml" >"$OUT/server-$1.log" 2>&1 &
  SERVER_PID=$!
  echo -n "waiting for trtllm-serve ($1)"
  for _ in $(seq 1 180); do
    if curl -sf "localhost:$PORT/health" >/dev/null; then echo " ready"; return; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo; tail -50 "$OUT/server-$1.log"; exit 1
    fi
    echo -n .; sleep 5
  done
  echo; tail -50 "$OUT/server-$1.log"; exit 1
}

run() {  # $1 = label, $2 = mode
  echo "== $1"
  rm -f "$OUT/$1.jsonl"
  python3 "$HERE/repro.py" --base-url "http://localhost:$PORT" --model "$MODEL" \
    --mode "$2" --show m-thermo-1 --out "$OUT/$1.jsonl" | tee "$OUT/$1.txt"
}

serve stock
run plain plain
run strict-stock strict

stop_server
python3 "$HERE/patch_parser.py" | tee "$OUT/patch.txt"
serve patched
run strict-patched strict
stop_server

echo "== summary ($MODEL)"
tail -qn1 "$OUT/plain.txt" "$OUT/strict-stock.txt" "$OUT/strict-patched.txt" | tee "$OUT/summary.txt"
