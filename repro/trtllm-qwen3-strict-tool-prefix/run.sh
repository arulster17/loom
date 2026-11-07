#!/usr/bin/env bash
# Repro: TensorRT-LLM's Qwen3 strict-tools structural tag forces `{"name":"` (no space),
# and Qwen3 then picks the wrong function more often.
#
# Needs: Docker with the NVIDIA runtime, one GPU that holds the model (an L40S 48 GB for
# Qwen3-8B BF16), internet for the image and weights. Prints three accuracies on the same
# 45 items: plain tools, strict tools (stock parser), strict tools (parser with both spaces
# restored). Everything else is identical between the last two runs.
#
#   ./run.sh                                   # Qwen/Qwen3-8B BF16
#   MODEL=nvidia/Qwen3-8B-FP8 REVISION=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a}"  # 1.4.0rc2
MODEL="${MODEL:-Qwen/Qwen3-8B}"
REVISION="${REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
OUT="${OUT:-$HERE/out}"
NAME=trtllm-qwen3-repro

mkdir -p "$OUT" "$HF_CACHE"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

in_image() {
  docker run --rm --gpus all -v "$HF_CACHE:/root/.cache/huggingface" -v "$HERE:/repro" \
    "$IMAGE" "$@"
}

echo "== versions"
in_image python3 -c 'import tensorrt_llm, xgrammar, transformers; print("tensorrt_llm", tensorrt_llm.__version__, "| xgrammar", xgrammar.__version__, "| transformers", transformers.__version__)' 2>&1 | tail -1 | tee "$OUT/versions.txt"
docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' | tee -a "$OUT/versions.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$OUT/versions.txt"

echo "== weights ($MODEL @ $REVISION)"
SNAPSHOT="$(in_image python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', revision='$REVISION'))" | tail -1)"
echo "$SNAPSHOT"

echo "== template vs forced prefix (no GPU work)"
in_image python3 /repro/template_check.py "$SNAPSHOT" 2>&1 | grep -v -i warn | tee "$OUT/template_check.txt"

serve() {  # $1 = stock | patched
  cleanup
  local pre=""
  [ "$1" = patched ] && pre="python3 /repro/patch_parser.py && "
  docker run -d --name "$NAME" --gpus all --ipc host -p "$PORT:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface" -v "$HERE:/repro" "$IMAGE" \
    bash -c "${pre}exec trtllm-serve '$SNAPSHOT' --backend pytorch --tool_parser qwen3 \
      --max_seq_len 8192 --served_model_name '$MODEL' --host 0.0.0.0 --port 8000 \
      --extra_llm_api_options /repro/options.yaml" >/dev/null
  echo -n "waiting for trtllm-serve ($1)"
  for _ in $(seq 1 180); do
    if curl -sf "localhost:$PORT/health" >/dev/null; then echo " ready"; return; fi
    if ! docker ps -q -f name="$NAME" | grep -q .; then
      echo; docker logs "$NAME" | tail -50; exit 1
    fi
    echo -n .; sleep 5
  done
  echo; docker logs "$NAME" | tail -50; exit 1
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
docker logs "$NAME" >"$OUT/server-stock.log" 2>&1

serve patched
docker logs "$NAME" 2>&1 | grep -m1 patched
run strict-patched strict
docker logs "$NAME" >"$OUT/server-patched.log" 2>&1

echo "== summary ($MODEL)"
tail -qn1 "$OUT/plain.txt" "$OUT/strict-stock.txt" "$OUT/strict-patched.txt" | tee "$OUT/summary.txt"
