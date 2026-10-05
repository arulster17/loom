# How to add a serving engine

Adding an engine is **code, not configuration**. vLLM and SGLang are wired into the
registry schema, the command-line renderer, the Prometheus name map and the host start
script. Switching an existing model between those two engines, or pinning a new version
of one, is configuration (a new `engine.version` and `engine.image` digest in
`config/models.yaml` or an experiment variant); this page is for a third engine.

## What the engine must provide

The Lab drives engines through an OpenAI-compatible HTTP server. Check these before
writing code:

| Requirement | Used by |
|---|---|
| A container image, pinned by digest, that serves on a given host/port | `engines.docker_run_argv`, `aws_scripts/start_engine.sh` |
| Loads Hugging Face weights from the HF cache at `/root/.cache/huggingface` with no network (`HF_HUB_OFFLINE=1`), at a given revision | weights are pre-downloaded; the engine never sees the HF token |
| `python3` with `huggingface_hub` inside the image | the cold-start weight download runs in the engine image |
| `GET /health` returns 200 when ready; `POST /v1/completions` answers | readiness and first-token checks in `start_engine.sh` |
| Streaming `POST /v1/chat/completions` and `/v1/completions` (SSE), with a final usage chunk for `stream_options: {include_usage: true}` | `client/openai_stream.py`; runs without usage are flagged untrusted |
| `ignore_eos` in the request body | synthetic profiles pin output length with it |
| `chat_template_kwargs` in the request body (if models use it) | registry `engine.chat_template_kwargs` |
| `GET /metrics` in Prometheus text format | `metrics/prometheus.py` |
| For the quality gate: `echo` + `logprobs` on `/v1/completions`, `response_format` with JSON schema, `tools` | `quality/divergence.py`, `quality/tasks/` |

## Code changes

1. **Registry schema** (`bench/src/loom_bench/registry.py`): add the name to
   `Engine.name: Literal["vllm", "sglang"]`. Also add it to
   `LocalProviderSpec.engine` in `experiment.py` so the `local` provider accepts it.
2. **Launch renderer** (`bench/src/loom_bench/engines.py`):
   - `ENTRYPOINTS[name]`: the container entrypoint (`("vllm", "serve")` for vLLM).
   - `_RESERVED_FLAGS[name]`: every flag rendered from registry fields, so
     `engine.args` cannot override them.
   - `_<name>_args(spec, extra)`: model repo and **revision** (and tokenizer revision),
     served model name = `spec.id`, tensor/pipeline parallel size, max context,
     `quantization_flag(spec)` or the engine's own mapping, KV-cache dtype (see
     `_SGLANG_KV_CACHE_DTYPE` for a name mapping), `--trust-remote-code` only via
     `_trust_remote_code(spec)`, metrics enabled, host `0.0.0.0`, port `ENGINE_PORT`.
   - `render_launch`: today it picks `_vllm_args` for `vllm` and `_sglang_args` for
     anything else. Make the dispatch explicit, or a new engine silently gets SGLang
     flags.
   - `docker_run_argv` assumes `--gpus N --ipc host`, the HF cache mount and offline
     mode; change it only if the engine needs something else.
3. **Engine metrics** (`bench/src/loom_bench/metrics/prometheus.py`): an
   `EngineMetricNames` with the engine's names for running and waiting requests,
   KV-cache usage, preemptions, prefix-cache queries/hits (or a hit-rate gauge) and the
   queue-time histogram, verified against the engine's source at the pinned version
   (write the version in the comment, as for vLLM and SGLang). Register it in
   `ENGINE_METRICS`; `summarize_scrapes` refuses unknown engines. Names it lacks are
   listed in `ServerMetrics.missing` instead of failing.
4. **Host scripts** (`bench/src/loom_bench/providers/aws_scripts/start_engine.sh`):
   adjust only if the engine differs from the requirements above (health route, first
   completion, metrics). Stage names (`image_pulled`, `weights_ready`,
   `engine_started`, `engine_healthy`, `first_token`) feed the cold-start records.
5. **Reports** (`bench/src/loom_bench/report/analyze.py`): if the engine names its
   tensor-parallel arg differently, add it to `TP_ARG_KEYS` so labels show `TPn`.
6. **Optional: its own benchmark tool.** To cross-check Loom's numbers against the
   engine's bundled load generator, add an `ExternalTool` wrapper in
   `bench/src/loom_bench/loadgen/` and register it in `LOAD_GENERATORS`
   ([load-generators.md](../load-generators.md#adding-a-tool-guidellm-genai-perf-)).
   Not needed for SLO and cost numbers, which use `native`.

## Configuration after the code

- A model entry or experiment variant with `engine: {name: <name>, version: "x.y.z",
  image: repo@sha256:<64 hex>}`. A variant that switches engine must give image and
  version; the other engine's `args` are dropped.
- Plan timings (`plan.AWS_TIMING`: image pull 300 s, engine init 240 s) are shared by
  all engines; adjust if the new one is much slower to start.

## Tests

- `bench/tests/aws/test_engines.py`: an exact-argv test like
  `test_vllm_qwen_exact_argv`, reserved-flag rejection, the quantization table, and
  `docker_run_argv` with the new entrypoint.
- `bench/tests/analysis/test_prometheus.py`: a scrape captured from the real engine,
  parsed into every `ServerMetrics` field.
- `bench/tests/config/test_registry.py`: `test_rejects_bad_model` uses
  `engine__name: tgi` as its unknown-engine case; pick another name if you add `tgi`.

```bash
uv run pytest -q bench/tests/aws bench/tests/analysis/test_prometheus.py bench/tests/config
```

## Prove it on a GPU

1. Start the engine yourself on a GPU machine and benchmark it with the `local`
   provider (`provider: {kind: local, base_url: http://127.0.0.1:8000/v1, metrics_url:
   http://127.0.0.1:8000/metrics, engine: <name>, served_model: <id>, tokenizer: hf}`;
   example experiment in [local-dev.md](../local-dev.md#benchmarking-a-running-endpoint-local-provider)).
   Check that `server.missing` in the run summaries is empty and that no result carries
   a missing-usage warning.
2. Run the model's quality suite against it (`bench quality run`) and gate it against
   the existing engine's results (`bench quality gate`).
3. Then an `aws_ec2` experiment with a variant on the new engine, planned and run per
   [runbook.md](../runbook.md).
