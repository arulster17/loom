# Load generators

A `LoadJob` names its generator in `loadgen`, a key in
`loom_bench.loadgen.base.LOAD_GENERATORS`. Every generator returns a `RunResult`
of `RequestRecord`s, so metrics, storage and reports don't depend on which one
produced the load.

## Native vs wrapped tools

| Key | What runs | Source |
|---|---|---|
| `native` | Loom's own async driver (`loadgen/native.py`) over the hand-parsed SSE client | this repo |
| `vllm_bench` | `vllm bench serve` | vLLM v0.30.0, `vllm/benchmarks/serve.py`, `vllm/benchmarks/datasets/datasets.py` |
| `sglang_bench` | `python -m sglang.benchmark.serving` | SGLang v0.5.21, `python/sglang/benchmark/serving.py`, `python/sglang/benchmark/datasets/` |

**Use `native`** for SLO and cost numbers. It is the only generator that
replays every arrival pattern exactly, honours `request_timeout_s` and the
open-loop drain deadline, and timestamps every request on one clock.

**Use a wrapped tool** to check that Loom's numbers agree with the tools
engine teams quote, or to reproduce a figure published with one of them. Run
the same job with `native` and with the wrapper, then compare. Both
wrappers can drive any OpenAI-compatible server, so vLLM's tool can benchmark
SGLang and SGLang's tool can benchmark vLLM.

## What each wrapper supports

Workloads (`prompts_source` says whose prompts were sent):

| Workload kind | `vllm_bench` | `sglang_bench` |
|---|---|---|
| `synthetic` | `random` dataset (tool prompts); needs `ignore_eos: true` | `random-ids` dataset (tool prompts) |
| `shared_prefix` | `prefix_repetition` (tool prompts); `range_ratio: 0` only | `generated-shared-prefix` (tool prompts); `range_ratio: 0` only |
| `chat_dataset` | `custom` JSONL (Loom prompts); single-turn conversations only | `openai` JSONL (Loom prompts), multi-turn included |
| `long_context_needle`, `long_generation` | `custom` (Loom prompts) | `openai` (Loom prompts) on the chat endpoint; completions endpoint rejected |
| `code_completion`, `trace` | `custom` (Loom prompts) | rejected (completions endpoint, no order-preserving dataset) |

Load:

| | `vllm_bench` | `sglang_bench` |
|---|---|---|
| Open loop | `poisson`, `gamma` (`--burstiness`), `constant` (`--burstiness inf`) | `poisson` (and `gamma` with burstiness 1) |
| Open-loop `max_inflight` | `--max-concurrency` | `--max-concurrency` |
| Closed loop | `--request-rate inf --max-concurrency N`; needs `num_requests` | same |
| `ramp`, `diurnal`, `onoff_burst`, `trace` arrivals | rejected | rejected |

Anything without a faithful mapping raises `UnsupportedByTool` before the tool
starts, rather than running a different experiment.

Details that differ from `native`:

- **Prompts.** With Loom prompts, the tool replays exactly the requests the
  native driver would measure: requests after the warmup ones, in send order.
  vLLM's `custom` rows are sent without a client-side chat template. With tool
  prompts, the text is the tool's own random tokens. Lengths match the profile,
  except that SGLang's `--random-range-ratio` means `[len*r, len]`, which Loom
  maps from its `[len*(1-r), len*(1+r)]` to within one token.
- **Arrivals.** The tool samples its own open-loop arrival times from the same
  process and rate, so the schedule differs from `plan.arrivals`
  (`meta["arrivals_source"] = "tool"`).
- **Warmup.** Open-loop arrivals before `warmup_s`, or closed-loop
  `warmup_requests`, become the tool's warmup count. The tools send their warmup
  requests together, without the arrival process, and all of them reuse the
  first prompt, so that prompt's prefix is cached before measuring starts.
- **Counts.** Shared-prefix runs are rounded down to whole prefix groups.
- **Timeouts.** Both tools use a fixed 6 h client timeout and have no
  per-request deadline.
- **Tokenizer.** The tools load the tokenizer from `TokenizerSpec.repo` at its
  latest revision; they have no revision flag. For a pinned revision, point the
  tool at a local snapshot path.
- **SGLang ignores EOS by default.** The wrapper passes `--disable-ignore-eos`
  and writes `ignore_eos` into each request whenever the profile doesn't ignore
  EOS.

## How results are normalised

`ExternalTool.parse` reads the tool's detailed per-request arrays (vLLM:
`--save-result --save-detailed`; SGLang: `--output-details`, last JSONL line)
and `external.result_from_arrays` rebuilds one `RequestRecord` per measured
request:

- Status: OK when the tool's `errors[i]` is empty (and, for vLLM, its latency
  is positive). Otherwise the record is ERROR, or TIMEOUT when the last line of
  the stored traceback names a timeout. That line becomes `error`.
- Times: `ttfts`, `itls` and `latencies`, in seconds. vLLM also stores
  `start_times` and `queue_times`, so t0 is the earliest arrival, `sent_at_s` is
  the actual send, and open-loop `scheduled_at_s` is the send time minus the wait
  for a concurrency slot. SGLang stores neither, so its requests sit at t=0 and
  end at their last content chunk (`meta["send_times"] = False`).
- Tokens: `input_lens` and `output_lens` become `prompt_tokens` and
  `completion_tokens` for successful requests. These come from the server's
  usage block when the tool gets one. SGLang on the completions endpoint
  reports the requested length (`meta["output_lens_source"] = "requested"`),
  because asking for a usage chunk there crashes its parser.
- Window: `[0, duration]`, the tool's own benchmark duration, so Loom's
  throughput uses the same denominator as the tool. The tool's aggregates are
  kept in `meta["tool_summary"]` for cross-checking.
- TTFT and ITL follow each tool's chunk rules. vLLM's chat client counts the
  first streamed chunk even when it carries no content. Native and SGLang
  count content chunks only.

## Running a wrapper

```python
from pathlib import Path

from loom_bench.loadgen.base import get_load_generator
from loom_bench.loadgen.external import ExternalLoadGenerator, WorkloadContext
from loom_bench.loadgen.vllm_bench import VllmBench

gen = get_load_generator(job.loadgen).bind(WorkloadContext.from_job(job))  # tool installed locally

# On a GPU host, inside the engine image; run files must be visible at the same path:
gen = ExternalLoadGenerator(
    VllmBench(),
    command_prefix=[
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        "/opt/loom/run:/opt/loom/run",
        "-e",
        "OPENAI_API_KEY",
        "--entrypoint",
        "",
        "vllm/vllm-openai:v0.30.0",
    ],
    workdir=Path("/opt/loom/run"),
).bind(WorkloadContext.from_job(job))
result = await gen.run(
    job.base_url,
    requests,
    plan,
    request_timeout_s=job.request_timeout_s,
    model=job.served_model,
    api_key=api_key,
)
```

`--entrypoint ""` clears the image's `vllm serve` entrypoint so the command runs
as given. The API key reaches the tool through `OPENAI_API_KEY` in its
environment, never on the command line. Cancelling `run()` sends SIGTERM to the
tool (which `docker run` forwards), then SIGKILL after a grace period.

## Adding a tool (GuideLLM, GenAI-Perf, ...)

The plug-in slot is `loom_bench.loadgen.external.ExternalTool`:

1. Write a class with `name`, `result_file`,
   `build_argv(run: ToolRun, result_path) -> list[str]` and
   `parse(run: ToolRun, result_path) -> RunResult`. `ToolRun` already holds the
   measured and warmup counts, concurrency, profile, arrival spec, seed and
   tokenizer. When the tool can replay prompts, write `run.requests` to
   `run.dataset_path` in the tool's dataset format. Raise `UnsupportedByTool`
   for anything the tool's flags cannot express.
2. In `parse`, map the tool's per-request output to the arrays
   `result_from_arrays` takes (`ttfts`, `itls`, `input_lens`, `output_lens`,
   `errors`, plus `start_times`, `latencies` and `queue_times` when available).
   If the tool exports only aggregates, it can't be used: Loom's metrics need
   per-request records.
3. Add a `generator()` factory returning `ExternalLoadGenerator(YourTool())`,
   and register it lazily in `LOAD_GENERATORS` next to `vllm_bench`.
4. Pin the tool version you mapped against, and test the exact argv for each
   workload and load mode, the unsupported combinations, and parsing of a
   hand-written result file. Also run a fake-tool script through
   `command_prefix` (see `bench/tests/loadgen_external/`).
