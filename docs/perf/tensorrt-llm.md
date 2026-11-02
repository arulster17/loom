# TensorRT-LLM in the Lab

Status (2026-11-02): the engine adapter is built and tested offline, including the
workaround that gives TensorRT-LLM the full quality gate. A smoke run
(`runpod-smoke-trtllm`) and a comparison run (`qwen3-8b-trtllm-vs-vllm-runpod`) are written
and priced. **Neither has been approved or run.** Nothing below is a measurement yet: it
says what we expect to see and why.

## The short version (how to explain this in an interview)

TensorRT-LLM is NVIDIA's own serving engine. It is the main engine at inference platforms
such as Baseten. vLLM is the open-source engine the Lab already uses as its reference.
Both engines do the same job: they take requests, batch them onto the GPU, and stream
tokens back. They differ in who wrote the GPU code and in their default policies.

- **vLLM** is built on PyTorch. It optimizes by scheduling cleverly: it admits requests
  optimistically and pauses some if memory runs out.
- **TensorRT-LLM** started out as a compiler. You "built" a model ahead of time into a
  fixed, fused GPU program. It now also has a PyTorch path, and in the release pinned
  here (1.4.0rc2) that path is the only one `trtllm-serve` offers. It still runs NVIDIA's
  own tuned kernels for attention and for matrix multiplies. Its default scheduler is
  conservative: it admits a request only when it can guarantee memory for the whole
  answer, so it never has to pause one.

The experiment runs vLLM in BF16, TensorRT-LLM in BF16, and TensorRT-LLM in FP8 on the
same model (Qwen3-8B), the same GPU (one NVIDIA L40S), the same realistic chat traffic and
the same latency targets. It asks three questions:

1. Which engine and precision serve more requests per second while still meeting the
   latency targets?
2. What does that make the cost per million tokens?
3. Does switching engine (or precision) change the answers?

On this setup I expect the two engines to land close on throughput in BF16 (the reasons
are below). FP8 should give TensorRT-LLM a large gain, as it gave vLLM in the config sweep.
The interesting parts are what the engines' default policies do to latency under load,
and whether TensorRT-LLM passes the same quality gate vLLM's own variants face. The
answers will be measured, not assumed.

A good one-liner: *"I added TensorRT-LLM as a third engine to my benchmark lab, pinned to
an exact image, and compared it with vLLM at a fixed latency target on realistic chat
traffic, in BF16 and FP8. The comparison reports cost per million tokens and includes a
quality gate, so a faster engine that changes answers can't win. TensorRT-LLM's server
can't return the prompt logprobs that gate needs, so I added a small route inside the
server that gets them from its own Python API: same process, same GPU, same weights."*

## Terms, defined once

- **Prefill**: the engine reads the whole prompt in one pass to build its memory of it.
  This step is compute-bound: the GPU's arithmetic speed limits it.
- **Decode**: producing the answer one token at a time. Each step re-reads all the model
  weights from GPU memory, so decode is memory-bandwidth-bound: on an 8B model the
  weight read alone is about 19 ms per step on an L40S.
- **KV cache**: the engine's memory of every token seen so far, kept so it does not
  recompute the prompt at each step. It is what fills the GPU when many requests run at
  once. **Paged KV** stores it in fixed-size blocks, like virtual memory, so requests of
  different lengths share the cache without waste. Both engines page it.
- **Continuous batching** (NVIDIA calls it **in-flight batching**): requests join and leave
  the running batch at every step. A short answer does not wait for a long one. Both
  engines do this.
- **Chunked prefill**: splitting a long prompt's prefill across several steps, so the
  requests already decoding are not stalled by one big prompt.
- **CUDA graph**: a recording of a whole GPU step that is replayed with one launch,
  instead of many small kernel launches from Python. It saves CPU time per step, which
  matters most at small batches.
- **Kernel fusion**: merging several GPU operations into one kernel, so intermediate
  results stay on-chip instead of making a round trip through memory.
- **TTFT** (time to first token) and **TPOT** (time per output token after the first) are
  what a user feels. The Lab's SLO (service-level objective) is TTFT p95 under 1 s and
  TPOT p95 under 50 ms.
- **Goodput at SLO**: the highest request rate at which the engine still meets the SLO,
  with a confidence interval over 3 repetitions. Cost per 1M tokens is the GPU's hourly
  price divided by the tokens served per hour at that rate
  ([cost-model.md](../cost-model.md)).
- **FP8**: 8-bit floating point. Storing weights (and optionally the KV cache) in 8 bits
  instead of 16 halves the bytes each decode step reads.
- **Logprobs**: the log-probability the model gives each candidate next token.
  **Teacher-forced prompt logprobs** are those distributions at every position of a
  given text (the model is fed the text, not its own guesses), which lets two engines be
  compared on exactly the same input.
- **Quality gate**: Loom's rule that a new config ships only if its eval scores are not
  worse than the BF16 reference's by more than each task's margin, with 95% confidence,
  and its logprobs do not drift from the reference's beyond calibrated limits
  ([quality-gate.md](../quality-gate.md)). BF16 vLLM stays the reference row.

## What TensorRT-LLM does differently from vLLM

| | vLLM v0.30.0 | TensorRT-LLM 1.4.0rc2 (`trtllm-serve`, PyTorch backend) |
|---|---|---|
| How the model runs | PyTorch, with torch.compile and CUDA graphs | PyTorch modules calling NVIDIA's own kernels, CUDA graphs, and an overlap scheduler that prepares step N+1 on the CPU while the GPU runs step N |
| Ahead-of-time compile | none | none. The older TensorRT engine-build flow (a fused engine per model, GPU and max shape) is not offered by `trtllm-serve` 1.4.0rc2 (`--backend` accepts only `pytorch`). |
| Admission policy | optimistic: admit, and preempt (pause) a request if the KV cache runs out | `GUARANTEED_NO_EVICT`: admit only when prompt + max_tokens fits, never preempt |
| Chunked prefill | always on; at most 2048 tokens per step on a GPU under 70 GB | off by default; at most 8192 tokens per step |
| CUDA graphs | padded to the nearest captured batch size | captured for decode-only steps of batch 1-32, 64 and 128 by default; padding is off, so other batch sizes and steps that include a prefill run without a graph |
| KV-cache size | 90% of total GPU memory, minus weights and activations | 90% of the memory that is free after loading |
| Prefix caching | on | on (`enable_block_reuse`) |
| Quantization | FP8 on load from BF16, or pre-quantized checkpoints (compressed-tensors, ModelOpt, ...) | pre-quantized checkpoints only (ModelOpt FP8 here); no compressed-tensors loader |

Sources for the TensorRT-LLM column are the v1.4.0rc2 source files:
`tensorrt_llm/commands/serve.py` (`--backend` choices `["pytorch"]`, chunked prefill off)
and `tensorrt_llm/llmapi/llm_args.py` (`max_num_tokens` 8192, `max_batch_size` 2048,
scheduler default `GUARANTEED_NO_EVICT`, `CudaGraphConfig.enable_padding: false`,
`disable_overlap_scheduler: false`, `enable_block_reuse: true`,
`free_gpu_memory_fraction: 0.9`). The vLLM column comes from the config sweep's notes
([benchmark-lab.md](../benchmark-lab.md)).

**Why the PyTorch backend.** In 1.4.0rc2 it is the only backend `trtllm-serve` offers, and
in 1.2.1 it was already the default. It loads the Hugging Face checkpoint directly, so the
Lab's pinned weights, tokenizer and chat template are exactly what vLLM loads. There is no
per-GPU compile step to pay for on a billed pod and no extra artifact to pin.

## What we expect, and why

- **Similar BF16 throughput at the 50 ms TPOT target.** On one L40S an 8B BF16 decode step
  is limited by reading 16.4 GB of weights plus the KV cache. Both engines read the same
  bytes, and better kernels cannot read memory faster. TensorRT-LLM's edge is lower CPU
  overhead per step (the overlap scheduler, its attention kernels). That edge shows most
  at small batches and fades as the batch grows.
- **Lower TTFT, possibly worse TPOT tails, at TensorRT-LLM's defaults.** With chunked
  prefill off and up to 8192 tokens per step, a new prompt is prefilled in one step, which
  helps its TTFT. But that step stalls every request that is decoding. vLLM caps a step at
  2048 tokens and spreads long prompts out. Which effect decides goodput depends on how
  often a chat prompt (up to 4096 tokens) arrives while many requests are decoding. That
  is what the run measures.
- **No-eviction admission costs little on these workloads.** The Lab pins each request's
  output length (`max_tokens` equals the real length, with `ignore_eos`). The reservation
  TensorRT-LLM makes therefore equals what the request uses. With real clients sending a
  large `max_tokens`, it would admit fewer requests than vLLM. That is worth saying in an
  interview, and the doc should not hide it.
- **FP8 should move goodput most.** The FP8 cell halves the weight bytes per step
  (9.4 GB) and the KV bytes per token. vLLM's FP8-weights-plus-FP8-KV cell served chat at
  2.4x its BF16 rate in the config sweep (5.239 against 2.213 req/s). The gate decides
  whether it may ship.

A tuned TensorRT-LLM config is the natural follow-up: chunked prefill on, CUDA-graph
padding on, perhaps `max_num_tokens` 2048. It is left out so the first comparison is
"engine defaults versus engine defaults", as vLLM versus SGLang was.

## Choosing the version (checked 2026-11-02)

The user's rule: pin the earliest **stable** release that fixes three 1.2.1 problems (strict
tool calls, the json_schema wrapper, the engine metrics), or 1.4.0rc2 if none does.

- **No stable release exists after 1.2.1.**
  - On GitHub (`gh release list -R NVIDIA/TensorRT-LLM`), every release after v1.2.1
    (2026-04-20, "Latest") is a pre-release: v1.3.0rc8 through v1.3.0rc29, then v1.4.0rc0
    to v1.4.0rc2 (2026-10-28).
  - The NGC release repository's non-rc tags (anonymous `GET /v2/nvidia/tensorrt-llm/release/tags/list`)
    stop at `1.2.1` (`0.20.0, 0.21.0, 1.0.0, 1.1.0, 1.2.0, 1.2.1`).
- **1.2.1 fails all three** (source at the tag):
  - `FunctionDefinition` has no `strict` under `extra="forbid"`, so a strict tool is a 400.
  - `response_format.json_schema` is passed to xgrammar whole, OpenAI's `{name, schema,
    strict}` wrapper included.
  - `metrics/collector.py` exports five request histograms and no running, waiting or
    KV-usage gauge.
- **1.4.0rc2 fixes all three** (source at the tag):
  - `FunctionDefinition.strict` exists, and `_build_tool_strict_guided_decoding_params`
    constrains strict tools' arguments with xgrammar structural tags; the `qwen3` parser
    provides `structure_info`.
  - The json_schema conversion unwraps `"schema"` ("OpenAI API spec wraps the actual JSON
    schema under a 'schema' key").
  - The collector exports `trtllm_num_requests_running`, `trtllm_num_requests_waiting`,
    `trtllm_kv_cache_utilization` and `trtllm_kv_cache_hit_rate`. They are fed from
    iteration stats, which need `enable_iter_perf_stats: true`, and Loom sets it.
  - It also adds `--served_model_name` and defaults `continuous_usage_stats` to false.
- **So both specs pin 1.4.0rc2.** The smoke mirrors the run's image, as it must.
- **The quality subset is `phase0-strict`,** the config sweep's, so the vLLM cell pairs
  with 7a8237d0's evals and strict tool calling is scored on TensorRT-LLM too.

## How the divergence check works on TensorRT-LLM

The gate's logprob check (quality/divergence.py) needs teacher-forced prompt logprobs: it
sends prompt + continuation to `/v1/completions` with `echo: true, logprobs: 5,
max_tokens: 1` and reads, for every token of the text, the top-5 distribution that
predicted it. vLLM and SGLang answer that. trtllm-serve does not, in any version:

- 1.2.1 rejects `logprobs` on completions ("logprobs is not supported").
- 1.4.0rc2 returns only the generated token's logprobs.
- No version exposes prompt logprobs over HTTP.

TensorRT-LLM's Python LLM API does have them. `SamplingParams(prompt_logprobs=k)` works on
the PyTorch backend: rc2 computes them from the context logits in
`executor/base_worker.py` `_compute_pytorch_prompt_logprobs`, and
`tests/unittest/llmapi/test_llm_pytorch.py::test_llm_return_logprobs` covers it.

**The workaround** (`bench/src/loom_bench/trtllm_scorer.py`):

1. The pod starts the engine as `python3 -c <Loom's wrapper> serve <the usual
   trtllm-serve arguments>` instead of the `trtllm-serve` script. The wrapper
   (`engines.TRTLLM_SERVE_SOURCE`) is `detokenize.py` plus `trtllm_scorer.py`, because
   loom_bench is not installed in the image.
2. Before handing over to trtllm-serve's own `main()`, the wrapper adds one ASGI
   middleware to trtllm-serve's FastAPI app.
3. The middleware answers only `POST /v1/completions` requests that carry `echo` and
   `logprobs`. It calls the server's own `LLM` object (`generate_async` with
   `prompt_logprobs=5`, `max_tokens=1`, temperature 0) and returns the result in vLLM's
   format. Every other request (load, eval tasks, health) passes through to trtllm-serve
   untouched.
4. **So the scores come from the served engine itself:** same process, GPU, weights,
   precision, FP8 KV cache, kernels and scheduler. It is not a second model instance that
   might differ, and it needs no second copy of the weights in GPU memory.
5. **No KV-cache reuse.** Each scoring request gets a fresh `cache_salt`, so no cached
   block is reused and every position's logits are computed.
6. **The answer is written in vLLM v0.30.0's exact form.** Tokens and top-k keys are
   rendered from each token id's raw bytes with vLLM's rule (`detokenize.incremental_text`,
   which handles characters split across byte-level tokens), with `text_offset` as vLLM
   computes it.
7. **Alignment.** TensorRT-LLM's entry *j* is the distribution that predicted token *j+1*,
   so the wrapper shifts it by one to match vLLM's echo layout. That is the detail most
   likely to go silently wrong, so it has its own test.

This is the first-choice method the user asked for. It needs no "weaker" label: it
measures the same quantity, from the same engine, as the vLLM and SGLang paths do.

**Verified offline** (`bench/tests/quality/test_trtllm_scorer.py`,
`bench/tests/aws/test_engines_trtllm.py`, `bench/tests/runpod/test_runpod_trtllm.py`):

- The rendering equals the mock backend's independent vLLM renderer on byte-split tokens.
- A fake TensorRT-LLM server, built to rc2's `prompt_logprobs` contract and behind the
  middleware, scores identically to a vLLM-style reference built from the same toy model:
  KL 0, top-1 100%, on texts with " √2" and CJK characters split across tokens.
- An off-by-one alignment does not pass as identical.
- Requests other than echo+logprobs reach the stock route untouched; a bad `k` is a 400.
- The wrapper source compiles as one program. Executed as the pod runs it, with stand-in
  `tensorrt_llm` modules and loom_bench unavailable, it installs the route and hands over
  to `trtllm-serve`'s `main()`.
- A server class lacking the expected attributes fails loudly at startup, which is the
  failure a TensorRT-LLM upgrade would cause.
- bash reads the multi-line engine command back exactly.
- On the pod simulator, both specs give both TensorRT-LLM cells a gate with a measured
  divergence and no `divergence_failed`. With the wrapper removed, divergence fails
  visibly and the gates are blocked.

What only a pod can show is that rc2's real `OpenAIServer`, `LLM.generate_async` and
`prompt_logprobs` behave as their source says. The smoke checks it: each TensorRT-LLM
cell's gate must report "KL ... nats, top-1 ...%".

## What the comparison checks

| Check | vLLM | TensorRT-LLM 1.4.0rc2 |
|---|---|---|
| Streaming chat and completions, final usage chunk | yes | yes (usage on the last chunk only, as rc2 defaults `continuous_usage_stats` to false) |
| `ignore_eos`, `chat_template_kwargs` (`enable_thinking: false`) | yes | yes |
| Tool calling (`tool_calling`, 135 items) | `hermes` parser | `--tool_parser qwen3` (Hermes-style `<tool_call>`) |
| Strict tools (`tool_calling_strict`) | xgrammar structural tag | xgrammar structural tag (`<tool_call>` trigger) |
| `response_format: json_schema` | the item's schema | the item's schema (wrapper unwrapped), xgrammar backend set by Loom |
| Teacher-forced logprob divergence | `echo` + `logprobs` | Loom's prompt-logprobs route (above) |
| Engine metrics | running, waiting, KV usage, preemptions, prefix hits, queue time | running, waiting, KV usage, prefix hit rate, queue time at `/prometheus/metrics`. No preemptions: the scheduler never preempts. |
| Served model name | `qwen3-8b` | `qwen3-8b` (`--served_model_name`) |

## Version, image and platform (verified 2026-11-02)

- **Image:** `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2`, multi-arch index digest
  `sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a` (linux/amd64
  `sha256:d5f390f4552d1489350868f1d5f56be3bf4d9a2071b8fc0750fb9dc5091d4086`). It is 17.3 GB
  compressed; vLLM 0.30.0's is 8.7 GB and SGLang 0.5.21's is 15.2 GB. The planner adds
  120 s of pull and 120 s of init for it (`plan.ENGINE_START_EXTRA_S`). The smoke measures
  the real figures.
- **Qwen3 and Ada support at v1.4.0rc2:** `docs/source/models/supported-models.md` lists
  `Qwen3ForCausalLM` with `Qwen/Qwen3-8B`, and `docs/source/supported-hardware.md` lists
  "NVIDIA Ada Lovelace: L20, L40/L40S".
- **NGC access:** no key needed. An anonymous token from `nvcr.io/proxy_auth` read the
  tag list, the manifests and the config blob.
- **CUDA and driver:** the image's config gives `CUDA_VERSION=13.4.1.012`, NVIDIA PyTorch
  26.09, `TRT_LLM_VERSION=1.4.0rc2` and a CUDA forward-compatibility driver 615.71.
  CUDA 13.x runs on driver 580 or newer through minor-version compatibility. The pods keep
  the vLLM runs' `allowedCudaVersions: ["13.0"]`, because changing it would also move the
  vLLM cell's config hash. **This is the largest risk the smoke settles:** 13.4 is four
  minor versions past the hosts' 13.0. Code paths that JIT-compile PTX at run time need a
  driver at least as new as the toolkit that produced the PTX.
- **How it starts on a pod:** the pod runs the image directly (no Docker inside), as for
  vLLM. The engine command is `bash -c <launcher> trtllm-launch <wrapper source> <options
  JSON> <snapshot dir> --backend pytorch --tp_size 1 --pp_size 1 --max_seq_len 32768
  --tool_parser qwen3 --served_model_name <id> --host 127.0.0.1 --port 8000`. The launcher
  writes the options to a private temp file and runs `python3 -c <wrapper> serve ...
  --extra_llm_api_options <file>`.
  - Options Loom always sets: `guided_decoding_backend: xgrammar`, `return_perf_metrics:
    true`, `enable_iter_perf_stats: true`, plus `kv_cache_config.dtype: fp8` on the FP8
    row.
  - The model argument is the Hugging Face cache snapshot directory, whose path names the
    pinned revision, so the engine runs offline with nothing to resolve.
- **Known issue to watch:** NVIDIA/TensorRT-LLM #16330 (open). It reports FlashInfer's
  PDL path failing on an sm_89 L20 with 1.2.0 ("requires .target sm_90 or higher"). If
  the smoke hits it, add `provider.engine_env: {TRTLLM_ENABLE_PDL: "0"}` to both specs.

## FP8: the second candidate

The FP8 cell (`trtllm-fp8`) serves registry row `qwen3-8b-fp8-modelopt`: NVIDIA's
`nvidia/Qwen3-8B-FP8` at `2cebc4c89e25abc17668c81b01dceaf3d8b914d5`, the repository's only
commit.

- **Format:** ModelOpt 0.35.0, `quant_algo: FP8`, so FP8 weights and activations with
  static per-tensor scales. `lm_head` is left in BF16. The KV cache is FP8 with calibrated
  scales (`kv_cache_quant_algo: FP8`), so the row says `kv_cache_dtype: fp8`.
- **Checkpoint:** Apache-2.0, not gated, safetensors only, 9,451,782,787 bytes.
- **Tokenizer:** tokenizer.json and the chat template are identical to the BF16 pin's.
  `tokenizer_config.json` names `<|endoftext|>` as EOS, not `<|im_end|>`, but
  `generation_config.json` stops on both, as BF16's does.
- **Ada support is not settled by the docs.** At v1.4.0rc2,
  `docs/source/features/quantization.md` marks per-tensor FP8 and FP8 KV cache supported
  on Ada and for Qwen-3, and block-scaled FP8 unsupported. The same page also says the
  PyTorch backend's FP8 targets "the latest Blackwell and Hopper GPUs". The smoke settles
  it on an L40S.
- **Why not RedHat's checkpoint:** its compressed-tensors format, which vLLM's FP8 cells
  used, does not load in TensorRT-LLM.

**What it adds:** TensorRT-LLM's FP8 path, gated against vLLM BF16 on the full gate,
divergence included. Unlike the config sweep's FP8 KV cells, this checkpoint ships
calibrated KV-cache scales. It ships only on a gate pass; the precision is in the row's
name.

**What it costs:** a third pod (`host_group: b`). On the second pod it would push that
pod past 80% of its TTL. That adds $3.81 to the estimate.

## The quote

`bench plan` against a copy of the results DB (Secure 1x L40S at $1.1010/h with disk;
$96.57 recorded against the $249.75 overall cap; $50 per-experiment cap):

| Spec | Pods (TTL each) | Estimate | Worst case (every pod to TTL) | Cap (`max_spend`) |
|---|---|---|---|---|
| `runpod-smoke-trtllm` | 3 (70 min) | $2.35 | $3.85 | $4 |
| `qwen3-8b-trtllm-vs-vllm-runpod` | 3 (285 min): vLLM 3.65 h, TRT-LLM BF16 3.70 h, TRT-LLM FP8 3.46 h | $11.90 | $15.69 | $16 |

Planned figures run high: the config sweep's pods took about 70% of their planned time.

## What the smoke cannot show

The smoke runs every path of the real run once:

- the same three pods and image digests;
- both workloads, with the dataset download;
- the geometric search with a descent;
- 3 repetitions;
- phase0-strict with 3 replicated passes;
- divergence capture, and scoring through the prompt-logprobs route on both TensorRT-LLM
  cells (including the two hard prompts);
- both gates.

What only the real run sees:

- full-length outputs and real overload on TensorRT-LLM (no preemption, but a queue);
- a search that bisects;
- full eval item counts, and prompt-logprobs scoring of all 48 divergence prompts at
  concurrency 16;
- a 4-5 hour pod.

The first two ran on vLLM in 7a8237d0 and 55a6222b. On TensorRT-LLM they are new
behavior, not new code paths.
