# TensorRT-LLM in the Lab

Status (2026-11-01): the engine adapter is built and tested offline. A smoke run
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
  here (1.2.1) that path is the default. It still runs NVIDIA's own tuned kernels for attention and for matrix
  multiplies. Its default scheduler is conservative: it admits a request only when it
  can guarantee memory for the whole answer, so it never has to pause one.

The experiment runs both engines on the same model (Qwen3-8B), the same GPU (one NVIDIA
L40S), the same realistic chat traffic and the same latency targets. It asks three
questions:

1. Which engine serves more requests per second while still meeting the latency targets?
2. What does that make the cost per million tokens?
3. Does switching engine change the answers?

On this setup I expect the two engines to land close on throughput (the reasons are
below). The interesting part is what the engines' default policies do to latency under
load. The answer will be measured, not assumed.

A good one-liner: *"I added TensorRT-LLM as a third engine to my benchmark lab, pinned to
an exact release and image, and compared it with vLLM at a fixed latency target on
realistic chat traffic. The comparison reports cost per million tokens and includes a
quality gate, so a faster engine that changes answers can't win."*

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
- **Quality gate**: Loom's rule that a new config ships only if its eval scores are not
  worse than the BF16 reference's by more than each task's margin, with 95% confidence
  ([quality-gate.md](../quality-gate.md)). BF16 vLLM stays the reference row.

## What TensorRT-LLM does differently from vLLM

| | vLLM v0.30.0 | TensorRT-LLM 1.2.1 (`trtllm-serve`, PyTorch backend) |
|---|---|---|
| How the model runs | PyTorch, with torch.compile and CUDA graphs | PyTorch modules calling NVIDIA's own kernels (attention backend `TRTLLM`), CUDA graphs, and an overlap scheduler that prepares step N+1 on the CPU while the GPU runs step N |
| Ahead-of-time compile | none | none on this path. The older `tensorrt` backend builds a fused TensorRT engine per model, GPU and max shape. It is not used here (see below). |
| Admission policy | optimistic: admit, and preempt (pause) a request if the KV cache runs out | `GUARANTEED_NO_EVICT`: admit only when prompt + max_tokens fits, never preempt |
| Chunked prefill | always on; at most 2048 tokens per step on a GPU under 70 GB | off by default; at most 8192 tokens per step |
| CUDA graphs | padded to the nearest captured batch size | captured for decode-only steps of batch 1-32, 64 and 128; padding is off, so other batch sizes and steps that include a prefill run without a graph |
| KV-cache size | 90% of total GPU memory, minus weights and activations | 90% of the memory that is free after loading |
| Prefix caching | on | on (`enable_block_reuse`) |
| Quantization | FP8 on load from BF16, or pre-quantized checkpoints (compressed-tensors, ModelOpt, ...) | pre-quantized checkpoints only (ModelOpt FP8, or `fp8` with block scales); no compressed-tensors loader in 1.2.1 |

Sources for the TensorRT-LLM column are the v1.2.1 source files:
`tensorrt_llm/commands/serve.py` (defaults: `--backend pytorch`, `max_num_tokens` 8192,
`GUARANTEED_NO_EVICT`, chunked prefill off) and `tensorrt_llm/llmapi/llm_args.py`
(`CudaGraphConfig`, `disable_overlap_scheduler: false`, `attn_backend: TRTLLM`,
`enable_block_reuse: true`, `free_gpu_memory_fraction: 0.9`). The vLLM column comes from
the config sweep's notes ([benchmark-lab.md](../benchmark-lab.md)).

**Why the PyTorch backend and not an engine build.** The PyTorch backend is the default
for `trtllm-serve` in 1.2.1. It loads the Hugging Face checkpoint directly, so the Lab's
pinned weights, tokenizer and chat template are exactly what vLLM loads. The engine-build
flow adds a per-GPU compile step (time on a billed pod) and another artifact to pin. It is
also no longer where NVIDIA puts new features. The PyTorch path is what someone serving
TensorRT-LLM today would run.

## What we expect, and why

- **Similar throughput at the 50 ms TPOT target.** On one L40S an 8B BF16 decode step is
  limited by reading 16.4 GB of weights plus the KV cache. Both engines read the same
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
- **The gate will be blocked, for API reasons, not quality ones** (next section).

A tuned TensorRT-LLM config is the natural follow-up: chunked prefill on, CUDA-graph
padding on, perhaps `max_num_tokens` 2048. It is left out so the first comparison is
"engine defaults versus engine defaults", as vLLM versus SGLang was.

## What the comparison can and cannot check

| Check | vLLM | TensorRT-LLM 1.2.1 | Effect |
|---|---|---|---|
| Streaming chat and completions, final usage chunk | yes | yes. Every chunk also carries usage, because 1.2.1 defaults `continuous_usage_stats` to true; the client keeps the last value. | none; a little extra server JSON per token |
| `ignore_eos`, `chat_template_kwargs` (`enable_thinking: false`) | yes | yes | pinned lengths, non-thinking mode on both |
| Tool calling (`tool_calling`, 135 items) | `hermes` parser | `--tool_parser qwen3` (Hermes-style `<tool_call>`) | scored |
| Strict tools (`tool_calling_strict`) | yes | **rejected**: its request schema forbids unknown fields and has no `strict` (added in 1.4.0rc2) | left out: the run uses subset `phase0`, not `phase0-strict` |
| `response_format: json_schema` | the item's schema | **the OpenAI wrapper `{name, schema, strict}` is passed to xgrammar as if it were the schema** (fixed in 1.4.0rc2) | scored, but it measures what an OpenAI client gets from 1.2.1, not constrained decoding on the item's schema |
| Teacher-forced logprob divergence | yes (`echo` + `logprobs` on completions) | **no**: completions rejects `logprobs` ("logprobs is not supported"), and no version returns prompt logprobs | `divergence_failed`. The gate is INCONCLUSIVE and blocked, whatever the task scores. Scores are stored, so `bench quality gate` can re-decide later. |
| Engine metrics | running, waiting, KV usage, preemptions, prefix hits, queue time | queue time only (`/prometheus/metrics`, with `return_perf_metrics`). 1.4.0rc2 adds running, waiting, KV usage and hit rate. | missing ones are listed in `server.missing`, not guessed |
| Served model name | `qwen3-8b` | the snapshot directory's name (the revision commit); requests' `model` is not checked | cosmetic |

The pod simulator test (`bench/tests/runpod/test_runpod_trtllm.py`) runs both specs with
an engine that rejects completions logprobs. It checks that the eval job keeps its task
scores, that `divergence_failed` names the reason, and that the gate comes out
INCONCLUSIVE rather than missing.

## Version, image and platform (verified 2026-11-01)

- **Release:** TensorRT-LLM **v1.2.1**, the latest stable release (GitHub releases, marked
  "Latest"; published 2026-04-20; notes: "Fixed an issue that caused KV cache corruption
  (#12770)"). The newest pre-release is v1.4.0rc2 (2026-10-28). It fixes four of the gaps
  above (strict tools, the json_schema wrapper, completions logprobs, the metrics) but it
  is a release candidate. It would still not return prompt logprobs. Moving to it is one
  line in both specs (tag `1.4.0rc2`, index digest
  `sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`).
- **Qwen3 and Ada support:** `docs/source/models/supported-models.md@v1.2.1` lists
  `Qwen3ForCausalLM` with `Qwen/Qwen3-8B`. `docs/source/supported-hardware.md@v1.2.1`
  lists "NVIDIA Ada Lovelace: L20, L40/L40S".
- **Image:** `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`, multi-arch index digest
  `sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9` (linux/amd64
  `sha256:cb4d8af81c586a90235ae3739b6d4ddc5d8336f2174c8a1c6b573d2e13faf5d7`). It is
  20.9 GB compressed; vLLM 0.30.0's is 8.7 GB and SGLang 0.5.21's is 15.2 GB. The planner
  adds 180 s of pull and 120 s of init for it (`plan.ENGINE_START_EXTRA_S`). The smoke
  measures the real figures.
- **NGC access:** no key needed. An anonymous token from `nvcr.io/proxy_auth` read the
  tag list, the manifests and the config blob.
- **CUDA and driver:** the image's config gives `CUDA_VERSION=13.1.0.036`, NVIDIA PyTorch
  25.12 and `TRT_LLM_VERSION=1.2.1`. CUDA 13.x needs driver 580 or newer (CUDA
  minor-version compatibility). The pods keep the vLLM runs' `allowedCudaVersions:
  ["13.0"]`, whose hosts run 580-series drivers. Changing it would also move the vLLM
  cell's config hash. If a 13.0 host refuses the 13.1 userspace, the smoke will show it.
- **How it starts on a pod:** the pod runs the image directly (no Docker inside), as for
  vLLM. The engine command is `bash -c <launcher>`. The launcher writes the LLM-API
  options (JSON, which is valid YAML) to a temp file and runs `trtllm-serve serve
  <snapshot dir> --backend pytorch --tp_size 1 --pp_size 1 --max_seq_len 32768
  --tool_parser qwen3 --host 127.0.0.1 --port 8000 --extra_llm_api_options <file>`. The
  model argument is the Hugging Face cache snapshot directory, whose path names the pinned
  revision. The engine therefore runs offline (`HF_HUB_OFFLINE=1`) with nothing to
  resolve. The image's `python3` has `huggingface_hub` (through `transformers==4.57.3`),
  which the weight download uses. `/health` returns 200 when ready.
- **Known issue to watch:** NVIDIA/TensorRT-LLM #16330 (open). It reports FlashInfer's
  PDL path failing on an sm_89 L20 with 1.2.0 ("requires .target sm_90 or higher"). If
  the smoke hits it, add `provider.engine_env: {TRTLLM_ENABLE_PDL: "0"}` to both specs.
  That changes the TensorRT-LLM config hash, as it should.

## FP8: the optional second candidate

TensorRT-LLM's quantization matrix at v1.2.1 marks Ada as supported for per-tensor FP8
and FP8 KV cache, and not for block-scaled FP8. The same page also says the PyTorch
backend's FP8 targets "the latest Blackwell and Hopper GPUs", so only a run settles it.
The checkpoint would be `nvidia/Qwen3-8B-FP8` at `2cebc4c89e25abc17668c81b01dceaf3d8b914d5`:

- ModelOpt 0.35.0, `quant_algo: FP8`, `kv_cache_quant_algo: FP8`, `lm_head` left in BF16
- Apache-2.0, not gated, safetensors only, 9,451,782,787 bytes
- tokenizer.json and chat template identical to the BF16 pin
- `tokenizer_config.json` names `<|endoftext|>` as its EOS token, not `<|im_end|>`; both
  are in `generation_config.json`'s stop list

RedHat's compressed-tensors checkpoint, which vLLM's FP8 cells used, does not load in
1.2.1.

**What it would add:** TensorRT-LLM's FP8 path against the BF16 reference. Unlike the
config sweep's FP8 KV cells, this checkpoint ships calibrated KV-cache scales. It would
be its own registry row (precision in the name, for example `qwen3-8b-fp8-modelopt`) and
would need its own gate pass to ship. That gate would also be blocked by the divergence
gap above.

**What it would cost:** a third pod. On the second pod it would push that pod past 80%
of its TTL. `bench plan` with a stand-in cell (same windows, the checkpoint's size, FP8
KV) puts the run at **$11.89 estimated, $15.69 worst case**, so a $16 cap, against $8.07
without it. It also needs the registry row, the suite's `also_models`, and a cell in the
smoke. None of that is built.

## The quote

`bench plan` (prices.yaml, Secure 1x L40S at $1.1010/h with disk):

| Spec | Pods | Estimate | Worst case (every pod to TTL) | Cap (`max_spend`) |
|---|---|---|---|---|
| `runpod-smoke-trtllm` | 2 (vLLM, TRT-LLM), TTL 70 min | $1.57 | $2.57 | $2.75 |
| `qwen3-8b-trtllm-vs-vllm-runpod` | 2, TTL 285 min | $8.07 (vLLM pod 3.63 h, TRT-LLM pod 3.70 h) | $10.46 | $11 |
| with the FP8 candidate (not built) | 3 | $11.89 | $15.69 | ~$16 |

Planned figures run high: the config sweep's pods took about 70% of their planned time.
The results database holds $96.57 of recorded spend against the $149.75 overall cap, so
both specs fit.

## What the smoke cannot show

The smoke runs every path of the real run once. It uses the same images by digest, both
workloads with the dataset download, the geometric search with a descent, 3 repetitions,
phase0 with 3 replicated passes, divergence capture and its expected failure, and the
gate. What only the real run sees:

- full-length outputs and real overload on TensorRT-LLM (no preemption, but a queue);
- a search that bisects;
- full eval item counts;
- a 4-5 hour pod.

The first two ran on vLLM in 7a8237d0 and 55a6222b. On TensorRT-LLM they are new
behavior, not new code paths.
