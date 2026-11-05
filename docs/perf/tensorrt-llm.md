# TensorRT-LLM in the Lab

Status (2026-11-04): **measured.** TensorRT-LLM 1.4.0rc2 (BF16 and FP8) and vLLM v0.30.0
(BF16) ran on AWS g6e.xlarge (1x L40S) under the Lab's SLO. Results are in "What we
measured" below; the report is [reports/8b-trtllm-v1](../../reports/8b-trtllm-v1/leaderboard.md)
with the roofline in [reports/8b-trtllm-v1/roofline](../../reports/8b-trtllm-v1/roofline/roofline.md).
Every figure traces to a run id: 46938d1f (vLLM BF16 and TensorRT-LLM BF16 load runs),
9647a23b (TensorRT-LLM BF16 evals), 08204560 (TensorRT-LLM FP8 load runs and evals).

Update (2026-11-05): the two open "why"s are answered by profiling and a padded-graph
variant ("Why: what the profiles and the padded variant showed", below; report
[reports/8b-trtllm-why-v1](../../reports/8b-trtllm-why-v1/README.md); runs 71e2c32c,
ad5b1408, smoke d35e631b).

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

**What we found, in plain words:**

1. **At engine defaults, TensorRT-LLM BF16 was slower than vLLM BF16 on this GPU.**
   - It held the SLO up to 1.355 req/s on chat, against vLLM's 2.25.
   - On 1k-in/1k-out it held 0.667 req/s, against vLLM's 1.0.
   - That makes it about 1.6–1.7x the cost per token.
   - The measured cause: each decode step costs more for every request in the batch. The
     cost of reading the weights is the same on both engines.
2. **TensorRT-LLM FP8 roughly matched vLLM BF16.** It held chat up to 2.49 req/s at 11%
   less cost per token, and tied on 1k-1k. Its steps are fast (about 17–19 ms) until the
   running batch grows past about 32 requests. Then they jump to about 48–50 ms. That
   jump, not the average speed, is what set its limit.
3. **Neither TensorRT-LLM config passes the quality gate,** for one reason: strict tool
   calling.
   - When asked for schema-constrained tool calls, TensorRT-LLM 1.4.0rc2 forces a slightly
     different JSON prefix from the one Qwen3 was trained on. The model then picks the
     wrong function on multi-tool requests.
   - Everything else matches vLLM BF16: math, plain tool calling, JSON schema, and the
     token-level logprobs.
   - So this is the engine's constrained decoding, not the weights. It is a finding about
     TensorRT-LLM 1.4.0rc2, not something we fixed.

A good one-liner: *"I added TensorRT-LLM as a third engine to my benchmark lab and
compared it with vLLM at a fixed latency target on realistic chat traffic, in BF16 and
FP8. At engine defaults on an L40S, TensorRT-LLM BF16 served about 40% less load at the
SLO. The roofline fit showed the extra cost was per sequence, not the weight read. Its
FP8 build matched vLLM BF16 but hit a step-time cliff above about 32 running requests.
The quality gate caught a real engine bug: TensorRT-LLM's strict tool-call grammar forces
a JSON prefix Qwen3 doesn't use, which drops strict tool-calling accuracy 11–14 points.
To run that gate at all, I added a small route inside the server that gets prompt
logprobs from its own Python API."*

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

## What we expected, and what happened

| Expectation (written before the runs) | What happened |
|---|---|
| Similar BF16 throughput: both engines read the same 16.4 GB of weights per step | Not borne out: TensorRT-LLM BF16 served 40% less chat load at the SLO. The weight read did cost the same; the per-sequence part did not (below). |
| Lower TTFT, possibly worse TPOT tails, from unchunked prefill | TTFT: no significant difference at equal load. TPOT: worse, and with more run-to-run spread. |
| No-eviction admission costs little here (pinned output lengths) | Consistent: neither engine queued (waiting about 0 at every passing point); TensorRT-LLM never preempted, vLLM preempted only in overload. |
| FP8 moves goodput most | Partly: FP8 steps were about 2x faster than BF16's, but a step-time cliff above about 32 running requests capped its goodput near vLLM BF16's. |

## What we measured (2026-11-04)

All three cells ran on AWS g6e.xlarge (1x L40S 48 GB, 4 vCPU, 32 GiB), on-demand at
$1.8829/h including 200 GB of storage. Each cell had its own host. The workloads, load
windows, search, 3 repetitions and SLO (TTFT p95 ≤ 1 s, TPOT p95 ≤ 50 ms, judged on the
upper end of each 95% CI) are the config sweep's. The engines ran at their defaults (see
"What TensorRT-LLM does differently").

### Goodput and cost at the SLO

Goodput is the highest tested load that met the SLO, with the load that failed after it.
Cost is per 1M tokens, blended at each workload's own input:output mix, with 95% CIs.
Sources: [reports/8b-trtllm-v1](../../reports/8b-trtllm-v1/leaderboard.md).

| Cell | Workload | Goodput | Out tok/s at goodput | TTFT p95 | TPOT p95 | $/1M blended | vs vLLM BF16 |
|---|---|---|---|---|---|---|---|
| vLLM BF16 (reference) | chat-sharegpt | 2.25 req/s (fails at 2.367) | 708 [608, 824] | 310 ms | 40.2 ms | $0.1859 [0.1650, 0.2092] | – |
| TensorRT-LLM BF16 | chat-sharegpt | 1.355 req/s (fails at 1.5) | 415 [340, 507] | 284 ms | 35.7 ms | $0.3118 [0.2504, 0.3882] | 1.68x the cost |
| TensorRT-LLM FP8 | chat-sharegpt | 2.49 req/s (fails at 2.619) | 799 [771, 827] | 164 ms | 21.3 ms | $0.1654 [0.1516, 0.1803] | 0.89x |
| vLLM BF16 (reference) | fixed-1k-1k | 1.0 req/s (fails at 1.107) | 1,068 [1,023, 1,114] | 255 ms | 44.7 ms | $0.2450 [0.2347, 0.2557] | – |
| TensorRT-LLM BF16 | fixed-1k-1k | 0.667 req/s (fails at 0.738) | 681 [509, 913] | 379 ms | 38.7 ms | $0.3838 [0.2865, 0.5143] | 1.57x |
| TensorRT-LLM FP8 | fixed-1k-1k | 1.0 req/s (fails at 1.107) | 1,068 [1,023, 1,114] | 125 ms | 20.1 ms | $0.2450 [0.2347, 0.2557] | tied (same search point) |

**How the rows were joined.** The three cells come from three experiments:
- vLLM BF16 and the TensorRT-LLM BF16 load runs from 46938d1f;
- TensorRT-LLM BF16's evals from 9647a23b (46938d1f lost them);
- TensorRT-LLM FP8 from 08204560.

`bench report -e 46938d1f -e 9647a23b -e 08204560` pairs results by config hash:
- `48f3d8d1…` for vLLM BF16, the stored baseline every gate used;
- `ae3b5705…` for TensorRT-LLM BF16, whose load runs and gate share that hash across
  46938d1f and 9647a23b;
- `55b35655…` for TensorRT-LLM FP8.

The report puts FP8 on its own board (model `nvidia/Qwen3-8B-FP8`). The table above lines
its rows up with vLLM BF16's by workload: same SLO, workloads, windows and search.

**Equal load** (the report's "Latency at equal load"):
- **Chat at 2.25 req/s:**
  - TensorRT-LLM BF16's TPOT p50 is 38.8 [37.0, 40.7] ms against vLLM's 34.8 [33.6, 36.0]:
    significantly higher.
  - Its TTFT p95 is 330 against 310 ms: no significant difference.
  - FP8 at the same load runs TPOT p50 17.8 ms and TTFT p95 162 ms.
- **1k-1k at 1.0 req/s:**
  - TensorRT-LLM BF16's TPOT p50 is 53.4 ms against vLLM's 42.7. The CI is too wide to
    call it significant.
  - FP8 runs at 19.1 ms.

### Why TensorRT-LLM BF16 is slower: the per-sequence cost

`bench roofline` fits each cell's typical decode step (ITL p50) as a fixed part plus a
per-sequence part times the batch. Source:
[roofline.md](../../reports/8b-trtllm-v1/roofline/roofline.md), "Fixed and per-sequence
step cost". The fixed part's floor is the weight read; the per-sequence part's floor is
that sequence's KV read.

| Cell | Workload | Fixed ms (floor 17.5) | Per sequence ms (floor) | Per-seq vs floor | R² |
|---|---|---|---|---|---|
| vLLM BF16 | chat | 22.6 | 0.317 (0.212) | 1.49x | 0.991 |
| TensorRT-LLM BF16 | chat | 22.3 | 0.401 (0.211) | 1.90x | 0.998 |
| vLLM BF16 | 1k-1k | 24.6 | 0.336 (0.261) | 1.29x | 1.000 |
| TensorRT-LLM BF16 | 1k-1k | 22.9 | 0.455 (0.261) | 1.74x | 0.997 |

- **The fixed part is the same or slightly lower on TensorRT-LLM.** Its weight-read path is
  not the problem.
- **Each extra running request costs TensorRT-LLM 27–35% more step time than vLLM.** At
  chat's batch of about 25 that is about 2 ms per step. Under the 50 ms TPOT target that
  is enough to move the knee.
- **At light load the two match.** At chat 1.0 req/s (batch about 9) TensorRT-LLM's TPOT
  p50 is 28.3 ms. vLLM BF16's, at the same config hash, is 27.1 ms (53e4c107, the
  spec-decode run's BF16 cell, another host and day). The gap
  opens as the batch grows, as a per-sequence cost would.
- **Queueing is not the cause.** Engine gauges show waiting at about 0 on both engines at
  every passing point, and the running batch at equal load is about the same (25 vs 23 at
  chat 2.25).
- **What is not established:** which part of the per-sequence work is slower. Candidates
  are attention over the KV cache on sm_89, per-request sampling and bookkeeping, or the
  missing CUDA graphs (next section). **Answered 2026-11-05 (71e2c32c): attention.**
  See "Why: what the profiles and the padded variant showed".

### FP8: fast steps, then a cliff above about 32 running requests

Per repetition, FP8's step time is bimodal (08204560):

| Running batch (mean over the window) | ITL p50 | TPOT p95 |
|---|---|---|
| about 11–20 (max ≤ 35) | 16–19 ms | 20–23 ms |
| about 34–50 | 47–50 ms | 47–53 ms |
| about 67–72 | 55 ms | 58 ms |

- **The flip happens within one load point.** At chat 2.619 req/s, repetitions 1 and 2 ran
  at 22–23 ms TPOT p95 and repetition 0 at 48.3 ms. Slower steps keep more requests in
  flight, which keeps the batch high: once a run crosses the line it tends to stay there.
- **That is what set FP8's goodput.** The SLO is judged on the upper end of the CI over
  3 repetitions, and one slow repetition widens the CI past 50 ms (2.619 req/s: TPOT p95
  CI [9.8, 86.5] ms). FP8's mean latency would allow far more load.
- **The likely cause, not proven:** TensorRT-LLM 1.4.0rc2 captures CUDA graphs only for
  decode-only batches of 1–31, 32, 64 and 128 by default, with padding off
  (`CudaGraphConfig` in `llmapi/llm_args.py`). Batches of 33–63 therefore run every step
  without a graph, launching kernels one by one from Python. FP8's short steps make that
  overhead visible. BF16's per-sequence growth may include the same effect at its larger
  batches.
- **Tested 2026-11-05 (ad5b1408): confirmed.** With `cuda_graph_config:
  {enable_padding: true, max_batch_size: 128}` the cliff is gone on the same arrivals
  that trigger it. See "Why: what the profiles and the padded variant showed".
- **Context only, not a paired comparison:** vLLM's own FP8 weights plus FP8 KV on an L40S
  held chat at 5.239 req/s in the config sweep (7a8237d0, RunPod, RedHat checkpoint). If
  the cliff is the graph-size default, it is what costs TensorRT-LLM FP8 most of its FP8
  advantage.

### The quality gate: blocked only by strict tool calling

Both TensorRT-LLM configs were gated against vLLM BF16 (46938d1f's stored evals and
divergence reference). The gate used three replicated passes of phase0-strict, full item
counts, and margins of gsm8k 1, ifeval 2, json_schema 3 and tool tasks 6 points.

| Task | TensorRT-LLM BF16 (9647a23b) | TensorRT-LLM FP8 (08204560) |
|---|---|---|
| gsm8k (1,319) | pass, +0.25 pts [−0.13, +0.66] | pass, −0.08 pts [−0.94, +0.78] |
| ifeval (541) | inconclusive, −0.55 pts [−2.16, +1.05] | inconclusive, −0.68 pts [−2.53, +1.17] |
| tool_calling (135) | pass, +0.00 pts [−2.22, +2.22] | pass, +0.49 pts [−1.73, +2.72] |
| **tool_calling_strict (135)** | **fail, −13.83 pts [−19.75, −8.15]** | **fail, −11.36 pts [−17.04, −6.17]** |
| json_schema (300) | pass, +0.67 pts [−0.67, +2.00] | pass, +0.00 pts [−1.89, +1.89] |
| divergence (48 prompts, 3,068 positions) | pass, KL 0.0006 nats, top-1 99.19% | pass, KL 0.0141 nats, top-1 95.60% |
| sanity (7,290 outputs) | pass | pass |

**In plain words.** The weights behave the same:
- TensorRT-LLM BF16's token-level probabilities match vLLM BF16's to within vLLM's own
  run-to-run noise (KL 0.0006 nats against a noise floor of ≤ 0.0007).
- FP8's are close too (KL 0.0141, under the 0.05 limit).
- Math, ordinary tool calling and JSON-schema output all pass.

What fails is one engine feature: strict tool calling, where the server constrains the
model's output to the tool's schema.

**Why it fails (stored samples, read-only):**
- **The forced prefix.** With strict tools, TensorRT-LLM 1.4.0rc2's `qwen3` parser builds
  an xgrammar structural tag whose forced prefix is `<tool_call>\n{"name":"` + name, with
  no space after `"name":` (`tensorrt_llm/serve/tool_parser/qwen3_tool_parser.py`,
  `structure_info`). Qwen3's chat template writes `{"name": "`, with a space, and so do
  vLLM v0.30's hermes structural tag and Qwen3's own training format.
- **Its effect.** Forced off its usual tokens at the point where it names the function, the
  model picks the wrong function when there are several to choose from:
  - TensorRT-LLM BF16 has 22 strict failures: 18 `wrong_function` and 1
    `missing_argument`, all on the 45 "multiple"-tool items, plus 3 `wrong_value` on
    simple ones. Simple items pass 87/90.
  - vLLM BF16's 3 strict failures are those same 3 `wrong_value` on simple items, so the
    entire drop is on multiple-tool items.
  - FP8 shows the same pattern: 16 `wrong_function`, all multiple.
  - Example: "set the thermostat to 19 °C" becomes `get_weather(city="house")`
    (`m-thermo-1`).
- **Plain tool calling is unaffected:** its 3 failures are the same 3 `wrong_value` items
  as the baseline's.
- **It reproduces.** TensorRT-LLM BF16 scored 0.837 strict in the divergence smoke
  a596326d (full items). The first eval pass of 46938d1f, kept in S3 when that experiment
  failed, read 0.844.

**What this does and does not say:**
- It is a TensorRT-LLM 1.4.0rc2 behavior with Qwen3, found by the gate. It is not a
  verdict on the FP8 checkpoint or the engine's numerics.
- The gate blocks shipping either config as it stands, as it should.
- Possible follow-ups, each needing the user's yes:
  - a patched parser variant with the space restored, gated again;
  - an upstream report to NVIDIA.
- Neither has been done.

### Weak spots, plainly

- **TensorRT-LLM BF16's rows are untrusted.** Run-to-run CV is above 10% at goodput:
  chat TTFT p95 13.9%; 1k-1k TTFT p95 13.1%, request rate and output tok/s 11.8%. Read its
  goodput as indicative; the direction (well below vLLM) holds at every equal-load point.
- **The goodputs are bracketed, not lower bounds.** Every cell has a load that failed above
  its goodput, so none is a "≥". But a bracket is as coarse as the search: 2.25 to 2.367
  for vLLM chat, 1.355 to 1.5 for TensorRT-LLM BF16.
- **One host per cell, different hosts and times on the same day.**
  - vLLM BF16 and TensorRT-LLM BF16 ran in 46938d1f on separate hosts in different
    regions (us-east-2 and us-east-1), hours apart.
  - FP8 ran in 08204560 that evening, in us-east-2.
  - TensorRT-LLM BF16's evals ran on a fourth host (9647a23b).
  - Host-to-host variation (a few percent in earlier runs) sits inside these comparisons.
    The effects reported here are 1.5–1.7x.
- **FP8 1k-1k ties vLLM BF16 by construction.** At a passing open-loop point throughput is
  the offered load, and both cells got the same seeded requests at the same search point.
  Its latencies there are far lower.
- **The cause of the BF16 per-sequence cost and of FP8's cliff is inferred from engine
  gauges and the roofline fit, not profiled.**
- **Engine defaults only.** No TensorRT-LLM tuning (chunked prefill, CUDA-graph padding,
  `max_num_tokens`) was tried, by design.

## Why: what the profiles and the padded variant showed (2026-11-05)

Three runs on AWS g6e.xlarge (1x L40S), $6.70 in all: smoke d35e631b ($1.33),
71e2c32c (TensorRT-LLM BF16 with torch-profiler windows, $2.19), ad5b1408 (FP8 stock vs
CUDA-graph padding, $3.18). vLLM's side is its own profile run 14d959e5 (same instance
type, same window shape), re-analysed with the same code. Every number below is in
[reports/8b-trtllm-why-v1/tables.md](../../reports/8b-trtllm-why-v1/tables.md).

### 1. Why TensorRT-LLM BF16 is slower than vLLM BF16: attention, not the GEMMs

**Verdict.** The extra time per decode step is almost all attention. TensorRT-LLM's
decode-attention kernel (XQA, `kernel_mha`) takes 14–19% longer per step than vLLM's
FlashAttention split-KV kernel. Its attention-side work as a whole is 17–46% higher, depending
on the batch. The matrix multiplies are the same kernels at the same speed in both engines.

Decode steps at matched batch sizes (ms per step; TensorRT-LLM / vLLM):

| Batch | Step | GEMM | Attention | Step gap | Of which attention |
|---|---|---|---|---|---|
| 1 | 22.34 / 22.19 | 20.85 / 20.85 | 1.09 / 0.83 | +0.16 | +0.26 |
| 16 (1k-1k) | 29.91 / 28.60 | 22.73 / 22.25 | 6.67 / 5.67 | +1.31 | +1.00 |
| 21–27 (chat, ~2.2 req/s) | 31.3–33.2 / 29.1–30.9 | 23.3–23.7 / 22.8–23.1 | 7.5–9.1 / 5.6–7.1 | +1.3 to +3.1 | +1.1 to +2.6 |
| 64 (1k-1k) | 50.77 / 43.76 | 24.55 / 24.00 | 25.58 / 18.93 | +7.01 | +6.65 |

- **The GEMM kernel is not the cause.** The trace shows TensorRT-LLM running the generic
  SM80 kernel `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2` (100% of its
  GEMM time at batch 16). vLLM runs that very kernel too: 93% of its GEMM time at batch
  16, and the same 32x32 variant leads both at chat loads.
  - For BF16 linear layers both engines go through PyTorch to cuBLAS. cuBLAS picks the
    kernel by matrix shape, not by engine.
  - GEMM time per step matches within 0.3–0.6 ms (2–3%) at every batch.
  - At batch 1 it is identical: 20.85 ms, reading 15.14 GB at 726 GB/s, 84% of the
    L40S's 864 GB/s peak.
  - An SM89-tuned BF16 kernel might be faster on both engines. That would not explain
    the gap between them.
- **The gap grows with the batch, through attention.**
  - Step gap: +0.16 ms at batch 1, +1.3 ms at 16, about +2 ms at chat's 21–27, +7.0 ms at 64.
  - Attention accounts for 76% of it at 16 and 95% at 64. GEMMs add 0.3–0.6 ms.
    TensorRT-LLM's normalization and activation kernels are 0.2–0.3 ms faster.
  - From 16 to 64 in flight each extra sequence costs 0.435 ms of step time on
    TensorRT-LLM and 0.316 ms on vLLM (1.38x). Attention alone is 0.394 vs 0.276 ms
    (1.43x).
  - The client-side fit from the load runs gave 0.455 vs 0.336 ms per sequence on 1k-1k
    (roofline). The trace accounts for that difference.
- **Why it moves the SLO knee:** under the 50 ms TPOT target, about 2 ms more per step at
  batch ~25 is enough to make TensorRT-LLM miss the SLO at a lower load (1.355 vs 2.25 req/s).
- **What we can't conclude:**
  - *Why* XQA is slower on this GPU. The torch trace times kernels, not their memory
    traffic or occupancy; Nsight Compute would say. Possible causes are its tiling or
    split heuristics for BF16 KV on sm_89, or the KV page size (32 tokens vs vLLM's 16).
    None is tested.
  - Whether another TensorRT-LLM attention backend (for example its FlashInfer backend)
    closes the gap. Not run.
  - Exact context lengths. The 1k-1k windows are closed loops whose requests started
    together, so the context at each window depends on timing. The chat windows (open
    loop, near-equal load) are the fairest pairing and show the same direction.
  - The batch-1 row pairs a TensorRT-LLM chat smoke window with vLLM's 1k-1k window.
    Contexts differ.
  - Per-sequence slopes fitted *inside* the chat windows (batches 20–27) are unreliable.
    They even point the other way, because the range is narrow and the context mix shifts
    with batch. The per-batch level gap is the result.
- **CUDA graphs are not the BF16 story at these loads.** The chat and 1k-1k steady
  batches (16, 21–27, 64) all have graphs. Batches without a graph appeared only while
  the 64-request batch refilled: one step each at 35, 42, 49 and 56. Those steps had 12–64
  ms of GPU idle, against about 0.1 ms with a graph. That is the same mechanism as the FP8
  cliff below.

### 2. The FP8 cliff is the CUDA-graph batch sizes: padding fixes it

**Verdict: yes.** The test used the same arrival traces that push stock FP8 into its slow
state. Padded FP8 stayed fast in every run.

- **Paired by construction.** Each load and repetition replays the same seeded arrivals in
  every cell; the measured request rates match to the digit.
  - Stock FP8 (55b35655) collapsed on exactly the same three of six traces in both its
    runs: chat 2.619 rep 0, and 3.375 reps 0 and 2. ITL p50 went to 49–51 ms or TPOT p95
    to 48–54 ms, with the running batch averaging 40–45 and peaking at 72–74.
  - So the cliff is deterministic given the arrivals, not host noise.
  - Padded FP8 (2d964304, `enable_padding: true, max_batch_size: 128`) ran those same
    six traces at ITL p50 16.8–18.4 ms and TPOT p95 22.5–24.8 ms, 0 of 6 slow.
- **Padded runs did go above 32 running requests** (peaks of 38–40). Those batches padded
  up to the 40-request graph and stayed fast. Stock runs that crossed 32 ran eagerly.
  Each slow step keeps more requests in flight, so the batch climbed to 72–74 and stayed
  slow.
- **The mechanism is shown, though not in an FP8 cliff trace.**
  - The BF16 trace shows batches without a graph leaving the GPU idle 12–64 ms per step,
    while the CPU launches kernels one by one.
  - FP8's GPU work per step is only about 17–20 ms, so that launch time decides its step:
    47–50 ms.
  - Both FP8 profile windows at 3.375 ran after their cell's measured runs, at batches
    15–30, where both configs have graphs. Their steps match to within 0.1 ms. The cliff
    state itself was not captured in a trace.
- **What it buys.** Padded FP8 held the SLO at every tested load. Its goodput is at least
  3.375 req/s on chat, against stock's 2.49–2.62.
  - Its cost is $0.1237 per 1M blended tokens, against $0.1859 for vLLM BF16, so at most
    0.67x vLLM's cost. That is an upper bound, because the knee was not bracketed.
  - It is a comparison point, not a ranked row:
    - The goodput search did not bracket the knee.
    - Its gate was GSM8K and divergence only. GSM8K was +0.28 points vs vLLM BF16, and
      divergence was KL 0.0141, the same as stock.
    - It would inherit stock FP8's tool_calling_strict failure, because that is the same
      engine and grammar.
  - The leaderboard keeps stock FP8 at "quality gate failed". As of this report, a later
    and narrower gate no longer clears an earlier failure (`report.analyze.effective_gate`).
- **What we can't conclude:**
  - Where the padded knee is.
  - Whether an explicit batch list without padding does as well.
  - The cliff's exact launch cost per step on FP8, since no trace of it was captured.

### 3. The batch-4 attention anomaly (FP8 KV cache, light load)

- **What happened.** In the smoke's padded FP8 windows, decode steps at batch 4 took
  79.5 ms. 67.4 ms of that was attention, in
  `mmha::masked_multihead_attention_kernel<bf16, fp8_e4m3>`, TensorRT-LLM's older
  masked multi-head attention (MMHA) kernel.
  - At batch 5 in the same window (padded to 8) the step took 14.0 ms, with 1.65 ms
    of attention in the XQA kernel.
  - BF16 at batch 4 used XQA: 1.63 ms.
  - So with an FP8 KV cache at small batches the engine falls back to MMHA, and MMHA is
    about 40x slower here.
- **It is not caused by padding.** Stock FP8's own smoke (e3694f45) ran at batches 1–3
  with ITL p50 of 38–79 ms in 11 of 12 runs, against about 24 ms for BF16. Padded FP8's
  smoke runs at batches 2–4 show the same (55.6 and 79.1 ms).
- **Why we think the fallback happens.** TensorRT-LLM's XQA runner (1.4.0rc2,
  `decoderXQARunner.cpp`, `mayHavePerfGain`) uses XQA only when KV heads x batch x
  multi-block count is large enough against the 142 SMs. Qwen3-8B has 8 KV heads, so a
  batch of 4 gives few blocks.
  - We did not trace the exact rule that runs in this path (rc2's `XqaDispatcher`), so
    the threshold is not confirmed.
- **Impact.** Light-load FP8 latency only. At the loads that set goodput (batches 15+)
  every step used XQA. Still, it is a large TensorRT-LLM FP8 latency problem at low
  concurrency, and worth reporting upstream.

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
  vLLM cell's config hash. **Settled on AWS:** the runs used the DLAMI's driver 595.91.07
  (CUDA 13.2), and the 13.4 userspace ran there through minor-version compatibility, BF16
  and FP8 alike (d549221f onward). RunPod's 13.0 hosts were never reached (no stock).
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
  It did not hit: no run needed the setting.

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
  PyTorch backend's FP8 targets "the latest Blackwell and Hopper GPUs". **Settled:**
  it loads and serves on the L40S (e3694f45, 08204560), with the step-time cliff described
  in "What we measured".
- **Why not RedHat's checkpoint:** its compressed-tensors format, which vLLM's FP8 cells
  used, does not load in TensorRT-LLM.

**What it adds:** TensorRT-LLM's FP8 path, gated against vLLM BF16 on the full gate,
divergence included. Unlike the config sweep's FP8 KV cells, this checkpoint ships
calibrated KV-cache scales. It ships only on a gate pass; the precision is in the row's
name. Measured (08204560): divergence passes (KL 0.0141 nats, top-1 95.60%) and the gate
fails only on strict tool calling, the same engine behavior as BF16's.

## How it ran: experiments, failures and spend

RunPod Secure had no 1x L40S stock throughout 2026-11-02 (every create refused for about
11 h; probes showed the request body was not the cause), so the runs moved to AWS
g6e.xlarge, on the same GPU. `qwen3-8b-trtllm-vs-vllm-aws.yaml` and its smokes are the
AWS twins of the RunPod specs above.

| Experiment | What | Spend | Outcome |
|---|---|---|---|
| d549221f | `aws-smoke-trtllm`, 3 hosts | $1.86 | vLLM BF16 and TensorRT-LLM BF16 ran every path (divergence measured through the prompt-logprobs route). The third host was refused with `VcpuLimitExceeded` 0.3 s after the second was terminated: AWS's quota trails `terminated`. Fixed: the quota is waited out for up to 10 min. |
| e3694f45 | `aws-smoke-trtllm-fp8`, the missing FP8 cell | $1.10 | FP8 (ModelOpt, FP8 KV) loads and serves on the L40S; all paths ran |
| 46938d1f | `qwen3-8b-trtllm-vs-vllm-aws`, the real run | $10.76 | Failed part-way after the vLLM BF16 cell (load and evals) and the TensorRT-LLM BF16 load runs completed, 27/27 each (details below) |
| a596326d | `aws-smoke-trtllm-divergence` | $1.51 | The TensorRT-LLM engine survived the full 48-prompt divergence, scored serially |
| 9647a23b | `qwen3-8b-trtllm-bf16-quality-aws`: TensorRT-LLM BF16 evals, 3 passes | $1.64 | Gate FAIL, on tool_calling_strict only |
| 08204560 | `qwen3-8b-trtllm-fp8-aws`: TensorRT-LLM FP8 load runs and evals | $5.52 | 27/27 runs. Gate FAIL, on tool_calling_strict only |

That totals $22.39 for the TensorRT-LLM track on AWS, as recorded in `bench_spend`.

**Why 46938d1f failed.** There were two causes, both fixed in d550c1e:
- **TensorRT-LLM BF16's evals were lost.**
  - Its first pass scored every task. Then the TensorRT-LLM engine died
    ("LLM is shutting down") under 16 concurrent prompt-logprobs requests.
  - Each such request holds a full-vocabulary fp32 row per prompt token, which the
    engine's startup memory profile does not count.
  - The scorer now sends them one at a time (`SCORE_CONCURRENCY = 1`), and divergence runs
    on the last pass, so losing the engine to it can no longer cost the task passes.
  - 9647a23b re-ran those evals on the same config hash.
- **The FP8 host was read as lost.** A new us-east-2 instance was described within its
  first second, got EC2's eventually consistent `InvalidInstanceID.NotFound`, and was
  terminated. `check_host` now allows `NOT_FOUND_GRACE_S` (300 s) after launch. 08204560
  ran FP8 on its own.

Two other infrastructure fixes came out of this track:
- RunInstances and teardown now survive EC2 network errors, with an idempotent
  `ClientToken` (014218c9, a profiling run, lost its experiment to one read timeout).
- The RunPod stock wait now creates only when a datacenter shows stock.
