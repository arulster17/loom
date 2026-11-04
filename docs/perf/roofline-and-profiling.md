# Why the numbers come out the way they do: roofline and profiling

This page explains Loom's measured latency and throughput from first principles: how
fast the hardware could possibly go (the *roofline*), how close the runs came, and where
the rest of the time goes. It starts with the plain-language version you can say in an
interview, then shows the math. Every measured figure names the run it came from; the
full tables are in [reports/roofline/roofline.md](../../reports/roofline/roofline.md),
regenerated with:

```bash
uv run bench roofline -e 7a8237d0 -e 9f0853d7 -e cf4d1614 -e 55102ddb \
  --db sqlite:///results/loom.db --out reports/roofline     # free: no GPU, DB read-only
```

The GPU-side measurement (where each millisecond of a step goes, from torch-profiler traces
of run 14d959e5) is in [reports/8b-profile-v1/](../../reports/8b-profile-v1/README.md), and
summarized below under [What the trace measured](#what-the-trace-measured-14d959e5).

## Terms (each defined once)

- **Decode step**: one pass of the model that produces the next token for *every*
  request currently generating. Requests generating together form the **batch**.
- **ITL (inter-token latency)**: the gap between two tokens of one request as the client
  sees it. A typical ITL is one decode step.
- **TPOT (time per output token)**: a request's average ITL. The SLO is TPOT p95 ≤ 50 ms.
- **TTFT (time to first token)**: from sending a request to its first token; mostly
  **prefill**, the pass that reads the whole prompt.
- **KV cache**: the keys and values the model saved for every earlier token of every
  request, so it does not recompute them. Each decode step reads all of it.
- **HBM bandwidth**: how fast the GPU can read its own memory (L40S 864 GB/s, H100 SXM
  3,350 GB/s; spec-sheet peaks).
- **Tensor throughput**: how many multiply-adds per second the GPU's matrix units do
  (L40S 362 TFLOP/s in BF16, 733 in FP8; H100 989 / 1,979).
- **BF16 / FP8**: number formats of 2 bytes and 1 byte. "fp8-kv8" means FP8 weights and
  an FP8 KV cache.
- **TP (tensor parallelism)**: one model split across several GPUs; after each layer
  they **all-reduce** (sum and share) their partial results over the link between them.
- **Roofline / floor**: the least time an operation can take on given hardware: bytes to
  move ÷ bandwidth, or FLOPs ÷ tensor throughput, whichever is larger.
- **Achieved %**: floor ÷ measured time. 100% would mean the hardware never waited.
- **Goodput**: the highest request rate at which the SLO still holds.

## The interview version

**"Decode is a memory-reading problem, not a math problem."** To produce one token for
everyone in the batch, the GPU has to read every weight of the model once, plus every
cached key and value of every request. For Qwen3-8B in BF16 that is 15.1 GB of weights.
At the L40S's 864 GB/s that alone takes 17.5 ms, so no decode step can be faster than
17.5 ms, whether the batch holds 1 request or 60. The math for 60 requests takes about
2 ms on the same GPU. That is why batching is nearly free up to a point, and why
"bytes per step" is the number to cut.

**"We get about 75% of that floor on one L40S in BF16."** At chat goodput (7a8237d0,
2.21 req/s) a typical step held 27 requests averaging 1,305 tokens of context: 23.6 ms of
floor (17.5 weights + 6.1 KV cache) against 31.1 ms measured, 76%. Fitting the step
time across load levels splits it into about 22 ms that every step pays (the 17.5 ms
weight read plus ~4-6 ms of overhead) and 0.33 ms per extra request (its KV read, which
alone would be 0.21-0.26 ms).

**"FP8 weights plus an FP8 KV cache gave 2.4x the goodput because it attacks both
terms."** FP8 weights cut the fixed part from ~22 ms to ~14.5 ms per step (8.2 GB of
weights instead of 15.1). An FP8 KV cache halves the bytes per request, so each extra
request costs 0.19 ms instead of 0.33. In the time BF16 spends on one step at its
goodput (31 ms), fp8-kv8 can run about 87 requests instead of 28: 3.1x more decode
capacity. We measured 2.4x (5.24 vs 2.21 req/s) because chat traffic brings prefill too:
2.4x the requests means 2.4x the prompts to prefill, prefill is compute-bound, and those
prefill chunks ride inside decode steps and stretch the slow tail that the p95 SLO
judges. KV capacity was not the limit: at goodput both caches were under 20% full.

**"The faster the step, the more the fixed overhead hurts."** fp8-kv8 reaches only
61-65% of its floor, against 75% for BF16, and the H100 70B BF16 run reaches 82-85%.
Every config pays roughly 3-6 ms per step that does not shrink with bytes; on a 15 ms
floor that is a bigger share than on a 24 ms one.

**"We profiled it: the overhead is mostly GEMMs below peak bandwidth, and FP8 adds a CPU
stall."** A GPU trace of every step (14d959e5) splits a batch-1 bf16 step (22.2 ms) into
20.9 ms of weight-reading GEMMs, which stream the 15.1 GB at 84% of the spec bandwidth, plus
0.8 ms attention, 0.4 ms of small kernels and under 0.1 ms with the GPU idle. The fp8-kv8 step
(16.3 ms) reads 8.2 GB, but its FP8 GEMMs reach only 57-72% of bandwidth (13.7 ms instead of
~10). It also pays two costs bf16 does not: 0.6 ms quantizing activations to FP8, and 1.0 ms
with the GPU idle, because in that configuration the CPU blocks on a GPU-to-CPU copy for the
whole step and cannot prepare the next one in parallel. Halving the bytes cut the step 1.36x,
not 1.85x.

**"The 2.4x checks out in the trace."** Per extra request, attention costs 0.29 ms in BF16 and
0.17 ms with the FP8 cache: 1.7x cheaper for half the bytes, because the FP8 attention kernel
runs at ~64% of bandwidth against ~74-90%. Decode alone gives ~2.8-3.1x. At its goodput
fp8-kv8 handles 2.4x the prompts, and FP8 prefill is only 1.3x faster per token (50 vs 67 µs).
So steps carrying a prompt chunk take 33% of its GPU time against 17% for bf16, and those
60-70 ms steps are the slow tokens the p95 counts.

**"On 2x H100 the 4x FP8 gain was mostly memory capacity, not speed."** Llama 3.3 70B
BF16 fills 141 GB of the pod's 160 GB, leaving room for only about 42k cached tokens,
about twenty 2k-token requests. At 0.6 req/s requests start queueing for cache space and
TTFT blows through 1 s while TPOT is still a comfortable 31 ms (9f0853d7). FP8 halves
the weights and leaves about 257k tokens of cache, so the limit moves back to decode
speed, where FP8 is also 2x faster per step.

**"Across two CPU sockets, 4x L40S can't hit 50 ms per token."** Each GPU still reads a
35 GB weight shard per step, a 40 ms floor; the 160 all-reduces per token have to go
through host memory because GPU-to-GPU transfers hang on that topology (cf4d1614), which
adds about 5 ms; and we measured about 60 ms, 78% of the floor. Prefill is worse: a
1k-token prompt's all-reduces move 2.7 GB at about 12 GB/s, 223 ms of pure
communication, so TTFT was never going to be far under the ~0.55 s we saw.

## The math

### Bytes per decode step

For a dense transformer with `L` layers, hidden size `h`, `n_q` query heads and `n_kv`
key-value heads of dimension `d`, MLP width `f` and vocabulary `V` (all from the pinned
`config.json`, `loom_bench/profile/arch.py`):

- linear weights per layer: `h·(n_q·d + 2·n_kv·d) + n_q·d·h + 3·h·f`
- every step reads all linear weights and the LM head (`V·h`, kept in BF16 even in the
  FP8 checkpoints); it does **not** read the embedding table (one row per request).
- KV per cached token: `2 · L · n_kv · d` elements.

| Model | Linear + LM head params | Step weights BF16 / FP8 | KV per token BF16 / FP8 |
|---|---|---|---|
| Qwen3-8B (L 36, h 4096, 32 q / 8 kv heads, d 128, f 12288, V 151936) | 7.57 B | 15.14 / 8.19 GB | 144 / 72 KiB |
| Llama 3.3 70B (L 80, h 8192, 64 / 8 heads, d 128, f 28672, V 128256) | 69.5 B | 139.0 / 70.6 GB | 320 / 160 KiB |

Check: adding back the embedding table and norms reproduces every registry checkpoint
size to within 0.15% (test `test_byte_accounting_reproduces_every_checkpoint_size`).

### The floors

Per GPU, with TP degree `t`, batch `B` and cached tokens `K = Σ context` over the batch:

```
memory_ms  = (W/t + K · kv_bytes/t) / HBM_bandwidth
compute_ms = (2 · params · B + 4 · L · n_q·d · K) / t / tensor_peak
comm_ms    = 2 · L · allreduce(B · h · 2 bytes)            (TP only)
step floor = max(memory_ms, compute_ms) + comm_ms
prefill floor (n tokens) = (2 · params · n + 2 · L · n_q·d · n²) / t / peak + 2·L·allreduce(n·h·2)
```

Decode is memory-bound until the batch is in the thousands: for Qwen3-8B BF16 at batch
64, memory is 34 ms and compute 3 ms. The all-reduce link is a straight line through two
measured points on the 4x L40S pods with P2P off (16 KB in 27.6 µs, 32 MB in 2.76 ms;
docs/benchmark-lab.md "70B tuning"); the H100 NVLink figure (10 µs + bytes / 450 GB/s)
is an assumption until a 2x H100 trace measures it.

Floors from `bench roofline` (context 1,536 tokens, ms per step):

| Config | B=1 | B=32 | B=64 | B=128 | Largest batch under a 50 ms floor |
|---|---|---|---|---|---|
| 8B bf16, 1x L40S | 17.8 | 25.9 | 34.3 | 51.1 | 124 |
| 8B fp8-kv8, 1x L40S | 9.6 | 13.7 | 17.9 | 26.3 | 309 |
| 70B bf16, 4x L40S TP4 (P2P off) | 44.8 | 55.9 | 67.4 | 90.4 | 16 |
| 70B bf16, 2x H100 TP2 | 22.4 | 24.9 | 27.5 | 32.7 | 342 |
| 70B fp8, 2x H100 TP2 | 12.2 | 14.7 | 17.3 | 22.5 | 468 |

### Measured against the floor

`bench roofline` rebuilds, from each run's per-request rows, how many requests were
decoding and how much context they held every 0.1 s of the measurement window, computes
the floor at each instant, and weights it by tokens produced. The typical measured step
is the ITL median (most steps carry no prefill). At each config's goodput:

| Run | Config | Load | Batch | Floor (weights / KV / all-reduce) | ITL p50 | Achieved |
|---|---|---|---|---|---|---|
| 7a8237d0 | 8B bf16, chat | 2.21 req/s | 27 | 23.6 (17.5 / 6.1 / 0) ms | 31.1 ms | 76% |
| 7a8237d0 | 8B fp8-kv8, chat | 5.24 req/s | 59 | 15.9 (9.5 / 6.4 / 0) ms | 25.9 ms | 61% |
| 7a8237d0 | 8B bf16, 1k/1k | 1 req/s | 44 | 28.9 (17.5 / 11.4 / 0) ms | 38.7 ms | 75% |
| 7a8237d0 | 8B fp8-kv8, 1k/1k | 2.25 req/s | 86 | 20.8 (9.5 / 11.3 / 0) ms | 31.7 ms | 65% |
| 9f0853d7 | 70B bf16, 2x H100, 1k/1k | 0.5 req/s | 16 | 23.6 (20.7 / 1.2 / 1.7) ms | 28.6 ms | 82% |
| 9f0853d7 | 70B fp8, 2x H100, 1k/1k | 2 req/s | 56 | 16.7 (10.5 / 4.2 / 1.9) ms | 21.8 ms | 77% |
| cf4d1614 | 70B bf16, 4x L40S, 1k/1k (no goodput) | 0.0625 req/s | 6 | 46.5 (40.2 / 0.8 / 5.4) ms | 59.8 ms | 78% |

The step-cost fit (ITL p50 less the modelled all-reduce = fixed + per-request × batch,
over each cell's 4-5 load points, R² ≥ 0.95 except 4x L40S at 0.89 with 3 points):

| Run / config | Fixed per step (floor) | Per request (floor) |
|---|---|---|
| 7a8237d0 bf16, chat / 1k-1k | 21.9 / 23.9 ms (17.5) | 0.33 / 0.34 ms (0.21 / 0.26), 1.3-1.5x |
| 7a8237d0 bf16-kv8 | 22.9 / 23.4 ms (17.5) | 0.20 / 0.21 ms (0.11 / 0.13), 1.6-1.9x |
| 7a8237d0 fp8 | 14.4 / 13.1 ms (9.5) | 0.30 / 0.35 ms (0.21 / 0.26), 1.4x |
| 7a8237d0 fp8-kv8 | 14.5 / 15.3 ms (9.5) | 0.19 / 0.20 ms (0.11 / 0.13), 1.5-1.8x |
| 9f0853d7 70B bf16 / fp8, 1k-1k | 25.6 / 13.7 ms (20.7 / 10.5) | 0.10 / 0.15 ms (0.075) |
| cf4d1614 70B bf16 4x L40S, 1k-1k | 52.7 ms (40.2) | 0.52 ms (0.14), 3.6x |

What the fit says:

1. **FP8 weights deliver most, not all, of their byte saving.** fp8-kv8's fixed part is
   7.4-8.6 ms below bf16's in 7a8237d0's fits; the weight floor drops by 8.0 ms. **Measured (14d959e5):**
   a batch-1 step drops by 5.9 ms, from 22.2 to 16.3 ms. Its GEMMs drop by 7.1 ms (20.9 → 13.7),
   and FP8 adds back 1.6 ms of GPU idle and activation quantization.
2. **Every config pays 3-6 ms per step on one GPU beyond the weight read** (bf16 +4.4 to
   +6.3, fp8-kv8 +5.0 to +5.8, H100 +3.2 to +4.9 after the modelled all-reduce). That
   overhead does not scale with bytes. **Measured (14d959e5, batch 1):** on bf16, 3.3 of its
   4.7 ms are GEMMs reading weights at 84% of the spec bandwidth instead of 100%. The rest
   is attention beyond its bytes (0.6), small kernels (0.4) and GPU idle (0.07). On fp8-kv8,
   4.3 of its 6.8 ms are FP8 GEMMs at 57-72% of bandwidth, plus 1.0 ms GPU idle (a CPU stall,
   below), 0.6 ms quantization, 0.6 ms attention beyond its bytes and 0.2 ms other kernels.
   Sampling is 0.03 ms on both.
3. **An FP8 KV cache halves the bytes but not the time.** Per request, FP8 KV costs
   0.19-0.21 ms against BF16's 0.30-0.35: 1.6-1.7x cheaper, not 2x. **Confirmed by the
   trace:** attention alone costs 0.288 ms per request on bf16 and 0.170 on fp8-kv8 (chat);
   that is 74% and 63% of the bandwidth their KV bytes need (90% and 65% on 1k/1k). The FP8
   cache is read by FlashInfer's kernel, which converts the values back to BF16 as it reads;
   BF16 uses FlashAttention.
4. **The 4x L40S TP4 run pays 12.5 ms per step beyond its floor and 3.6x the KV floor per
   request.** Its all-reduces go through host memory; at batch 21 each carries 344 KB,
   between the two message sizes the link was measured at, so mid-size messages are
   likely slower than the straight line assumes. Profiling a TP4 pod would settle it
   (not in the proposed run).

### The 2.4x, step by step (7a8237d0, chat)

| | bf16 | fp8-kv8 | ratio |
|---|---|---|---|
| Goodput (measured) | 2.21 req/s | 5.24 req/s | **2.37x** |
| Fixed cost per step (fit) | 21.9 ms | 14.5 ms | |
| Cost per request (fit) | 0.326 ms | 0.190 ms | |
| Requests in a 31.1 ms step (bf16's at goodput) | 28 | 87 | 3.09x |
| TPOT p95 / ITL p50 at goodput (the tail) | 1.31 | 1.60 | |
| Prefill share of GPU time: floor / implied by ITL mean vs median | 10% / 15% | 11% / 25% | |
| KV cache full at the first failing load | 19% (~178k-token cache) | 18% (~439k) | |

So: decode alone would give ~3.1x. The SLO is a p95, and fp8-kv8 runs 2.4x the prompts
per second; prefill is FLOP-bound, and on Ada FP8 prefill is only partly faster (the
implied prefill share rises from 15% to 25% while its floor barely moves, so FP8
prefill reaches a smaller share of its doubled peak). Mixed prefill+decode steps
stretch the tail, and the realized gain lands at 2.37x. FP8 weights alone (3.87 req/s)
and an FP8 cache alone (3.67 req/s) each gave ~1.7x; together they compound.

**The trace confirms each step of that** (14d959e5, both cells at both chat rates on one
host; [report](../../reports/8b-profile-v1/README.md#4-the-237x-step-by-step)):

| At its goodput | bf16 @ 2.21 req/s | fp8-kv8 @ 5.24 req/s |
|---|---|---|
| Decode step line from the trace (ms) | 22.3 + 0.323 × batch | 15.4 + 0.204 × batch |
| Typical decode batch, step | 20-28, 28.7-31.1 ms | 59-79, 26.3-31.5 ms |
| Steps carrying prefill | 7% of steps, 17% of GPU time | 17% of steps, 33% of GPU time |
| Prefill cost per prompt token | 67 µs (57% of BF16 peak) | 50 µs (38% of FP8 peak) |
| TPOT p95 / ITL p50 (unprofiled) | 1.34 | 1.56 |

The bf16 line matches the client fit to 0.4 ms. The fp8-kv8 line is 0.9 ms higher, which
is the profiler (2.5-3.8% on fp8-kv8) and possibly the host's CPU. Within a 31.1 ms step the
lines fit 27 vs 77 requests, 2.8x (the unprofiled fits: 3.1x). FP8 prefill is only 1.34x faster per token. So at
2.37x the prompts, pure prefill work is ~25% of fp8-kv8's GPU time against ~14% of bf16's,
which is the 15% / 25% the client data implied.

The fixed-1k-1k version of the same comparison: 2.69x decode capacity, 2.25x measured.
On 2x H100 (9f0853d7) decode alone would give 3.2x, and the measured 4.0x comes from
BF16 failing early on TTFT: its KV cache (~42k tokens, estimated from the engine's own
usage gauge on runs without preemptions or prefix sharing; BF16 at 0.595 req/s had
4 preemptions per run) fills before decode speed is the limit.

### Time to first token

At the lightest load each run tried, TTFT p50 was 127-186 ms on the 8B against a prefill
floor of 20-44 ms: the prompt shares its step with a running decode batch (23-66
requests at those loads), waits for the step in flight, and pays tokenization and
streaming. On 4x L40S, TTFT p50 was 544 ms against a 322 ms floor, of which 223 ms is
the measured host-memory all-reduce: TP over that link cannot give a fast TTFT whatever
the engine does.

### Speculative decoding (55102ddb)

With EAGLE3 a step verifies several draft tokens per request, so one client ITL is one
step that delivers ~2.28 tokens on average. Steps took 64-88 ms (vs 57-72 ms without),
so per token the median request ran at 29-48 ms, but requests the draft cannot predict
stay at the ~60 ms floor. `bench roofline` marks these rows "n/a": the step floor would
need the draft model and the verification batch in it.

## Profiling: measuring where the rest goes

The client-side data can say how much time is beyond the floor, not what it is. A trace
of the GPU can. Loom captures one through vLLM's built-in torch profiler.

**How a profile window works** (`loom_bench/profile/`, experiment `profile:` section):

1. The variant starts vLLM with `--profiler-config` (`profiler: torch`, a trace
   directory, Python stacks off). Nothing is recorded until asked.
2. After a cell's measured load points, the runner runs one short extra job per
   (workload, load value): load for `settle_s` to reach steady state, then
   `POST /start_profile` with `max_iterations` (how many engine steps to record) and
   `POST /stop_profile`, which writes the trace. vLLM v0.30 accepts `max_iterations` but
   ignores it (smoke a183a857). Each window records every step for `capture_s`: in
   14d959e5, 209-1,218 steps in 20 s, 6-22 MB per trace, flushed in 9-51 s.
3. On RunPod a root script archives the trace directory and uploads it through the same
   presigned-S3 path the jobs use (`collect_profile.sh`; on AWS the same over SSM); the
   runner downloads it to `results/<experiment>/profiles/<run id>/trace.tar.gz`.
4. `bench profile analyze` (run automatically, also by hand on any trace) puts every GPU
   kernel in a category by name: GEMM, attention, all-reduce/NCCL, quantize/dequantize,
   sampling, memory copies, other kernels; time with no kernel running is **GPU idle**
   (the CPU scheduling the next step). Overlapping kernels count once, so only exposed
   communication counts as communication. torch.compile fuses FP8 activation
   quantization into the RMSNorm and SiLU kernels. Those count as quantize, not GEMM.
   vLLM v0.30 labels every step `execute_context_<n>(<tokens>)_generation_<n>(<tokens>)`.
   From those labels the analysis counts steps and splits the window step by step, from
   one step's start to the next. Pure-decode steps are grouped by batch size, and steps
   carrying prefill are grouped as "mixed" (`step_shapes`). It writes per-step
   milliseconds, the step shapes and the top kernels to `window.json` and a `profile` event.

**Profiled windows never count.** They are not recorded in `bench_runs`, so they are not
in goodput, latency, cost or the search. The profiler slows the steps it records; each
window keeps its own ITL next to the ITL just before it (`profiler_slowdown`) only to
show that cost.

**Nsight Systems.** vLLM also supports `nsys profile ... vllm serve --profiler-config.
profiler cuda` with `--capture-range=cudaProfilerApi`. On RunPod the engine runs as a
plain process in the pod's container, so nsys would have to be installed into the image
at pod start and wrap the engine command; CUDA kernel tracing uses CUPTI and needs no
extra privileges, but CPU sampling needs `perf_event_paranoid` ≤ 2 and GPU hardware
metrics need admin profiling rights, neither of which RunPod's containers are known to
grant. Loom uses the torch profiler (no install, kernel-level timing, small traces);
nsys stays an option to try on one pod if the torch trace leaves a gap.

### The profiling run (14d959e5, ran 2026-11-03)

`bench/experiments/qwen3-8b-profile-aws.yaml`, after its smoke `aws-smoke-profile`
(a183a857, $0.96). bf16 and fp8-kv8 on one AWS g6e.xlarge (1x L40S 48 GB, 4 vCPU, on-demand;
fp8-kv8 through a warm restart). fixed-1k-1k closed loop at 1, 16 and 64 concurrent requests
(known batch sizes) and chat at 2.213 and 5.239 req/s (each config's goodput in 7a8237d0).
Two measured repetitions per point, all 20 completed. One 20 s profile window per point, all
10 captured. **$4.59** of a $5.25 cap; the instance is terminated. The RunPod twins
(`qwen3-8b-profile-runpod`, `runpod-smoke-profile`) never got a pod: Secure L40S creates were
refused all of 2026-11-02 ($0).

Report, tables and per-window breakdowns:
[reports/8b-profile-v1/](../../reports/8b-profile-v1/README.md).

#### What the trace measured (14d959e5)

A decode step at batch 1 (fixed-1k-1k, ~1.5k tokens of context), ms per step from one step's
start to the next:

| | bf16 | fp8-kv8 |
|---|---|---|
| Step | 22.17 | 16.28 |
| Weight-read floor | 17.52 | 9.48 |
| GEMM (36 layers + LM head) | 20.85 (726 GB/s, 84% of peak) | 13.71 (597 GB/s, 69%) |
| Attention | 0.83 | 0.77 |
| FP8 activation quantization | 0 | 0.61 |
| Other kernels | 0.41 | 0.17 |
| Sampling | 0.03 | 0.03 |
| GPU idle | 0.07 | 1.04 |

What each line means:

- **The GEMMs are the step.** They are memory-bound at every batch measured: from batch 1 to
  64 they grow only 20.9 → 24.0 ms (bf16) and 13.7 → 14.6 ms (fp8), and their compute floor at
  batch 64 is 2.7 / 1.3 ms. Per layer, BF16's cuBLAS kernels stream at 77-86% of the 864 GB/s
  spec peak. The FP8 CUTLASS kernels stream at 57-72%: qkv and o projections are the worst
  (57%, 63%), gate_up the best (72%). So half the bytes buys 1.59x per layer, not 2x. The LM
  head stays BF16 in the FP8 checkpoint: 1.7 ms in both.
- **fp8-kv8's GPU idles ~1 ms per step; bf16's does not.** In bf16 the CPU prepares a step in
  1.9 ms, then waits on an event while the GPU runs, so its work is hidden. In fp8-kv8 the CPU
  blocks in a GPU-to-CPU copy (`aten::to`, 13.1 ms per step) for the whole GPU step, so its
  ~1 ms of preparation runs while the GPU waits. It also makes five pinned-memory copies and a
  stream sync per step that bf16 does not. Which of FP8 weights, the FP8 cache or the FlashInfer
  backend causes it is not separated yet: the windows were taken without Python stacks.
- **Quantization is small in decode, not in prefill.** 0.6 ms per decode step (108 fused
  norm/SiLU + quantize kernels). In a step carrying an 800-token prompt chunk it is 5.4 ms
  of ~72.
- **Attention is the per-request cost.** 0.288 ms per extra request in BF16, 0.170 with the
  FP8 cache (chat). At batch 64 (1k/1k) that is 18.9 of a 40.9 ms step on bf16 and 13.3 of
  30.0 on fp8-kv8. At bf16's overload point (5.24 req/s, batch ~125), 36.8 of 63 ms.
- **Prefill is where FP8 gains least.** Per prompt token, 67 µs on bf16 (57% of the BF16
  tensor peak) against 50 µs on fp8-kv8 (38% of the FP8 peak): 1.34x, not 2x.

The profiler cost: profiled steps were 0-5% slower than the 22 s just before them in the same
job (fp8-kv8 2.5-3.8%, its step being CPU-bound). The one window reading 0.88 (bf16 at 64
concurrent) caught a closed-loop wave's prefill burst, with shorter contexts, not a speed-up.
Two things not settled: fp8-kv8 at 5.24 req/s ran 6% slower on this host than in 7a8237d0
(ITL p50 27.5 vs 25.9 ms; bf16 matched), plausibly the 4-vCPU host and the CPU-bound step. And
the CPU stall's call site is not named yet. Next, not proposed for spend yet: one fp8-kv8 window
with Python stacks, and the `fp8` and `bf16-kv8` cells profiled, which separate the
two FP8 halves.
