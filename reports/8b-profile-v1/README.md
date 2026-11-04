# Where a decode step's time goes: Qwen3-8B bf16 vs fp8-kv8, measured (14d959e5)

**Run.** Experiment `14d959e5-5a13-4939-a56d-71a8d87d9d2a` (`bench/experiments/qwen3-8b-profile-aws.yaml`),
2026-11-03, one AWS g6e.xlarge (1x L40S, us-east-1, on-demand), vLLM v0.30.0, perf-track
`b0cf448`. Two cells on one host: **bf16** (BF16 weights and KV cache) and **fp8-kv8**
(RedHatAI FP8-dynamic weights, FP8 KV cache). fixed-1k-1k at 1, 16 and 64 concurrent
requests; chat-sharegpt at 2.213 and 5.239 req/s (bf16's and fp8-kv8's goodput in
7a8237d0). 20 measured runs (2 per point), all completed; one 20 s torch-profiler window per
point (10 windows, all captured). Spend **$4.59** against a $5.25 cap. Smoke: `a183a857`
($0.96). The instance is terminated.

Files here:

- `tables.md`: every number below, generated from the traces by `build.py`.
- `breakdowns/`: one JSON per profile window (`bench profile analyze` output plus the window's
  capture facts).
- `roofline/`: `bench roofline -e 14d959e5 -e 7a8237d0` (client-side floors, fits and the
  2.37x explanation).

Regenerate (free, no GPU):

```bash
uv run python reports/8b-profile-v1/build.py results/14d959e5-5a13-4939-a56d-71a8d87d9d2a
uv run bench roofline -e 14d959e5 -e 7a8237d0 --db sqlite:///results/loom.db --out reports/8b-profile-v1/roofline
```

## The short version

1. **A decode step is almost all weight reading, and BF16 does it well.** At batch 1, a bf16
   step is 22.2 ms, and 20.9 ms of that is matrix-vector work (the GEMMs) reading 15.1 GB of
   weights at 726 GB/s, 84% of the L40S's 864 GB/s spec peak. Attention is 0.8 ms. The GPU
   is idle for less than 0.1 ms. The "5 ms of overhead" the client data showed is mostly the
   GEMMs running at 84% of peak bandwidth instead of 100% (3.3 ms), not the CPU.
2. **FP8 halves the bytes but not the GEMM time.** The FP8 step reads 8.2 GB, yet its GEMMs
   take 13.7 ms, not ~10: the FP8 kernels reach 57-72% of peak bandwidth per layer, against
   77-86% for BF16. Halving the bytes bought 1.52x on the GEMMs, not 2x.
3. **fp8-kv8 pays two costs bf16 does not: ~1 ms of GPU idle per step and 0.6 ms of
   quantization.** The idle time is at the start of every step: in this configuration the
   CPU blocks for the whole GPU step in a GPU-to-CPU copy, so it cannot prepare the next
   step while the GPU works. In bf16 the CPU finishes in 1.9 ms and waits, so its work is
   hidden.
4. **An FP8 KV cache halves the bytes, but attention gets 1.4-1.7x cheaper, not 2x.** Each
   extra request costs 0.288 ms of attention on bf16 and 0.170 ms on fp8-kv8 (chat). BF16
   attention runs at 74-90% of the bandwidth its KV bytes need; FP8-KV attention (a
   FlashInfer kernel) at 63-65%.
5. **The 2.37x is confirmed, and the trace shows where the gap to ~3x goes.** Decode alone
   would give fp8-kv8 about 2.8x the requests per step budget (3.1x from the unprofiled
   client fit). At its goodput fp8-kv8 handles 2.37x the prompts, and its prefill is only
   1.3x faster per token than bf16's. So steps carrying prefill take 33% of its GPU time
   against 17% for bf16, and 17% of its steps against 7%. Those 60-70 ms steps are the slow
   tokens the p95 TPOT counts.

## Method

- **Profile window**: during each load point, after 45 s of steady load, vLLM's torch profiler
  recorded 20 s of every engine step (vLLM v0.30 ignores the `max_iterations` bound; see
  `loom_bench/profile/trace.py`). Windows are separate jobs and never enter `bench_runs`.
- **Steps**: vLLM v0.30 names each step `execute_context_<n>(<tokens>)_generation_<n>(<tokens>)`
  (prefill requests and tokens, decode requests). A step runs from its start to the next
  step's start, so it includes the LM head and sampling (which run after the annotated
  forward) and any time the GPU waits for the CPU. Pure-decode steps are grouped by batch.
  Steps carrying prefill are grouped as "mixed".
- **Categories**: every GPU kernel is put in one category by name. Where kernels overlap, the
  time goes to one category. Time with no kernel running is **idle** (the GPU waiting on the
  CPU). FP8 activation quantization is fused by torch.compile into the RMSNorm / SiLU kernels
  before each GEMM. It is counted as **quant** (the norm or SiLU math in those kernels comes
  with it).
- **Floors** use spec-sheet peaks: 864 GB/s HBM, 362 TFLOP/s BF16, 733 TFLOP/s FP8 (dense).
  Weights read per step are 15.14 GB for bf16 and 8.19 GB for fp8; the latter includes the
  1.25 GB LM head, which stays BF16 in the FP8 checkpoint. KV per token is 144 KiB for bf16
  and 72 KiB for fp8.
- **Profiler cost**: each window compares the client's inter-token gap while recording with
  the 22 s before it, in the same job. It was 0-5% slower (`tables.md`, "Windows"). The one
  outlier is explained under "What looks off".

## 1. A decode step at batch 1: what each millisecond is

fixed-1k-1k, one request at a time (context ~1.5k tokens): windows `224038b8` (bf16) and
`1b155770` (fp8-kv8). The profiled steps are 3.8% slower than unprofiled ones on fp8-kv8
and 0.2% on bf16. Unprofiled client ITL p50: 22.1 and 15.7 ms (14d959e5's measured runs).

| ms per step | bf16 | fp8-kv8 | What it is |
|---|---|---|---|
| **Step (start to next start)** | **22.17** | **16.28** | |
| Weight-read floor | 17.52 | 9.48 | 15.14 / 8.19 GB at 864 GB/s |
| GEMM, linear layers | 19.13 | 12.04 | 36 layers × (qkv, o, gate_up, down) |
| GEMM, LM head (BF16 in both) | 1.72 | 1.75 | 1.25 GB at 725 / 711 GB/s |
| Attention (incl. RoPE, KV write) | 0.83 | 0.77 | its KV floor: 0.26 / 0.13 |
| Quantize (fused norm/SiLU + FP8 quant) | 0 | 0.61 | 108 small kernels |
| Other (norms, SiLU, residual adds) | 0.41 | 0.17 | in fp8 most of it moved into "quant" |
| Sampling | 0.03 | 0.03 | |
| **GPU idle** | **0.07** | **1.04** | fp8: ~0.9 ms before the step's first layer |

Per layer (median over 200 steps; one layer's kernels, B=1):

| Projection | Bytes BF16 / FP8 | bf16 µs (GB/s, % of peak) | fp8 µs (GB/s, % of peak) |
|---|---|---|---|
| qkv (4096 → 6144) | 50.3 / 25.2 MB | 70.0 (719, 83%) | 51.5 (489, 57%) |
| o (4096 → 4096) | 33.6 / 16.8 MB | 50.5 (665, 77%) | 31.0 (541, 63%) |
| gate_up (4096 → 24576) | 201.3 / 100.7 MB | 270.8 (743, 86%) | 162 (622, 72%) |
| down (12288 → 4096) | 100.7 / 50.3 MB | 140.2 (718, 83%) | 90 (559, 65%) |
| **Layer** | 386 / 193 MB | **531** | **335** (1.59x faster, bytes 2x) |

Plain words:

- **bf16** uses cuBLAS matrix-vector kernels (`gemvx`) at batch 1. They stream weights at
  77-86% of the spec peak. That is against the spec sheet: we have not measured this card's
  practical streaming ceiling (a copy benchmark would), so the true headroom is smaller. Over the
  floor: 3.3 ms from GEMMs below peak, 0.6 ms of attention beyond its bytes, 0.4 ms of small
  kernels, 0.1 ms idle. The CPU is not the problem.
- **fp8-kv8** uses CUTLASS sm89 FP8 GEMMs with a per-token-scale epilogue. On one row of input
  they read weights at 57-72% of peak. The two smallest projections (qkv, o) are the worst.
  Over its 9.5 ms floor: 4.3 ms from GEMMs below peak, 1.0 ms idle, 0.6 ms quantization, 0.6 ms
  of attention beyond its bytes, 0.2 ms of other kernels. The faster step pays more overhead,
  in absolute terms and as a share: 6.8 ms (42% of the step) against 4.7 ms (21%).

**Why fp8-kv8's GPU idles.** In the bf16 trace the CPU's step call lasts 1.9 ms (median). It
then waits in `cudaEventSynchronize` while the GPU runs the 20 ms step, so the CPU has the next
step ready before the GPU needs it. In the fp8-kv8 trace the CPU's step call lasts as long as
the GPU step (15.3 vs 14.5 ms median). 13.1 ms of it is one blocking GPU-to-CPU copy
(`aten::to` → `cudaMemcpyAsync`) inside the step. Each fp8-kv8 step also makes five
pinned-memory copies and one `cudaStreamSynchronize` that bf16 steps don't. So the ~1 ms of CPU
work to prepare a step cannot overlap the GPU, and the GPU sits idle for it: 0.19 ms then
0.69 ms of gaps before the first layer's kernel (window `1b155770`). Not settled: which call
does the blocking copy. The trace was taken without Python stacks. It shows up only in the
fp8-kv8 configuration (FP8 weights, FP8 KV cache, FlashInfer attention backend), and this run
cannot say which of the three causes it (see "Next").

## 2. What each extra request costs: attention

Fitting step time = fixed + per-request × batch over each window's pure-decode steps
(`tables.md`; batches 1-137):

| | bf16 chat | bf16 1k/1k | fp8-kv8 chat | fp8-kv8 1k/1k |
|---|---|---|---|---|
| Fixed (ms) | 22.27 | 22.64 | 15.38 | 15.97 |
| Per request (ms) | 0.323 | 0.297 | 0.204 | 0.218 |
| of which attention | 0.288 | 0.287 | 0.170 | 0.199 |
| KV floor per request (context ~1.25k / ~1.5k) | 0.214 | 0.259 | 0.107 | 0.130 |
| Attention vs its floor | 74% | 90% | 63% | 65% |

The client-side fits (unprofiled) agree: 7a8237d0 bf16 chat 21.9 + 0.326, fp8-kv8 chat
14.5 + 0.190. On this host the fixed1k/1k fits are bf16 22.0 + 0.370 and fp8-kv8 15.4 + 0.199
(`roofline/roofline.md`). The trace's bf16 line matches 7a8237d0 to 0.4 ms. Its fp8-kv8
line sits 0.9 ms higher. Most of that is the profiler slowing fp8-kv8 more (2.5-3.8%; its step is
CPU-bound), and some may be this host's 4-vCPU CPU.

Plain words: each request's KV cache is read once per step. In BF16 that read runs close to
bandwidth. The FP8 cache is half the bytes, but its attention kernel has to turn the FP8
values back into BF16 as it reads, and it runs at about 64% of bandwidth. So a request costs
1.4-1.7x less on fp8-kv8, not 2x. At batch 64 (1k/1k), attention is 18.9 ms of a 40.9 ms step
on bf16 and 13.3 of 30.0 on fp8-kv8. At bf16's overload point (5.24 req/s, batch ~125),
attention is 36.8 of 63 ms.

The GEMMs barely grow with batch: bf16 20.9 → 24.0 ms from batch 1 to 64, fp8-kv8 13.7 →
14.6. At batch 64 their compute floor is 2.7 ms (bf16) and 1.3 ms (FP8), so decode stays
memory-bound. One step-change: past 64 requests the FP8 GEMMs switch tile and cost 1.1 ms
more (14.6 → 15.7 ms at batch 65; chat 5.24 window). bf16 does the same at 129 (24.9 → 27.8).

## 3. Steps that carry prefill

| Window | Mixed steps | Share of GPU time | Prompt tokens / step | Mixed step (mean) | Cost per prompt token | vs compute floor |
|---|---|---|---|---|---|---|
| bf16 chat 2.21 req/s | 7% (41 of 591) | 17% | 785 | 81 ms | 67 µs | 57% of 362 TFLOP/s |
| fp8-kv8 chat 2.21 req/s | 4% (40 of 992) | 12% | 804 | 59 ms | 51 µs | 37% of 733 |
| fp8-kv8 chat 5.24 req/s | 17% (89 of 537) | 33% | 842 | 72 ms | 50 µs | 38% of 733 |
| bf16 chat 5.24 req/s (overloaded) | 23% (49 of 209) | 48% | 1269 | 194 ms | 101 µs* | |

\* includes the attention of ~125 decoding requests riding along. See `tables.md`.

FP8 prefill costs 50 µs per prompt token against BF16's 67 µs, 1.34x faster, although FP8's
tensor peak is 2x. Prefill is compute-bound. The FP8 GEMMs reach about 38% of the FP8 peak,
the BF16 ones about 57% of theirs. On top of that, quantizing activations costs 5.4 ms of a
72 ms mixed step on fp8-kv8: the per-token scales are computed for every prompt token at every
layer.

## 4. The 2.37x, step by step

Measured (7a8237d0): bf16 serves 2.21 req/s at the chat SLO and fp8-kv8 5.24 req/s, a 2.37x
gain. In this run, at the same rates on one host, the client numbers repeat that result:

| At its goodput | bf16 @ 2.21 | fp8-kv8 @ 5.24 |
|---|---|---|
| ITL p50 / TPOT p95 (unprofiled, 14d959e5) | 30.8 / 41.3 ms | 27.5 / 42.9 ms |
| Decode batch (typical) | 20-28 | 59-79 |
| Pure-decode step | 28.7-31.1 ms | 26.3-31.5 ms |
| of which GEMM / attention / idle | 22.8-23.0 / 4.9-7.3 / 0.05 | 14.6-15.7 / 9.8-13.6 / 1.3 |
| Steps carrying prefill | 7% of steps, 17% of time | 17% of steps, 33% of time |
| TPOT p95 / ITL p50 | 1.34 | 1.56 |

1. **Decode alone would give ~2.8-3.1x.** Within bf16's typical step (31.1 ms), the trace lines
   fit 27 requests on bf16 and 77 on fp8-kv8 (2.8x). The unprofiled client fits give 28 and
   87 (3.1x; `roofline/roofline.md`). Where the gain comes from: the fixed part drops from 22.3
   to 15.4 ms (GEMMs 22.1 → 13.3; idle and quant add 1.6 back), and each request costs 0.204 ms
   instead of 0.323.
2. **Prefill takes part of it back.** fp8-kv8 at its goodput admits 2.37x the prompts per
   second, and each prompt token costs it 75% of what it costs bf16 (50 vs 67 µs). Pure
   prefill work is then about 25% of fp8-kv8's GPU time (5.24 req/s × ~950 tokens × 50 µs)
   against 14% of bf16's (2.21 × ~950 × 67 µs). This is the share the client-side
   mean-vs-median gap implied (15% vs 25%).
3. **The SLO is a p95, so the slow steps decide.** A step carrying a prompt chunk lasts 60-70 ms
   on fp8-kv8. A request that shares 17% of its steps with prefill (fp8-kv8) has a slower tail
   than one that shares 7% (bf16): TPOT p95 is 1.56x the median ITL on fp8-kv8 against 1.34x
   on bf16. That is what holds the realized gain at 2.37x instead of ~3x.
4. **Not the limit**: KV cache capacity (under 20% full at both goodputs in 7a8237d0), and
   sampling (0.03-0.08 ms per step).

## What looks off, and what it means

- **bf16 at 64 concurrent shows a profiler "speed-up" (slowdown 0.880, window `5650aa2e`).**
  This is not a profiler effect. The 1k/1k closed loop at 64 starts and finishes requests in
  waves. The window caught a wave's prefill burst (34 mixed steps, 30% of its time), after
  which contexts were short and decode steps fast (p50 40.9 ms, mean 43.8). The 22 s before
  it was late in the previous wave, with long contexts (ITL p50 47.2). The other nine windows
  show 1.002-1.051.
- **fp8-kv8 at 5.24 req/s is 6% slower here than in 7a8237d0** (ITL p50 27.5 vs 25.9 ms; TPOT
  p95 42.9 vs 41.4, still under 50). bf16 at 2.21 matches (30.8 vs 31.1). Different host
  (AWS g6e.xlarge, 4 vCPUs, vs a RunPod pod) and different config hash (the armed profiler).
  The config whose step is CPU-bound is the one that moved, which points at the host's CPU.
  Not tested.
- **The fp8-kv8 trace fixed cost (15.4-16.0 ms) is ~1 ms above 7a8237d0's client fit
  (14.5-15.3).** Profiled steps run 2.5-3.8% slower on fp8-kv8, plus the host point above.
  The breakdown's shares are not affected.

## Next (not run, not proposed for spend yet)

- One fp8-kv8 window with `torch_profiler_with_stack: true` would name the blocking GPU-to-CPU
  copy. It is about 1 ms per step (~6% of a 16 ms step) and fixing it could give fp8-kv8 that
  back.
- Profile `fp8` (BF16 KV, FlashAttention) and `bf16-kv8` (FP8 KV, FlashInfer). That separates
  "FP8 weights" from "FP8 KV / FlashInfer" for both the CPU block and the attention efficiency.
- FP8 GEMMs at 57-72% of bandwidth at small batch: try vLLM's other FP8 linear kernels for sm89
  (e.g. Marlin FP8 weight-only for decode) and measure the per-layer µs the same way.
