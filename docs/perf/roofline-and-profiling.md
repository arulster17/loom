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
floor that is a bigger share than on a 24 ms one. That is the next thing to profile.

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

1. **FP8 weights deliver their full byte saving.** The fixed part drops by 8-10 ms; the
   weight floor drops by 8.0 ms.
2. **Every config pays 3-6 ms per step on one GPU beyond the weight read** (bf16 +4.4 to
   +6.3, fp8-kv8 +5.0 to +5.8, H100 +3.2 to +4.9 after the modelled all-reduce). That
   overhead does not scale with bytes. Candidates: GEMMs below peak bandwidth, small
   kernels (norms, activations, rotary, FP8 activation quantization: four per layer),
   sampling, and the GPU idling while the CPU schedules the next step. This is what the
   profiling run measures.
3. **An FP8 KV cache halves the bytes but not the time.** Per request, FP8 KV costs
   0.19-0.21 ms against BF16's 0.30-0.35: 1.6-1.7x cheaper, not 2x. Attention over an
   FP8 cache on Ada runs further from peak bandwidth (1.5-1.9x its floor vs 1.3-1.5x),
   consistent with the dequantization work, but the trace has to confirm it.
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
   `POST /stop_profile`, which writes the trace. A trace of 200 steps is a few MB.
3. On RunPod a root script archives the trace directory and uploads it through the same
   presigned-S3 path the jobs use (`collect_profile.sh`); the runner downloads it to
   `results/<experiment>/profiles/<run id>/trace.tar.gz`.
4. `bench profile analyze` (run automatically, also by hand on any trace) puts every GPU
   kernel in a category by name: GEMM, attention, all-reduce/NCCL, quantize/dequantize,
   sampling, memory copies, other kernels; time with no kernel running is **GPU idle**
   (the CPU scheduling the next step). Overlapping kernels count once, so only exposed
   communication counts as communication. It reports per-step milliseconds and the
   top kernels, in `window.json` and a `profile` event.

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

### The proposed run (not approved, not run)

`bench/experiments/qwen3-8b-profile-runpod.yaml`, after its smoke
`bench/experiments/runpod-smoke-profile.yaml`: bf16 and fp8-kv8 on one RunPod Secure
1x L40S pod; fixed-1k-1k closed loop at 1, 16 and 64 concurrent requests (known batch
sizes: the step-time-vs-batch line from the trace's side) and chat at 2.213 and 5.239
req/s (each config's goodput in 7a8237d0); two measured repetitions per point and one
profile window each (200 steps).

| Spec | Pod time (plan) | Estimate | Worst case (pod to TTL) | Cap |
|---|---|---|---|---|
| runpod-smoke-profile | 38.6 min of a 55 min TTL | $0.71 | $1.01 | $1.05 |
| qwen3-8b-profile-runpod | 2.23 h of a 2.67 h TTL | $2.45 | $2.94 | $3.00 |
| Total | | $3.16 | $3.95 | $4.05 |

What it answers: the per-step split of fixed overhead (GEMM below bandwidth vs small
kernels vs GPU idle), whether FP8-KV attention is the per-request excess, and the
prefill share at each config's chat goodput. Both variants add `--profiler-config`, so
their config hashes differ from 7a8237d0's; `bench compare <id> 7a8237d0 --match-by
workload` at the shared chat rates checks that an armed but idle profiler costs nothing.

#### On AWS instead (2026-11-02, proposed, not approved, not run)

RunPod Secure 1x L40S was out of stock all of 2026-11-02, so the run has AWS twins on one
g6e.xlarge on-demand host (the same L40S 48 GB; 4 vCPU and 32 GiB against the pod's 16 and
94; $1.861/h plus 200 GB gp3, accrued at $1.8829/h). The account's on-demand G/VT quota is
4 vCPU, exactly one such host, so each spec uses one host, with fp8-kv8 through a warm
restart. Same variants, workloads, windows, repetitions and SLO as the RunPod specs; only
the provider block and budget differ (`bench/tests/aws/test_aws_perf_track.py`).

On AWS the engine runs in Docker, so a profiled vLLM container bind-mounts its
`torch_profiler_dir` at the same host path and `aws_scripts/collect_profile.sh` archives
and uploads it over SSM (new code; the smoke runs it first).

| Spec | Host time (plan) | Estimate | Worst case (host to TTL) | Cap |
|---|---|---|---|---|
| aws-smoke-profile | 42.6 min of a 55 min TTL | $1.34 | $1.73 | $1.75 |
| qwen3-8b-profile-aws | 2.29 h of a 2.75 h TTL | $4.32 | $5.18 | $5.25 |
| Total | | $5.66 | $6.91 | $7.00 |
