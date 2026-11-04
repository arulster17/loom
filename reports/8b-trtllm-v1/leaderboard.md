# Loom leaderboard: cost at SLO

## Summary

What one replica costs us to serve at the SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%), at the on-demand list price including its storage, at the highest tested load that met the SLO; 95% confidence intervals in brackets. $/1M input and $/1M output split the replica's cost by measured prefill time (methodology at the end); $/1M blended is the replica's cost over all tokens at that workload's own input:output mix and needs no split. Each workload quotes one config: rank 1, ranked among configs that passed every check (a candidate ranks only on a quality gate pass against its reference). With nothing ranked, an untrusted config of verified quality is quoted as a comparison point with its reason; a config whose gate failed or is inconclusive is never quoted. Nothing here relaxes a check, and the ranking is the leaderboard's.

### Qwen/Qwen3-8B

Hardware: 1×L40S (aws g6e.xlarge, $1.8829/h on-demand incl. 200 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | **vllm-bf16** (quoted) | rank 1 | 2.25 req/s (fails at 2.367) | $0.0952 [0.0783, 0.1158] | $0.4556 [0.3688, 0.5586] | $0.1859 [0.1650, 0.2092] | baseline |
| chat-sharegpt | trtllm-bf16 | quality gate failed | 1.355 req/s (fails at 1.5) | $0.0849 [0.0529, 0.1362] | $1.0016 [0.7610, 1.2921] | $0.3118 [0.2504, 0.3882] | -0.138 (tool_calling_strict) · fail |
| fixed-1k-1k | **vllm-bf16** (quoted) | rank 1 | 1 req/s (fails at 1.107) | $0.0968 [0.0869, 0.1077] | $0.3932 [0.3705, 0.4166] | $0.2450 [0.2347, 0.2557] | baseline |
| fixed-1k-1k | trtllm-bf16 | quality gate failed | 0.6667 req/s (fails at 0.7378) | $0.1060 [0.0688, 0.1635] | $0.6616 [0.4819, 0.9051] | $0.3838 [0.2865, 0.5143] | -0.138 (tool_calling_strict) · fail |

**chat-sharegpt** (935 input / 314 output tokens per request)

- vllm-bf16: $0.0952 [0.0783, 0.1158] per 1M input tokens, $0.4556 [0.3688, 0.5586] per 1M output tokens, $0.1859 [0.1650, 0.2092] per 1M tokens blended at this mix, holding the SLO up to 2.25 req/s (fails at 2.367).
- What limits vllm-bf16: at 2.367 req/s, TPOT p95 is 45.1 [39.8, 51.1] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Caveat: trtllm-bf16 is untrusted, so the leaderboard does not rank it: ttft_ms.p95 at goodput: run-to-run CV 13.9% exceeds 10%. Treat its figure as indicative and rerun before relying on it.
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.93×); OpenRouter $0.2020 (aggregator, our cost 0.92×). Our blended cost is $0.1859; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**fixed-1k-1k** (1,024 input / 1,024 output tokens per request)

- vllm-bf16: $0.0968 [0.0869, 0.1077] per 1M input tokens, $0.3932 [0.3705, 0.4166] per 1M output tokens, $0.2450 [0.2347, 0.2557] per 1M tokens blended at this mix, holding the SLO up to 1 req/s (fails at 1.107).
- What limits vllm-bf16: at 1.107 req/s, TPOT p95 is 48.2 [41.9, 55.5] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Caveat: trtllm-bf16 is untrusted, so the leaderboard does not rank it: ttft_ms.p95 at goodput: run-to-run CV 13.1% exceeds 10%; throughput.request_rate at goodput: run-to-run CV 11.8% exceeds 10%; throughput.output_tok_s at goodput: run-to-run CV 11.8% exceeds 10%. Treat its figure as indicative and rerun before relying on it.
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 1.22×); OpenRouter $0.2860 (aggregator, our cost 0.86×). Our blended cost is $0.2450; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**Quality**

- vllm-bf16 (BF16 weights, KV cache auto (model dtype)): the reference the quality gate compares against; gsm8k 0.902, ifeval 0.826, json_schema 0.910, tool_calling 0.978, tool_calling_strict 0.978.
- trtllm-bf16 (BF16 weights, KV cache auto (model dtype)): quality gate vs BF16 (AWS g6e.xlarge) is fail: gsm8k, tool_calling, json_schema, divergence, sanity pass; ifeval inconclusive (delta -0.55 pts [-2.16 pts, +1.05 pts], n=541: CI crosses -2.00 pts, more samples needed); tool_calling_strict fail (delta -13.83 pts [-19.75 pts, -8.15 pts], n=135 is a drop of more than 6.00 pts). It is not ranked.

### nvidia/Qwen3-8B-FP8

Hardware: 1×L40S (aws g6e.xlarge, $1.8829/h on-demand incl. 200 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | trtllm-fp8 | quality gate failed | 2.49 req/s (fails at 2.619) | $0.0559 [0.0450, 0.0693] | $0.4895 [0.4539, 0.5250] | $0.1654 [0.1516, 0.1803] | -0.114 (tool_calling_strict) · fail |
| fixed-1k-1k | trtllm-fp8 | quality gate failed | 1 req/s (fails at 1.107) | $0.0510 [0.0464, 0.0560] | $0.4389 [0.4180, 0.4608] | $0.2450 [0.2347, 0.2557] | -0.114 (tool_calling_strict) · fail |

**chat-sharegpt** (957 input / 323 output tokens per request)

- No cost at the declared SLO: no config with a cost at SLO has verified quality (trtllm-fp8: failed the quality gate vs BF16 (AWS g6e.xlarge)); a candidate ranks only on a gate pass.
- What limits trtllm-fp8: at 2.619 req/s, TPOT p95 is 29.2 [9.8, 86.5] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).

**fixed-1k-1k** (1,024 input / 1,024 output tokens per request)

- No cost at the declared SLO: no config with a cost at SLO has verified quality (trtllm-fp8: failed the quality gate vs BF16 (AWS g6e.xlarge)); a candidate ranks only on a gate pass.
- What limits trtllm-fp8: at 1.107 req/s, TPOT p95 is 35.4 [11.2, 112.2] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).

**Quality**

- trtllm-fp8 (FP8 weights, KV cache FP8): quality gate vs BF16 (AWS g6e.xlarge) is fail: gsm8k, tool_calling, json_schema, divergence, sanity pass; ifeval inconclusive (delta -0.68 pts [-2.53 pts, +1.17 pts], n=541: CI crosses -2.00 pts, more samples needed); tool_calling_strict fail (delta -11.36 pts [-17.04 pts, -6.17 pts], n=135 is a drop of more than 6.00 pts). It is not ranked.

## Leaderboards

Each table lists one model on one workload. $/1M input and $/1M output split the replica's cost by measured prefill time; $/1M blended is its cost over all tokens at that workload's own input:output mix (methodology below). Rows are ranked by the rank key, the cost under the experiments' declared allocation (all_output charges the whole replica to output tokens), at the on-demand list price, cheapest first; spot, committed-1y and as-run costs of the rank key are shown where available. Every price includes the replica's storage volume. Values are point estimates (geometric means for latency and throughput) with 95% confidence intervals in brackets. Goodput is the highest tested load that met the SLO; raw peak throughput ignores the SLO and is not goodput. Only configs that passed every check are ranked: a candidate needs a quality gate pass against its reference. Unranked rows (quality gate failed or not verified, untrusted, or no cost) are listed last with the reason.

Goodput is searched on a grid of loads, so it is known only to a bracket: at least the goodput load, below the load that failed (shown as "fails at"). Configs whose brackets overlap are tied within the search resolution: their goodput, throughput and cost at SLO come from the same grid point and are not a measured equality.

### Qwen/Qwen3-8B: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, spot | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **vllm-bf16**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | $0.0952 [0.0783, 0.1158] | $0.4556 [0.3688, 0.5586] | $0.1859 [0.1650, 0.2092] | $0.7391 [0.6346, 0.8608] | n/a | $0.7391 [0.6346, 0.8608] | 707.7 [607.6, 824.2] | 707.7 [607.6, 824.2] | 2.25 req/s (fails at 2.367) | 310 [298, 323] ms | 40.2 [36.2, 44.7] ms | 1,085.4 [947.9, 1,242.8] at 3.375 req/s | baseline | 724 s (median of 1) | ranked | Only ranked config at SLO; quality baseline |
| – | **trtllm-bf16**<br>trtllm 1.4.0rc2 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | $0.0849 [0.0529, 0.1362] | $1.0016 [0.7610, 1.2921] | $0.3118 [0.2504, 0.3882] | $1.2597 [1.0316, 1.5383] | $1.2448 [1.0193, 1.5200] | $1.2597 [1.0316, 1.5383] | 415.2 [340.0, 507.0] | 415.2 [340.0, 507.0] | 1.355 req/s (fails at 1.5) | 284 [201, 399] ms | 35.7 [28.1, 45.3] ms | 707.7 [607.6, 824.2] at 2.25 req/s | -0.138 (tool_calling_strict) · fail | 1,025 s (median of 2) | quality gate failed | Not ranked: failed the quality gate vs BF16 (AWS g6e.xlarge) (worst: tool_calling_strict -0.138) |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | vllm-bf16 | trtllm-bf16 | Verdict |
|---|---|---|---|---|
| 2.25 req/s | SLO at this load | met | failed |  |
| 2.25 req/s | TTFT p50 | 164 [138, 196] ms | 171 [153, 191] ms | no significant difference |
| 2.25 req/s | TTFT p95 | 310 [298, 323] ms | 330 [310, 352] ms | no significant difference |
| 2.25 req/s | TPOT p50 | **34.8 [33.6, 36.0] ms** | 38.8 [37.0, 40.7] ms | vllm-bf16 lower |
| 2.25 req/s | TPOT p95 | 40.2 [36.2, 44.7] ms | 46.2 [42.2, 50.7] ms | no significant difference |
| 2.25 req/s | E2E p95 | 24,326 [20,750, 28,519] ms | 27,051 [22,776, 32,129] ms | no significant difference |

**Warnings**

- trtllm-bf16: ttft_ms.p95 at goodput: run-to-run CV 13.9% exceeds 10%

### Qwen/Qwen3-8B: fixed-1k-1k (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, spot | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **vllm-bf16**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | $0.0968 [0.0869, 0.1077] | $0.3932 [0.3705, 0.4166] | $0.2450 [0.2347, 0.2557] | $0.4900 [0.4695, 0.5114] | n/a | $0.4900 [0.4695, 0.5114] | 1,067.5 [1,022.8, 1,114.1] | 1,067.5 [1,022.8, 1,114.1] | 1 req/s (fails at 1.107) | 255 [240, 271] ms | 44.7 [44.3, 45.1] ms | 1,524.3 [1,428.8, 1,626.2] at 1.5 req/s | baseline | 724 s (median of 1) | ranked | Only ranked config at SLO; quality baseline |
| – | **trtllm-bf16**<br>trtllm 1.4.0rc2 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | $0.1060 [0.0688, 0.1635] | $0.6616 [0.4819, 0.9051] | $0.3838 [0.2865, 0.5143] | $0.7676 [0.5729, 1.0286] | $0.7585 [0.5661, 1.0163] | $0.7676 [0.5729, 1.0286] | 681.3 [508.5, 912.9] | 681.3 [508.5, 912.9] | 0.6667 req/s (fails at 0.7378) | 379 [274, 524] ms | 38.7 [32.8, 45.5] ms | 1,067.5 [1,022.8, 1,114.1] at 1 req/s | -0.138 (tool_calling_strict) · fail | 1,025 s (median of 2) | quality gate failed | Not ranked: failed the quality gate vs BF16 (AWS g6e.xlarge) (worst: tool_calling_strict -0.138) |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | vllm-bf16 | trtllm-bf16 | Verdict |
|---|---|---|---|---|
| 1 req/s | SLO at this load | met | failed |  |
| 1 req/s | TTFT p50 | 184 [180, 188] ms | 245 [135, 443] ms | no significant difference |
| 1 req/s | TTFT p95 | 255 [240, 271] ms | 408 [133, 1,257] ms | no significant difference |
| 1 req/s | TPOT p50 | 42.7 [41.5, 43.9] ms | 53.4 [41.2, 69.2] ms | no significant difference |
| 1 req/s | TPOT p95 | 44.7 [44.3, 45.1] ms | 57.8 [39.7, 84.2] ms | no significant difference |
| 1 req/s | E2E p95 | 45,897 [45,510, 46,287] ms | 59,453 [40,815, 86,603] ms | no significant difference |

**Warnings**

- trtllm-bf16: ttft_ms.p95 at goodput: run-to-run CV 13.1% exceeds 10%
- trtllm-bf16: throughput.request_rate at goodput: run-to-run CV 11.8% exceeds 10%
- trtllm-bf16: throughput.output_tok_s at goodput: run-to-run CV 11.8% exceeds 10%

### nvidia/Qwen3-8B-FP8: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **trtllm-fp8**<br>trtllm 1.4.0rc2 · FP8 weights, KV cache FP8 · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | $0.0559 [0.0450, 0.0693] | $0.4895 [0.4539, 0.5250] | $0.1654 [0.1516, 0.1803] | $0.6548 [0.6321, 0.6783] | $0.6548 [0.6321, 0.6783] | 798.8 [771.1, 827.4] | 798.8 [771.1, 827.4] | 2.49 req/s (fails at 2.619) | 164 [135, 200] ms | 21.3 [19.6, 23.2] ms | 1,085.4 [947.9, 1,242.8] at 3.375 req/s | -0.114 (tool_calling_strict) · fail | 956 s (median of 1) | quality gate failed | Not ranked: failed the quality gate vs BF16 (AWS g6e.xlarge) (worst: tool_calling_strict -0.114) |

### nvidia/Qwen3-8B-FP8: fixed-1k-1k (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **trtllm-fp8**<br>trtllm 1.4.0rc2 · FP8 weights, KV cache FP8 · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | $0.0510 [0.0464, 0.0560] | $0.4389 [0.4180, 0.4608] | $0.2450 [0.2347, 0.2557] | $0.4900 [0.4695, 0.5114] | $0.4900 [0.4695, 0.5114] | 1,067.5 [1,022.8, 1,114.1] | 1,067.5 [1,022.8, 1,114.1] | 1 req/s (fails at 1.107) | 125 [119, 131] ms | 20.1 [19.5, 20.8] ms | 1,524.3 [1,428.8, 1,626.2] at 1.5 req/s | -0.114 (tool_calling_strict) · fail | 956 s (median of 1) | quality gate failed | Not ranked: failed the quality gate vs BF16 (AWS g6e.xlarge) (worst: tool_calling_strict -0.114) |

## Competitiveness: our cost at SLO vs public list prices

> Public list prices only: competitor numbers are the per-token prices each provider publishes on its pricing page, with the source and the date it was checked. No competitor endpoint was called or benchmarked, and list prices say nothing about a provider's latency, quality or quantization unless the page discloses it.

Competitor prices last checked 2026-10-04. Flags leave out aggregators and entries whose availability is unverified; they are listed for reference. Our cost is the quoted config's cost at SLO (rank 1; with nothing ranked, an untrusted config of verified quality as a comparison point, with its reason) at the on-demand list price, storage included: $/1M input and $/1M output split the replica's cost by measured prefill time, and $/1M blended is its cost over all tokens at the workload's own input:output mix. Each public price is also shown at every workload's mix, so blended compares like with like. With no price set, the break-even price is the lowest price that covers our cost: at the point estimate, and at the cost CI high bound.

### Qwen3 8B (`qwen3-8b`), ours unquantized

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | vllm-bf16 (rank 1) | 935 / 314 | $0.0952 [0.0783, 0.1158] | $0.4556 [0.3688, 0.5586] | $0.1859 [0.1650, 0.2092] | $0.0770 | $0.3680 | $0.0952 (CI high $0.1158) | $0.4556 (CI high $0.5586) | -$0.0182 (-24% of price); -$0.0388 at cost CI high | -$0.0876 (-24% of price); -$0.1906 at cost CI high |
| fixed-1k-1k (open loop) | vllm-bf16 (rank 1) | 1,024 / 1,024 | $0.0968 [0.0869, 0.1077] | $0.3932 [0.3705, 0.4166] | $0.2450 [0.2347, 0.2557] | $0.0770 | $0.3680 | $0.0968 (CI high $0.1077) | $0.3932 (CI high $0.4166) | -$0.0198 (-26% of price); -$0.0307 at cost CI high | -$0.0252 (-6.8% of price); -$0.0486 at cost CI high |

**Public list prices** (per 1M tokens)

| Provider | Provider model | $/1M in | $/1M out | Precision | At chat-sharegpt mix (935 / 314) | At fixed-1k-1k mix (1,024 / 1,024) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | $0.2000 (our cost 0.93×) | $0.2000 (our cost 1.22×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | $0.2020 (our cost 0.92×) | $0.2860 (our cost 0.86×) | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_public_comparison`: qwen3-8b: no eligible public list price
- [chat-sharegpt (open loop)] `negative_margin`: qwen3-8b input: price $0.0770/1M is $0.0182 below measured cost $0.0952/1M
- [chat-sharegpt (open loop)] `negative_margin`: qwen3-8b output: price $0.3680/1M is $0.0876 below measured cost $0.4556/1M
- [fixed-1k-1k (open loop)] `negative_margin`: qwen3-8b input: price $0.0770/1M is $0.0198 below measured cost $0.0968/1M
- [fixed-1k-1k (open loop)] `negative_margin`: qwen3-8b output: price $0.3680/1M is $0.0252 below measured cost $0.3932/1M

### Qwen3 8B FP8 (ModelOpt, FP8 KV) (`qwen3-8b-fp8-modelopt`), ours not recorded

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | no config with a cost at SLO has verified quality (trtllm-fp8: failed the quality gate vs BF16 (AWS g6e.xlarge)); a candidate ranks only on a gate pass | n/a | n/a | n/a | n/a | no price set | no price set | n/a | n/a | n/a | n/a |
| fixed-1k-1k (open loop) | no config with a cost at SLO has verified quality (trtllm-fp8: failed the quality gate vs BF16 (AWS g6e.xlarge)); a candidate ranks only on a gate pass | n/a | n/a | n/a | n/a | no price set | no price set | n/a | n/a | n/a | n/a |

**Public list prices** (per 1M tokens)

Public list prices are those of qwen3-8b, the model this entry serves at another precision: providers price the model, and the precision column says which listings match ours.

| Provider | Provider model | $/1M in | $/1M out | Precision | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_price_set`: qwen3-8b-fp8-modelopt: no price set
- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_cost_measurement`: qwen3-8b-fp8-modelopt: no measured cost at SLO
- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_public_comparison`: qwen3-8b-fp8-modelopt: no eligible public list price

### Providers with no public per-token price

- Groq: Enterprise plans only; contact sales. (checked 2026-10-04)
- Cerebras: No public per-token price. (checked 2026-10-04)
- Lambda: Inference offering winding down. (checked 2026-10-04)
- Hyperbolic: Inference offering retired. (checked 2026-10-04)
- Nebius: Neither model is in its catalog. (checked 2026-10-04)

## Methodology and provenance

- **SLO:** TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%
- **Load generation:** open loop (requests arrive on a fixed schedule at a set rate, req/s)
- **Content:** realistic, synthetic
- **Dataset:** ShareGPT_Vicuna_unfiltered (ShareGPT_V3_unfiltered_cleaned_split.json) @ 192ab2185289094fc556ec8ce5ce1e8e587154ca (license: HF card: apache-2.0. The conversations were scraped from sharegpt.com and contain ChatGPT outputs, and the card's license does not settle OpenAI's terms of use. Loom uses them only as request shapes for load (prompt text in, lengths out), never ships or trains on them, and publishes only latency and cost.); source: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- **Repetitions:** 3 per load point
- **Confidence intervals:** Repetitions of a load point are combined per metric, with a two-sided 95% Student-t interval on the scale that fits the metric. Latencies, throughputs and rates are strictly positive and right-skewed: the value shown is the geometric mean and the interval is the t-interval of ln(value), exponentiated (geometric mean ×/÷ a factor), so both bounds are positive. Proportions such as error rate and SLO attainment use the arithmetic mean of the per-run proportions with the t-interval clipped to [0, 1]; it measures run-to-run variation, not binomial sampling, so it is [0, 0] when no repetition saw an error. Counts use the arithmetic t-interval clipped at 0, as does a positive metric when a repetition measured 0. A load meets the SLO only if the CI upper bound of every target is within it; goodput is the highest passing load below the first failing one. Cost CI bounds come from the goodput throughput CI (low cost from high throughput), so they are finite whenever the throughput lower bound is above zero. Results with a single repetition have no CI and are never trusted.
  - request counts and sample sizes: arithmetic mean, Student-t interval clipped at 0 (t_clipped)
  - latency percentiles and means (ms): geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - throughput and goodput (tok/s, req/s): geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - requests in prefill (request rate × mean TTFT): geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - tokens per decode step (speculative decoding) and per streamed chunk: geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - error rate, SLO attainment, cache fractions and hit rates: arithmetic mean of per-run proportions, Student-t interval clipped to [0, 1] (t_clipped)
  - server queue time, measurement window: geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - GPU utilization (%): arithmetic mean, Student-t interval clipped to [0, 100] (t_clipped)
  - other server and GPU gauges: arithmetic mean, Student-t interval clipped at 0 (t_clipped)
  - anything else: arithmetic mean, Student-t interval (t)
- **Cost allocation (ranking):** all_output: all of the replica's hourly cost is charged to output tokens, so input tokens have no separate price ($/1M input is n/a, not $0); its $/1M output is the rank key, while the published $/1M input and output prices use the measured prefill-time split
- **Input/output split (headline):** Headline $/1M input and $/1M output use the prefill_time split, measured at the goodput point: input tokens pay for the share φ of the replica's time spent prefilling prompts, output tokens for the rest (decode steps, and any idle headroom the SLO needs). φ is the mean number of requests in their prefill phase, request rate × mean TTFT (Little's law), per repetition, combined like a throughput (geometric mean, log-t CI) and capped at 1. Input = φ × hourly price / input tok/s; output = (1 − φ) × hourly price / output tok/s; at the measured mix they bill exactly the replica's hourly price. TTFT includes queueing and the first decode step and concurrent prefills are each counted, so φ tends to overstate prefill time (input high, output low); idle headroom is charged to output (output high). The price CIs combine the φ and throughput CIs at the ends that push each price the same way, an outer bound. When φ is high (prefill-heavy traffic, e.g. long shared prompts or chat at high load) its CI can reach 1, the cap: the measurements then cannot rule out that prefill takes all of the replica's time, so the output price's low end is $0 and is marked as such. That $0 is a property of the split, not a measured free token; the point estimate and the high end are unaffected. Blended $/1M is the hourly price over all tokens at the workload's own input:output mix and needs no split. The ranking still uses the declared cost allocation, shown in its own column.
- **Price book last checked:** 2026-11-03

### vllm-bf16 · chat-sharegpt, open loop (`48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-east-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-east-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 15; provenance digests: `289e8c546a9ab01f33a561f943b0dd9461839478102a32770319bb5307af87de`, `3f450771c839b3bf59e54c0a68004cdc75578641a1d67d85712d605d47994e77`, `4c6cfa3ec3c455370daef59e613891550fe3da156385d933bea10dbaf8c7255b`, `562cf6d2729121e324128673dc452f4bd90b3ca027cb810662817bb386f48177`, `91da15015a5adb08167c04ad7b3a3dc543bdd44d91a3e4ebb00c5277cda19ab6`, `92c5f44183c41f157af623628e096ca40ef203b718bc9901850407c7aa2360c4`, `c1e0706c4b4da1de0534949d99e7123378d517ce08c1323754fd2e155d5fd8f2`, `dadf895cc236da949cd70ebaa7dc8fd6a44e566cb41d222312b373c7b3ef94d4`, `de35d346734f4408095e53b6ae0a886593fd5865b2fd2fb238fe4905a1eb76b3`, `e11261cf1486a2af030243792409ab55a2689df5cdf29b986eb5068066281d81`, `e87de27bdd0851c566982a45c96d3f8dadbd45d15fc32dd262508b1b428d0067`, `edca6f29398529268269a0c658748c432c2822fd43fa531f68f9408c2202f055`, `f1e1e3d722f4d6cab6babe27f253693bd189ea5d8c8c347484a545830e6f0155`, `f588d416352f2d897d21b1680fc5574a2fdc4d6ca415a534486759a513378667`, `f61b452441d1b6d84eccb0f0987c5de0bca8fa1ccec1383f4be52c007832311d`
- **Reproduce:** `bench reproduce 47743400-4a4e-4272-892c-f50c30041848`

### trtllm-bf16 · chat-sharegpt, open loop (`ae3b5705d2fa54e053eeadaf507ef46ed681a1fbcf79631dc7f65f4f5457fb8a`)

- **Config:** trtllm 1.4.0rc2 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** trtllm 1.4.0rc2; image `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`; digest `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-east-1 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-east-1 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot $1.8605/h; committed 1y n/a; source: https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json, https://instances.vantage.sh/aws/ec2/g6e.xlarge, https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json, https://aws.amazon.com/ebs/pricing/; last checked 2026-10-04; the spot price is an indicative average and moves hourly. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 15; provenance digests: `1ec1a6946ca4471b3d963e07db2a5e57e90e17741360fef59b6fe801f77b91a0`, `201a13d1a9a88887125de31e948ead750b89f269ef6aabec66d414cd8a52c2dc`, `2dc0433c8748b47c7f565e45d9d212e697b31c9e8d2025d302c40d44d067fc76`, `41529500fd131c63cbbf7f44860f7f54216d00dbf01e9004bd39653bc6f95621`, `6385a9c28e029c485cf66bebff88261a1fb1cc19f1bfc4a56894ee96f7fe1b96`, `92c23a0d67d3b6bedde32937ada22fbbb370e85f8871d4ffebab2321a2e1411a`, `942cc31639d0378235bfd4419a22747444ba3a6e3a302b40f4362a150e3c4ff1`, `9af7b0a3ccb1cf8d9af952e4958972267fdc174b1a2d7da21fb527990a444983`, `a03a7a7df4ebc5f4701c16b1de0f4265f09816b9982b51ed6ccfd70e31a3e3a8`, `aef7d962bdc54ee3b416f6aa7b1ddb47438b0d29430193d5e36100a6b0a013b3`, `cc7c89162ddefce9d84b443fcca62311ebee62a003e2c89d3af9e658b5fff26b`, `d1e558227b992c3f45a05fe9949fb67d871880967e3d3fd141e918f2e8cfacf1`, `d5af71d2db04d3e29eb53ffb1a9930be5971b348327bbc2a4652ac4f898b4d77`, `e0fa38ee19a20276980cc91cbfed25f3c44f48569ff60d02ce322cad2a7c12d6`, `fdfc0868141a4215c3fa0cb4ffac5827a50952b1af346bfc48d1596fb13df102`
- **Reproduce:** `bench reproduce 89858e6c-a39a-445d-8873-4acbd4b55a84`

### vllm-bf16 · fixed-1k-1k, open loop (`48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-east-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-east-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 12; provenance digests: `03a33eafd0a5ce5084d1d357fe9df555a367f7808a8b098cc421022853e3bcdc`, `17dba3e7f8e5773e5b4c3123b250b28fd3a383e7c17745e21d946b6ac8038c90`, `38e8db80ab4933604b4da6263f5e5fc297adb289f6cce32f317559af95140538`, `409afa3f5f8f5681700ff6e3815ce000d33406ddf3d5c6cf3e7dce0baf4d6759`, `7474a1880a7a70f7d0b3435ccb21d44c13b48b9e35b7bd7a6c6e5833d2cd0f4e`, `7d01e71443b5762d1c6b7245ce4febd1661dc55a8f8e0e1920a03c5a7de60428`, `8b18b97c8bddb4a34200e57a63e82b6a1f614b6713626acee95e00c5b1ff1682`, `9dbd2833778494484ef62488e06aa86d54e89b7401d4b414417579c8f5a4dd6e`, `9f80536c9dbf8fb5cd8f3c7bd3e7f34adbc439c619d89367cf9874e8c8dd1695`, `c5e82b2cfe3b9cb107ce47447be94240a3c59ea6771a1bf53fb193b308f2138d`, `ce84ce96805390bd8f34809c592efec472d94a0e5eb435446df01f9d30b0b0eb`, `d0898537a44ba2d9980d9337c075b82490651bea22d02f2e37cf4ad8a38feebe`
- **Reproduce:** `bench reproduce 4c8cb291-07e5-4801-a3ef-9bb504432640`

### trtllm-bf16 · fixed-1k-1k, open loop (`ae3b5705d2fa54e053eeadaf507ef46ed681a1fbcf79631dc7f65f4f5457fb8a`)

- **Config:** trtllm 1.4.0rc2 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** trtllm 1.4.0rc2; image `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`; digest `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-east-1 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-east-1 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot $1.8605/h; committed 1y n/a; source: https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json, https://instances.vantage.sh/aws/ec2/g6e.xlarge, https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json, https://aws.amazon.com/ebs/pricing/; last checked 2026-10-04; the spot price is an indicative average and moves hourly. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 12; provenance digests: `032c2cc249cc0f96c9717da0063622a74552dc9cb076211ef6ece6f68a16d4ce`, `12702de1e5152f4578aba0624bfa64d7466315d1c6e7fb773864dd0c14605e41`, `24de4b3cd4bc19e686418ea22be50ac73224febb969b6f5b3c5c9049fa0dcf09`, `252036597f9504a3a77b3caecb742eda0f2ec6162e7c9ac871784b4b06712cf8`, `25e3a2a160b6c93e3ee086fbead04b6a95e1465e3488bf44b4c7bebf88bcfb8d`, `68aac6eb006cfd002c2393ad94741c413c5ce1aedffcd7eed6085ad5bddb15a7`, `7ee8c9fb57000eb1c51d1b37943a40820efd23bf7420535be0ea97be0c30e16b`, `910765a54bb28f7b02e54554e11fc5228bcde9f8c083a93a7d0d4f963811001d`, `9c921ac7ebffd6670576919c9f94db69c8b01694af1adc450f3ad8edb3641551`, `b448c8e5634e9091a0a2631f997371774897b1217e73ced1d9a20796b3945073`, `d134e10ca04767493321d6d21ea9c3cd47dcdcc1c07be67ccb11fdae1c39901e`, `df420b0af24ff174babb697a587d13f97a251da50c7031a541c633a7fbc17d20`
- **Reproduce:** `bench reproduce e84e31d6-4340-4cfe-8b81-786d0aa9319e`

### trtllm-fp8 · chat-sharegpt, open loop (`55b356557da002ca376de348fdcdd39422699bc205e2f9016776877ca4509b80`)

- **Config:** trtllm 1.4.0rc2 · FP8 weights, KV cache FP8 · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** trtllm 1.4.0rc2; image `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`; digest `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`
- **Model:** nvidia/Qwen3-8B-FP8 @ 2cebc4c89e25abc17668c81b01dceaf3d8b914d5
- **Hardware:** 1×L40S · aws / us-east-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `21acec9aba86818891be7ec81eb17f891803eaec`; loom-bench 0.1.0
- **Price:** aws/us-east-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 15; provenance digests: `014c5a3defe15abda6dc7f2ed78d4808ac0b3038955c1c49523b6ea83d13b2b9`, `1455142f673068e5a2586f0fe5d3f980a07cccf18dd294868407c2fb080f7f53`, `1de23b5ca1eecc4242f165dfe649a3d5660fbe0101c67b83bdc74039ff2747ff`, `3af51beb48b4019b6936ef88709855e502d119286edecc77dd5e4e703b5eeade`, `7eebbd3b51143d1540c7064ecce6611d80b3a52dd202753389cd3811424af077`, `8165819b998d56d047d7b7f044d4d57825862a16ee164d570f0fc7f2165aba0d`, `8a161263b531de89b59b8f9dd257759db5bf4bc7860ceae1d354d0a3c52330af`, `8b1599b23429b64926b1841f82bc626afe9df745ec1ade076693517bc34bfde8`, `9b2a4c00da9b5c0b62d81d895701108934c12c5a3ab637597a674334a326bf45`, `bad196a62e9490f3ca5ed6f758c7336cc609f10404960c80d037a21b49d21750`, `cea3a737b08fa43fb2a675caefda455cdb39f69462ad5c2b5d6a15fe59bb4f16`, `da8c95ee072c356080c10882745a6d513d1910696df2b4ed8840e1245f5bba4c`, `ea4629f627d444a4f784fbd1ebcebe248cb428484e757242e7ea03eacf490b98`, `f06080820229f75a07dab3e9c142a8b1142296326ddfe75ae14158a06fd208af`, `fb42e881d6703a8085b7e696958e9cd35db78db8891279cf90258e0e66ff9ca0`
- **Reproduce:** `bench reproduce e1eaad3f-7c38-460e-9a1b-e2364941ca14`

### trtllm-fp8 · fixed-1k-1k, open loop (`55b356557da002ca376de348fdcdd39422699bc205e2f9016776877ca4509b80`)

- **Config:** trtllm 1.4.0rc2 · FP8 weights, KV cache FP8 · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** trtllm 1.4.0rc2; image `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`; digest `nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`
- **Model:** nvidia/Qwen3-8B-FP8 @ 2cebc4c89e25abc17668c81b01dceaf3d8b914d5
- **Hardware:** 1×L40S · aws / us-east-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `21acec9aba86818891be7ec81eb17f891803eaec`; loom-bench 0.1.0
- **Price:** aws/us-east-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 12; provenance digests: `0e5e259ed2b86a397e10f50214fecfb4264834e198b59080b3f23bd048984c2f`, `1d0a6d53feb73e9d641ae918eaebfc0890d751ad9d8c0a02e42889f897f21d47`, `87ee3ea59084089ad8b5381426946aae0c42b5fd1c6468a723b6dfc66706694e`, `8f4ae30f53df21ce367c318b6968d95ae51ce7b1711c7f6f8e24741219d99c34`, `925d94d0a5787b24cb5eaf6848c8fb1f5f6034376b092b673bc21c2567f8ede7`, `93fbee56a70ae4e14016cee7bf0053ce84c08d32acdb32716f69e3ac80446f2c`, `96fbc96464133fe8828ebb039fa9cb5d55d6a22ec4a53c47dc9e2d3ce55cb378`, `9e700926ade2b04592b06752d4fa77351431cc0ac29c5015fc302f6e07a03bed`, `aead6dee67ab8a14eea1bc9b2b10192f301ba5e538e5249e8caa88f109c9ff89`, `c7e77b1857b33761ff63d7f03d22e2ceca1c88655a140a85e6b518a90e982eca`, `ec441f738bce4cd31deb0ec9c8a8a8a75faa375c4f823479ec9315e66512b6f3`, `f55364fd5fd36c8747b05ea83a360e842306eed37f5baf9ed2fdda1d8c972466`
- **Reproduce:** `bench reproduce 1818a642-b4e2-4048-a441-bd81185a1d88`
