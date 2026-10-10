# Loom leaderboard: cost at SLO

## Summary

What one replica costs us to serve at the SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%), at the on-demand list price including its storage, at the highest tested load that met the SLO; 95% confidence intervals in brackets. $/1M input and $/1M output split the replica's cost by measured prefill time (methodology at the end); $/1M blended is the replica's cost over all tokens at that workload's own input:output mix and needs no split. Each workload quotes one config: trusted first, then quality verified (the gate's reference or a gate pass), then leaderboard rank; a config that failed the quality gate is never quoted. Nothing here relaxes a check: untrusted figures are quoted only with their reason, and the ranking is the leaderboard's.

### Qwen/Qwen3-8B

Hardware: 1×L40S (runpod l40s-x1, $1.1010/h on-demand incl. 80 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| fixed-1k-1k | sglang | rank 1 | 1 req/s (fails at 1.091) | $0.0508 [0.0342, 0.0756] | $0.2458 [0.1926, 0.3106] | $0.1483 [0.1218, 0.1806] | -0.008 (ifeval) · inconclusive |
| fixed-1k-1k | **vllm** (quoted) | rank 2 | 1 req/s (fails at 1.091) | $0.0639 [0.0402, 0.1015] | $0.2328 [0.1752, 0.3016] | $0.1483 [0.1218, 0.1806] | baseline |
| code-completion | **vllm** (quoted) | untrusted, not ranked | 1.297 req/s (fails at 1.414) | $0.0378 [0.0171, 0.0835] | $3.6321 [2.1004, 5.8063] | $0.1468 [0.1002, 0.2151] | baseline |
| code-completion | sglang | untrusted, not ranked | 1.091 req/s (fails at 1.189) | $0.0378 [0.0202, 0.0710] | $4.7465 [3.1472, 6.8942] | $0.1787 [0.1333, 0.2397] | -0.008 (ifeval) · inconclusive |
| shared-prefix | **sglang** (quoted) | rank 1 | 2 req/s (fails at 2.378) | $0.0286 [0.0180, 0.0455] | $0.7295 [0.4826, 1.0104] | $0.0695 [0.0580, 0.0832] | -0.008 (ifeval) · inconclusive |
| shared-prefix | vllm | untrusted, not ranked | 2 req/s (fails at 2.378) | $0.0276 [0.0172, 0.0444] | $0.7447 [0.4954, 1.0274] | $0.0695 [0.0580, 0.0832] | baseline |

**fixed-1k-1k** (1,024 input / 1,024 output tokens per request)

- vllm: $0.0639 [0.0402, 0.1015] per 1M input tokens, $0.2328 [0.1752, 0.3016] per 1M output tokens, $0.1483 [0.1218, 0.1806] per 1M tokens blended at this mix, holding the SLO up to 1 req/s (fails at 1.091). sglang ranks first but its quality is not verified (-0.008 (ifeval) · inconclusive), so the figure quoted is vllm's (baseline, rank 2).
- vllm and sglang are tied: their goodput brackets overlap, so the load search cannot separate them on cost; the latency at equal load table compares them.
- What limits vllm: at 1.091 req/s, TPOT p95 is 47.7 [43.3, 52.4] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.74×); OpenRouter $0.2860 (aggregator, our cost 0.52×). Our blended cost is $0.1483; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**code-completion** (1,537 input / 48 output tokens per request)

- vllm: $0.0378 [0.0171, 0.0835] per 1M input tokens, $3.6321 [2.1004, 5.8063] per 1M output tokens, $0.1468 [0.1002, 0.2151] per 1M tokens blended at this mix, holding the SLO up to 1.297 req/s (fails at 1.414). No config on this board is trusted, so this figure is indicative only.
- What limits vllm: at 1.414 req/s, TPOT p95 is 38.7 [28.6, 52.4] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Caveat: vllm is untrusted, so the leaderboard does not rank it: throughput.request_rate at goodput: run-to-run CV 14.4% exceeds 10%; throughput.output_tok_s at goodput: run-to-run CV 14.7% exceeds 10%. Treat its figure as indicative and rerun before relying on it.
- Caveat: sglang is untrusted, so the leaderboard does not rank it: throughput.request_rate at goodput: run-to-run CV 12.0% exceeds 10%; throughput.output_tok_s at goodput: run-to-run CV 12.2% exceeds 10%. Treat its figure as indicative and rerun before relying on it.
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.73×); OpenRouter $0.1272 (aggregator, our cost 1.15×). Our blended cost is $0.1468; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**shared-prefix** (2,065 input / 128 output tokens per request)

- sglang: $0.0286 [0.0180, 0.0455] per 1M input tokens, $0.7295 [0.4826, 1.0104] per 1M output tokens, $0.0695 [0.0580, 0.0832] per 1M tokens blended at this mix, holding the SLO up to 2 req/s (fails at 2.378). Its quality is not verified against the reference (-0.008 (ifeval) · inconclusive); see Quality below.
- sglang and vllm are tied: their goodput brackets overlap, so the load search cannot separate them on cost; the latency at equal load table compares them.
- What limits sglang: at 2.378 req/s, TPOT p95 is 49.2 [34.4, 70.4] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Caveat: vllm is untrusted, so the leaderboard does not rank it: ttft_ms.p95 at goodput: run-to-run CV 11.0% exceeds 10%. Treat its figure as indicative and rerun before relying on it.
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.35×); OpenRouter $0.1367 (aggregator, our cost 0.51×). Our blended cost is $0.0695; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**Quality**

- sglang (unquantized): quality gate vs vllm is inconclusive: gsm8k, tool_calling, json_schema, divergence, sanity pass; ifeval inconclusive (delta -0.80 pts [-2.03 pts, +0.37 pts], n=541: CI crosses -2.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match vllm.
- vllm (unquantized): the reference the quality gate compares against; gsm8k 0.903, ifeval 0.818, json_schema 0.911, tool_calling 0.967.

## Leaderboards

Each table lists one model on one workload. $/1M input and $/1M output split the replica's cost by measured prefill time; $/1M blended is its cost over all tokens at that workload's own input:output mix (methodology below). Rows are ranked by the rank key, the cost under the experiments' declared allocation (all_output charges the whole replica to output tokens), at the on-demand list price, cheapest first; spot, committed-1y and as-run costs of the rank key are shown where available. Every price includes the replica's block storage. Values are point estimates (geometric means for latency and throughput) with 95% confidence intervals in brackets. Goodput is the highest tested load that met the SLO; raw peak throughput ignores the SLO and is not goodput. Unranked rows (quality gate failed, untrusted, or no cost) are listed last.

Goodput is searched on a grid of loads, so it is known only to a bracket: at least the goodput load, below the load that failed (shown as "fails at"). Configs whose brackets overlap are tied within the search resolution: their goodput, throughput and cost at SLO come from the same grid point and are not a measured equality.

### Qwen/Qwen3-8B: fixed-1k-1k (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **sglang**<br>sglang 0.5.21 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0508 [0.0342, 0.0756] | $0.2458 [0.1926, 0.3106] | $0.1483 [0.1218, 0.1806] | $0.2966 [0.2436, 0.3613] | $0.2966 [0.2436, 0.3613] | 1,030.9 [846.5, 1,255.5] | 1,030.9 [846.5, 1,255.5] | 1 req/s (fails at 1.091); tied with vllm | 240 [207, 277] ms | 46.6 [43.6, 49.9] ms | 1,430.1 [1,157.9, 1,766.3] at 1.414 req/s | -0.008 (ifeval) · inconclusive | 337 s (median of 3) | ranked | Tied for cheapest at SLO with vllm: goodput brackets overlap (1 req/s (fails at 1.091)), so the search cannot separate them; compare latency at equal load; quality gate inconclusive vs vllm (worst: ifeval -0.008) |
| 2 | **vllm**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0639 [0.0402, 0.1015] | $0.2328 [0.1752, 0.3016] | $0.1483 [0.1218, 0.1806] | $0.2966 [0.2436, 0.3613] | $0.2966 [0.2436, 0.3613] | 1,030.9 [846.5, 1,255.5] | 1,030.9 [846.5, 1,255.5] | 1 req/s (fails at 1.091); tied with sglang | 298 [238, 374] ms | 44.0 [41.0, 47.1] ms | 1,430.1 [1,157.9, 1,766.3] at 1.414 req/s | baseline | 299 s (median of 3) | ranked | Tied with sglang at SLO: goodput brackets overlap (1 req/s (fails at 1.091)), so the search cannot separate them; compare latency at equal load; quality baseline |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | sglang | vllm | Verdict |
|---|---|---|---|---|
| 0.5 req/s | SLO at this load | met | met |  |
| 0.5 req/s | TTFT p50 | **140 [131, 149] ms** | 161 [156, 165] ms | sglang lower |
| 0.5 req/s | TTFT p95 | 198 [113, 347] ms | 222 [143, 344] ms | no significant difference |
| 0.5 req/s | TPOT p50 | 29.1 [26.1, 32.4] ms | 28.6 [25.8, 31.8] ms | no significant difference |
| 0.5 req/s | TPOT p95 | 31.0 [28.3, 34.0] ms | 30.5 [27.9, 33.2] ms | no significant difference |
| 0.5 req/s | E2E p95 | 31,876 [29,096, 34,922] ms | 31,354 [28,671, 34,288] ms | no significant difference |
| 1 req/s | SLO at this load | met | met |  |
| 1 req/s | TTFT p50 | **163 [153, 174] ms** | 200 [190, 211] ms | sglang lower |
| 1 req/s | TTFT p95 | 240 [207, 277] ms | 298 [238, 374] ms | no significant difference |
| 1 req/s | TPOT p50 | 44.6 [38.9, 51.1] ms | 42.1 [36.8, 48.2] ms | no significant difference |
| 1 req/s | TPOT p95 | 46.6 [43.6, 49.9] ms | 44.0 [41.0, 47.1] ms | no significant difference |
| 1 req/s | E2E p95 | 47,879 [44,729, 51,250] ms | 45,165 [42,136, 48,412] ms | no significant difference |
| 2 req/s | SLO at this load | failed | failed |  |
| 2 req/s | TTFT p50 | 15,639 [9,439, 25,909] ms | 9,141 [4,546, 18,378] ms | no significant difference |
| 2 req/s | TTFT p95 | 38,828 [27,751, 54,326] ms | 29,226 [20,576, 41,512] ms | no significant difference |
| 2 req/s | TPOT p50 | **67.6 [66.5, 68.7] ms** | 80.3 [79.8, 80.8] ms | sglang lower |
| 2 req/s | TPOT p95 | 76.6 [47.9, 122.5] ms | 83.6 [82.8, 84.4] ms | no significant difference |
| 2 req/s | E2E p95 | 111,503 [96,935, 128,261] ms | 111,228 [100,384, 123,244] ms | no significant difference |

### Qwen/Qwen3-8B: code-completion (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **vllm**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0378 [0.0171, 0.0835] | $3.6321 [2.1004, 5.8063] | $0.1468 [0.1002, 0.2151] | $4.8415 [3.3676, 6.9604] | $4.8415 [3.3676, 6.9604] | 63.2 [43.9, 90.8] | 63.2 [43.9, 90.8] | 1.297 req/s (fails at 1.414) | 295 [245, 356] ms | 38.6 [31.6, 47.2] ms | 93.8 [81.8, 107.5] at 2 req/s | baseline | 299 s (median of 3) | untrusted | Not ranked: untrusted (high run-to-run variance); rerun before relying on it |
| – | **sglang**<br>sglang 0.5.21 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0378 [0.0202, 0.0710] | $4.7465 [3.1472, 6.8942] | $0.1787 [0.1333, 0.2397] | $5.9735 [4.4154, 8.0813] | $5.9735 [4.4154, 8.0813] | 51.2 [37.8, 69.3] | 51.2 [37.8, 69.3] | 1.091 req/s (fails at 1.189) | 283 [262, 305] ms | 36.5 [33.6, 39.7] ms | 93.8 [81.8, 107.5] at 2 req/s | -0.008 (ifeval) · inconclusive | 337 s (median of 3) | untrusted | Not ranked: untrusted (high run-to-run variance); rerun before relying on it |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | vllm | sglang | Verdict |
|---|---|---|---|---|
| 1 req/s | SLO at this load | met | met |  |
| 1 req/s | TTFT p50 | 179 [156, 205] ms | 185 [162, 212] ms | no significant difference |
| 1 req/s | TTFT p95 | 252 [173, 368] ms | 266 [183, 387] ms | no significant difference |
| 1 req/s | TPOT p50 | 25.6 [23.9, 27.4] ms | 26.3 [24.1, 28.8] ms | no significant difference |
| 1 req/s | TPOT p95 | 33.3 [30.1, 36.8] ms | 36.3 [31.7, 41.6] ms | no significant difference |
| 1 req/s | E2E p95 | 1,899 [1,512, 2,385] ms | 2,005 [1,559, 2,580] ms | no significant difference |
| 2 req/s | SLO at this load | failed | failed |  |
| 2 req/s | TTFT p50 | 195 [174, 219] ms | 203 [180, 229] ms | no significant difference |
| 2 req/s | TTFT p95 | 385 [258, 574] ms | 428 [243, 754] ms | no significant difference |
| 2 req/s | TPOT p50 | 31.0 [28.2, 34.1] ms | 33.6 [29.4, 38.5] ms | no significant difference |
| 2 req/s | TPOT p95 | 45.4 [33.9, 61.0] ms | 52.9 [39.3, 71.2] ms | no significant difference |
| 2 req/s | E2E p95 | 2,519 [2,048, 3,100] ms | 2,821 [2,329, 3,416] ms | no significant difference |

**Warnings**

- vllm: throughput.request_rate at goodput: run-to-run CV 14.4% exceeds 10%
- vllm: throughput.output_tok_s at goodput: run-to-run CV 14.7% exceeds 10%
- sglang: throughput.request_rate at goodput: run-to-run CV 12.0% exceeds 10%
- sglang: throughput.output_tok_s at goodput: run-to-run CV 12.2% exceeds 10%

### Qwen/Qwen3-8B: shared-prefix (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **sglang**<br>sglang 0.5.21 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0286 [0.0180, 0.0455] | $0.7295 [0.4826, 1.0104] | $0.0695 [0.0580, 0.0832] | $1.1908 [0.9942, 1.4263] | $1.1908 [0.9942, 1.4263] | 256.8 [214.4, 307.6] | 256.8 [214.4, 307.6] | 2 req/s (fails at 2.378); tied with vllm | 310 [252, 380] ms | 41.9 [38.6, 45.5] ms | 490.4 [445.9, 539.4] at 4 req/s | -0.008 (ifeval) · inconclusive | 337 s (median of 3) | ranked | Only ranked config at SLO; quality gate inconclusive vs vllm (worst: ifeval -0.008) |
| – | **vllm**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0276 [0.0172, 0.0444] | $0.7447 [0.4954, 1.0274] | $0.0695 [0.0580, 0.0832] | $1.1908 [0.9942, 1.4263] | $1.1908 [0.9942, 1.4263] | 256.8 [214.4, 307.6] | 256.8 [214.4, 307.6] | 2 req/s (fails at 2.378); tied with sglang | 297 [227, 390] ms | 36.6 [34.5, 38.8] ms | 490.4 [445.9, 539.4] at 4 req/s | baseline | 299 s (median of 3) | untrusted | Not ranked: untrusted (high run-to-run variance); rerun before relying on it |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | sglang | vllm | Verdict |
|---|---|---|---|---|
| 1 req/s | SLO at this load | met | met |  |
| 1 req/s | TTFT p50 | 165 [160, 170] ms | 157 [153, 161] ms | no significant difference |
| 1 req/s | TTFT p95 | 224 [160, 314] ms | 206 [144, 297] ms | no significant difference |
| 1 req/s | TPOT p50 | 27.3 [25.9, 28.7] ms | 26.3 [25.4, 27.3] ms | no significant difference |
| 1 req/s | TPOT p95 | 30.8 [28.2, 33.5] ms | 28.9 [27.3, 30.7] ms | no significant difference |
| 1 req/s | E2E p95 | 4,073 [3,752, 4,421] ms | 3,829 [3,610, 4,062] ms | no significant difference |
| 2 req/s | SLO at this load | met | met |  |
| 2 req/s | TTFT p50 | 172 [167, 178] ms | 167 [158, 176] ms | no significant difference |
| 2 req/s | TTFT p95 | 310 [252, 380] ms | 297 [227, 390] ms | no significant difference |
| 2 req/s | TPOT p50 | 33.7 [31.2, 36.5] ms | 31.2 [29.3, 33.3] ms | no significant difference |
| 2 req/s | TPOT p95 | 41.9 [38.6, 45.5] ms | 36.6 [34.5, 38.8] ms | no significant difference |
| 2 req/s | E2E p95 | 5,537 [5,199, 5,896] ms | **4,839 [4,565, 5,129] ms** | vllm lower |
| 4 req/s | SLO at this load | failed | failed |  |
| 4 req/s | TTFT p50 | 224 [187, 268] ms | 222 [179, 274] ms | no significant difference |
| 4 req/s | TTFT p95 | 450 [384, 528] ms | 456 [377, 551] ms | no significant difference |
| 4 req/s | TPOT p50 | 58.3 [46.1, 73.6] ms | 46.4 [40.4, 53.3] ms | no significant difference |
| 4 req/s | TPOT p95 | 75.7 [67.3, 85.1] ms | **58.4 [53.9, 63.4] ms** | vllm lower |
| 4 req/s | E2E p95 | 9,921 [8,649, 11,380] ms | 7,769 [6,915, 8,728] ms | no significant difference |

**Warnings**

- vllm: ttft_ms.p95 at goodput: run-to-run CV 11.0% exceeds 10%

## Competitiveness: our cost at SLO vs public list prices

> Public list prices only: competitor numbers are the per-token prices each provider publishes on its pricing page, with the source and the date it was checked. No competitor endpoint was called or benchmarked, and list prices say nothing about a provider's latency, quality or quantization unless the page discloses it.

Competitor prices last checked 2026-10-04. Flags leave out aggregators and entries whose availability is unverified; they are listed for reference. Our cost is the headline config's cost at SLO (see the summary) at the on-demand list price, storage included: $/1M input and $/1M output split the replica's cost by measured prefill time, and $/1M blended is its cost over all tokens at the workload's own input:output mix. Each public price is also shown at every workload's mix, so blended compares like with like. With no price set, the break-even price is the lowest price that covers our cost: at the point estimate, and at the cost CI high bound.

### Qwen3 8B (`qwen3-8b`), ours unquantized

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-1k-1k (open loop) | vllm (leaderboard rank 2; tied with sglang) | 1,024 / 1,024 | $0.0639 [0.0402, 0.1015] | $0.2328 [0.1752, 0.3016] | $0.1483 [0.1218, 0.1806] | no price set | no price set | $0.0639 (CI high $0.1015) | $0.2328 (CI high $0.3016) | n/a | n/a |
| code-completion (open loop) | vllm (untrusted, not ranked: high run-to-run variance) | 1,537 / 48 | $0.0378 [0.0171, 0.0835] | $3.6321 [2.1004, 5.8063] | $0.1468 [0.1002, 0.2151] | no price set | no price set | $0.0378 (CI high $0.0835) | $3.6321 (CI high $5.8063) | n/a | n/a |
| shared-prefix (open loop) | sglang (leaderboard rank 1; tied with vllm) | 2,065 / 128 | $0.0286 [0.0180, 0.0455] | $0.7295 [0.4826, 1.0104] | $0.0695 [0.0580, 0.0832] | no price set | no price set | $0.0286 (CI high $0.0455) | $0.7295 (CI high $1.0104) | n/a | n/a |

**Public list prices** (per 1M tokens)

| Provider | Provider model | $/1M in | $/1M out | Precision | At fixed-1k-1k mix (1,024 / 1,024) | At code-completion mix (1,537 / 48) | At shared-prefix mix (2,065 / 128) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | $0.2000 (our cost 0.74×) | $0.2000 (our cost 0.73×) | $0.2000 (our cost 0.35×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | $0.2860 (our cost 0.52×) | $0.1272 (our cost 1.15×) | $0.1367 (our cost 0.51×) | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [fixed-1k-1k (open loop); code-completion (open loop); shared-prefix (open loop)] `no_price_set`: qwen3-8b: no price set
- [fixed-1k-1k (open loop); code-completion (open loop); shared-prefix (open loop)] `no_public_comparison`: qwen3-8b: no eligible public list price

### Providers with no public per-token price

- Groq: Enterprise plans only; contact sales. (checked 2026-10-04)
- Cerebras: No public per-token price. (checked 2026-10-04)
- Lambda: Inference offering winding down. (checked 2026-10-04)
- Hyperbolic: Inference offering retired. (checked 2026-10-04)
- Nebius: Neither model is in its catalog. (checked 2026-10-04)

## Methodology and provenance

- **SLO:** TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%
- **Load generation:** open loop (requests arrive on a fixed schedule at a set rate, req/s)
- **Content:** synthetic
- **Dataset:** unnamed dataset (license: not recorded)
- **Repetitions:** 3 per load point
- **Confidence intervals:** Repetitions of a load point are combined per metric, with a two-sided 95% Student-t interval on the scale that fits the metric. Latencies, throughputs and rates are strictly positive and right-skewed: the value shown is the geometric mean and the interval is the t-interval of ln(value), exponentiated (geometric mean ×/÷ a factor), so both bounds are positive. Proportions such as error rate and SLO attainment use the arithmetic mean of the per-run proportions with the t-interval clipped to [0, 1]; it measures run-to-run variation, not binomial sampling, so it is [0, 0] when no repetition saw an error. Counts use the arithmetic t-interval clipped at 0, as does a positive metric when a repetition measured 0. A load meets the SLO only if the CI upper bound of every target is within it; goodput is the highest passing load below the first failing one. Cost CI bounds come from the goodput throughput CI (low cost from high throughput), so they are finite whenever the throughput lower bound is above zero. Results with a single repetition have no CI and are never trusted.
  - request counts and sample sizes: arithmetic mean, Student-t interval clipped at 0 (t_clipped)
  - latency percentiles and means (ms): geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - throughput and goodput (tok/s, req/s): geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - requests in prefill (request rate × mean TTFT): geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - server queue time, measurement window: geometric mean, Student-t interval on the log scale (log_t); arithmetic interval clipped at 0 if a repetition is 0
  - error rate, SLO attainment, cache fractions and hit rates: arithmetic mean of per-run proportions, Student-t interval clipped to [0, 1] (t_clipped)
  - GPU utilization (%): arithmetic mean, Student-t interval clipped to [0, 100] (t_clipped)
  - other server and GPU gauges: arithmetic mean, Student-t interval clipped at 0 (t_clipped)
  - anything else: arithmetic mean, Student-t interval (t)
- **Cost allocation (ranking):** all_output: all of the replica's hourly cost is charged to output tokens, so input tokens have no separate price ($/1M input is n/a, not $0); $/1M output is the headline number
- **Input/output split (headline):** Headline $/1M input and $/1M output use the prefill_time split, measured at the goodput point: input tokens pay for the share φ of the replica's time spent prefilling prompts, output tokens for the rest (decode steps, and any idle headroom the SLO needs). φ is the mean number of requests in their prefill phase, request rate × mean TTFT (Little's law), per repetition, combined like a throughput (geometric mean, log-t CI) and capped at 1. Input = φ × hourly price / input tok/s; output = (1 − φ) × hourly price / output tok/s; at the measured mix they bill exactly the replica's hourly price. TTFT includes queueing and the first decode step and concurrent prefills are each counted, so φ tends to overstate prefill time (input high, output low); idle headroom is charged to output (output high). The price CIs combine the φ and throughput CIs at the ends that push each price the same way, an outer bound. Blended $/1M is the hourly price over all tokens at the workload's own input:output mix and needs no split. The ranking still uses the declared cost allocation, shown in its own column.
- **Price book last checked:** 2026-10-09

### sglang · fixed-1k-1k, open loop (`67df38a32da2c3963020756df69ca25e6244c4099ee6ce4fce59bc954eaadbc0`)

- **Config:** sglang 0.5.21 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** sglang 0.5.21; image `lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`; digest `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `aa7d38d77b7d74ad448ca49b6559732eb26dd30c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-07T08:18:12.389030+00:00) + 80 GB block storage
- **Runs:** 18; provenance digests: `00f714baa3d1f4ede368eacd633034c424b467f8e41ae404a51a278022a76bee`, `0fa6fbad698c723789c87e313dec16357b7f9ba136c32ee7f5d9dc37aa2036a0`, `3e404f616c7697d103b9ac9764a139878a9c033f40e9b9350e2771cedfc54944`, `628503998749ae0cc0ded2481c8fc75c5ad52eaaed79c42ded37ecfa4fd454e2`, `646d3dfd0d7bfc0c8cc40303f6ada3167474c908f2850544156a12d3307557d0`, `65c316520228b7fd7b0b404a815cdb6ec43a876884441e09a60695f4ee628860`, `773b6c704b5d8ed446bc065ab9adb9dcb2688fd1b280116057367a72ef67b16a`, `7a153309e15c85fa33cae4d65b0e93c2672d945f2e61b1536657c8dba4ed98ac`, `94de0efde15fd9f6173504a6ea590ead33d7644f41d21e8254b8fe12ac125f49`, `a140c64a9f9945e05e93ea77234c8b0ad3e6497939316aa4ab906f216a00891c`, `b4cf086080d4ca403454460e29d7a639aadf72e12de1b7a12c4e7843bd55dbbe`, `b6f1408103f26700ab1d2fc62be21a37c80f9156371aba5e8394524c048c5cd7`, `c55a1e2d7e264c453039febbb0d71c721a7e38c2a38d241281b5852fa517aef5`, `d16dbe036fd0c96c8cbc2b1b67826b5d64a14899a8e962c5e14924cb9e871da0`, `e1699fd7c4102a33cb4563f5cf16165cb642ec8cb14454d34581919a2a0a3448`, `f1c881056d047f57781f25010180326c2b0b321e14dfdf5c02810e2616701c8c`, `f4e703cc30fb743ad4e9a7da838aa03bb2297c689916657d4325f19e297f61cf`, `f9483d3e73805c087054289447ab2f79a6a63b6e47c8d7908ffe1572d0b7d139`
- **Reproduce:** `bench reproduce b83314e7-82f6-4a5c-af85-2b5b0ac83e17`

### vllm · fixed-1k-1k, open loop (`7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `aa7d38d77b7d74ad448ca49b6559732eb26dd30c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-07T05:40:31.406858+00:00) + 80 GB block storage
- **Runs:** 18; provenance digests: `0cb7ec2456e9210f64ca0abf04890e4f3741caf84eed1dc45b94f3f007b30e4b`, `218c828f41305070963f2c9d8128878f2dd53ef915e6dbeb559fdbaab8ff3cac`, `266972031fc241072f837382576ca849be9dafb790f8498871b2941d9784ef27`, `2e366ab112cbdc9050d6764e737621e7bee22e870a1cfe108cc45f180c503a89`, `322743a4041b8ba12e332a04b70a2c40a8197a558ccb9544f58579a86678843e`, `32d22eaa2f6ffccef92d9a07ba732e1d766e535ee08d1f8eba25db532b56d1a3`, `410b62514d8572b66d648c14d0d2eeaba2937112dddc345951f47d4f435c027a`, `463a08e020340083d7284ccf515910fffc262f525a2e18f546f1c776b24b020c`, `7d85b863ee0ffc1231817651f9ec5b3720fe4205e7f361d21017688574798fb3`, `8612421cb935561ba97e821ab8d170578d182ad830adb40950e5e5c864c15020`, `8ada7e8247ac75e33baeb62845a7ac99106b51c4ca217222ac194692ccdf09a2`, `a2b1fe6fd164940c82af513ed172d4eb5ff69404e56522f1d47123af74db3c0c`, `af5ac085ff1899eb1e82e65640b26ee0c8d6e7e8eb6c5fd51e4f7bbc9ced1f94`, `d02de35477be78b9700f08f151cb4152f35da3687efbb2a85db681e0fa654ea7`, `d90980b3226b96f0827e578d5054e5345ab7caeb48fe8f8166473043f433d24c`, `dee2556c964aa9a138feadf0803770d6bd8925d764ef73d556cce036283a32d5`, `dfac7a7271ae4d0b8b1f99cf2d4387fc84954a97947445d2fdc4fa8873c1cd50`, `e9bd02aa433910a08220a993232c9f9b57043db1586d520a511947720d763c2f`
- **Reproduce:** `bench reproduce 29f42d2c-c2aa-41bf-89c8-78f3ae8a303b`

### vllm · code-completion, open loop (`7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `aa7d38d77b7d74ad448ca49b6559732eb26dd30c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-07T05:40:31.406858+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `26f52fb97f2debfa9b553395195f85d7c82aef39ed1541b593f09ddc4d9234f4`, `28bf2a4936c2cab057b937afff0fb3b3ecd850714c12b34f3f235bafe9635aed`, `29c204d0eac08a4393147edf87a79592b59a6d0e0aa7a47addcca63db8c2a2d0`, `334c1213a4557fe0e58f6caf64835249672547f44ba13d5f9e74b44dd7d99dcd`, `3a56cfa145f8fe6ea75b11dc5c5636e128af67a30dd3b42da9b75c07a4abd780`, `645b5676cb1b0034aef2f54ea3c666adbf7cddd2eb7468e8e893835cd90458b8`, `73288e56267a3de2d0986fac6757afd97efe7cbd1b8d37431a2f1aa49ddf4dda`, `75af781ca7b6ae90846f62416038886fd7b8e7e66b798541d5966c48ca5a0d0b`, `79ebaab81694b3899dbabde6b8b675725bae3a4b5dbe0580bf9ca1cea77e3f68`, `9d9f52f73fac28d67ae03187f2397b9e99a2d3ad3e8fecf6e554aa14a9a0db4e`, `cc6e3b45e2a1e1acf1997dec5f324f1eb9df1c88e7e072c4852ed91a607952cd`, `ddcbbe61516d716a367ba9c0dbcb68496e835ffc7a9f44f15ab4fc1a3e65b2a9`, `e5e1842201bcaa60e1cb0d56487a6e2913dcaa69e8eac0a11b8d353a7d864384`, `f5699a07289e2acdc945923b54b6cb70b27f6b10353ce9b22721113161fcd4f1`, `ffac13717de63bee3d8b339e05b9aa4eded2c240ab010b29960b460127fe8a5e`
- **Reproduce:** `bench reproduce ced8b807-f7f5-4927-8cee-c0dbbb23fc5e`

### sglang · code-completion, open loop (`67df38a32da2c3963020756df69ca25e6244c4099ee6ce4fce59bc954eaadbc0`)

- **Config:** sglang 0.5.21 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** sglang 0.5.21; image `lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`; digest `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `aa7d38d77b7d74ad448ca49b6559732eb26dd30c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-07T08:18:12.389030+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `31f48d15e024c455060e133fc6e0b466b88a46843ea81c07d48961526cf6b8a0`, `403e696934af1b20ec9ed286b8f5d8ceae731ce3ad8869452d8826cc7ae75af2`, `437e52b0aba2a188940206d76ea9aad450b4de50a7c293315e716d220ccab84c`, `489d38fdc65737ec192b2dc8e83529f4d1855168af591485869d44a4e87607d9`, `520e6cf1b11b012623235f549e71f530f5cb31b4ce149d809d060eaeba3a4da6`, `6b6d26848f770704a335dca2d532569c5d1e22e746e64d4a28edcc5c046426dd`, `713abc724751fbb22ce260425e234d6832062c0f2f2c49861cdf89ada1738be0`, `75898d1ec532c98741a12bd6026863bc9f4c8f6cdbdefc8bc0c6a5c69aa03971`, `7a9118a427c193858e793bf0ded27387e8b4c3d02669d1eb310b0585473dd9c3`, `8a6d07d16b9b65ddaa6360d3a210ff1e64e99b1da346e0786f978995db06f6a0`, `97542f6a19ac0f54e649ac73ee3cabd59131ab9fbdbfe66cbaf1b7a7db3c6da7`, `a1e1ffcbc5931b95ca1b698807bb426c8baf2f74918c30ab34015efae65d41ba`, `b0aa1d909611e0ee97f6ea532ed8c0fcd457428f877a765f88da3dc82663798b`, `e32f7c6154cf942736cf0c81835644e25b6736924d2d57b84f01a1dc1b82f274`, `f88b46a20ebe3511bc32ebe4df7d21f1efc9c7f20ab4fae65ff98607c5722efc`
- **Reproduce:** `bench reproduce c6f370e1-a7a5-4f69-b329-7c368e2c0f60`

### sglang · shared-prefix, open loop (`67df38a32da2c3963020756df69ca25e6244c4099ee6ce4fce59bc954eaadbc0`)

- **Config:** sglang 0.5.21 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** sglang 0.5.21; image `lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`; digest `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `aa7d38d77b7d74ad448ca49b6559732eb26dd30c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-07T08:18:12.389030+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `0504bb3b50fd8b32411d11ce61abcd58e3973d3ed764cb81065b9ac0c3cb3110`, `2246383fddf3e8edce86d10048ed9df48b7ba02fb917521db103ab1f698a87f1`, `33d2d524cbe0e473be1a63fc20486d405137e4946812478e480f9b6bdf4d7222`, `33de9b76903aa672cf8f5b4f2ffff5de1a811ca237fe650fb85c5715ff24bbdf`, `50a0cb3da6efa285699a110e34454cfd4fc1dbc8daca49bd5625f7d4cea4a239`, `6a69050051c3ff65c2202c86e095256b1a7f3224361e67c179cec097994ae546`, `72233ff9802bf044c6f82ad6b41cd8f2f03a969514bae9cddebb7b24b66e7db7`, `7f6f66c4ff177a655fa2bb6bdeca7aa34ea2696b541bd2c5eb4a9a0ac97af75c`, `8d925655b5be51c08b7b3d91d19788fcd8d87d6368512b25808bca2ed6c416c7`, `9f55065eb1a0110c3605f1af8950d5af065db7b582060ab975ebcd086e5ba5c4`, `a17b8c08f066b0a9c079b8ce3d12adcf87df6f6c229d7d6fd74e280b648664a9`, `a5d6b2358af28e25ebf7eb21c6447828cf30d410e4c09f450566095b37ae0afe`, `b51ed43327e96e9ce53be615e7036a43ffc6dee1104114245901d2f8757b9cf5`, `bc3ff6ce35bcc885cf792a12c16cf1361fb34eea9ed6394f5c2d88900462168d`, `e8ac9d4c111c2be4a90c795d6d564879ee49a8234f2ec4b6ef6ca0a62fa67fac`
- **Reproduce:** `bench reproduce f484a1f4-74bb-4e6b-9f39-984847102106`

### vllm · shared-prefix, open loop (`7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `aa7d38d77b7d74ad448ca49b6559732eb26dd30c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-07T05:40:31.406858+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `0338f3916eecdbe49eaedfa46e206318ed969e3f5a835e91c4fecfb2bd59043c`, `0677055c8da73f2b37301f80fc6666cd9840f98a6c7ca2319c7d7ff007760e44`, `0afeec6b7fadb641f2e9439bbbdc5afb14dcc269a20a8bde59723e031eb8023a`, `161ea0eedaccfd36d0683c679acd3dbbec1e7052d65fa2f85e6b4214c204a643`, `2ef1a9ac5f424952b3938aa8e4f081311a717f7f21184b96beeb47edfb9631f9`, `45860f17965094d64e8db296804688b43cc9bcc124d03939a8c6af2bd36fc27f`, `53ba158542203e6668375e3b40a9cc940a253b14ac7681e7e73d5f67fc653d69`, `56b938f63988a4e65180821527098296f7c52a41d1b32f7c51a4cce83411f777`, `6c12200ebeb0610a1e17631d98cdf110e9087a70db9e6ab40a358e5227772377`, `a0b7622ea2d804c5baf4b4cc502215e9ba87c1d5864c8185b7ead5bd067305ee`, `c3c3ecdf9b38e1044dd777928f1d09583f385b8bca66c14840c350dba60f92b4`, `d27b9dd8104b3340634afc679ea4ced64fa311575419adf0cf83bbc9dfe3f3ef`, `d4d32f70458a22f801c36a59ec2df61ad2557b46a4ecbfbe7ecebc81cbc71995`, `df7704e2d7b8498864fbde5a669acb410581c93ad5e6d68448c7cd0cf1252fb6`, `e64b588d3d980290c78fbb7fb9b8a348fbbebb2f747ffbf557e012198373626d`
- **Reproduce:** `bench reproduce 1b07f0b8-b191-4090-913d-207420cd72cf`
