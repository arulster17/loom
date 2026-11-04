# Alternative SLO view, not the experiments' SLO: TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 20 ms, error rate ≤ 1.00% (the experiments declare TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%; the main leaderboard uses that)

## Summary

Alternative SLO view (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 20 ms, error rate ≤ 1.00%), not the declared SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%). What one replica costs us to serve at the SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 20 ms, error rate ≤ 1.00%), at the on-demand list price including its storage, at the highest tested load that met the SLO; 95% confidence intervals in brackets. $/1M input and $/1M output split the replica's cost by measured prefill time (methodology at the end); $/1M blended is the replica's cost over all tokens at that workload's own input:output mix and needs no split. Each workload quotes one config: rank 1, ranked among configs that passed every check (a candidate ranks only on a quality gate pass against its reference). With nothing ranked, an untrusted config of verified quality is quoted as a comparison point with its reason; a config whose gate failed or is inconclusive is never quoted. Nothing here relaxes a check, and the ranking is the leaderboard's.

### Qwen/Qwen3-8B

Hardware: 1×L40S (aws g6e.xlarge, $1.8829/h on-demand incl. 200 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | bf16 | no cost at SLO | none met the SLO (fails at 0.25) | n/a | n/a | n/a | baseline |
| chat-sharegpt | bf16-eagle3 | no cost at SLO | none met the SLO (fails at 0.25) | n/a | n/a | n/a | -0.005 (ifeval) · inconclusive |
| chat-sharegpt | bf16-ngram | no cost at SLO | none met the SLO (fails at 0.25) | n/a | n/a | n/a | -0.006 (ifeval) · inconclusive |
| code-edit | bf16-eagle3 | quality not verified, not ranked | ≥4 req/s (none failed) | $0.1957 [0.1614, 0.2373] | $0.3731 [0.3222, 0.4285] | $0.2783 [0.2536, 0.3055] | -0.005 (ifeval) · inconclusive |
| code-edit | bf16-ngram | quality not verified, not ranked | ≥4 req/s (none failed) | $0.1257 [0.1027, 0.1539] | $0.4534 [0.4017, 0.5095] | $0.2783 [0.2536, 0.3055] | -0.006 (ifeval) · inconclusive |
| code-edit | bf16 | no cost at SLO | none met the SLO (fails at 0.5) | n/a | n/a | n/a | baseline |

**chat-sharegpt** (991 input / 325 output tokens per request)

- No cost at this view's SLO: no tested load met the SLO.
- bf16 misses the SLO even at the lowest tested load, 0.25 req/s: TPOT p95 is 25.0 [23.7, 26.4] ms against the 20 ms target. Missing a latency target at the lowest tested load points to the per-request latency of this config on this hardware, not to load.
- bf16-eagle3 misses the SLO even at the lowest tested load, 0.25 req/s: TPOT p95 is 15.7 [11.5, 21.6] ms against the 20 ms target (the SLO is judged on the CI upper bound, which is over it). Missing a latency target at the lowest tested load points to the per-request latency of this config on this hardware, not to load.
- bf16-ngram misses the SLO even at the lowest tested load, 0.25 req/s: TPOT p95 is 25.7 [25.4, 26.1] ms against the 20 ms target. Missing a latency target at the lowest tested load points to the per-request latency of this config on this hardware, not to load.

**code-edit** (252 input / 220 output tokens per request)

- No cost at this view's SLO: no config with a cost at SLO has verified quality (bf16-eagle3, bf16-ngram: quality gate inconclusive vs BF16 (AWS g6e.xlarge)); a candidate ranks only on a gate pass.
- bf16-eagle3 met the SLO at every tested load: goodput is at least 4 req/s, so its cost is an upper bound.
- bf16-ngram met the SLO at every tested load: goodput is at least 4 req/s, so its cost is an upper bound.
- bf16 misses the SLO even at the lowest tested load, 0.5 req/s: TPOT p95 is 24.2 [23.3, 25.2] ms against the 20 ms target. Missing a latency target at the lowest tested load points to the per-request latency of this config on this hardware, not to load.
- Caveat: bf16-eagle3 is untrusted, so the leaderboard does not rank it: every tested load met the SLO; goodput is at least 4 req/s and cost at SLO is an upper bound. Treat its figure as indicative and rerun before relying on it.
- Caveat: bf16-ngram is untrusted, so the leaderboard does not rank it: every tested load met the SLO; goodput is at least 4 req/s and cost at SLO is an upper bound. Treat its figure as indicative and rerun before relying on it.

**Quality**

- bf16 (BF16 weights, KV cache auto (model dtype)): the reference the quality gate compares against; gsm8k 0.904, ifeval 0.823, json_schema 0.910, tool_calling 0.978, tool_calling_strict 0.978.
- bf16-eagle3 (BF16 weights, KV cache auto (model dtype)): quality gate vs BF16 (AWS g6e.xlarge) is inconclusive: gsm8k, tool_calling, tool_calling_strict, json_schema, divergence, sanity pass; ifeval inconclusive (delta -0.49 pts [-2.16 pts, +1.11 pts], n=541: CI crosses -2.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match BF16 (AWS g6e.xlarge), so it is not ranked.
- bf16-ngram (BF16 weights, KV cache auto (model dtype)): quality gate vs BF16 (AWS g6e.xlarge) is inconclusive: gsm8k, tool_calling, tool_calling_strict, json_schema, divergence, sanity pass; ifeval inconclusive (delta -0.55 pts [-2.03 pts, +0.92 pts], n=541: CI crosses -2.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match BF16 (AWS g6e.xlarge), so it is not ranked.

## Leaderboards

Each table lists one model on one workload. $/1M input and $/1M output split the replica's cost by measured prefill time; $/1M blended is its cost over all tokens at that workload's own input:output mix (methodology below). Rows are ranked by the rank key, the cost under the experiments' declared allocation (all_output charges the whole replica to output tokens), at the on-demand list price, cheapest first; spot, committed-1y and as-run costs of the rank key are shown where available. Every price includes the replica's storage volume. Values are point estimates (geometric means for latency and throughput) with 95% confidence intervals in brackets. Goodput is the highest tested load that met the SLO; raw peak throughput ignores the SLO and is not goodput. Only configs that passed every check are ranked: a candidate needs a quality gate pass against its reference. Unranked rows (quality gate failed or not verified, untrusted, or no cost) are listed last with the reason.

Goodput is searched on a grid of loads, so it is known only to a bracket: at least the goodput load, below the load that failed (shown as "fails at"). Configs whose brackets overlap are tied within the search resolution: their goodput, throughput and cost at SLO come from the same grid point and are not a measured equality.

≥ before a goodput load: not bracketed. Every tested load met the SLO, so goodput is at least that load (it may be higher) and cost at SLO is an upper bound.

### Qwen/Qwen3-8B: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **bf16**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | n/a | n/a | n/a | n/a | n/a | n/a | none met the SLO (fails at 0.25) | n/a | n/a | 967.9 [903.2, 1,037.3] at 3 req/s | baseline | 877 s (median of 1) | no cost at SLO | No cost at SLO: no tested load met the SLO; test lower loads |
| – | **bf16-eagle3**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'eagle3', 'model': 'RedHatAI/Qwen3-8B-speculator.eagle3', 'revision': '08610ffa01dd9f16731fe8f627b85905b6aa51c4', 'num_speculative_tokens': 3} | n/a | n/a | n/a | n/a | n/a | n/a | none met the SLO (fails at 0.25) | n/a | n/a | 967.9 [903.2, 1,037.3] at 3 req/s | -0.005 (ifeval) · inconclusive | n/a | no cost at SLO | No cost at SLO: no tested load met the SLO; test lower loads |
| – | **bf16-ngram**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'ngram', 'num_speculative_tokens': 3, 'prompt_lookup_max': 4, 'prompt_lookup_min': 2} | n/a | n/a | n/a | n/a | n/a | n/a | none met the SLO (fails at 0.25) | n/a | n/a | 967.9 [903.2, 1,037.3] at 3 req/s | -0.006 (ifeval) · inconclusive | n/a | no cost at SLO | No cost at SLO: no tested load met the SLO; test lower loads |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | bf16 | bf16-eagle3 | bf16-ngram | Verdict |
|---|---|---|---|---|---|
| 0.25 req/s | SLO at this load | failed | failed | failed |  |
| 0.25 req/s | TTFT p50 | 119 [93, 153] ms | 126 [95, 165] ms | 99 [71, 138] ms | no significant difference |
| 0.25 req/s | TTFT p95 | 203 [156, 265] ms | 210 [160, 274] ms | 179 [133, 241] ms | no significant difference |
| 0.25 req/s | TPOT p50 | 24.0 [23.8, 24.2] ms | **12.0 [11.7, 12.4] ms** | 22.5 [20.9, 24.1] ms | bf16-eagle3 lower |
| 0.25 req/s | TPOT p95 | 25.0 [23.7, 26.4] ms | **15.7 [11.5, 21.6] ms** | 25.7 [25.4, 26.1] ms | bf16-eagle3 lower |
| 0.25 req/s | E2E p95 | 15,981 [12,983, 19,671] ms | **8,094 [6,592, 9,938] ms** | 14,979 [13,935, 16,102] ms | bf16-eagle3 lower |
| 1 req/s | SLO at this load | failed | met | failed |  |
| 1 req/s | TTFT p50 | 126 [100, 158] ms | 132 [99, 177] ms | 104 [84, 129] ms | no significant difference |
| 1 req/s | TTFT p95 | 231 [207, 259] ms | 242 [214, 274] ms | 197 [180, 216] ms | no significant difference |
| 1 req/s | TPOT p50 | 27.1 [25.1, 29.2] ms | **13.8 [13.1, 14.5] ms** | 26.7 [25.1, 28.4] ms | bf16-eagle3 lower |
| 1 req/s | TPOT p95 | 29.7 [27.6, 31.9] ms | **17.7 [16.3, 19.3] ms** | 30.8 [27.5, 34.4] ms | bf16-eagle3 lower |
| 1 req/s | E2E p95 | 18,818 [17,171, 20,623] ms | **9,025 [7,803, 10,438] ms** | 16,483 [14,738, 18,433] ms | bf16-eagle3 lower |
| 2 req/s | SLO at this load | failed | failed | failed |  |
| 2 req/s | TTFT p50 | 161 [153, 170] ms | 161 [154, 169] ms | **133 [125, 141] ms** | bf16-ngram lower |
| 2 req/s | TTFT p95 | 287 [279, 294] ms | 293 [279, 306] ms | 259 [229, 293] ms | no significant difference |
| 2 req/s | TPOT p50 | 34.0 [30.4, 38.1] ms | **16.4 [14.9, 18.1] ms** | 33.1 [28.8, 38.1] ms | bf16-eagle3 lower |
| 2 req/s | TPOT p95 | 38.2 [33.5, 43.6] ms | **21.5 [19.2, 24.1] ms** | 40.5 [36.0, 45.6] ms | bf16-eagle3 lower |
| 2 req/s | E2E p95 | 24,117 [20,613, 28,217] ms | **11,122 [9,880, 12,520] ms** | 20,893 [17,491, 24,958] ms | bf16-eagle3 lower |
| 3 req/s | SLO at this load | failed | failed | failed |  |
| 3 req/s | TTFT p50 | 212 [171, 263] ms | 195 [161, 237] ms | 166 [128, 214] ms | no significant difference |
| 3 req/s | TTFT p95 | 406 [293, 563] ms | 375 [271, 520] ms | 364 [232, 573] ms | no significant difference |
| 3 req/s | TPOT p50 | 46.5 [38.5, 56.1] ms | **20.1 [17.5, 23.0] ms** | 43.1 [36.6, 50.7] ms | bf16-eagle3 lower |
| 3 req/s | TPOT p95 | 55.2 [43.7, 69.9] ms | **27.2 [24.5, 30.2] ms** | 55.9 [47.0, 66.5] ms | bf16-eagle3 lower |
| 3 req/s | E2E p95 | 32,984 [27,449, 39,635] ms | **13,870 [12,677, 15,177] ms** | 28,100 [24,726, 31,935] ms | bf16-eagle3 lower |

#### Speculative decoding by load

Speculative decoding by load. Accepted per step is 1 + accepted / drafts (vLLM's mean acceptance length, the target's own token included, over steps that drafted); draft acceptance is accepted / drafted tokens, then the share of drafts kept at each position. Tokens per chunk is measured by the client (completion tokens per streamed chunk, one chunk per engine step): its lowest request shows content the draft cannot predict, which the server's mean hides. ITL is the time between chunks, so under speculation it is the verify step's time, not a token's. Speed-up is the reference's TPOT p50 over the config's at the same load (point estimates).

| Load | Config | Accepted per step | Draft acceptance (by position) | Tokens per chunk: mean, lowest request | TPOT p50 | TPOT p95 | ITL p50 (step) | TPOT p50 speed-up |
|---|---|---|---|---|---|---|---|---|
| 0.25 req/s | bf16 | n/a | n/a | 1.01 [1.00, 1.01]; 1.00 [1.00, 1.00] | 24.0 [23.8, 24.2] ms | 25.0 [23.7, 26.4] ms | 23.8 [23.5, 24.1] ms | 1.00x |
| 0.25 req/s | bf16-eagle3 | 2.29 [2.18, 2.41] | 0.43 [0.39, 0.47] (0.64, 0.42, 0.24) | 2.31 [2.26, 2.37]; 1.44 [0.63, 3.32] | 12.0 [11.7, 12.4] ms | 15.7 [11.5, 21.6] ms | 27.9 [27.6, 28.1] ms | 2.00x |
| 0.25 req/s | bf16-ngram | 1.84 [1.77, 1.93] | 0.28 [0.25, 0.31] (0.42, 0.25, 0.17) | 1.23 [1.18, 1.29]; 1.01 [0.99, 1.03] | 22.5 [20.9, 24.1] ms | 25.7 [25.4, 26.1] ms | 26.0 [25.7, 26.3] ms | 1.07x |
| 1 req/s | bf16 | n/a | n/a | 1.01 [1.01, 1.01]; 1.00 [1.00, 1.00] | 27.1 [25.1, 29.2] ms | 29.7 [27.6, 31.9] ms | 25.6 [24.6, 26.7] ms | 1.00x |
| 1 req/s | bf16-eagle3 | 2.31 [2.30, 2.33] | 0.44 [0.43, 0.44] (0.65, 0.42, 0.25) | 2.29 [2.25, 2.33]; 1.30 [0.93, 1.84] | 13.8 [13.1, 14.5] ms | 17.7 [16.3, 19.3] ms | 29.5 [28.4, 30.7] ms | 1.96x |
| 1 req/s | bf16-ngram | 1.92 [1.86, 1.97] | 0.31 [0.29, 0.32] (0.45, 0.27, 0.19) | 1.24 [1.20, 1.28]; 1.00 [1.00, 1.00] | 26.7 [25.1, 28.4] ms | 30.8 [27.5, 34.4] ms | 29.1 [27.1, 31.2] ms | 1.01x |
| 2 req/s | bf16 | n/a | n/a | 1.01 [1.01, 1.01]; 1.00 [1.00, 1.00] | 34.0 [30.4, 38.1] ms | 38.2 [33.5, 43.6] ms | 30.1 [27.4, 33.0] ms | 1.00x |
| 2 req/s | bf16-eagle3 | 2.34 [2.32, 2.36] | 0.45 [0.44, 0.45] (0.65, 0.43, 0.26) | 2.32 [2.30, 2.35]; 1.19 [0.98, 1.45] | 16.4 [14.9, 18.1] ms | 21.5 [19.2, 24.1] ms | 33.8 [31.8, 36.0] ms | 2.07x |
| 2 req/s | bf16-ngram | 1.97 [1.91, 2.03] | 0.32 [0.30, 0.34] (0.47, 0.29, 0.21) | 1.26 [1.24, 1.28]; 1.00 [1.00, 1.00] | 33.1 [28.8, 38.1] ms | 40.5 [36.0, 45.6] ms | 34.9 [31.0, 39.4] ms | 1.03x |
| 3 req/s | bf16 | n/a | n/a | 1.01 [1.01, 1.01]; 1.00 [1.00, 1.00] | 46.5 [38.5, 56.1] ms | 55.2 [43.7, 69.9] ms | 38.0 [33.1, 43.7] ms | 1.00x |
| 3 req/s | bf16-eagle3 | 2.33 [2.31, 2.35] | 0.44 [0.44, 0.45] (0.65, 0.43, 0.25) | 2.32 [2.27, 2.36]; 1.11 [1.03, 1.20] | 20.1 [17.5, 23.0] ms | 27.2 [24.5, 30.2] ms | 38.1 [35.9, 40.5] ms | 2.31x |
| 3 req/s | bf16-ngram | 1.97 [1.95, 1.99] | 0.32 [0.32, 0.33] (0.47, 0.29, 0.21) | 1.26 [1.22, 1.30]; 1.00 [1.00, 1.00] | 43.1 [36.6, 50.7] ms | 55.9 [47.0, 66.5] ms | 42.2 [37.7, 47.3] ms | 1.08x |

- Reference (no speculation): bf16.
- bf16-eagle3: TPOT p50 speed-up 2.00x at 0.25 req/s, 1.96x at 1 req/s, 2.07x at 2 req/s, 2.31x at 3 req/s: faster at every compared load (up to 3 req/s).
- bf16-ngram: TPOT p50 speed-up 1.07x at 0.25 req/s, 1.01x at 1 req/s, 1.03x at 2 req/s, 1.08x at 3 req/s: faster at every compared load (up to 3 req/s).

**Warnings**

- bf16: no tested load met the SLO (lowest tested 0.25 req/s)
- bf16-eagle3: no tested load met the SLO (lowest tested 0.25 req/s)
- bf16-ngram: no tested load met the SLO (lowest tested 0.25 req/s)

### Qwen/Qwen3-8B: code-edit (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **bf16-eagle3**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'eagle3', 'model': 'RedHatAI/Qwen3-8B-speculator.eagle3', 'revision': '08610ffa01dd9f16731fe8f627b85905b6aa51c4', 'num_speculative_tokens': 3} | $0.1957 [0.1614, 0.2373] | $0.3731 [0.3222, 0.4285] | $0.2783 [0.2536, 0.3055] | $0.5977 [0.5484, 0.6514] | $0.5977 [0.5484, 0.6514] | 875.1 [803.0, 953.7] | 875.1 [803.0, 953.7] | ≥4 req/s (none failed); tied with bf16-ngram | 118 [107, 131] ms | 12.3 [12.2, 12.4] ms | 875.1 [803.0, 953.7] at 4 req/s | -0.005 (ifeval) · inconclusive | n/a | quality not verified | Not ranked: quality gate inconclusive vs BF16 (AWS g6e.xlarge), so its quality is not shown to match; a candidate ranks only on a gate pass; also untrusted (goodput not bracketed) |
| – | **bf16-ngram**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'ngram', 'num_speculative_tokens': 3, 'prompt_lookup_max': 4, 'prompt_lookup_min': 2} | $0.1257 [0.1027, 0.1539] | $0.4534 [0.4017, 0.5095] | $0.2783 [0.2536, 0.3055] | $0.5977 [0.5484, 0.6514] | $0.5977 [0.5484, 0.6514] | 875.1 [803.0, 953.7] | 875.1 [803.0, 953.7] | ≥4 req/s (none failed); tied with bf16-eagle3 | 80 [72, 90] ms | 15.4 [15.0, 15.8] ms | 875.1 [803.0, 953.7] at 4 req/s | -0.006 (ifeval) · inconclusive | n/a | quality not verified | Not ranked: quality gate inconclusive vs BF16 (AWS g6e.xlarge), so its quality is not shown to match; a candidate ranks only on a gate pass; also untrusted (goodput not bracketed) |
| – | **bf16**<br>vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | none met the SLO (fails at 0.5) | n/a | n/a | 875.1 [803.0, 953.7] at 4 req/s | baseline | 877 s (median of 1) | no cost at SLO | No cost at SLO: no tested load met the SLO; test lower loads |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | bf16-eagle3 | bf16-ngram | bf16 | Verdict |
|---|---|---|---|---|---|
| 0.5 req/s | SLO at this load | met | met | failed |  |
| 0.5 req/s | TTFT p50 | 76 [73, 79] ms | **46 [45, 47] ms** | 75 [72, 78] ms | bf16-ngram lower |
| 0.5 req/s | TTFT p95 | 94 [87, 101] ms | **67 [64, 71] ms** | 89 [86, 92] ms | bf16-ngram lower |
| 0.5 req/s | TPOT p50 | 8.8 [8.5, 9.0] ms | **7.7 [7.1, 8.3] ms** | 23.7 [23.5, 23.9] ms | bf16-ngram lower |
| 0.5 req/s | TPOT p95 | 10.8 [9.8, 11.9] ms | 12.9 [9.7, 17.2] ms | 24.2 [23.3, 25.2] ms | no significant difference |
| 0.5 req/s | E2E p95 | 3,139 [2,856, 3,449] ms | 3,286 [2,874, 3,756] ms | 8,113 [7,358, 8,947] ms | no significant difference |
| 2 req/s | SLO at this load | met | met | failed |  |
| 2 req/s | TTFT p50 | 87 [86, 88] ms | **56 [55, 57] ms** | 80 [79, 81] ms | bf16-ngram lower |
| 2 req/s | TTFT p95 | 105 [103, 107] ms | **74 [71, 77] ms** | 97 [94, 100] ms | bf16-ngram lower |
| 2 req/s | TPOT p50 | 9.3 [9.0, 9.5] ms | **8.2 [8.0, 8.3] ms** | 25.0 [24.5, 25.5] ms | bf16-ngram lower |
| 2 req/s | TPOT p95 | **11.2 [11.1, 11.3] ms** | 13.6 [12.6, 14.7] ms | 25.7 [25.2, 26.2] ms | bf16-eagle3 lower |
| 2 req/s | E2E p95 | 3,342 [3,108, 3,593] ms | 3,352 [3,104, 3,619] ms | 8,735 [8,294, 9,200] ms | no significant difference |
| 4 req/s | SLO at this load | met | met | failed |  |
| 4 req/s | TTFT p50 | 93 [89, 97] ms | **59 [55, 64] ms** | 87 [83, 92] ms | bf16-ngram lower |
| 4 req/s | TTFT p95 | 118 [107, 131] ms | **80 [72, 90] ms** | 111 [98, 126] ms | bf16-ngram lower |
| 4 req/s | TPOT p50 | 10.1 [10.0, 10.1] ms | **9.0 [8.9, 9.1] ms** | 27.1 [26.4, 27.8] ms | bf16-ngram lower |
| 4 req/s | TPOT p95 | **12.3 [12.2, 12.4] ms** | 15.4 [15.0, 15.8] ms | 28.6 [26.3, 31.2] ms | bf16-eagle3 lower |
| 4 req/s | E2E p95 | 3,636 [3,444, 3,838] ms | 3,892 [3,575, 4,237] ms | 9,482 [8,688, 10,348] ms | no significant difference |

#### Speculative decoding by load

Speculative decoding by load. Accepted per step is 1 + accepted / drafts (vLLM's mean acceptance length, the target's own token included, over steps that drafted); draft acceptance is accepted / drafted tokens, then the share of drafts kept at each position. Tokens per chunk is measured by the client (completion tokens per streamed chunk, one chunk per engine step): its lowest request shows content the draft cannot predict, which the server's mean hides. ITL is the time between chunks, so under speculation it is the verify step's time, not a token's. Speed-up is the reference's TPOT p50 over the config's at the same load (point estimates).

| Load | Config | Accepted per step | Draft acceptance (by position) | Tokens per chunk: mean, lowest request | TPOT p50 | TPOT p95 | ITL p50 (step) | TPOT p50 speed-up |
|---|---|---|---|---|---|---|---|---|
| 0.5 req/s | bf16 | n/a | n/a | 1.02 [1.01, 1.02]; 1.00 [1.00, 1.00] | 23.7 [23.5, 23.9] ms | 24.2 [23.3, 25.2] ms | 23.6 [23.3, 23.8] ms | 1.00x |
| 0.5 req/s | bf16-eagle3 | 3.06 [3.00, 3.13] | 0.69 [0.67, 0.71] (0.81, 0.69, 0.56) | 3.03 [2.96, 3.11]; 2.15 [1.90, 2.43] | 8.8 [8.5, 9.0] ms | 10.8 [9.8, 11.9] ms | 26.8 [26.7, 27.0] ms | 2.70x |
| 0.5 req/s | bf16-ngram | 3.65 [3.54, 3.75] | 0.88 [0.85, 0.92] (0.92, 0.88, 0.85) | 3.13 [2.93, 3.35]; 1.56 [1.24, 1.95] | 7.7 [7.1, 8.3] ms | 12.9 [9.7, 17.2] ms | 25.1 [25.0, 25.2] ms | 3.09x |
| 2 req/s | bf16 | n/a | n/a | 1.02 [1.01, 1.03]; 1.00 [1.00, 1.00] | 25.0 [24.5, 25.5] ms | 25.7 [25.2, 26.2] ms | 24.4 [24.1, 24.7] ms | 1.00x |
| 2 req/s | bf16-eagle3 | 3.06 [3.05, 3.07] | 0.69 [0.68, 0.69] (0.81, 0.68, 0.56) | 3.05 [3.00, 3.10]; 1.94 [1.82, 2.07] | 9.3 [9.0, 9.5] ms | 11.2 [11.1, 11.3] ms | 27.9 [27.4, 28.5] ms | 2.70x |
| 2 req/s | bf16-ngram | 3.62 [3.57, 3.68] | 0.88 [0.86, 0.89] (0.91, 0.87, 0.84) | 3.14 [3.05, 3.23]; 1.21 [0.93, 1.57] | 8.2 [8.0, 8.3] ms | 13.6 [12.6, 14.7] ms | 26.2 [25.5, 26.9] ms | 3.06x |
| 4 req/s | bf16 | n/a | n/a | 1.02 [1.01, 1.02]; 1.00 [1.00, 1.00] | 27.1 [26.4, 27.8] ms | 28.6 [26.3, 31.2] ms | 26.0 [25.5, 26.5] ms | 1.00x |
| 4 req/s | bf16-eagle3 | 3.06 [3.00, 3.12] | 0.69 [0.67, 0.71] (0.82, 0.68, 0.56) | 3.04 [2.98, 3.10]; 1.81 [1.45, 2.26] | 10.1 [10.0, 10.1] ms | 12.3 [12.2, 12.4] ms | 30.0 [29.6, 30.4] ms | 2.69x |
| 4 req/s | bf16-ngram | 3.61 [3.58, 3.64] | 0.87 [0.86, 0.88] (0.91, 0.87, 0.84) | 3.09 [3.05, 3.14]; 1.11 [0.99, 1.25] | 9.0 [8.9, 9.1] ms | 15.4 [15.0, 15.8] ms | 28.1 [27.3, 28.9] ms | 3.02x |

- Reference (no speculation): bf16.
- bf16-eagle3: TPOT p50 speed-up 2.70x at 0.5 req/s, 2.70x at 2 req/s, 2.69x at 4 req/s: faster at every compared load (up to 4 req/s).
- bf16-ngram: TPOT p50 speed-up 3.09x at 0.5 req/s, 3.06x at 2 req/s, 3.02x at 4 req/s: faster at every compared load (up to 4 req/s).

**Warnings**

- bf16-eagle3: every tested load met the SLO; goodput is at least 4 req/s and cost at SLO is an upper bound
- bf16-ngram: every tested load met the SLO; goodput is at least 4 req/s and cost at SLO is an upper bound
- bf16: no tested load met the SLO (lowest tested 0.5 req/s)

## Competitiveness: our cost at SLO vs public list prices

> Public list prices only: competitor numbers are the per-token prices each provider publishes on its pricing page, with the source and the date it was checked. No competitor endpoint was called or benchmarked, and list prices say nothing about a provider's latency, quality or quantization unless the page discloses it.

Competitor prices last checked 2026-10-04. Flags leave out aggregators and entries whose availability is unverified; they are listed for reference. Our cost is the quoted config's cost at SLO (rank 1; with nothing ranked, an untrusted config of verified quality as a comparison point, with its reason) at the on-demand list price, storage included: $/1M input and $/1M output split the replica's cost by measured prefill time, and $/1M blended is its cost over all tokens at the workload's own input:output mix. Each public price is also shown at every workload's mix, so blended compares like with like. With no price set, the break-even price is the lowest price that covers our cost: at the point estimate, and at the cost CI high bound.

### Qwen3 8B (`qwen3-8b`), ours not recorded

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | no tested load met the SLO | n/a | n/a | n/a | n/a | $0.0770 | $0.3680 | n/a | n/a | n/a | n/a |
| code-edit (open loop) | no config with a cost at SLO has verified quality (bf16-eagle3, bf16-ngram: quality gate inconclusive vs BF16 (AWS g6e.xlarge)); a candidate ranks only on a gate pass | n/a | n/a | n/a | n/a | $0.0770 | $0.3680 | n/a | n/a | n/a | n/a |

**Public list prices** (per 1M tokens)

| Provider | Provider model | $/1M in | $/1M out | Precision | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop); code-edit (open loop)] `no_cost_measurement`: qwen3-8b: no measured cost at SLO
- [chat-sharegpt (open loop); code-edit (open loop)] `no_public_comparison`: qwen3-8b: no eligible public list price

### Providers with no public per-token price

- Groq: Enterprise plans only; contact sales. (checked 2026-10-04)
- Cerebras: No public per-token price. (checked 2026-10-04)
- Lambda: Inference offering winding down. (checked 2026-10-04)
- Hyperbolic: Inference offering retired. (checked 2026-10-04)
- Nebius: Neither model is in its catalog. (checked 2026-10-04)

## Methodology and provenance

- **SLO:** TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 20 ms, error rate ≤ 1.00%
- **Load generation:** open loop (requests arrive on a fixed schedule at a set rate, req/s)
- **Content:** realistic
- **Dataset:** ShareGPT_Vicuna_unfiltered (ShareGPT_V3_unfiltered_cleaned_split.json) @ 192ab2185289094fc556ec8ce5ce1e8e587154ca (license: HF card: apache-2.0. The conversations were scraped from sharegpt.com and contain ChatGPT outputs, and the card's license does not settle OpenAI's terms of use. Loom uses them only as request shapes for load (prompt text in, lengths out), never ships or trains on them, and publishes only latency and cost.); source: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- **Dataset:** CommitPackFT, Python (data/python/data.jsonl) @ fc56fe33c030c6daa414c2b112c932b8eed085e6 (license: HF card: MIT. Each record keeps its repository's license; this profile uses only MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense and CC0 records. Loom uses them only as request text for load (prompt in, lengths out), never ships or trains on them, and publishes only latency and cost.); source: https://huggingface.co/datasets/bigcode/commitpackft
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

### bf16 · chat-sharegpt, open loop (`48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-west-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-west-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 12; provenance digests: `413a33c5c363cd9b062ea9b60ade87f1b30d0586eedda2ad4ac786e7198cba99`, `490f5ca6611dad032e468f8c6749c729956d40e609990b1ebb8983fea4b905d1`, `5f34e6d92dc6343950f9cc7bd613c015535da64ec77d4da3a2c1154fb8d7d2b0`, `77865aed880a5308ef688261f42d0d10c74c46142a381b84aa1083e2c228cfc5`, `7dcbdc9fe4d453f242a300653e3ac82560b94b67ba99dbaf13bbe33f09c65103`, `951d623229aaf2c1fa353bdbc25adb53eb8ec0a5d828b16392b1f0af531e87f9`, `9576bab23b01cf35d0a0e1a59396f6dddd57a639b588ede96b1f5d11d9f36961`, `9cb80532cb2e0e6eb53548021143b73a289bde771aa196484baac248bf906b81`, `a7addb6e4ee11eccb7cf502cdc601f82e998cf1f7666afa63a8e827d2f4805be`, `c556c68f0a90ea19fead7a37604c630c30ccdec5dd695f1d47df0ce191b3637e`, `cefa87b60f1a5d8357c3127bac68ccd7b9aac53ab2e477fc80c21cf3ead8500d`, `df95dd24258400d7eb85b64943682a09c6c652edf333b27d42993e70dbbb7c3e`
- **Reproduce:** `bench reproduce 7b4a5902-beeb-4b6d-a98c-fc8730750d84`

### bf16-eagle3 · chat-sharegpt, open loop (`e70e300e705c8d628046bd3559c0490cc43071b8203c6665200fd3d125c0ef9f`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'eagle3', 'model': 'RedHatAI/Qwen3-8B-speculator.eagle3', 'revision': '08610ffa01dd9f16731fe8f627b85905b6aa51c4', 'num_speculative_tokens': 3}
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-west-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-west-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 12; provenance digests: `0a330b6e96117e88a99b1f2d4354cf4620f70ee843a5df9132657123c76a0b59`, `1a75912c133bf09c51148a069a8e67b33b9b039de7a79c7b9e26c0d53a200a09`, `2ff064d4070e0f4db833e995ea8e7d2ecb62b44253af7a28774a00d8ce860975`, `4b5d07f374eeefc6b90cd7e9106e0081c66f760da193a2af80a49e85053e9b7a`, `575c87c0fed0c02f75462e1405c02f23eb000a75f73795b9d8c0a27d2e0bfdd9`, `8da7981883d844ddc275d78ce9a40ada8e8af7f2c2d9bc22bea80767389a1f51`, `bbe77c694eeae5cfe85d584609c5459b3317ac45798adf5e102f724be996e2ca`, `c6e4ddded8880734e9c6d133cfb0bbe23ebe6120d89c425e786119e487dd4087`, `d706002c55cf4f0e29651ee6cfee25ac2eb0842192c7c91bb78ece5db6fa20f9`, `db1d340cc30958be372dca1aa5dbc4c4c74752ad0cecab06a26f1b1e64bd532b`, `eb8e246282cb9210628502fb0e036eb1b5a17c5cae9564eb721b48b6b9902306`, `f52dd6bc2fcba2929b4af00ff83d6172607ba91defb58d5ac742386bbc957205`
- **Reproduce:** `bench reproduce c6c2fe56-f257-44ce-8237-e6669d77ba8e`

### bf16-ngram · chat-sharegpt, open loop (`97447e81a1b595df53e3bcc67aef4b14c7161df1c93df9425042373f480099d8`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'ngram', 'num_speculative_tokens': 3, 'prompt_lookup_max': 4, 'prompt_lookup_min': 2}
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-west-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-west-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 12; provenance digests: `1042bf6c1b1c4bb1d96cc7ac5414045767560aa07d5148874728230c8f560696`, `1d3dbadbb18db0970970e18857b85a9e70c433f032d8c60560b8d4204fb5eb5c`, `1eff3f2fb1e4eab0d9bfeb7af3d8be31f95078cda06c6ecae467a597bd13de27`, `235664378a3a582469323795717f9eab1b91c45912beda02aaf48e058139fd9b`, `375cbbf9f17cd6cd66ac616d72df61ccb8722205c87224210c9505531540ce95`, `3937382476da6e8e19c799611734d8d4c04e19b9456f9ca77a48dff506478c09`, `51db04488df10bbbe4cb4c9e24e718a6cf3d7ccf05150b9025f622626c29f276`, `51ef2f357fe08812a2c2d4166f2484ade85dc2fd603605b3063f2b1463939599`, `7ce7888d624f345c48bbf904e855f89ba942e633de63e25e81d0dd41e91b1098`, `bcb2473f51e7d0174159786dbfb5c03158cd15eb3b4bb7eb7aed67e3eda75f16`, `e40fbc0a8eb747b1582bc1b8ecd74cae7474bde9ab0bf550dc1dd910b4330dd2`, `e71ef6b465c362b32b62ac6acdff5b8a97a981722d4766bf347b1197352766d5`
- **Reproduce:** `bench reproduce 79c93f64-ddb0-4f56-9ec2-da896a78f015`

### bf16-eagle3 · code-edit, open loop (`e70e300e705c8d628046bd3559c0490cc43071b8203c6665200fd3d125c0ef9f`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'eagle3', 'model': 'RedHatAI/Qwen3-8B-speculator.eagle3', 'revision': '08610ffa01dd9f16731fe8f627b85905b6aa51c4', 'num_speculative_tokens': 3}
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-west-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-west-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 9; provenance digests: `1ca4680eac25c2b3b29dcbd74ee1219d59f3aa5fee444565f1df18a74427e4d9`, `365fffb3b28a8ff86b9b14778a327886babecbf70361b11eed1d42ccaf261f26`, `3692308a162768bfd70f846ec237656021b0f030b3fd07a39fbac4237b87f2dd`, `4054887ee5480e9e2dda550bfdfb24fc406b7a3fad8c5056e6dee8d748946cba`, `6cf00387a0d43f3ae7b8338a2bf876ee51adaf0935cfcd38c20a5e44850a2bdd`, `9152f20ab77ab5a1be93c7dbaf6256d567ed477ef26ad852e9985c78f318131d`, `a36bdc0e30bf225ea6c5d37d99884a8760156d944fbd7d1692fb780ea592c299`, `b4d564a24a5202dc1dca3a9e0debfd0f7b446788522e87b8b6e8fb6b4205fcdf`, `eaf11c647285a6a56f9482d31b9de473e16144d73cc56a562db368faf0d7328f`
- **Reproduce:** `bench reproduce 8dcbea3c-23c3-4c2d-a66f-103bcbbfc99c`

### bf16-ngram · code-edit, open loop (`97447e81a1b595df53e3bcc67aef4b14c7161df1c93df9425042373f480099d8`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand) · speculative_config={'method': 'ngram', 'num_speculative_tokens': 3, 'prompt_lookup_max': 4, 'prompt_lookup_min': 2}
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-west-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-west-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 9; provenance digests: `461cb8d53d47f418344fcd5e81e94ad65ea411b7bdb412ca0a1d8eb89c6d0b87`, `583b3e04edc31012bc6b60bac72f1e6eb4442e427219df5a287644b701b2b224`, `61b2c81bcf1ff3108c69f82ddee5a7d50cd12771193dcb3aea66340362224cce`, `6547ddc9a002ae8260694617c1c7496596b172d4ce489d85a0ff19f32004b757`, `6a794468a67c7eba2c79e35ee99a524a4095630144c3b1aee02e0220c79b0186`, `7857da018430807d9c3b519c64037d5adce2af2a1d60cfacc13e482a3d404d08`, `8e5d7d7eee3ff418fd284c9e7423b1d756e8ad52b9824702218a682a7047a5d0`, `c3a68cd3226f00126b7caf159a089b347e208cdf6c46f5543d37e0527835be21`, `db63ba8822ca0ef81c32026f0820a1e1eeab1a5e19d3ac263f9be02e11b40122`
- **Reproduce:** `bench reproduce b4ebc8b9-e7fa-4ee9-8b46-174d0bfdb13b`

### bf16 · code-edit, open loop (`48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375`)

- **Config:** vllm 0.30.0 · BF16 weights, KV cache auto (model dtype) · TP1 · 1×L40S · AWS g6e.xlarge (on-demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-west-2 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `7fbfdb6bd606839eda1b0a60c7a341a8cdbd592b`; loom-bench 0.1.0
- **Price:** aws/us-west-2 g6e.xlarge + 200 GB EBS gp3 volume, from the price book: on-demand $1.8829/h; spot n/a; committed 1y n/a; source: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-west-2/index.json, https://aws.amazon.com/ec2/pricing/on-demand/, https://aws.amazon.com/ebs/pricing/; last checked 2026-11-03. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB EBS gp3 volume
- **Runs:** 9; provenance digests: `31428e28cbccca2a47901464eda58d92a8a111ad10e7582d21ab002531e15768`, `3a9b1fd15e724a74c205519dbe8ed6c97a3c841fcd631b659f7f26a260b38b7e`, `52821f061306aff199149c2b4e55f03372169b4fb440431379d3d6a0656342e8`, `553bfa7b3de3e110b60cce457b5c1e336dc7132168c0b3861c998efe9547b715`, `725cb5c880792f46bebab97281efd67d40f528ae8630f1b9576bf38cf1051f03`, `73f76470e89bb1d2951d2c7b958b9debfb4bac0c6828b7d217c6696e32b55ab4`, `c9934c7aec81b1248e2240ba0723818e737b36c0df64fd9bdff191ae9ec30322`, `e27ee1d3cf5aef32b50725241dafab3624ad098a1536c18ed41739021bfebedb`, `e4351c95d6590c3b7ffe3cf2bf44a95515c420bfa448ce4cb5d7d7922c80eea5`
- **Reproduce:** `bench reproduce 3f077b17-8782-42bd-87ac-c87e720fdd67`
