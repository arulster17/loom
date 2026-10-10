# Loom leaderboard: cost at SLO

## Summary

What one replica costs us to serve at the SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%), at the on-demand list price including its storage, at the highest tested load that met the SLO; 95% confidence intervals in brackets. $/1M input and $/1M output split the replica's cost by measured prefill time (methodology at the end); $/1M blended is the replica's cost over all tokens at that workload's own input:output mix and needs no split. Each workload quotes one config: trusted first, then quality verified (the gate's reference or a gate pass), then leaderboard rank; a config that failed the quality gate is never quoted. Nothing here relaxes a check: untrusted figures are quoted only with their reason, and the ranking is the leaderboard's.

### Qwen/Qwen3-8B

Hardware: 1×L40S (runpod l40s-x1, $1.1010/h on-demand incl. 80 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | bf16-kv8 | rank 1 | 3.674 req/s (fails at 3.865) | $0.0655 [0.0591, 0.0726] | $0.0713 [0.0569, 0.0865] | $0.0670 [0.0639, 0.0702] | -0.009 (ifeval) · inconclusive |
| chat-sharegpt | **bf16** (quoted) | rank 2 | 2.213 req/s (fails at 2.449) | $0.0557 [0.0486, 0.0638] | $0.2517 [0.2049, 0.3060] | $0.1032 [0.0967, 0.1099] | baseline |
| fixed-1k-1k | bf16-kv8 | rank 1 | 1.66 req/s (fails at 1.837) | $0.0593 [0.0512, 0.0687] | $0.1187 [0.1067, 0.1315] | $0.0890 [0.0836, 0.0947] | -0.009 (ifeval) · inconclusive |
| fixed-1k-1k | **bf16** (quoted) | rank 2 | 1 req/s (fails at 1.107) | $0.0572 [0.0527, 0.0621] | $0.2293 [0.2175, 0.2416] | $0.1432 [0.1372, 0.1495] | baseline |

**chat-sharegpt** (1,007 input / 322 output tokens per request)

- bf16: $0.0557 [0.0486, 0.0638] per 1M input tokens, $0.2517 [0.2049, 0.3060] per 1M output tokens, $0.1032 [0.0967, 0.1099] per 1M tokens blended at this mix, holding the SLO up to 2.213 req/s (fails at 2.449). bf16-kv8 ranks first but its quality is not verified (-0.009 (ifeval) · inconclusive), so the figure quoted is bf16's (baseline, rank 2).
- What limits bf16: at 2.449 req/s, TPOT p95 is 44.6 [36.5, 54.6] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.52×); OpenRouter $0.1989 (aggregator, our cost 0.52×). Our blended cost is $0.1032; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**fixed-1k-1k** (1,024 input / 1,024 output tokens per request)

- bf16: $0.0572 [0.0527, 0.0621] per 1M input tokens, $0.2293 [0.2175, 0.2416] per 1M output tokens, $0.1432 [0.1372, 0.1495] per 1M tokens blended at this mix, holding the SLO up to 1 req/s (fails at 1.107). bf16-kv8 ranks first but its quality is not verified (-0.009 (ifeval) · inconclusive), so the figure quoted is bf16's (baseline, rank 2).
- What limits bf16: at 1.107 req/s, TPOT p95 is 47.2 [40.7, 54.7] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.72×); OpenRouter $0.2860 (aggregator, our cost 0.50×). Our blended cost is $0.1432; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**Quality**

- bf16-kv8 (unquantized): quality gate vs bf16 is inconclusive: tool_calling, tool_calling_strict, json_schema, divergence, sanity pass; gsm8k inconclusive (delta -0.58 pts [-1.29 pts, +0.13 pts], n=1319: CI crosses -1.00 pts, more samples needed); ifeval inconclusive (delta -0.86 pts [-2.34 pts, +0.49 pts], n=541: CI crosses -2.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match bf16.
- bf16 (unquantized): the reference the quality gate compares against; gsm8k 0.904, ifeval 0.821, json_schema 0.910, tool_calling 0.978, tool_calling_strict 0.978.

### RedHatAI/Qwen3-8B-FP8-dynamic

Hardware: 1×L40S (runpod l40s-x1, $1.1010/h on-demand incl. 80 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | **fp8-kv8** (quoted) | rank 1 | 5.239 req/s (fails at 5.511) | $0.0544 [0.0389, 0.0710] | $0.0242 [0.0000, 0.0584] | $0.0467 [0.0418, 0.0522] | -0.003 (json_schema) · inconclusive |
| chat-sharegpt | fp8-kv8-mbt1024 | rank 2 | 4.734 req/s (fails at 4.98) | $0.0603 [0.0451, 0.0777] | $0.0278 [0.0000, 0.0585] | $0.0522 [0.0479, 0.0568] | -0.012 (ifeval) · inconclusive |
| chat-sharegpt | fp8 | rank 3 | 3.865 req/s (fails at 4.066) | $0.0525 [0.0394, 0.0702] | $0.0961 [0.0511, 0.1412] | $0.0636 [0.0592, 0.0683] | -0.005 (gsm8k) · inconclusive |
| fixed-1k-1k | **fp8-kv8** (quoted) | rank 1 | 2.25 req/s (fails at 2.756) | $0.0495 [0.0445, 0.0550] | $0.0827 [0.0762, 0.0894] | $0.0661 [0.0635, 0.0689] | -0.003 (json_schema) · inconclusive |
| fixed-1k-1k | fp8-kv8-mbt1024 | rank 2 | 2.25 req/s (fails at 2.756) | $0.0637 [0.0562, 0.0721] | $0.0686 [0.0605, 0.0767] | $0.0661 [0.0635, 0.0689] | -0.012 (ifeval) · inconclusive |
| fixed-1k-1k | fp8 | rank 3 | 1.5 req/s (fails at 1.66) | $0.0542 [0.0467, 0.0630] | $0.1464 [0.1327, 0.1609] | $0.1003 [0.0940, 0.1070] | -0.005 (gsm8k) · inconclusive |

**chat-sharegpt** (959 input / 326 output tokens per request)

- fp8-kv8: $0.0544 [0.0389, 0.0710] per 1M input tokens, $0.0242 [0.0000, 0.0584] per 1M output tokens, $0.0467 [0.0418, 0.0522] per 1M tokens blended at this mix, holding the SLO up to 5.239 req/s (fails at 5.511). Its quality is not verified against the reference (-0.003 (json_schema) · inconclusive); see Quality below.
- What limits fp8-kv8: at 5.511 req/s, TPOT p95 is 44.9 [34.2, 59.0] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Market: public list prices of qwen3-8b (the model this config serves at another precision; providers price the model) at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.23×); OpenRouter $0.2028 (aggregator, our cost 0.23×). Our blended cost is $0.0467; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**fixed-1k-1k** (1,024 input / 1,024 output tokens per request)

- fp8-kv8: $0.0495 [0.0445, 0.0550] per 1M input tokens, $0.0827 [0.0762, 0.0894] per 1M output tokens, $0.0661 [0.0635, 0.0689] per 1M tokens blended at this mix, holding the SLO up to 2.25 req/s (fails at 2.756). Its quality is not verified against the reference (-0.003 (json_schema) · inconclusive); see Quality below.
- fp8-kv8 and fp8-kv8-mbt1024 are tied: their goodput brackets overlap, so the load search cannot separate them on cost; the latency at equal load table compares them.
- What limits fp8-kv8: at 2.756 req/s, TPOT p95 is 49.1 [42.4, 57.0] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Market: public list prices of qwen3-8b (the model this config serves at another precision; providers price the model) at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.33×); OpenRouter $0.2860 (aggregator, our cost 0.23×). Our blended cost is $0.0661; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**Quality**

- fp8-kv8 (fp8): quality gate vs 7a203a3ffacc is inconclusive: ifeval, tool_calling, tool_calling_strict, json_schema, divergence, sanity pass; gsm8k inconclusive (delta -0.28 pts [-1.14 pts, +0.61 pts], n=1319: CI crosses -1.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match 7a203a3ffacc.
- fp8-kv8-mbt1024 (fp8): quality gate vs 7a203a3ffacc is inconclusive: tool_calling, tool_calling_strict, json_schema, divergence, sanity pass; gsm8k inconclusive (delta -0.66 pts [-1.54 pts, +0.25 pts], n=1319: CI crosses -1.00 pts, more samples needed); ifeval inconclusive (delta -1.23 pts [-2.96 pts, +0.49 pts], n=541: CI crosses -2.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match 7a203a3ffacc.
- fp8 (fp8): quality gate vs 7a203a3ffacc is inconclusive: ifeval, tool_calling, tool_calling_strict, json_schema, divergence, sanity pass; gsm8k inconclusive (delta -0.51 pts [-1.29 pts, +0.33 pts], n=1319: CI crosses -1.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match 7a203a3ffacc.

## Leaderboards

Each table lists one model on one workload. $/1M input and $/1M output split the replica's cost by measured prefill time; $/1M blended is its cost over all tokens at that workload's own input:output mix (methodology below). Rows are ranked by the rank key, the cost under the experiments' declared allocation (all_output charges the whole replica to output tokens), at the on-demand list price, cheapest first; spot, committed-1y and as-run costs of the rank key are shown where available. Every price includes the replica's block storage. Values are point estimates (geometric means for latency and throughput) with 95% confidence intervals in brackets. Goodput is the highest tested load that met the SLO; raw peak throughput ignores the SLO and is not goodput. Unranked rows (quality gate failed, untrusted, or no cost) are listed last.

Goodput is searched on a grid of loads, so it is known only to a bracket: at least the goodput load, below the load that failed (shown as "fails at"). Configs whose brackets overlap are tied within the search resolution: their goodput, throughput and cost at SLO come from the same grid point and are not a measured equality.

### Qwen/Qwen3-8B: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **bf16-kv8**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0655 [0.0591, 0.0726] | $0.0713 [0.0569, 0.0865] | $0.0670 [0.0639, 0.0702] | $0.2646 [0.2510, 0.2789] | $0.2646 [0.2510, 0.2789] | 1,155.9 [1,096.7, 1,218.3] | 1,155.9 [1,096.7, 1,218.3] | 3.674 req/s (fails at 3.865) | 373 [352, 396] ms | 47.1 [44.5, 49.8] ms | 1,454.4 [1,304.6, 1,621.3] at 4.5 req/s | -0.009 (ifeval) · inconclusive | n/a | ranked | Cheapest at SLO; 38% cheaper than bf16; quality gate inconclusive vs bf16 (worst: ifeval -0.009) |
| 2 | **bf16**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0557 [0.0486, 0.0638] | $0.2517 [0.2049, 0.3060] | $0.1032 [0.0967, 0.1099] | $0.4258 [0.3721, 0.4873] | $0.4258 [0.3721, 0.4873] | 718.1 [627.6, 821.8] | 718.1 [627.6, 821.8] | 2.213 req/s (fails at 2.449) | 339 [318, 361] ms | 40.8 [38.9, 42.7] ms | 1,454.4 [1,304.6, 1,621.3] at 4.5 req/s | baseline | 256 s (median of 1) | ranked | 61% more expensive than bf16-kv8 at SLO; 9.2% lower p95 TTFT at goodput; quality baseline |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | bf16-kv8 | bf16 | Verdict |
|---|---|---|---|---|
| 4.5 req/s | SLO at this load | failed | failed |  |
| 4.5 req/s | TTFT p50 | 224 [213, 236] ms | 1,142 [31, 41,387] ms | no significant difference |
| 4.5 req/s | TTFT p95 | **446 [415, 480] ms** | 19,698 [9,076, 42,752] ms | bf16-kv8 lower |
| 4.5 req/s | TPOT p50 | **50.2 [48.3, 52.3] ms** | 93.7 [77.4, 113.4] ms | bf16-kv8 lower |
| 4.5 req/s | TPOT p95 | **58.9 [52.9, 65.5] ms** | 112.5 [108.0, 117.3] ms | bf16-kv8 lower |
| 4.5 req/s | E2E p95 | **36,274 [32,970, 39,909] ms** | 72,295 [58,608, 89,177] ms | bf16-kv8 lower |

### Qwen/Qwen3-8B: fixed-1k-1k (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **bf16-kv8**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0593 [0.0512, 0.0687] | $0.1187 [0.1067, 0.1315] | $0.0890 [0.0836, 0.0947] | $0.1780 [0.1673, 0.1895] | $0.1780 [0.1673, 0.1895] | 1,717.7 [1,614.0, 1,828.0] | 1,717.7 [1,614.0, 1,828.0] | 1.66 req/s (fails at 1.837) | 270 [258, 284] ms | 45.4 [42.3, 48.7] ms | 2,313.3 [2,220.6, 2,409.8] at 2.25 req/s | -0.009 (ifeval) · inconclusive | n/a | ranked | Cheapest at SLO; 38% cheaper than bf16; quality gate inconclusive vs bf16 (worst: ifeval -0.009) |
| 2 | **bf16**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0572 [0.0527, 0.0621] | $0.2293 [0.2175, 0.2416] | $0.1432 [0.1372, 0.1495] | $0.2865 [0.2745, 0.2990] | $0.2865 [0.2745, 0.2990] | 1,067.5 [1,022.8, 1,114.1] | 1,067.5 [1,022.8, 1,114.1] | 1 req/s (fails at 1.107) | 256 [245, 266] ms | 43.7 [43.1, 44.3] ms | 1,524.3 [1,428.8, 1,626.2] at 1.5 req/s | baseline | 256 s (median of 1) | ranked | 61% more expensive than bf16-kv8 at SLO; 5.5% lower p95 TTFT at goodput; quality baseline |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | bf16-kv8 | bf16 | Verdict |
|---|---|---|---|---|
| 1.5 req/s | SLO at this load | met | failed |  |
| 1.5 req/s | TTFT p50 | **184 [181, 186] ms** | 325 [190, 558] ms | bf16-kv8 lower |
| 1.5 req/s | TTFT p95 | **269 [256, 282] ms** | 10,738 [9,382, 12,291] ms | bf16-kv8 lower |
| 1.5 req/s | TPOT p50 | **41.1 [39.0, 43.4] ms** | 67.2 [54.1, 83.6] ms | bf16-kv8 lower |
| 1.5 req/s | TPOT p95 | **43.4 [42.8, 44.0] ms** | 75.1 [63.7, 88.7] ms | bf16-kv8 lower |
| 1.5 req/s | E2E p95 | **44,573 [43,974, 45,180] ms** | 80,541 [62,234, 104,234] ms | bf16-kv8 lower |

### RedHatAI/Qwen3-8B-FP8-dynamic: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **fp8-kv8**<br>vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0544 [0.0389, 0.0710] | $0.0242 [0.0000, 0.0584] | $0.0467 [0.0418, 0.0522] | $0.1840 [0.1711, 0.1978] | $0.1840 [0.1711, 0.1978] | 1,662.5 [1,546.5, 1,787.2] | 1,662.5 [1,546.5, 1,787.2] | 5.239 req/s (fails at 5.511) | 336 [272, 414] ms | 41.3 [35.5, 48.2] ms | 2,158.1 [2,123.4, 2,193.3] at 6.75 req/s | -0.003 (json_schema) · inconclusive | n/a | ranked | Cheapest at SLO; 12% cheaper than fp8-kv8-mbt1024; quality gate inconclusive vs bf16 (worst: json_schema -0.003) |
| 2 | **fp8-kv8-mbt1024**<br>vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) · max_num_batched_tokens=1024 | $0.0603 [0.0451, 0.0777] | $0.0278 [0.0000, 0.0585] | $0.0522 [0.0479, 0.0568] | $0.2081 [0.2055, 0.2108] | $0.2081 [0.2055, 0.2108] | 1,469.7 [1,451.0, 1,488.5] | 1,469.7 [1,451.0, 1,488.5] | 4.734 req/s (fails at 4.98) | 376 [304, 466] ms | 36.9 [32.2, 42.3] ms | 2,158.1 [2,123.4, 2,193.3] at 6.75 req/s | -0.012 (ifeval) · inconclusive | 183 s (median of 1) | ranked | 13% more expensive than fp8-kv8 at SLO; quality gate inconclusive vs bf16 (worst: ifeval -0.012) |
| 3 | **fp8**<br>vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0525 [0.0394, 0.0702] | $0.0961 [0.0511, 0.1412] | $0.0636 [0.0592, 0.0683] | $0.2502 [0.2259, 0.2770] | $0.2502 [0.2259, 0.2770] | 1,222.4 [1,103.9, 1,353.6] | 1,222.4 [1,103.9, 1,353.6] | 3.865 req/s (fails at 4.066) | 304 [244, 378] ms | 41.6 [38.3, 45.1] ms | 1,454.4 [1,304.6, 1,621.3] at 4.5 req/s | -0.005 (gsm8k) · inconclusive | 221 s (median of 1) | ranked | 36% more expensive than fp8-kv8 at SLO; 9.5% lower p95 TTFT at goodput; quality gate inconclusive vs bf16 (worst: gsm8k -0.005) |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | fp8-kv8 | fp8-kv8-mbt1024 | fp8 | Verdict |
|---|---|---|---|---|---|
| 4.5 req/s | SLO at this load | met | met | failed |  |
| 4.5 req/s | TTFT p50 | **144 [133, 155] ms** | 186 [173, 200] ms | 199 [173, 229] ms | fp8-kv8 lower |
| 4.5 req/s | TTFT p95 | **295 [276, 315] ms** | 368 [341, 397] ms | 397 [342, 462] ms | fp8-kv8 lower |
| 4.5 req/s | TPOT p50 | 29.1 [27.9, 30.4] ms | 29.8 [28.5, 31.0] ms | 47.5 [40.3, 55.8] ms | no significant difference |
| 4.5 req/s | TPOT p95 | 35.4 [31.0, 40.5] ms | 36.5 [32.0, 41.7] ms | 60.6 [46.6, 78.8] ms | no significant difference |
| 4.5 req/s | E2E p95 | 21,369 [18,992, 24,045] ms | 21,950 [19,676, 24,487] ms | 34,865 [28,587, 42,522] ms | no significant difference |

### RedHatAI/Qwen3-8B-FP8-dynamic: fixed-1k-1k (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **fp8-kv8**<br>vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0495 [0.0445, 0.0550] | $0.0827 [0.0762, 0.0894] | $0.0661 [0.0635, 0.0689] | $0.1322 [0.1269, 0.1377] | $0.1322 [0.1269, 0.1377] | 2,313.3 [2,220.6, 2,409.8] | 2,313.3 [2,220.6, 2,409.8] | 2.25 req/s (fails at 2.756); tied with fp8-kv8-mbt1024 | 241 [223, 261] ms | 39.7 [38.6, 40.9] ms | 3,448.9 [3,288.2, 3,617.5] at 3.375 req/s | -0.003 (json_schema) · inconclusive | n/a | ranked | Tied for cheapest at SLO with fp8-kv8-mbt1024: goodput brackets overlap (2.25 req/s (fails at 2.756)), so the search cannot separate them; compare latency at equal load; quality gate inconclusive vs bf16 (worst: json_schema -0.003) |
| 2 | **fp8-kv8-mbt1024**<br>vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) · max_num_batched_tokens=1024 | $0.0637 [0.0562, 0.0721] | $0.0686 [0.0605, 0.0767] | $0.0661 [0.0635, 0.0689] | $0.1322 [0.1269, 0.1377] | $0.1322 [0.1269, 0.1377] | 2,313.3 [2,220.6, 2,409.8] | 2,313.3 [2,220.6, 2,409.8] | 2.25 req/s (fails at 2.756); tied with fp8-kv8 | 300 [275, 328] ms | 40.0 [39.1, 40.9] ms | 3,448.9 [3,288.2, 3,617.5] at 3.375 req/s | -0.012 (ifeval) · inconclusive | 183 s (median of 1) | ranked | Tied with fp8-kv8 at SLO: goodput brackets overlap (2.25 req/s (fails at 2.756)), so the search cannot separate them; compare latency at equal load; quality gate inconclusive vs bf16 (worst: ifeval -0.012) |
| 3 | **fp8**<br>vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) | $0.0542 [0.0467, 0.0630] | $0.1464 [0.1327, 0.1609] | $0.1003 [0.0940, 0.1070] | $0.2006 [0.1881, 0.2140] | $0.2006 [0.1881, 0.2140] | 1,524.3 [1,428.8, 1,626.2] | 1,524.3 [1,428.8, 1,626.2] | 1.5 req/s (fails at 1.66) | 258 [245, 272] ms | 44.6 [43.9, 45.2] ms | 1,880.2 [1,711.1, 2,066.0] at 1.837 req/s | -0.005 (gsm8k) · inconclusive | 221 s (median of 1) | ranked | 52% more expensive than fp8-kv8 at SLO; quality gate inconclusive vs bf16 (worst: gsm8k -0.005) |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | fp8-kv8 | fp8-kv8-mbt1024 | fp8 | Verdict |
|---|---|---|---|---|---|
| 1.5 req/s | SLO at this load | met | met | met |  |
| 1.5 req/s | TTFT p50 | **127 [124, 130] ms** | 160 [156, 165] ms | 173 [158, 190] ms | fp8-kv8 lower |
| 1.5 req/s | TTFT p95 | **188 [176, 201] ms** | 222 [210, 234] ms | 258 [245, 272] ms | fp8-kv8 lower |
| 1.5 req/s | TPOT p50 | 26.3 [25.7, 26.8] ms | 26.3 [25.7, 26.9] ms | 40.5 [35.4, 46.3] ms | no significant difference |
| 1.5 req/s | TPOT p95 | 27.5 [25.5, 29.6] ms | 27.5 [25.6, 29.5] ms | 44.6 [43.9, 45.2] ms | no significant difference |
| 1.5 req/s | E2E p95 | 28,262 [26,235, 30,446] ms | 28,288 [26,325, 30,396] ms | 45,771 [45,116, 46,435] ms | no significant difference |
| 2.25 req/s | SLO at this load | met | met | failed |  |
| 2.25 req/s | TTFT p50 | 154 [153, 156] ms | 199 [190, 209] ms | 5,331 [35, 805,929] ms | no significant difference |
| 2.25 req/s | TTFT p95 | **241 [223, 261] ms** | 300 [275, 328] ms | 30,145 [26,547, 34,230] ms | fp8-kv8 lower |
| 2.25 req/s | TPOT p50 | 34.5 [31.5, 37.8] ms | 34.7 [31.7, 38.0] ms | 92.7 [90.5, 95.0] ms | no significant difference |
| 2.25 req/s | TPOT p95 | 39.7 [38.6, 40.9] ms | 40.0 [39.1, 40.9] ms | 98.6 [86.3, 112.7] ms | no significant difference |
| 2.25 req/s | E2E p95 | 40,796 [39,680, 41,943] ms | 41,161 [40,272, 42,070] ms | 130,232 [108,777, 155,920] ms | no significant difference |

## Competitiveness: our cost at SLO vs public list prices

> Public list prices only: competitor numbers are the per-token prices each provider publishes on its pricing page, with the source and the date it was checked. No competitor endpoint was called or benchmarked, and list prices say nothing about a provider's latency, quality or quantization unless the page discloses it.

Competitor prices last checked 2026-10-04. Flags leave out aggregators and entries whose availability is unverified; they are listed for reference. Our cost is the headline config's cost at SLO (see the summary) at the on-demand list price, storage included: $/1M input and $/1M output split the replica's cost by measured prefill time, and $/1M blended is its cost over all tokens at the workload's own input:output mix. Each public price is also shown at every workload's mix, so blended compares like with like. With no price set, the break-even price is the lowest price that covers our cost: at the point estimate, and at the cost CI high bound.

### Qwen3 8B (`qwen3-8b`), ours unquantized

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | bf16 (leaderboard rank 2) | 1,007 / 322 | $0.0557 [0.0486, 0.0638] | $0.2517 [0.2049, 0.3060] | $0.1032 [0.0967, 0.1099] | no price set | no price set | $0.0557 (CI high $0.0638) | $0.2517 (CI high $0.3060) | n/a | n/a |
| fixed-1k-1k (open loop) | bf16 (leaderboard rank 2) | 1,024 / 1,024 | $0.0572 [0.0527, 0.0621] | $0.2293 [0.2175, 0.2416] | $0.1432 [0.1372, 0.1495] | no price set | no price set | $0.0572 (CI high $0.0621) | $0.2293 (CI high $0.2416) | n/a | n/a |

**Public list prices** (per 1M tokens)

| Provider | Provider model | $/1M in | $/1M out | Precision | At chat-sharegpt mix (1,007 / 322) | At fixed-1k-1k mix (1,024 / 1,024) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | $0.2000 (our cost 0.52×) | $0.2000 (our cost 0.72×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | $0.1989 (our cost 0.52×) | $0.2860 (our cost 0.50×) | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_price_set`: qwen3-8b: no price set
- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_public_comparison`: qwen3-8b: no eligible public list price

### Qwen3 8B FP8 (`qwen3-8b-fp8`), ours fp8

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | fp8-kv8 (leaderboard rank 1) | 959 / 326 | $0.0544 [0.0389, 0.0710] | $0.0242 [0.0000, 0.0584] | $0.0467 [0.0418, 0.0522] | no price set | no price set | $0.0544 (CI high $0.0710) | $0.0242 (CI high $0.0584) | n/a | n/a |
| fixed-1k-1k (open loop) | fp8-kv8 (leaderboard rank 1; tied with fp8-kv8-mbt1024) | 1,024 / 1,024 | $0.0495 [0.0445, 0.0550] | $0.0827 [0.0762, 0.0894] | $0.0661 [0.0635, 0.0689] | no price set | no price set | $0.0495 (CI high $0.0550) | $0.0827 (CI high $0.0894) | n/a | n/a |

**Public list prices** (per 1M tokens)

Public list prices are those of qwen3-8b, the model this entry serves at another precision: providers price the model, and the precision column says which listings match ours.

| Provider | Provider model | $/1M in | $/1M out | Precision | At chat-sharegpt mix (959 / 326) | At fixed-1k-1k mix (1,024 / 1,024) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | $0.2000 (our cost 0.23×) | $0.2000 (our cost 0.33×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | $0.2028 (our cost 0.23×) | $0.2860 (our cost 0.23×) | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_price_set`: qwen3-8b-fp8: no price set
- [chat-sharegpt (open loop); fixed-1k-1k (open loop)] `no_public_comparison`: qwen3-8b-fp8: no eligible public list price

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
- **Dataset:** ShareGPT_Vicuna_unfiltered (ShareGPT_V3_unfiltered_cleaned_split.json) @ 192ab2185289094fc556ec8ce5ce1e8e587154ca (license: HF card: apache-2.0. The conversations were scraped from sharegpt.com and contain ChatGPT outputs, and the card's license does not settle OpenAI's terms of use. Loom uses them only as request shapes for load (prompt text in, lengths out), never ships or trains on them, and publishes only latency and cost. Verify before publishing.); source: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
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

### bf16-kv8 · chat-sharegpt, open loop (`c9579bddaaee92cbc8124d1a58f8f1ffd27980e41210ccc69bd0fdf36a81ea5b`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in location SE (datacenter not reported) (2026-10-09T23:28:33.837501+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `0ba7e9e126f38769292178311e4f48aead64bc9110a5f39a6b640467633feb25`, `127912c219c7833b49f2115eb7391bb4ba4295e93e94840e4de6938e46c38d26`, `1ed2ed72977fb887c255ecaae98d791f1cfb605bbad394756c65d3bded38c8ce`, `1ee251d02b2348e7758095643bac13dc0e6d7a5594b96b88ae719592759a9047`, `40487865aa7c9b55c3d1d68339897bf1bf99c113e0c7da4cc4ba3c874ed2c528`, `4a05cb0267714aab9468435a8d348a760c581f71ea498a6b9f3ce64fd2aa1431`, `4cadacc0eeabe2bcab9b7dcb6d3f8d2e1111a6740c33d84c80fc29072d744f98`, `636d49750e15c434edf414ed3cd7d1e18ceb4b61f92c38b0dc89410c2f9f6025`, `79a3259eac76636516b9de360e4692efb489dc22746327132a744fb35013779d`, `98226c29e50c64a7d61904d5b50741207bce16b539c074655845032b5d614c7a`, `c34050a76d83dd80550ec018fa704f2c5ce9e7ce73e7601a97f6a677f9db03df`, `c88b5bcce7ca6c208e7a518bdcf8b064583e0f0df83f696fe1dddec963492a87`, `c96d4223c0f04dc5ad3b2b1036396526f8fab678772965e10a15413b72015185`, `da000f34398288bcc037e2da1c6d78c79dd93aebdd6e0cc0ed0a6b4ad88536a9`, `e4cd7fa9c370cd4b86b49ecd3d6decb50e9bdfdb26fde192620c580ae7808f88`
- **Reproduce:** `bench reproduce 5e7c5c68-f66a-4650-98f5-5486fbfd5708`

### bf16 · chat-sharegpt, open loop (`7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in location SE (datacenter not reported) (2026-10-09T23:28:33.837501+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `0530975ae53bbf8cf6c6ddf9ae695a0cdc4f8a05840059f1cd38f6eafdd122c0`, `1cd3e0580407afe943115adb77b9a2bdba1e925bbcff40b6f2035dbe0ff86999`, `2b3f5d435423f74784f1e69186fab735b447484b8b70bcdbdd7dc40bda80a659`, `58f7261af1a7413f8426058c3df45c823f44bd6bfa6fe30c8cf0c4fb28234b35`, `6aa798b940208bc7156095437008ce55772d07bcc02133e694a33ef28ccd79d8`, `6e98674d5a35da54a92a21dc61b46202d1001c8a86b1ca932175d349d1087722`, `95d6249e6ba627bfcc63513d6581f83df18bc38500e7a57d03786e250628fa9a`, `976dd5a25a70f7900ac154a135d4b4ab8a49452a4e6dec9e17246f2d723398ea`, `a89a7f5f37e033c6b96e96268fa3bafe74d43de0e1445a8c409dedef941a54ef`, `be771274b2a0d01dbba74827d36d0e4897e2a3abd6b3a7e041ecd39ad5dd3a34`, `cc07015c9448f7a4aaa9cab829dcc8a1fceaa91fc201a74999fd9bfe81faff28`, `cd0b7b59198b17a06c48be89f1941301d2bc045122f898b27348871de948e3d2`, `d18020028bc402008ffa2b09c15938a4116fdb903338171448e389c8d6111431`, `d9a816b452f34b6b90b224f9267d979da06fbe6363201d91e16e681f40891aa6`, `fa5454db8f969d524fe63e8e259078dc85ff54283bbadd745894f32efec2cd1f`
- **Reproduce:** `bench reproduce 5d02e387-d3d0-4970-bcf9-a6ebb8eeafba`

### bf16-kv8 · fixed-1k-1k, open loop (`c9579bddaaee92cbc8124d1a58f8f1ffd27980e41210ccc69bd0fdf36a81ea5b`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in location SE (datacenter not reported) (2026-10-09T23:28:33.837501+00:00) + 80 GB block storage
- **Runs:** 12; provenance digests: `24dd96fa52b2ecac9ffd1ae1bee36f22ba56ca1f41cfc8c126e0f5b5d78d7b65`, `4c4627b92acf75702e7437164945ca181b1056e3976c4e315828c732f2980aeb`, `5315f3e4ca1f81f7839824513bab9471995cfb7b30f83733cb47985bd382142f`, `636957da3ea31368253c6e17dcd0ae772f0b18d45fb433e5293f72ec433a8faf`, `74750a06de2fb0c99267e69f5e7eac70ad89019ff7379e12691eb2da45600637`, `94d82b92d2d57dc02f29f4ccccdbc70c43e9351b2e42f2fab624182caa493733`, `c99d87730d913f04880c7ed59a6cedf33a2b6063d44226108fd8009bcad417a6`, `db690ec070afc0d31b9af3ad3d66331a98014cc0d68a6a9cad80c4e19ccd3443`, `f14686584bf05cf104a09acc1c8459430e80e2ab66c4ef570d55694ea3b39d0a`, `f579307fd820602e49214c3eba91219cb74b6b67e2ce602d4766194db0e5cf0d`, `f9c81df98945d5e6fb7f83c3b3da9487384f2762858ea1147341e2372e0ae47a`, `fc29b5ce4c68b58828ab78755700635dacab1a247e8c298083b112e5a89773cd`
- **Reproduce:** `bench reproduce 4dff471c-34d6-4d7d-9414-e9e99fa3590a`

### bf16 · fixed-1k-1k, open loop (`7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in location SE (datacenter not reported) (2026-10-09T23:28:33.837501+00:00) + 80 GB block storage
- **Runs:** 12; provenance digests: `1a1032d85d8e463403b5b6eec52fb27606009e83f038e3fbd8f28f35a07ca301`, `1f90dae7d3c01316d368a92a3ce4db6d7aba87cccdee7e4b90b2c4d4eeca8f5c`, `2d7667e06c980ef9b6fa6713bc4a7d6715dd11a90252f77beb9a4b26106b0b3c`, `650c33ede94a2145a8543f2565c36c575479d4f6419324ac21f703feeec74172`, `6ebb6ce4589927b1adaf7fac75250f0de36ab03f1db800ec09057cc6f4c7f363`, `b940b78872eb065f3227c1e533dec6fa3d9d2c6c7900f295d48eced1cc6a4ba3`, `baccde5c5f872af38ae91f33315a7476e07e5d1dfde3618ece85461af5457ab3`, `bb0bfd7efc9e28c520965d9a572c9d6ab38bad52ceb32a44ddf87a07a4d98080`, `bf04dc1977928aa4c5241c5c4fab0cf6d512f093c346c1934f6fd5704a441571`, `cf5eb2adac8d64b469579425e97be542e58ca6b55efe52b362367f6bf025af9d`, `d34dcde3c5d280104cd4ab707e98d84b51e47b32cb0f34c314477465c5d6c115`, `e36a9b22b5ae2e61ee4ef7b6950af7dbfa0687f2bdf1233278779a6e11d8c012`
- **Reproduce:** `bench reproduce 76b4dea5-19e1-4fdb-9ecf-291393a97e6b`

### fp8-kv8 · chat-sharegpt, open loop (`c5e726712cab37cfff7e0ae8108575dc26ec5c7883c835e53149eddac42a4028`)

- **Config:** vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Qwen3-8B-FP8-dynamic @ 05233ce1e0565b5fdc9cfa000ab840152ed30c70
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.178.04
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in OC-AU-1 (2026-10-10T04:51:18.672362+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `0a0eb4b31010acfcb56eddf59fbcba76a87b2c9382b88255f23684e8f5ab5684`, `0d36ce6da40e731226aef47670f36b5dbeb2d0ffae2fa40d7fd28aadb0908817`, `3bc726620dc07cd0dec6694507acef71ed9ef920316f64a17b8f1b068c16329f`, `3edafcfe379e33b2ea324329c0e0c1350193987cf8b92fa827efee0f01a77670`, `4873b5d81c11d43f988eeebd928530a57594f33690d89965897ca8981ccf54ad`, `60eb5f4eb10fd2309f7dae57cc69d116607a5cdbc84eb98e5cd487877fe04ee7`, `6425869f738f4b653bad66a790ea7c3d203279eea651c1677567807bd4351db9`, `985edd11c6cd7fede98d62363bd805ce70668c11a7dbbef16be0265bfcc903db`, `9ec62002f9953ce66bfe9673e944aac90de73ccbc63b7392ddaf68bc005c8c9a`, `a7998ae03d3699d2ccb023932d9628d656bfc0ea2ee1a3445f60cdd87fc82213`, `c274d956b686f3518f6d83602d30b726506d6ccb51406c107b62a2595d591fea`, `dd533fadc71e3c5b7f95027279467f0b952ccb285428a474d29807b26f06609e`, `e06c907914501bafc57fa325911f9d29080f0613d2b3fde2342b192227db3cde`, `edf6589b666c7cdaf769ed1d0aad22d82fb02af3b0aa78a461ff550e604a0439`, `efc1dba814a009c2c5ddf046c0e012df5c5f17062e529a2cd90b820686934f52`
- **Reproduce:** `bench reproduce 62e584e8-4dd6-4381-9013-d84664e91646`

### fp8-kv8-mbt1024 · chat-sharegpt, open loop (`4bad051517ed6624d52bb407e1e310e567e58d240110a0b91232fdf97ae77ea6`)

- **Config:** vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) · max_num_batched_tokens=1024
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Qwen3-8B-FP8-dynamic @ 05233ce1e0565b5fdc9cfa000ab840152ed30c70
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-10T10:04:20.822340+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `0bab28e2f23534c5dbb8df65e2a5c84552a63e7b1f1dce7f0d39c838849796d8`, `3f374e272af1ac6451229b71fb5129ea7ef0456925019112db18ba5a6d9bf59a`, `4489fc8cf410b5cc3dabecf828ae11316b740c74d2c3d760cd80a3dc8eacf450`, `5bf3edee916c4e97bb30932cee433228d566616bc86f54d8545c7762928218ba`, `71b91b1c988f1b6e97b4d39946b6bec35eda63157b38423bbc25a2208eab9112`, `7e5d79dbd3991c99dd51c2da71f7a8f3658b9a881d9a379c8909043a55e658e1`, `83535b58f2fad33dac686fbdb5ace12a6b36b8d9cead69f7d10f8a01c0c73d46`, `8ade72f82aab68dbed8debd28dc05c1a3f5c3e1ab50af73c54ab2685be160a63`, `93a48980d2e78a928f35a3b288e63420cb7aa31acee9779a25887d510bf03260`, `a2c6122b13f879378786e3d154f835cbf06d74a996b50ef5345429742a534966`, `a34bc11c34cf09a313febeef29dc148b566d92a82204a7291c408fca40b5c93d`, `b702d14c207f00414f0b6dd4c99c90134f0241e1fc15c8f04613d00175a0801d`, `e033fa1f1b9019ed2cb2cfd8d69ba85665163a57d58bbbf704ab2d15ff618d25`, `e0dd0b7fc0c4ad261185017210c151ea59e8a919724326c26cb5449ccf9a4567`, `f48b6585450ca5dbc357543f5875a8385dd4a4ec1fbc8929c077b1d4b71d6641`
- **Reproduce:** `bench reproduce 3ade7d70-5902-4ceb-b292-412e9d6b4b14`

### fp8 · chat-sharegpt, open loop (`ca01dee38d78f73b9491c94023ffb6b076634fae1d290ad17151d3ce4788d4a6`)

- **Config:** vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Qwen3-8B-FP8-dynamic @ 05233ce1e0565b5fdc9cfa000ab840152ed30c70
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.178.04
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in OC-AU-1 (2026-10-10T04:51:18.672362+00:00) + 80 GB block storage
- **Runs:** 15; provenance digests: `00c71392a5268b8d65d45ab040e70c544b58be1e1837105c15c765fe2dbd8fd7`, `0e16c9ec18ec39c50d794304040bf508cb5e1d4a928ee8bcd91a4c2468addf95`, `3e9b10c5e3bf3f485a3f30e79cd0f5b4f20cfa8e0a0036975f803bc5bcfbbb33`, `41d163792651972a29aa0d6502fb09a5ada7850e39c5ebf6e4d398371f845f53`, `45be4c0a31ddc777c9cf91b89aafae64e7b2d798e7a2cf238b9692d652af0df0`, `4e72d01ede9dca161701644664bda62e268158340463fe5b2ef61c1ca794846c`, `6014ff67bd672508409f208f4942fa7f71850ab6fb09478d685b2050602c3714`, `66690b4289cf2f90b246ee98ca8a90b439ba66a658af119c7b834fa58089fc15`, `837dbabbba51272982df4d79c021df9f5546db93847194995d0c9eaf0644a3d3`, `967e933bf47515ac993b4fcd4c9912291b52a5a24a12eda018c153a9eda8eae3`, `a0e151b5b45c10ca7d2c80fb28b00b70a020f2bd4866f16eb3e1952b7573f912`, `a261182734a1e3ba85020b4d7266616a0aec5c6ba53eb79c5fef0ee3c3b0ebd6`, `b64e6ee75852763983facee3de4c0e7b1548ebc2c032849270ef0b8b2d492bf8`, `c3f76086a13942a36dddea3e2d51b383df420172f3614d78836694596b28753c`, `e6a935deef15c73864e90a0898423f313bea2643cc450450baffc74046009393`
- **Reproduce:** `bench reproduce 34d26308-8ab8-4034-b496-0c128692d32c`

### fp8-kv8 · fixed-1k-1k, open loop (`c5e726712cab37cfff7e0ae8108575dc26ec5c7883c835e53149eddac42a4028`)

- **Config:** vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Qwen3-8B-FP8-dynamic @ 05233ce1e0565b5fdc9cfa000ab840152ed30c70
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.178.04
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in OC-AU-1 (2026-10-10T04:51:18.672362+00:00) + 80 GB block storage
- **Runs:** 12; provenance digests: `2ad37a9b2b2a88629b1cbc09c8516a9d856c0acd8c60a4860abc74b66abd545e`, `3e9ac53f2bd0c20af28478b3ebf2c5073ff292caef00ed6e0ce34ba7cf1ddcb1`, `550ad08b9be5134ce8db03adb2a3ae1223b32f7afa052375ab937b70425e7bd8`, `6230ee7afa95585706736ced6d9bbefd547436785e1ab25638865d6ba067f272`, `65ab2b697f4f47a0360e0702a90a2ad86f4c5505fef4b12056a0d34a5912a2b3`, `835ab92f4b3b13b828c8292ab9e5ce61e27ba940f522be48020d21c7c99d3911`, `98d2e9121820a0bb88864985c46a4fdcfffd64849897638efa8c43e90f458db5`, `a945da1cb0a39ddf62e2bbc59f5b81a415bfff87f6a6e08d33a9490b655bc6f9`, `d932ca9080ac43be2250a730022873dbff96ba77c8f4c8f17cd2bf0a7a7233f5`, `dc8f168091b81604d700a6cc62540deef9935073e9d2440954c985d295e5f5c6`, `f11b6caf98da975454b95c9e2eddb3d45b742a0e782623d55f00670a94a235e6`, `fa0f5a988563a5ea07a7f2a6f49d15553880239a17a4cec6db6bbbffab69bff3`
- **Reproduce:** `bench reproduce 34c0f9b8-9d85-4630-b9c3-959eb90b5017`

### fp8-kv8-mbt1024 · fixed-1k-1k, open loop (`4bad051517ed6624d52bb407e1e310e567e58d240110a0b91232fdf97ae77ea6`)

- **Config:** vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand) · max_num_batched_tokens=1024
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Qwen3-8B-FP8-dynamic @ 05233ce1e0565b5fdc9cfa000ab840152ed30c70
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.159.03
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-10T10:04:20.822340+00:00) + 80 GB block storage
- **Runs:** 12; provenance digests: `00179f12f22469d5d99ae462cb30083bc30f4b85d9644be2ad7f899625f2a34c`, `189c06215aab8f232fa497dc5c218ab7e917febe75e34988e2e40539d0cf409a`, `2d09da4c9cd5f22e08528c071aa1f71c452b83e934ee382854913f13fb4e0775`, `304747b3024f72783145bfde8e94b7d7acebb906cb0e7cddd6036505aa9e1608`, `7f07cd74921a760fb03e78ba72fb3a2624ceb748ce084648553142d0059c47a8`, `90655619ec134ee64385caf962e1de522ad41b5ad46b75c5d1aa964f95076ba0`, `a2d88e2542a000fdad1b1ea28f0f092107e094babd17333ccede0ac6b16a79ce`, `ae3a2e7206bcd7b5a65239ac3bcf0330189e58d211ece2015d811229f0937152`, `c36d7f46e54c81816cfa72269b89daedbeaa547c50f01d1ab09e4ceb2ed81856`, `ecbece0aa48c59a8dc9ec31abac8db614a807aa5a9d08338892ca39c8720bfbc`, `f155db3498ed030c62b712f0d646f753576a7a32a105a55c442de3cb10f55c85`, `f42644d43d7e279d42856af9613126cf72e56b5577b5e274223aed6947eb3f58`
- **Reproduce:** `bench reproduce fbb5fd79-4151-4552-8f01-d0b26a9f208a`

### fp8 · fixed-1k-1k, open loop (`ca01dee38d78f73b9491c94023ffb6b076634fae1d290ad17151d3ce4788d4a6`)

- **Config:** vllm 0.30.0 · fp8 · TP1 · 1×L40S · l40s-x1 (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Qwen3-8B-FP8-dynamic @ 05233ce1e0565b5fdc9cfa000ab840152ed30c70
- **Hardware:** 1×L40S · runpod / secure · l40s-x1 · on_demand; CUDA 13.0, driver 580.178.04
- **Code:** commit `8a4bdb150efdf04b651bad069c8bf2b65519aa5c`; loom-bench 0.1.0
- **Price:** runpod/secure l40s-x1 + 80 GB block storage, from the price book: on-demand $1.1010/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-06. As run: $1.1010/h, on_demand host: runpod API price observed at launch in OC-AU-1 (2026-10-10T04:51:18.672362+00:00) + 80 GB block storage
- **Runs:** 12; provenance digests: `06eae444ce452555847721c85a0d82dc242017c28472ad89851a0e09d1186ab2`, `24792557dac07c450005c0e6668e52304a937316efd53e1b3567c8e94e453b6a`, `29fae4ef9d950dcefbcc56a92ca51b9ea0cd6b30ea879b7f82f89392d898a16e`, `2c1ee544d1bd65392199d5871d5085410ec7ad2039f9bc72dd37f303b84cd298`, `442733155f881638dd49771a6dbe653e97b79000f9fe1b9a78d94247e53bb7da`, `4d42b9a1f98973be05e39358233d19a87efb78bfb813b11423eb16f7383b7f5b`, `60dc9aed41e54a7b438dbb159ebc285bb22184cde0f2bc861b3112ad2b101496`, `65acbea216287cecbcea821ec14aafe803e8ca3ccc70ac26445c14f149791ac0`, `750c58c30d999c3e042ec8ffefd2f1c5699abe4077592d218c2e6183460191db`, `7d91d54b2f18ffaef7b95fd563ec405303d95e75f6e6734ea2fe8fcf1eb840d9`, `cec7f5656452be13a3b185714c5c23d8467029440bda5be215303fd1383e0519`, `df42fea8e4926873068e8b2a6381d2292647c6df1fb88af64e0834b259912904`
- **Reproduce:** `bench reproduce ffbfc939-6526-4189-9d24-53978f5e6427`
