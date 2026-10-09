# Loom leaderboard: cost at SLO

## Summary

What one replica costs us to serve at the SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%), at the on-demand list price including its storage, at the highest tested load that met the SLO; 95% confidence intervals in brackets. $/1M input and $/1M output split the replica's cost by measured prefill time (methodology at the end); $/1M blended is the replica's cost over all tokens at that workload's own input:output mix and needs no split. Each workload quotes one config: trusted first, then quality verified (the gate's reference or a gate pass), then leaderboard rank; a config that failed the quality gate is never quoted. Nothing here relaxes a check: untrusted figures are quoted only with their reason, and the ranking is the leaderboard's.

### RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic

Hardware: 2×H100 (runpod h100-sxm-x2, $8.0156/h on-demand incl. 260 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| fixed-1k-1k | **vllm-tp2-fp8** (quoted) | rank 1 | 2 req/s (fails at 4) | $0.4548 [0.4350, 0.4755] | $0.6381 [0.6133, 0.6636] | $0.5464 [0.5320, 0.5612] | -0.017 (tool_calling_strict) · inconclusive |
| shared-prefix | **vllm-tp2-fp8** (quoted) | rank 1 | 4 req/s (fails at 8) | $0.2551 [0.2082, 0.3069] | $0.3879 [0.0000, 0.8952] | $0.2628 [0.2390, 0.2891] | -0.017 (tool_calling_strict) · inconclusive |

**fixed-1k-1k** (1,025 input / 1,024 output tokens per request)

- vllm-tp2-fp8: $0.4548 [0.4350, 0.4755] per 1M input tokens, $0.6381 [0.6133, 0.6636] per 1M output tokens, $0.5464 [0.5320, 0.5612] per 1M tokens blended at this mix, holding the SLO up to 2 req/s (fails at 4). Its quality is not verified against the reference (-0.017 (tool_calling_strict) · inconclusive); see Quality below.
- What limits vllm-tp2-fp8: at 4 req/s, TTFT p95 is 61,747.5 [45,107.8, 84,525.4] ms against the 1000 ms target; TPOT p95 is 79.9 [70.0, 91.2] ms against the 50 ms target.
- Market: public list prices of llama-3.3-70b-instruct (the model this config serves at another precision; providers price the model) at this mix, per 1M tokens: Together AI $1.0400 (our cost 0.53×); Fireworks AI $0.9000 (availability unverified, our cost 0.61×); DeepInfra $0.2099 (fp8, same precision as ours: like for like, our cost 2.60×); Novita AI $0.2674 (our cost 2.04×); OpenRouter $0.2099 (aggregator, our cost 2.60×). Our blended cost is $0.5464, 2.60× the lowest eligible list price (DeepInfra, $0.2099).

**shared-prefix** (2,083 input / 128 output tokens per request)

- vllm-tp2-fp8: $0.2551 [0.2082, 0.3069] per 1M input tokens, $0.3879 [0.0000, 0.8952] per 1M output tokens, $0.2628 [0.2390, 0.2891] per 1M tokens blended at this mix, holding the SLO up to 4 req/s (fails at 8). Its quality is not verified against the reference (-0.017 (tool_calling_strict) · inconclusive); see Quality below.
- What limits vllm-tp2-fp8: at 8 req/s, TTFT p95 is 2,155.1 [518.8, 8,952.2] ms against the 1000 ms target; TPOT p95 is 188.9 [143.2, 249.1] ms against the 50 ms target.
- Market: public list prices of llama-3.3-70b-instruct (the model this config serves at another precision; providers price the model) at this mix, per 1M tokens: Together AI $1.0400 (our cost 0.25×); Fireworks AI $0.9000 (availability unverified, our cost 0.29×); DeepInfra $0.1127 (fp8, same precision as ours: like for like, our cost 2.33×); Novita AI $0.1503 (our cost 1.75×); OpenRouter $0.1127 (aggregator, our cost 2.33×). Our blended cost is $0.2628, 2.33× the lowest eligible list price (DeepInfra, $0.1127).

**Quality**

- vllm-tp2-fp8 (fp8): quality gate vs 2f7fb3f9372b is inconclusive: gsm8k, ifeval, tool_calling, json_schema, divergence, sanity pass; tool_calling_strict inconclusive (delta -1.67 pts [-6.67 pts, +3.33 pts], n=60: CI crosses -6.00 pts, more samples needed). An inconclusive gate blocks the config: its quality is not shown to match 2f7fb3f9372b.

### meta-llama/Llama-3.3-70B-Instruct

Hardware: 2×H100 (runpod h100-sxm-x2, $8.0156/h on-demand incl. 260 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| fixed-1k-1k | **vllm-tp2** (quoted) | untrusted, not ranked | 0.5 req/s (fails at 0.5946) | $0.4466 [0.2789, 0.7153] | $4.2886 [3.4461, 5.2857] | $2.3666 [1.9702, 2.8429] | baseline |
| shared-prefix | **vllm-tp2** (quoted) | untrusted, not ranked | 2 req/s (fails at 2.828) | $0.2679 [0.1586, 0.4526] | $4.3098 [2.1042, 6.6817] | $0.5019 [0.4190, 0.6012] | baseline |

**fixed-1k-1k** (1,025 input / 1,024 output tokens per request)

- vllm-tp2: $0.4466 [0.2789, 0.7153] per 1M input tokens, $4.2886 [3.4461, 5.2857] per 1M output tokens, $2.3666 [1.9702, 2.8429] per 1M tokens blended at this mix, holding the SLO up to 0.5 req/s (fails at 0.5946). No config on this board is trusted, so this figure is indicative only.
- What limits vllm-tp2: at 0.5946 req/s, TTFT p95 is 673.1 [13.4, 33,816.3] ms against the 1000 ms target (the SLO is judged on the CI upper bound, which is over it).
- Caveat: vllm-tp2 is untrusted, so the leaderboard does not rank it: ttft_ms.p95 at goodput: run-to-run CV 21.8% exceeds 10%. Treat its figure as indicative and rerun before relying on it.
- Market: public list prices at this mix, per 1M tokens: Together AI $1.0400 (our cost 2.28×); Fireworks AI $0.9000 (availability unverified, our cost 2.63×); DeepInfra $0.2099 (fp8, ours unquantized: not like for like, our cost 11.27×); Novita AI $0.2674 (our cost 8.85×); OpenRouter $0.2099 (aggregator, our cost 11.27×). Our blended cost is $2.3666, 11.27× the lowest eligible list price (DeepInfra, $0.2099).

**shared-prefix** (2,083 input / 128 output tokens per request)

- vllm-tp2: $0.2679 [0.1586, 0.4526] per 1M input tokens, $4.3098 [2.1042, 6.6817] per 1M output tokens, $0.5019 [0.4190, 0.6012] per 1M tokens blended at this mix, holding the SLO up to 2 req/s (fails at 2.828). No config on this board is trusted, so this figure is indicative only.
- What limits vllm-tp2: at 2.828 req/s, TTFT p95 is 972.1 [88.4, 10,683.5] ms against the 1000 ms target (the SLO is judged on the CI upper bound, which is over it); TPOT p95 is 53.2 [44.4, 63.6] ms against the 50 ms target.
- Caveat: vllm-tp2 is untrusted, so the leaderboard does not rank it: ttft_ms.p95 at goodput: run-to-run CV 10.8% exceeds 10%. Treat its figure as indicative and rerun before relying on it.
- Market: public list prices at this mix, per 1M tokens: Together AI $1.0400 (our cost 0.48×); Fireworks AI $0.9000 (availability unverified, our cost 0.56×); DeepInfra $0.1127 (fp8, ours unquantized: not like for like, our cost 4.45×); Novita AI $0.1503 (our cost 3.34×); OpenRouter $0.1127 (aggregator, our cost 4.45×). Our blended cost is $0.5019, 4.45× the lowest eligible list price (DeepInfra, $0.1127).

**Quality**

- vllm-tp2 (unquantized): the reference the quality gate compares against; gsm8k 0.955, ifeval 0.889, json_schema 0.970, tool_calling 0.450, tool_calling_strict 0.450.

## Leaderboards

Each table lists one model on one workload. $/1M input and $/1M output split the replica's cost by measured prefill time; $/1M blended is its cost over all tokens at that workload's own input:output mix (methodology below). Rows are ranked by the rank key, the cost under the experiments' declared allocation (all_output charges the whole replica to output tokens), at the on-demand list price, cheapest first; spot, committed-1y and as-run costs of the rank key are shown where available. Every price includes the replica's block storage. Values are point estimates (geometric means for latency and throughput) with 95% confidence intervals in brackets. Goodput is the highest tested load that met the SLO; raw peak throughput ignores the SLO and is not goodput. Unranked rows (quality gate failed, untrusted, or no cost) are listed last.

Goodput is searched on a grid of loads, so it is known only to a bracket: at least the goodput load, below the load that failed (shown as "fails at"). Configs whose brackets overlap are tied within the search resolution: their goodput, throughput and cost at SLO come from the same grid point and are not a measured equality.

### RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic: fixed-1k-1k (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **vllm-tp2-fp8**<br>vllm 0.30.0 · fp8 · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95 | $0.4548 [0.4350, 0.4755] | $0.6381 [0.6133, 0.6636] | $0.5464 [0.5320, 0.5612] | $1.0933 [1.0645, 1.1229] | $1.0933 [1.0645, 1.1229] | 2,036.5 [1,982.9, 2,091.6] | 1,018.3 [991.5, 1,045.8] | 2 req/s (fails at 4) | 304 [275, 336] ms | 29.4 [25.2, 34.3] ms | 2,932.8 [2,790.6, 3,082.3] at 4 req/s | -0.017 (tool_calling_strict) · inconclusive | n/a | ranked | Only ranked config at SLO; quality gate inconclusive vs vllm-tp2 (worst: tool_calling_strict -0.017) |

### RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic: shared-prefix (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **vllm-tp2-fp8**<br>vllm 0.30.0 · fp8 · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95 | $0.2551 [0.2082, 0.3069] | $0.3879 [0.0000, 0.8952] | $0.2628 [0.2390, 0.2891] | $4.5400 [4.1275, 4.9938] | $4.5400 [4.1275, 4.9938] | 490.4 [445.9, 539.4] | 245.2 [222.9, 269.7] | 4 req/s (fails at 8) | 417 [345, 502] ms | 38.1 [36.5, 39.8] ms | 1,014.7 [897.9, 1,146.6] at 8 req/s | -0.017 (tool_calling_strict) · inconclusive | n/a | ranked | Only ranked config at SLO; quality gate inconclusive vs vllm-tp2 (worst: tool_calling_strict -0.017) |

### meta-llama/Llama-3.3-70B-Instruct: fixed-1k-1k (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **vllm-tp2**<br>vllm 0.30.0 · unquantized · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95 | $0.4466 [0.2789, 0.7153] | $4.2886 [3.4461, 5.2857] | $2.3666 [1.9702, 2.8429] | $4.7356 [3.9423, 5.6885] | $4.7356 [3.9423, 5.6885] | 470.2 [391.4, 564.8] | 235.1 [195.7, 282.4] | 0.5 req/s (fails at 0.5946) | 271 [158, 462] ms | 31.1 [29.0, 33.3] ms | 969.2 [912.6, 1,029.3] at 1 req/s | baseline | 989 s (median of 1) | untrusted | Not ranked: untrusted (high run-to-run variance); rerun before relying on it |

**Warnings**

- vllm-tp2: ttft_ms.p95 at goodput: run-to-run CV 21.8% exceeds 10%

### meta-llama/Llama-3.3-70B-Instruct: shared-prefix (open loop, synthetic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **vllm-tp2**<br>vllm 0.30.0 · unquantized · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95 | $0.2679 [0.1586, 0.4526] | $4.3098 [2.1042, 6.6817] | $0.5019 [0.4190, 0.6012] | $8.6696 [7.2381, 10.3840] | $8.6696 [7.2381, 10.3840] | 256.8 [214.4, 307.6] | 128.4 [107.2, 153.8] | 2 req/s (fails at 2.828) | 411 [315, 537] ms | 42.0 [39.3, 44.7] ms | 490.4 [445.9, 539.4] at 4 req/s | baseline | 989 s (median of 1) | untrusted | Not ranked: untrusted (high run-to-run variance); rerun before relying on it |

**Warnings**

- vllm-tp2: ttft_ms.p95 at goodput: run-to-run CV 10.8% exceeds 10%

## Competitiveness: our cost at SLO vs public list prices

> Public list prices only: competitor numbers are the per-token prices each provider publishes on its pricing page, with the source and the date it was checked. No competitor endpoint was called or benchmarked, and list prices say nothing about a provider's latency, quality or quantization unless the page discloses it.

Competitor prices last checked 2026-10-04. Flags leave out aggregators and entries whose availability is unverified; they are listed for reference. Our cost is the headline config's cost at SLO (see the summary) at the on-demand list price, storage included: $/1M input and $/1M output split the replica's cost by measured prefill time, and $/1M blended is its cost over all tokens at the workload's own input:output mix. Each public price is also shown at every workload's mix, so blended compares like with like. With no price set, the break-even price is the lowest price that covers our cost: at the point estimate, and at the cost CI high bound.

### Llama 3.3 70B Instruct (`llama-3.3-70b-instruct`), ours unquantized

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-1k-1k (open loop) | vllm-tp2 (untrusted, not ranked: high run-to-run variance) | 1,025 / 1,024 | $0.4466 [0.2789, 0.7153] | $4.2886 [3.4461, 5.2857] | $2.3666 [1.9702, 2.8429] | no price set | no price set | $0.4466 (CI high $0.7153) | $4.2886 (CI high $5.2857) | n/a | n/a |
| shared-prefix (open loop) | vllm-tp2 (untrusted, not ranked: high run-to-run variance) | 2,083 / 128 | $0.2679 [0.1586, 0.4526] | $4.3098 [2.1042, 6.6817] | $0.5019 [0.4190, 0.6012] | no price set | no price set | $0.2679 (CI high $0.4526) | $4.3098 (CI high $6.6817) | n/a | n/a |

**Public list prices** (per 1M tokens)

| Provider | Provider model | $/1M in | $/1M out | Precision | At fixed-1k-1k mix (1,025 / 1,024) | At shared-prefix mix (2,083 / 128) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|---|
| Together AI |  | $1.0400 | $1.0400 | not disclosed | $1.0400 (our cost 2.28×) | $1.0400 (our cost 0.48×) | listed | yes | https://www.together.ai/pricing | 2026-10-04 |
| Fireworks AI |  | $0.9000 | $0.9000 | not disclosed | $0.9000 (our cost 2.63×) | $0.9000 (our cost 0.56×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| DeepInfra | meta-llama/Llama-3.3-70B-Instruct-Turbo | $0.1000 | $0.3200 | fp8 (ours unquantized: not like for like) | $0.2099 (our cost 11.27×) | $0.1127 (our cost 4.45×) | listed | yes | https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct-Turbo | 2026-10-04 |
| Novita AI |  | $0.1350 | $0.4000 | not disclosed | $0.2674 (our cost 8.85×) | $0.1503 (our cost 3.34×) | listed | yes | https://novita.ai/pricing | 2026-10-04 |
| OpenRouter (aggregator) | meta-llama/llama-3.3-70b-instruct | $0.1000 | $0.3200 | not disclosed | $0.2099 (our cost 11.27×) | $0.1127 (our cost 4.45×) | listed | no | https://openrouter.ai/meta-llama/llama-3.3-70b-instruct | 2026-10-04 |

**Flags**

- [fixed-1k-1k (open loop); shared-prefix (open loop)] `no_price_set`: llama-3.3-70b-instruct: no price set
- [fixed-1k-1k (open loop)] `cost_above_market`: llama-3.3-70b-instruct input: our cost at SLO $0.4466/1M is $0.3466 (346.6%) above the lowest public list price, $0.1000/1M (DeepInfra); median $0.1350/1M of 3 providers
- [fixed-1k-1k (open loop)] `cost_above_market`: llama-3.3-70b-instruct output: our cost at SLO $4.2886/1M is $3.9686 (1240.2%) above the lowest public list price, $0.3200/1M (DeepInfra); median $0.4000/1M of 3 providers
- [fixed-1k-1k (open loop)] `cost_above_market`: llama-3.3-70b-instruct blended: our cost at SLO $2.3666/1M is $2.1567 (1027.3%) above the lowest public list price at the workload's token mix, $0.2099/1M (DeepInfra); median $0.2674/1M of 3 providers
- [shared-prefix (open loop)] `cost_above_market`: llama-3.3-70b-instruct input: our cost at SLO $0.2679/1M is $0.1679 (167.9%) above the lowest public list price, $0.1000/1M (DeepInfra); median $0.1350/1M of 3 providers
- [shared-prefix (open loop)] `cost_above_market`: llama-3.3-70b-instruct output: our cost at SLO $4.3098/1M is $3.9898 (1246.8%) above the lowest public list price, $0.3200/1M (DeepInfra); median $0.4000/1M of 3 providers
- [shared-prefix (open loop)] `cost_above_market`: llama-3.3-70b-instruct blended: our cost at SLO $0.5019/1M is $0.3892 (345.2%) above the lowest public list price at the workload's token mix, $0.1127/1M (DeepInfra); median $0.1503/1M of 3 providers

### Llama 3.3 70B FP8 (`llama-3.3-70b-instruct-fp8`), ours fp8

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-1k-1k (open loop) | vllm-tp2-fp8 (leaderboard rank 1) | 1,025 / 1,024 | $0.4548 [0.4350, 0.4755] | $0.6381 [0.6133, 0.6636] | $0.5464 [0.5320, 0.5612] | no price set | no price set | $0.4548 (CI high $0.4755) | $0.6381 (CI high $0.6636) | n/a | n/a |
| shared-prefix (open loop) | vllm-tp2-fp8 (leaderboard rank 1) | 2,083 / 128 | $0.2551 [0.2082, 0.3069] | $0.3879 [0.0000, 0.8952] | $0.2628 [0.2390, 0.2891] | no price set | no price set | $0.2551 (CI high $0.3069) | $0.3879 (CI high $0.8952) | n/a | n/a |

**Public list prices** (per 1M tokens)

Public list prices are those of llama-3.3-70b-instruct, the model this entry serves at another precision: providers price the model, and the precision column says which listings match ours.

| Provider | Provider model | $/1M in | $/1M out | Precision | At fixed-1k-1k mix (1,025 / 1,024) | At shared-prefix mix (2,083 / 128) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|---|
| Together AI |  | $1.0400 | $1.0400 | not disclosed | $1.0400 (our cost 0.53×) | $1.0400 (our cost 0.25×) | listed | yes | https://www.together.ai/pricing | 2026-10-04 |
| Fireworks AI |  | $0.9000 | $0.9000 | not disclosed | $0.9000 (our cost 0.61×) | $0.9000 (our cost 0.29×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| DeepInfra | meta-llama/Llama-3.3-70B-Instruct-Turbo | $0.1000 | $0.3200 | fp8 (same precision as ours: like for like) | $0.2099 (our cost 2.60×) | $0.1127 (our cost 2.33×) | listed | yes | https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct-Turbo | 2026-10-04 |
| Novita AI |  | $0.1350 | $0.4000 | not disclosed | $0.2674 (our cost 2.04×) | $0.1503 (our cost 1.75×) | listed | yes | https://novita.ai/pricing | 2026-10-04 |
| OpenRouter (aggregator) | meta-llama/llama-3.3-70b-instruct | $0.1000 | $0.3200 | not disclosed | $0.2099 (our cost 2.60×) | $0.1127 (our cost 2.33×) | listed | no | https://openrouter.ai/meta-llama/llama-3.3-70b-instruct | 2026-10-04 |

**Flags**

- [fixed-1k-1k (open loop); shared-prefix (open loop)] `no_price_set`: llama-3.3-70b-instruct-fp8: no price set
- [fixed-1k-1k (open loop)] `cost_above_market`: llama-3.3-70b-instruct-fp8 input: our cost at SLO $0.4548/1M is $0.3548 (354.8%) above the lowest public list price, $0.1000/1M (DeepInfra); median $0.1350/1M of 3 providers
- [fixed-1k-1k (open loop)] `cost_above_market`: llama-3.3-70b-instruct-fp8 output: our cost at SLO $0.6381/1M is $0.3181 (99.4%) above the lowest public list price, $0.3200/1M (DeepInfra); median $0.4000/1M of 3 providers
- [fixed-1k-1k (open loop)] `cost_above_market`: llama-3.3-70b-instruct-fp8 blended: our cost at SLO $0.5464/1M is $0.3364 (160.3%) above the lowest public list price at the workload's token mix, $0.2099/1M (DeepInfra); median $0.2674/1M of 3 providers
- [shared-prefix (open loop)] `cost_above_market`: llama-3.3-70b-instruct-fp8 input: our cost at SLO $0.2551/1M is $0.1551 (155.1%) above the lowest public list price, $0.1000/1M (DeepInfra); median $0.1350/1M of 3 providers
- [shared-prefix (open loop)] `cost_above_market`: llama-3.3-70b-instruct-fp8 output: our cost at SLO $0.3879/1M is $0.0679 (21.2%) above the lowest public list price, $0.3200/1M (DeepInfra); median $0.4000/1M of 3 providers
- [shared-prefix (open loop)] `cost_above_market`: llama-3.3-70b-instruct-fp8 blended: our cost at SLO $0.2628/1M is $0.1501 (133.1%) above the lowest public list price at the workload's token mix, $0.1127/1M (DeepInfra); median $0.1503/1M of 3 providers

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
- **Price book last checked:** 2026-10-08

### vllm-tp2-fp8 · fixed-1k-1k, open loop (`0381825d763025fdf1752037fbd925abe0a992e191215227ccb01e48039e5afd`)

- **Config:** vllm 0.30.0 · fp8 · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic @ f50dbad2c84590ca17dc51e207c34321b65ff14b
- **Hardware:** 2×H100 · runpod / secure · h100-sxm-x2 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `7374a4b618fb5c4d941a5102dfa8d8c0a290ddf8`; loom-bench 0.1.0
- **Price:** runpod/secure h100-sxm-x2 + 260 GB block storage, from the price book: on-demand $8.0156/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-08. As run: $8.0156/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-09T04:44:27.075823+00:00) + 260 GB block storage
- **Runs:** 12; provenance digests: `14db207a4f0bbeabccea5beb309dab583e0a5d0f3726419e67e3846e71410138`, `3446d34d791b3f1f2babe01742b9925c2a7d714e1cf488180e91532a59b7a0b8`, `36ffecbfc87698e144aa51dc6d76907b8aed0260d89e44cc2ed52e8d7ceb2a9d`, `41c1967e0fbd44556b86cbfdc42de1d9bf3c3fb14d87a2cab66e5254c0b360b3`, `638f0131e317a9938c86f654255fd66236ea231712929d1ba207d8866a9a9850`, `77b58e7fcbcca938d693bc14f07518ba11894ee6442ffebab1acf258b2b9be32`, `77ff616a8ee750c5aa5dbd8375fca772cebdd61dba5d280129ba7d842779d430`, `9270e72fec59c5abea4470b13ecca4c71aac67f14586ce3d335369d91f627f78`, `ab230123446ecba993c05480ce38c1cfc047d597276fcf0d9b6e11ca90bc2da3`, `e3909f5d63f03e14263086ce729f3716d12bb67e83c733f02bb7b12a3956d8b4`, `e944714572ec66d67a2ac9066a722ded25a2aab1baebc3a4f6db1eb455bfe52e`, `fd045a2d4d66ec58463f858a352232691681f5ac9dd1c4a7c4e79a5aee84b10f`
- **Reproduce:** `bench reproduce 3d413b1f-c5eb-48d8-8e44-279b742de24d`

### vllm-tp2-fp8 · shared-prefix, open loop (`0381825d763025fdf1752037fbd925abe0a992e191215227ccb01e48039e5afd`)

- **Config:** vllm 0.30.0 · fp8 · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic @ f50dbad2c84590ca17dc51e207c34321b65ff14b
- **Hardware:** 2×H100 · runpod / secure · h100-sxm-x2 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `7374a4b618fb5c4d941a5102dfa8d8c0a290ddf8`; loom-bench 0.1.0
- **Price:** runpod/secure h100-sxm-x2 + 260 GB block storage, from the price book: on-demand $8.0156/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-08. As run: $8.0156/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-09T04:44:27.075823+00:00) + 260 GB block storage
- **Runs:** 12; provenance digests: `0f0df246941dfbb4f01cff87b824ba55ab0ac7e6cb71b708611e99ee3bf7456a`, `47534095783006c44a4a6e07d60a63c1495ee0815a49f1a31e2c747733a0ef1b`, `4b71e71d7875436ccddde271317343d573f7b7eaaee9cae5d3a2e091cf5a542d`, `5ff203dc793f311f1493365c0886dc8aab0be0426936185d6dff6dd8fddee7bc`, `6329ca0de38c7c344bcf3e19e1f7f89dd4d4b5778fb913013f3dc9aa316e7c2b`, `6b061f5cbfcb01cf71b2b113c06580bcb7d0744e626f3323d6b341f13765edee`, `759d6da25d6e5d56cf17af4a41b162f1a4362203f89df3e3a0a39979da97600e`, `78ae560f78be6adbf6de2997f9ca579c8024bc29b90c5853a008bd4b6f5dcccb`, `b5f25dae52d50eaf756fbdbe68a409e9bc927f02144db21399615bca4da84149`, `c0c5d3d104bf61bf7deacb8ca4d93bd25cadf4bd95f03c0387ea181651d255cf`, `c5fc7e18e18c6a1274d874a0c5001d8616370277d792844678e49ab27d448ab9`, `e64bdf17d2561e98fcbd49f716bb8140cd73817e79d2abb165def0b181cee968`
- **Reproduce:** `bench reproduce edc55500-bfb3-41ef-9a8f-323e6b8f4bc4`

### vllm-tp2 · fixed-1k-1k, open loop (`2f7fb3f9372ba09550198b14978459e35c68866db852f44d4856c293a5adf8f9`)

- **Config:** vllm 0.30.0 · unquantized · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** meta-llama/Llama-3.3-70B-Instruct @ 6f6073b423013f6a7d4d9f39144961bfbfbc386b
- **Hardware:** 2×H100 · runpod / secure · h100-sxm-x2 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `7374a4b618fb5c4d941a5102dfa8d8c0a290ddf8`; loom-bench 0.1.0
- **Price:** runpod/secure h100-sxm-x2 + 260 GB block storage, from the price book: on-demand $8.0156/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-08. As run: $8.0156/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-09T04:44:27.075823+00:00) + 260 GB block storage
- **Runs:** 12; provenance digests: `09783ef8cb9ae185eb109175e9d420c7b41bb204790503d7bd4aababccaa0ce0`, `3db93366ba0eb7e3d0966083da719d9ccbd5f0498f9bcc480cb68fbf59ecb56d`, `54ebaa78fdb34498f9159a9b8eba2e4f595bf1c953a8353a7e6b4b38aa3df050`, `5b1188e1f2f8b73b44cd8cd993d5442e7d0c5655ec6a20443e881ec014da0626`, `65845c060a094e84ace2b9174ebf72162ffeac69f450b0360c4c071dbabe9f39`, `75b150ce8ff766453ae1f716832ce027c4d773a23b9baab5c75cb09b4b68a6dd`, `7f057e01fb94ba2810b51e00b9690d99105f71c2ff1516382d7bae6c2b4ec1a6`, `cb7e3994be3e1429f3b98d423f18d8da1ef9ddecfcd9fef7bdcc82087e40d85c`, `d969d96226fdb765e26009694604a185e431aca63998577c55d5c498078059e5`, `e03e3970dcf2682b525c34a48e90decc923b9af82f096e820ee9ec4e654deaf6`, `ec2e646d82e744f495945747274d274a086e7c776f288474cac1941493d41712`, `fd8643b45f5837c843647497b778f1f1573856535502e187017654dc5a614ff8`
- **Reproduce:** `bench reproduce 9fc516b1-4050-463e-b181-793f50824f90`

### vllm-tp2 · shared-prefix, open loop (`2f7fb3f9372ba09550198b14978459e35c68866db852f44d4856c293a5adf8f9`)

- **Config:** vllm 0.30.0 · unquantized · TP2 · 2×H100 · h100-sxm-x2 (on_demand) · gpu_memory_utilization=0.95
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** meta-llama/Llama-3.3-70B-Instruct @ 6f6073b423013f6a7d4d9f39144961bfbfbc386b
- **Hardware:** 2×H100 · runpod / secure · h100-sxm-x2 · on_demand; CUDA 13.0, driver 580.126.09
- **Code:** commit `7374a4b618fb5c4d941a5102dfa8d8c0a290ddf8`; loom-bench 0.1.0
- **Price:** runpod/secure h100-sxm-x2 + 260 GB block storage, from the price book: on-demand $8.0156/h; spot n/a; committed 1y n/a; source: https://www.runpod.io/pricing, https://docs.runpod.io/pods/pricing; last checked 2026-10-08. As run: $8.0156/h, on_demand host: runpod API price observed at launch in US-MO-1 (2026-10-09T04:44:27.075823+00:00) + 260 GB block storage
- **Runs:** 12; provenance digests: `08728df1e4ff5cded1929cc2932dd216b526689a4b45a8e2be59520c1bdd805c`, `11ecbae1ec83341159291185fe575b0324431fa3236114871831b724ef0b21ec`, `18f3db412ea37c26c450dd80996e5457b31fce2e8a3673d71c9f8d066d7c16ba`, `692befa28b8c800f09b0898535ebf89751487076452d999404b955c9b31764c3`, `6fd64a536908c9bbb3664533a56ef0f0debee810446518572d7e6ef031082335`, `7f203d2b31990ae53fe664eb1e8c4e9bf0d8bdf41d02705f19d88e7c38ba393e`, `8a8f37aad182e234f309d8312be2f839ee4ed559cb5195c52083b4a5f49ecc94`, `bebaabb2d860688d4b296578d9854ddc902dcd18322f8fba3df09e6664a42a3e`, `e246471cca908fac31829c8cfb343c5e767614cf83c32c54a5ccdcace7ebcbd9`, `e6a939eb1e9d2c699cde2f143346a1c62af823aca2aa463d9415f3940a485164`, `f15f02238d50ff798910df9c649ed922f2dc1222779b71b9c0f4ada0696482f8`, `fc5c4709e3c9334e68d38ca1bcf28546a7755398caf833edc26163721df58a24`
- **Reproduce:** `bench reproduce 3457581f-bfdd-4c36-a302-43b92bffc196`
