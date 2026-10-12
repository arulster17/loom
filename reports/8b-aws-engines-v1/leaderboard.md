# Loom leaderboard: cost at SLO

## Summary

What one replica costs us to serve at the SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%), at the on-demand list price including its storage, at the highest tested load that met the SLO; 95% confidence intervals in brackets. $/1M input and $/1M output split the replica's cost by measured prefill time (methodology at the end); $/1M blended is the replica's cost over all tokens at that workload's own input:output mix and needs no split. Each workload quotes one config: trusted first, then quality verified (the gate's reference or a gate pass), then leaderboard rank; a config that failed the quality gate is never quoted. Nothing here relaxes a check: untrusted figures are quoted only with their reason, and the ranking is the leaderboard's.

### Qwen/Qwen3-8B

Hardware: 1×L40S (aws g6e.xlarge, $1.8829/h on-demand incl. 200 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | **bf16** (quoted) | rank 1 | 2.213 req/s (fails at 2.449) | $0.0950 [0.0805, 0.1121] | $0.4313 [0.3425, 0.5333] | $0.1765 [0.1654, 0.1880] | baseline |
| chat-sharegpt | sglang | quality gate failed | 1.633 req/s (fails at 2) | $0.0838 [0.0561, 0.1250] | $0.7299 [0.5642, 0.9266] | $0.2453 [0.2019, 0.2980] | -0.111 (tool_calling_strict) · fail |

**chat-sharegpt** (1,007 input / 322 output tokens per request)

- bf16: $0.0950 [0.0805, 0.1121] per 1M input tokens, $0.4313 [0.3425, 0.5333] per 1M output tokens, $0.1765 [0.1654, 0.1880] per 1M tokens blended at this mix, holding the SLO up to 2.213 req/s (fails at 2.449).
- What limits bf16: at 2.449 req/s, TPOT p95 is 45.6 [37.8, 55.0] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.88×); OpenRouter $0.1989 (aggregator, our cost 0.89×). Our blended cost is $0.1765; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**Quality**

- bf16 (unquantized): the reference the quality gate compares against; gsm8k 0.904, ifeval 0.826, json_schema 0.909, tool_calling 0.978, tool_calling_strict 0.978.
- sglang (unquantized): quality gate vs bf16 is fail: gsm8k, ifeval, tool_calling, json_schema, divergence, sanity pass; tool_calling_strict fail (delta -11.11 pts [-16.30 pts, -5.93 pts], n=135 is a drop of more than 6.00 pts). It is not ranked.

## Leaderboards

Each table lists one model on one workload. $/1M input and $/1M output split the replica's cost by measured prefill time; $/1M blended is its cost over all tokens at that workload's own input:output mix (methodology below). Rows are ranked by the rank key, the cost under the experiments' declared allocation (all_output charges the whole replica to output tokens), at the on-demand list price, cheapest first; spot, committed-1y and as-run costs of the rank key are shown where available. Every price includes the replica's block storage. Values are point estimates (geometric means for latency and throughput) with 95% confidence intervals in brackets. Goodput is the highest tested load that met the SLO; raw peak throughput ignores the SLO and is not goodput. Unranked rows (quality gate failed, untrusted, or no cost) are listed last.

Goodput is searched on a grid of loads, so it is known only to a bracket: at least the goodput load, below the load that failed (shown as "fails at"). Configs whose brackets overlap are tied within the search resolution: their goodput, throughput and cost at SLO come from the same grid point and are not a measured equality.

### Qwen/Qwen3-8B: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, spot | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **bf16**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · g6e.xlarge (on_demand) | $0.0950 [0.0805, 0.1121] | $0.4313 [0.3425, 0.5333] | $0.1765 [0.1654, 0.1880] | $0.7283 [0.6365, 0.8334] | $0.7196 [0.6289, 0.8235] | $0.7283 [0.6365, 0.8334] | 718.1 [627.6, 821.8] | 718.1 [627.6, 821.8] | 2.213 req/s (fails at 2.449) | 341 [333, 349] ms | 41.7 [39.3, 44.3] ms | 1,454.4 [1,304.6, 1,621.3] at 4.5 req/s | baseline | 587 s (median of 1) | ranked | Only ranked config at SLO; quality baseline |
| – | **sglang**<br>sglang 0.5.21 · unquantized · TP1 · 1×L40S · g6e.xlarge (on_demand) | $0.0838 [0.0561, 0.1250] | $0.7299 [0.5642, 0.9266] | $0.2453 [0.2019, 0.2980] | $0.9812 [0.8212, 1.1723] | $0.9695 [0.8115, 1.1584] | $0.9812 [0.8212, 1.1723] | 533.1 [446.1, 636.9] | 533.1 [446.1, 636.9] | 1.633 req/s (fails at 2) | 287 [286, 288] ms | 38.6 [35.9, 41.4] ms | 1,454.4 [1,304.6, 1,621.3] at 4.5 req/s | -0.111 (tool_calling_strict) · fail | n/a | quality gate failed | Not ranked: failed the quality gate vs bf16 (worst: tool_calling_strict -0.111) |

#### Latency at equal load

Latency at equal load: each config at the loads every config on this board ran (the loads at or below the lowest goodput, plus the highest common load). Geometric means across repetitions with 95% CIs; a config is lower on a metric only when its CI lies entirely below every other config's, otherwise there is no significant difference.

| Load | Metric | bf16 | sglang | Verdict |
|---|---|---|---|---|
| 4.5 req/s | SLO at this load | failed | failed |  |
| 4.5 req/s | TTFT p50 | 1,308 [22, 79,600] ms | 4,046 [710, 23,049] ms | no significant difference |
| 4.5 req/s | TTFT p95 | 21,812 [11,193, 42,504] ms | 17,336 [10,324, 29,112] ms | no significant difference |
| 4.5 req/s | TPOT p50 | 96.5 [81.7, 114.2] ms | 89.3 [85.3, 93.5] ms | no significant difference |
| 4.5 req/s | TPOT p95 | 113.5 [110.1, 116.9] ms | **97.5 [91.6, 103.9] ms** | sglang lower |
| 4.5 req/s | E2E p95 | 74,510 [60,888, 91,179] ms | 68,619 [59,879, 78,635] ms | no significant difference |

## Competitiveness: our cost at SLO vs public list prices

> Public list prices only: competitor numbers are the per-token prices each provider publishes on its pricing page, with the source and the date it was checked. No competitor endpoint was called or benchmarked, and list prices say nothing about a provider's latency, quality or quantization unless the page discloses it.

Competitor prices last checked 2026-10-04. Flags leave out aggregators and entries whose availability is unverified; they are listed for reference. Our cost is the headline config's cost at SLO (see the summary) at the on-demand list price, storage included: $/1M input and $/1M output split the replica's cost by measured prefill time, and $/1M blended is its cost over all tokens at the workload's own input:output mix. Each public price is also shown at every workload's mix, so blended compares like with like. With no price set, the break-even price is the lowest price that covers our cost: at the point estimate, and at the cost CI high bound.

### Qwen3 8B (`qwen3-8b`), ours unquantized

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | bf16 (leaderboard rank 1) | 1,007 / 322 | $0.0950 [0.0805, 0.1121] | $0.4313 [0.3425, 0.5333] | $0.1765 [0.1654, 0.1880] | $0.0770 | $0.3680 | $0.0950 (CI high $0.1121) | $0.4313 (CI high $0.5333) | -$0.0180 (-23% of price); -$0.0351 at cost CI high | -$0.0633 (-17% of price); -$0.1653 at cost CI high |

**Public list prices** (per 1M tokens)

| Provider | Provider model | $/1M in | $/1M out | Precision | At chat-sharegpt mix (1,007 / 322) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | $0.2000 (our cost 0.88×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | $0.1989 (our cost 0.89×) | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop)] `no_public_comparison`: qwen3-8b: no eligible public list price
- [chat-sharegpt (open loop)] `negative_margin`: qwen3-8b input: price $0.0770/1M is $0.0180 below measured cost $0.0950/1M
- [chat-sharegpt (open loop)] `negative_margin`: qwen3-8b output: price $0.3680/1M is $0.0633 below measured cost $0.4313/1M

### Providers with no public per-token price

- Groq: Enterprise plans only; contact sales. (checked 2026-10-04)
- Cerebras: No public per-token price. (checked 2026-10-04)
- Lambda: Inference offering winding down. (checked 2026-10-04)
- Hyperbolic: Inference offering retired. (checked 2026-10-04)
- Nebius: Neither model is in its catalog. (checked 2026-10-04)

## Methodology and provenance

- **SLO:** TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%
- **Load generation:** open loop (requests arrive on a fixed schedule at a set rate, req/s)
- **Content:** realistic
- **Dataset:** ShareGPT_Vicuna_unfiltered (ShareGPT_V3_unfiltered_cleaned_split.json) @ 192ab2185289094fc556ec8ce5ce1e8e587154ca (license: HF card: apache-2.0. The conversations were scraped from sharegpt.com and contain ChatGPT outputs, and the card's license does not settle OpenAI's terms of use. Loom uses them only as request shapes for load (prompt text in, lengths out), never ships or trains on them, and publishes only latency and cost. Verify before publishing.); source: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
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

### bf16 · chat-sharegpt, open loop (`48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375`)

- **Config:** vllm 0.30.0 · unquantized · TP1 · 1×L40S · g6e.xlarge (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-east-1 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `d6b9ffcbfbe33ffe077bff026f1b94d1e74a54fb`; loom-bench 0.1.0
- **Price:** aws/us-east-1 g6e.xlarge + 200 GB block storage, from the price book: on-demand $1.8829/h; spot $1.8605/h; committed 1y n/a; source: https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json, https://instances.vantage.sh/aws/ec2/g6e.xlarge, https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json, https://aws.amazon.com/ebs/pricing/; last checked 2026-10-04; the spot price is an indicative average and moves hourly. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB block storage
- **Runs:** 15; provenance digests: `1f1092310f58ce0e4d1b5420a808a77ebd993669c975f0851ea59231c0265256`, `376318327aa6d9b5ca28d4090981155ba5aa408370e8efd0bc5f7fe332e8dfc5`, `5d26235d5cccc5f38078c0a901b583795107df70968b0bea633dfcdaca255a6e`, `5f59ed464241287a6ac8697683310de4fedae36449e50c80b29e2fbc2b7abf78`, `66b67ef20227e9bb82091e1a94e5a5e7962384212299bc6d29e659f4fb082aba`, `66de41cb07a76b3ebb5dc61812241d2e678fe96dab2647ed8e4fec6d961a9c4f`, `7c4a18183683e469746d7021546d0745d131509df75cd96fbe9a631b71e1599d`, `8090aad88d9eecd655862a31d9410d4dac68e3061ad3bbe853fca204b5f9240e`, `89550274ab0a91cb807c7d941854b520349ca3c27f149c0cb7d4e91f97ce6d39`, `8c253c7022c82f81ab0f154647b5f33c9f3f38191767bce3a34fa617378560e4`, `a2aacd8365bf461d62c29783f238ef6e4cf36f7363a9e827440e259e35fa15a4`, `afc52c567ae1a87ed9342753977a558b35305157d0cce6970bc45dc945115776`, `cf28d34fb5b707b87aebb0f62ced671440b105d9345cf43b6a46f268f4d69206`, `d30130618edf96c0be60562e939bfdce3791453de43700b1aff91b9e4485a7ca`, `e9731ff8d21cdc5d2363a04bb5a687765089842b24b0edc905ec08aacb33aff6`
- **Reproduce:** `bench reproduce 18708748-b2db-4837-b266-c65d9d287f2f`

### sglang · chat-sharegpt, open loop (`990c5aeefc1f9ee40734b592f9a407dab8d64d3fd45196870fceef42bba776e7`)

- **Config:** sglang 0.5.21 · unquantized · TP1 · 1×L40S · g6e.xlarge (on_demand)
- **Engine:** sglang 0.5.21; image `lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`; digest `lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`
- **Model:** Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218
- **Hardware:** 1×L40S · aws / us-east-1 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `d6b9ffcbfbe33ffe077bff026f1b94d1e74a54fb`; loom-bench 0.1.0
- **Price:** aws/us-east-1 g6e.xlarge + 200 GB block storage, from the price book: on-demand $1.8829/h; spot $1.8605/h; committed 1y n/a; source: https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json, https://instances.vantage.sh/aws/ec2/g6e.xlarge, https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json, https://aws.amazon.com/ebs/pricing/; last checked 2026-10-04; the spot price is an indicative average and moves hourly. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB block storage
- **Runs:** 15; provenance digests: `02fd35c41f0128275fead6721f9aaa172e87e52db823f9d0f208cb72860befd9`, `19d7b60aa87f282dfc7115023cdcda334d41aabe1e1915b16c36719ca3265752`, `1dbd780f6b2b94135a79c91e3f77895fa927a435c07f59df7db41b25e7f0e980`, `3eb92cbc6a07dfe9a4cf1bb7b28b1f28e2c0f00032d8e06823028a87bc9f2dba`, `431d133f685de6b7db3001a128595072711eed4035683e1fcf02ee3e944275d5`, `53c0710f8e51fc01f6b706e8ec47d9c3adbdb2609cbeeb7db38573b3d181485b`, `a8c73827187f4fc0bd744846ed98d458aecf67265b447d2ebd1b74346c26a87a`, `aedfc491cc7b935302531dea99a13f7b92149e180d6d50942cc8912365215aca`, `c9989ebc4977ff1dfa1ae54457d51d6f9eb5ecf2d803d1171f1f210150fbb7ba`, `cf3427eeb9ef3cb582cbff3067456509acb8a022514f00802311ca197dbc048a`, `d1428d7fb1813666d77ba2b316cbb574852b2904e342fb6769c1d63edf5753b2`, `e8ad17a4a6c39da49c17df4bcd52938cc7649562cb3f46fb9ce59ca9eef1502c`, `eeacb1d6b81c66b4d8681a0604d1d507b1e985bfb27ec40c0518fa17e5966253`, `f40cc89e2fdfebb979218986614b49747a3a7ad0b13b3261a8c25eb3be396253`, `fd058da17ce4f6f18dc19c64434f52fe340ce439e89b762ac3844b22bdaee3b9`
- **Reproduce:** `bench reproduce ddf1f748-596a-4404-9c79-f09a14181656`
