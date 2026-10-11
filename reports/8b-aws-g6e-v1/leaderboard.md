# Loom leaderboard: cost at SLO

## Summary

What one replica costs us to serve at the SLO (TTFT p95 ≤ 1000 ms, TPOT p95 ≤ 50 ms, error rate ≤ 1.00%), at the on-demand list price including its storage, at the highest tested load that met the SLO; 95% confidence intervals in brackets. $/1M input and $/1M output split the replica's cost by measured prefill time (methodology at the end); $/1M blended is the replica's cost over all tokens at that workload's own input:output mix and needs no split. Each workload quotes one config: trusted first, then quality verified (the gate's reference or a gate pass), then leaderboard rank; a config that failed the quality gate is never quoted. Nothing here relaxes a check: untrusted figures are quoted only with their reason, and the ranking is the leaderboard's.

### Qwen/Qwen3-8B

Hardware: 1×L40S (aws g6e.xlarge, $1.8829/h on-demand incl. 200 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | **bf16** (quoted) | rank 1 | 2.213 req/s (fails at 2.449) | $0.0917 [0.0790, 0.1063] | $0.4417 [0.3578, 0.5386] | $0.1765 [0.1654, 0.1880] | baseline |

**chat-sharegpt** (1,007 input / 322 output tokens per request)

- bf16: $0.0917 [0.0790, 0.1063] per 1M input tokens, $0.4417 [0.3578, 0.5386] per 1M output tokens, $0.1765 [0.1654, 0.1880] per 1M tokens blended at this mix, holding the SLO up to 2.213 req/s (fails at 2.449).
- What limits bf16: at 2.449 req/s, TPOT p95 is 44.9 [36.6, 54.9] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).
- Market: public list prices at this mix, per 1M tokens: Fireworks AI $0.2000 (availability unverified, our cost 0.88×); OpenRouter $0.1989 (aggregator, our cost 0.89×). Our blended cost is $0.1765; no listed price is eligible for the flags (aggregators and unverified listings are left out).

**Quality**

- bf16 (unquantized): the reference the quality gate compares against; gsm8k 0.902, ifeval 0.826, json_schema 0.911, tool_calling 0.978, tool_calling_strict 0.978.

### RedHatAI/Qwen3-8B-FP8-dynamic

Hardware: 1×L40S (aws g6e.xlarge, $1.8829/h on-demand incl. 200 GB storage).

| Workload | Config | Standing | Goodput at SLO | $/1M input | $/1M output | $/1M blended | Quality |
|---|---|---|---|---|---|---|---|
| chat-sharegpt | fp8-kv8 | quality gate failed | 5.239 req/s (fails at 5.511) | $0.0937 [0.0657, 0.1214] | $0.0393 [0.0000, 0.1030] | $0.0799 [0.0714, 0.0893] | -0.020 (ifeval) · fail |

**chat-sharegpt** (959 input / 326 output tokens per request)

- No cost at the declared SLO: every config with a cost at SLO failed the quality gate.
- What limits fp8-kv8: at 5.511 req/s, TPOT p95 is 45.2 [35.0, 58.3] ms against the 50 ms target (the SLO is judged on the CI upper bound, which is over it).

**Quality**

- fp8-kv8 (fp8): quality gate vs 48f3d8d1850a is fail: tool_calling, tool_calling_strict, json_schema, divergence, sanity pass; gsm8k inconclusive (delta -0.30 pts [-1.24 pts, +0.66 pts], n=1319: CI crosses -1.00 pts, more samples needed); ifeval fail (delta -2.03 pts [-4.37 pts, +0.25 pts], n=541 is a drop of more than 2.00 pts). It is not ranked.

## Leaderboards

Each table lists one model on one workload. $/1M input and $/1M output split the replica's cost by measured prefill time; $/1M blended is its cost over all tokens at that workload's own input:output mix (methodology below). Rows are ranked by the rank key, the cost under the experiments' declared allocation (all_output charges the whole replica to output tokens), at the on-demand list price, cheapest first; spot, committed-1y and as-run costs of the rank key are shown where available. Every price includes the replica's block storage. Values are point estimates (geometric means for latency and throughput) with 95% confidence intervals in brackets. Goodput is the highest tested load that met the SLO; raw peak throughput ignores the SLO and is not goodput. Unranked rows (quality gate failed, untrusted, or no cost) are listed last.

Goodput is searched on a grid of loads, so it is known only to a bracket: at least the goodput load, below the load that failed (shown as "fails at"). Configs whose brackets overlap are tied within the search resolution: their goodput, throughput and cost at SLO come from the same grid point and are not a measured equality.

### Qwen/Qwen3-8B: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, spot | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **bf16**<br>vllm 0.30.0 · unquantized · TP1 · 1×L40S · g6e.xlarge (on_demand) | $0.0917 [0.0790, 0.1063] | $0.4417 [0.3578, 0.5386] | $0.1765 [0.1654, 0.1880] | $0.7283 [0.6365, 0.8334] | $0.7196 [0.6289, 0.8235] | $0.7283 [0.6365, 0.8334] | 718.1 [627.6, 821.8] | 718.1 [627.6, 821.8] | 2.213 req/s (fails at 2.449) | 326 [317, 335] ms | 41.0 [39.3, 42.8] ms | 1,454.4 [1,304.6, 1,621.3] at 4.5 req/s | baseline | 571 s (median of 1) | ranked | Only ranked config at SLO; quality baseline |

### RedHatAI/Qwen3-8B-FP8-dynamic: chat-sharegpt (open loop, realistic content)

| # | Config | $/1M in at SLO (measured split) | $/1M out at SLO (measured split) | $/1M blended at this mix | Rank key: $/1M out at SLO, on-demand (all_output) | $/1M out at SLO, spot | $/1M out at SLO, as run | Goodput out tok/s per replica | per GPU | Goodput load (search bracket) | p95 TTFT at goodput | p95 TPOT at goodput | Raw peak out tok/s (no SLO) | Quality Δ / gate | Cold start | Status | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | **fp8-kv8**<br>vllm 0.30.0 · fp8 · TP1 · 1×L40S · g6e.xlarge (on_demand) | $0.0937 [0.0657, 0.1214] | $0.0393 [0.0000, 0.1030] | $0.0799 [0.0714, 0.0893] | $0.3146 [0.2927, 0.3382] | $0.3109 [0.2892, 0.3342] | $0.3146 [0.2927, 0.3382] | 1,662.5 [1,546.5, 1,787.2] | 1,662.5 [1,546.5, 1,787.2] | 5.239 req/s (fails at 5.511) | 328 [259, 415] ms | 41.8 [36.3, 48.1] ms | 2,158.1 [2,123.4, 2,193.3] at 6.75 req/s | -0.020 (ifeval) · fail | n/a | quality gate failed | Not ranked: failed the quality gate vs bf16 (worst: ifeval -0.020) |

## Competitiveness: our cost at SLO vs public list prices

> Public list prices only: competitor numbers are the per-token prices each provider publishes on its pricing page, with the source and the date it was checked. No competitor endpoint was called or benchmarked, and list prices say nothing about a provider's latency, quality or quantization unless the page discloses it.

Competitor prices last checked 2026-10-04. Flags leave out aggregators and entries whose availability is unverified; they are listed for reference. Our cost is the headline config's cost at SLO (see the summary) at the on-demand list price, storage included: $/1M input and $/1M output split the replica's cost by measured prefill time, and $/1M blended is its cost over all tokens at the workload's own input:output mix. Each public price is also shown at every workload's mix, so blended compares like with like. With no price set, the break-even price is the lowest price that covers our cost: at the point estimate, and at the cost CI high bound.

### Qwen3 8B (`qwen3-8b`), ours unquantized

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | bf16 (leaderboard rank 1) | 1,007 / 322 | $0.0917 [0.0790, 0.1063] | $0.4417 [0.3578, 0.5386] | $0.1765 [0.1654, 0.1880] | no price set | no price set | $0.0917 (CI high $0.1063) | $0.4417 (CI high $0.5386) | n/a | n/a |

**Public list prices** (per 1M tokens)

| Provider | Provider model | $/1M in | $/1M out | Precision | At chat-sharegpt mix (1,007 / 322) | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | $0.2000 (our cost 0.88×) | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | $0.1989 (our cost 0.89×) | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop)] `no_price_set`: qwen3-8b: no price set
- [chat-sharegpt (open loop)] `no_public_comparison`: qwen3-8b: no eligible public list price

### Qwen3 8B FP8 (`qwen3-8b-fp8`), ours not recorded

| Workload | Our config at SLO (basis) | Tokens per request in / out | Our cost $/1M in | Our cost $/1M out | Our cost $/1M blended | Our price $/1M in | Our price $/1M out | Break-even $/1M in | Break-even $/1M out | Margin in | Margin out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat-sharegpt (open loop) | every config with a cost at SLO failed the quality gate | n/a | n/a | n/a | n/a | no price set | no price set | n/a | n/a | n/a | n/a |

**Public list prices** (per 1M tokens)

Public list prices are those of qwen3-8b, the model this entry serves at another precision: providers price the model, and the precision column says which listings match ours.

| Provider | Provider model | $/1M in | $/1M out | Precision | Availability | In flag comparison | Source | Last checked |
|---|---|---|---|---|---|---|---|---|
| Fireworks AI |  | $0.2000 | $0.2000 | not disclosed | unverified | no | https://docs.fireworks.ai/serverless/pricing | 2026-10-04 |
| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed | listed | no | https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |

**Flags**

- [chat-sharegpt (open loop)] `no_price_set`: qwen3-8b-fp8: no price set
- [chat-sharegpt (open loop)] `no_cost_measurement`: qwen3-8b-fp8: no measured cost at SLO
- [chat-sharegpt (open loop)] `no_public_comparison`: qwen3-8b-fp8: no eligible public list price

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
- **Code:** commit `41599d0a756d0883da9e959f317c8c585395b456`; loom-bench 0.1.0
- **Price:** aws/us-east-1 g6e.xlarge + 200 GB block storage, from the price book: on-demand $1.8829/h; spot $1.8605/h; committed 1y n/a; source: https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json, https://instances.vantage.sh/aws/ec2/g6e.xlarge, https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json, https://aws.amazon.com/ebs/pricing/; last checked 2026-10-04; the spot price is an indicative average and moves hourly. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB block storage
- **Runs:** 15; provenance digests: `08685c1abfdbbe7487ef261029e5f4a9f11562045637017281c1fffa8eebed80`, `2fc5b931dd13893430476f1440108566a58b121fa4f90c6f2a3f1e5b87ceaf04`, `34f3a4670e0f57c3fe4339b0d7f419e0e45b59be81adf2de0df765a4ef7d691c`, `61b0be9c84615727b2e783b683bbedee68474390fd4fa49fe03866453531b851`, `749cd1b93c5f153c45a007a74da524e8e8691f3364e33183e93564253d7929c9`, `781cd0f4191f1204294e774b36642771496e2c7b9ca30d3bdb6237c0778dd5d6`, `7fb801eac123b8735b9bd6b3197991c00805907ef1653147c1ee6c83e5b93e7f`, `87db3d173189d713769d5f11f16a738f7a3a51f5c09a319c4d87ae08dbf0d13c`, `88ed2231bcc126de897e51543100650028f394732fd034ae07afdf5315013966`, `a4dd8e5488db99b477b41b7629a3fb6589231db462cc092d003dbb6a95a3e66e`, `a8cc5cbcdc6e7d5c781ab7f6882cfb3b72df92114fce25324f3abc160e737f6d`, `ba88da0c3c47b3966f40cac406ad756fb923bf73c00bee95f86c220054651782`, `bd4f5c886f4a048c764eb2e9fe63372332dae9c5d6adfe6bcc538d00ea26f469`, `d341881dd24f794946ccb4b035f841c1cab09df1f754a000d2bf782fb7c29cc8`, `eeb212a7b2263a5110a3e3efeb660d0ea45852320e45851e7973c7a9dc9ddd83`
- **Reproduce:** `bench reproduce 773feb5c-0369-4629-a4e7-72806ca95d0b`

### fp8-kv8 · chat-sharegpt, open loop (`b5e45b872bf71aa084f8d9af220a2e3b9855cfa6cbc24777919eddd6b9fc7507`)

- **Config:** vllm 0.30.0 · fp8 · TP1 · 1×L40S · g6e.xlarge (on_demand)
- **Engine:** vllm 0.30.0; image `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; digest `vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`
- **Model:** RedHatAI/Qwen3-8B-FP8-dynamic @ 05233ce1e0565b5fdc9cfa000ab840152ed30c70
- **Hardware:** 1×L40S · aws / us-east-1 · g6e.xlarge · on_demand; CUDA 13.2, driver 595.91.07
- **Code:** commit `41599d0a756d0883da9e959f317c8c585395b456`; loom-bench 0.1.0
- **Price:** aws/us-east-1 g6e.xlarge + 200 GB block storage, from the price book: on-demand $1.8829/h; spot $1.8605/h; committed 1y n/a; source: https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json, https://instances.vantage.sh/aws/ec2/g6e.xlarge, https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json, https://aws.amazon.com/ebs/pricing/; last checked 2026-10-04; the spot price is an indicative average and moves hourly. As run: $1.8829/h, on_demand host: price-book on-demand price + 200 GB block storage
- **Runs:** 15; provenance digests: `02d28aae3751758f55111da11153c722788fcdf63e86d1e53c6323beed058d63`, `0cf0fb3a89a377789376e699d669e4b0e79c90312db81300e58919e597c079ac`, `16e5f4304615eaa9778358dbd553df99ff8317e939be86eace1187302a2a57bc`, `2a78cd608673c9073c94736899639bbd7aae0874e408a476c0bac1e43f9cc03a`, `3304da8fa5bee1ee05ae0a44d2f0ec9e3077f05c9ced379aab5a14d5c9641096`, `457a92c7f5433942735f973fd978908ddbf967874ae614ddaff667895c2c50fa`, `5d839286cbc267ddab5e992d96fc3f9ed35f1e510bc4aafe819c5d7315e6b953`, `6ac372d92c5bf909be1683f5b9d1927a446ab7bf0bbacba4b096493c530e565f`, `72affc4dccd7e941ab2fac90b2f2b0395371ad6c31cc685ad23a88a0a7d5a09e`, `7e58dcd34a4c5bc07c559c8ffec196e936aedc1b7376fc0b1759ba6a6917a4ee`, `b9f94e0aa0117f76e4ce126f607021ebfc1cf5236c7b9fa8847ece18b877ccdd`, `cabc6d249d84c07a46738d5e8115d2410cc4b708b5e297a4c3e7fac7db242359`, `ede4c3b8849187fe2177490641521874f9211cbac1a93d6daf7b4537873c91b5`, `f0019d54cd872129256399add4982b489ce2e702ae954997876d485b6856c39c`, `fe30768a332ef959c2762f11d920d6ef0f575763b5cb042948c4747120c1b41e`
- **Reproduce:** `bench reproduce 35270f5d-e2e1-4b15-8ccc-a4bac44c601e`
