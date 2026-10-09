# Cost model

How the Benchmark Lab turns a load sweep into "$ per 1M tokens at SLO", what the hourly
price contains, and how that cost is compared with our planned price and public list
prices.

Code: `money.py` (integer micro-dollar math), `prices.py` (price book), `slo.py`
(goodput), `cost.py` (which hourly prices, `replica_prices`, and prices at goodput),
`report/analyze.py` (one priced result per sweep), `competitiveness.py` and
`report/competitiveness.py` (margins and flags).

## Money

All money is integer micro-dollars (1 USD = 1,000,000), never floats:

- Config files: `bench/prices.yaml`, `bench/competitors.yaml` and registry `pricing` use
  strict integers (`1_838_600` = $1.8386); a float such as `1.86` is a validation error.
  Experiment and budget files take strings (`"$40"`), parsed by `money.parse_usd`, which
  rejects anything finer than one micro-dollar.
- Database: every amount is `BIGINT` micros.
- Math: inputs are converted to `Fraction` (throughputs are floats, converted exactly),
  the whole formula is evaluated exactly, and the result is rounded once, half-up
  (`money.round_half_up`). Display goes through `money.format_usd` only.

## 1. Goodput at the SLO

An experiment declares an SLO, e.g. `{ttft_ms: {p95: 1000}, tpot_ms: {p95: 50},
max_error_rate: 0.01}`, and a load sweep per workload: fixed `values`, or a bisection
`search: {lo, hi, rel_tol, max_points}` (`slo.bisect_next_load`: test `lo`, then `hi`,
then halve the bracket between the highest pass and the lowest failure until it is within
`rel_tol`; `scale: geometric` climbs from `lo` instead, and `descend: N` steps down from a
failing `lo` up to N times before giving up).

1. Each load point runs `repetitions` times. Each run yields its own p95 TTFT, p95 TPOT,
   error rate, throughput, ... (`metrics/summary.py`, warmup excluded).
2. Per load point the repetitions are combined per metric (`metrics/aggregate.py`):
   latencies and throughputs as a geometric mean with a Student-t interval on the log
   scale (`log_t`); proportions with a t-interval clipped to [0, 1].
3. A load point **meets the SLO** only if, for every target, the upper end of the 95% CI
   is within the target (`slo.slo_met`). With one repetition the mean is used and the
   result is flagged untrusted.
4. **Goodput** is the highest passing load below the first failing one
   (`slo.find_goodput`); loads above the first failure never count. If no tested load
   failed, goodput is "at least" the highest load (shown `≥`), the result is flagged
   `unbracketed_goodput`, and the cost at SLO is an upper bound.
5. The throughputs at the goodput point (`throughput.output_tok_s`, `input_tok_s`,
   `request_rate`, each an estimate with a CI) are what gets priced.

Goodput compares like with like only within one load mode: open loop (req/s) and closed
loop (concurrency) results are separate rows.

## 2. Allocation: splitting the replica's cost

A replica produces input (prefill) and output (decode) tokens at the same time, so how
its hourly cost H is split between them is a choice, set per experiment
(`cost_allocation`) and printed with every result. With throughputs in tokens per second
at goodput, and prices in micros per 1M tokens:

| Method | Input price | Output price |
|---|---|---|
| `all_output` (default) | n/a | H × 10⁶ / (out_tok_s × 3600) |
| `all_input` | H × 10⁶ / (in_tok_s × 3600) | n/a |
| `weighted`, ratio r | H × 10⁶ / (E × 3600) | r × H × 10⁶ / (E × 3600) |
| `prefill_time` (measured) | φ × H × 10⁶ / (in_tok_s × 3600) | (1 − φ) × H × 10⁶ / (out_tok_s × 3600) |

where E = in_tok_s + r × out_tok_s, and φ is the measured share of the replica's time
spent prefilling (below). The weighted and prefill_time prices bill exactly H per hour at
the measured token mix. "n/a" is not $0: that side is assigned no cost, so there is
nothing to price or to compare a margin against (`MicrosRange.na_reason`). Every result
also has:

- blended total: H × 10⁶ / ((in_tok_s + out_tok_s) × 3600), the replica's cost over all
  tokens at the workload's own input:output mix. It needs no allocation, so it is the
  like-for-like number to compare across allocations and with list prices taken at the
  same mix;
- per 1,000 requests: H × 1000 / (request_rate × 3600).

Leaderboards rank by $/1M output tokens (by input under `all_input`) at the on-demand
price (section 4), under the allocation the experiments declare (`all_output` for every
experiment so far). That ranking is unchanged. The **headline** $/1M input and $/1M
output in reports, in the summary and in the competitiveness view use `prefill_time`,
computed for every result whatever the declared allocation (`ConfigResult.split_cost`).

### The headline split: `prefill_time`

`all_output` puts the whole replica on output tokens, so a short-output workload reads as
expensive output and free input: Qwen3-8B code-completion (1,537 in / 48 out tokens per
request) showed $4.84 per 1M output tokens and "n/a" per 1M input, while the replica
cost $0.15 per 1M tokens of that traffic. The brief asks for $ per 1M input **and** per 1M
output tokens at SLO, so the split has to be one that can be defended from what was
measured.

What was measured per request: TTFT, inter-token gaps and token counts (Parquet); per
run, their summaries. A request spends TTFT in its prefill phase and the rest of its
life decoding. By Little's law the mean number of requests in prefill is

    L_prefill = request_rate × mean TTFT

(successful requests over the measurement window, mean TTFT over the same requests;
`RunSummary.prefill_in_flight`). While L_prefill is below 1 it is the share of the
run's time the replica had a prompt in prefill. Prefill and decode share one GPU in
continuous batching, and a step that carries a 1k-token prefill chunk is dominated by
it, so that time is the replica's prefill time. So:

- φ = L_prefill at the goodput point, per repetition, combined like a throughput
  (geometric mean with a log-t CI; `CI_RULES`), and capped at 1;
- input tokens pay φ × H, output tokens pay (1 − φ) × H, each over its measured
  throughput.

Equivalently, an output token costs (1 − φ)/φ × in_tok_s/out_tok_s input tokens: a
`weighted` ratio, but measured per sweep instead of assumed.

Assumptions and known biases, all stated in the report's methodology:

- **TTFT includes more than prefill**: client and server queueing (the server queue time
  at these goodput points is under 1 ms), waiting for the step in flight, and the first
  decode step. That overstates φ: input a little high, output a little low.
- **Concurrent prefills are each counted.** λ × TTFT counts two prompts prefilling at the
  same time twice. Checked against the Parquet rows of the 8B sweep 565b8d3f: the union
  of the requests' prefill intervals is 10-30% below λ × TTFT at the goodput loads
  (fixed-1k-1k at 1 req/s: 0.12-0.17 of the span vs 0.16-0.23). Same direction as above.
- **Idle headroom is charged to output.** At goodput a latency-bound replica is not busy
  all the time (8B code-completion: no request in flight 15-25% of the span).
  That headroom exists to keep TPOT within the SLO, so it goes to output. This
  overstates output and offsets the two biases above.
- **When prefills overlap** (φ ≥ 1, normally past saturation, never at a measured
  goodput so far) prefill time cannot be separated from decode time, and both sides are
  "n/a" with that reason (`NA_PREFILL_SATURATED`); the blended cost still applies. A run
  with no TTFT gives "n/a" too (`NA_NO_PREFILL_TIME`).
- Prefix caching is captured, not modelled: a cached prefix shortens TTFT, so it lowers
  φ and the input price (shared-prefix).

Measured φ in the 8B sweep (565b8d3f, point estimates of sglang and vllm): fixed-1k-1k
0.17 and 0.22, code-completion 0.21 and 0.25, shared-prefix 0.39 and 0.37. The implied
output:input cost ratio on fixed-1k-1k is 4.8 (sglang) and 3.6 (vllm), in the range
providers publish (OpenRouter's Qwen3-8B listing is 3.9).

Other choices considered: a fixed ratio (not measured); a FLOPs model (a model, not a
measurement, and blind to batching); fitting per-token costs across workloads from
their goodputs (a measurement, but it needs two or more workloads per config at the same
SLO, the goodput brackets are coarse, and two of the three 8B code-completion results
are untrusted). `all_output`, `all_input` and `weighted` stay available as declared
allocations, and the CSV keeps every declared-allocation column.

## 3. Confidence intervals on cost

Cost falls as throughput rises, so the **low** cost bound comes from the **high**
throughput bound and vice versa; when a price depends on both input and output
throughput, both are taken at the same end of their CIs (`cost.cost_at_slo`). Throughput
intervals are `log_t` (positive bounds), so the cost interval is exactly the reciprocal
of the throughput interval, scaled.

Under `prefill_time` φ has its own CI, and each price bound takes φ and the throughput at
the ends that push that price the same way: input low = φ_lo × H / in_tok_s_hi, input
high = φ_hi × H / in_tok_s_lo, output low = (1 − φ_hi) × H / out_tok_s_hi, output high
= (1 − φ_lo) × H / out_tok_s_lo (`cost._split_prices`). That is an outer bound, wider
than a joint 95% interval, since φ and throughput come from the same runs. A φ bound
above 1 is capped at 1 (so an output low bound can be $0). With one repetition φ has no
interval and the prices have no bounds, as everywhere else. Exceptions:

- a repetition measured zero throughput: that metric falls back to an arithmetic
  interval clipped at 0, and the high cost bound is unbounded (`None`);
- one repetition: no interval at all, and the result is untrusted.

The interval is the run-to-run variation of throughput at the goodput load. It does not
include the uncertainty of where goodput lies between the passing load and the first
failing one; `rel_tol` bounds that bracket. Reports show the bracket next to goodput and
mark configs whose brackets overlap as tied within the search resolution, since their
costs then come from the same grid point (`docs/benchmark-lab.md`, "Reading the reports").

## 4. The hourly price H

Every result is priced in four columns, all computed by one function,
`cost.replica_prices`, from the run's provenance and `bench/prices.yaml`. Each column's
H is an instance price plus the run's block storage, amortised per hour and rounded
once, half-up (`PriceBook.with_storage`):

    H = round_half_up(instance $/h + per_gb_month × storage GB / 730)

where 730 is `prices.HOURS_PER_MONTH`, the hours per month AWS uses to convert GB-month
prices, and storage GB is the root volume the host ran with, recorded in the
provenance's `price_basis.storage_gb` (max(`disk_gb`, `root_volume_gb` 200) on
`aws_ec2`; the container disk, `container_disk_gb`, on `runpod`). us-east-1 storage is
EBS gp3 at $0.08/GB-month, from the AWS price list; RunPod container disk is
$0.10/GB-month, from RunPod's pricing docs.

| Column | Instance price | Use |
|---|---|---|
| **on-demand** | `on_demand_per_hour` | ranks leaderboards; the headline, the site and margins |
| **spot** | `spot_per_hour` (indicative average across us-east-1 AZs) | shown next to it |
| **committed 1y** | `committed_1y_per_hour`, where present (no entry has one today) | shown next to it |
| **as run** | the price recorded at launch (below) | shown next to it |

On-demand is the ranking price because it is a public list price: anyone can recompute
a ranking from the price book alone, and it does not depend on when, where or in which
market a run happened. A spot run is ranked at the on-demand price like any other; what
it actually paid is its as-run column.

A **mock or local host** has no list price. Its on-demand and as-run columns are the
price its experiment declares (`provider.hourly_price`, simulated for the mock) and it
has no spot or committed column; without a declared price it has no cost at SLO and is
not ranked (warning `no_price`).

### As run: what the provenance records

At launch the provider records the host's cost price in each run's provenance:
`hourly_micros` (instance plus storage, as above) and `price_basis` saying where it came
from:

| `price_basis.source` | `hourly_micros` |
|---|---|
| `observed_spot` | the spot price of the host's AZ read at launch (`describe_spot_price_history`; `spot_price_usd`, `observed_at` and `availability_zone` are recorded) plus storage |
| `observed_api` | a RunPod pod: the `costPerHr` its create (or first read) returned, plus container disk; `observed_at` and the datacenter (`availability_zone`) are recorded |
| `prices_yaml` | an on-demand host: `on_demand_per_hour` plus storage, the same as its on-demand column (also a RunPod pod whose API returned no price) |
| `experiment` | a mock or local host's declared price |
| `unobserved` | none: a spot host in an AZ with no spot price to read; its as-run cost is unknown |

`hourly_micros` never includes the budget guard's safety multiplier. A record written
before provenance schema 2 has no `price_basis`; its storage volume is unknown, so it
gets no cost (warning `no_price`) rather than a guessed one.

### One price everywhere

`bench report` (md, html, csv), `bench competitiveness`, the table printed after
`bench run`, the site snapshot and `results/<experiment>/goodput.json` all price through
`report.analyze.default_price_resolver`, which calls `cost.replica_prices`, so the same
column has the same value in every artefact. The CSV has every column
(`on_demand_*`, `spot_*`, `committed_1y_*`, `as_run_*`, plus `storage_gb`); the md and
html tables show spot, committed 1y and as run where a board has a value in them. A
price entry marked `verified: false` makes the report refuse rather than use it
(`UnverifiedPriceError`); `goodput.json` records the refusal in `price_error`.

### Budget accrual (what the guard records as spend)

The budget guard does not use these prices. `providers/aws_ec2.py` (`_candidates`)
gives each host an accrual rate, `Host.hourly_micros`:

- spot: ⌈current spot price of the chosen AZ × 1.25 × 10⁶⌉, where 1.25 is
  `AwsSettings.spot_price_multiplier`; an AZ with no spot price is accrued at the
  on-demand rate;
- on-demand: `on_demand_per_hour`;
- plus the root EBS volume: ⌈`per_gb_month` × volume GB / 730⌉.

`providers/runpod.py` accrues a pod at the larger of its API `costPerHr` and
`on_demand_per_hour`, plus the container disk the same way, and terminates a pod whose
API price is above `on_demand_per_hour` × `RunpodSettings.max_price_ratio` (1.25) before
any work. The raw values are in the host's `accrual_basis`.

The multiplier keeps recorded spend above the real bill, since spot prices move during a
run; it stays inside accrual and never reaches a reported cost. The planner estimates
the same way from `prices.yaml` (`plan.host_price`; `bench plan` shows "accrued at
$x/h"). Each `bench_spend` row records the accrual basis (spot price and timestamp,
multiplier, EBS rate) under `accrual_basis`.

### Not included anywhere

- **Network**: load is generated on the GPU host itself, so benchmark traffic has no
  internet egress; image and weight downloads are inbound. `prices.yaml` accepts a
  per-region `data_transfer` block (`ingress_per_gb`, `egress_internet_per_gb`), but no
  region sets one and no cost uses it.
- **Instance-store NVMe** (weights cache) is part of the instance price.
- **Fixed costs**: the S3 bucket, Secrets Manager ($0.40 per secret per month), the
  reaper Lambda and its logs.

## 5. Worked example

Qwen3-8B on `g6e.xlarge` in us-east-1 with the default 200 GB root volume, `all_output`.
The throughputs are illustrative (no GPU results are published yet): three repetitions
at the goodput load measured 1480, 1510 and 1530 output tok/s, 1470, 1505 and 1532
input tok/s, and 1.44, 1.47 and 1.49 req/s.

1. H, on-demand: `on_demand_per_hour: 1_861_000` + 80,000 × 200 / 730 = 1,861,000 +
   21,917.8 = 1,882,918 micros ($1.882918/h), rounded once.
2. Output throughput, `log_t`: geometric mean 1506.53 tok/s, 95% CI [1445.19, 1570.46].
3. $/1M output tokens:
   1,882,918 × 10⁶ / (1506.53 × 3600) = 347,178 micros = **$0.3472**.
4. CI: low bound from the high throughput, 1,882,918 × 10⁶ / (1570.46 × 3600) = 333,043;
   high bound from the low throughput, 1,882,918 × 10⁶ / (1445.19 × 3600) = 361,913.
   Reported: $0.3472 [$0.3330, $0.3619] per 1M output tokens, on-demand.
5. Blended total: input geometric mean 1502.12, so 1,882,918 × 10⁶ /
   ((1502.12 + 1506.53) × 3600) = 173,843 micros ($0.1738 per 1M tokens).
6. Per 1,000 requests: 1,882,918 × 1000 / (1.4665 × 3600) = 356,648 micros ($0.3566).
7. Under `weighted` with r = 4: E = 1502.12 + 4 × 1506.53 = 7528.22, input
   1,882,918 × 10⁶ / (7528.22 × 3600) = 69,476 micros ($0.0695), output 4 × that,
   computed exactly: 277,905 micros ($0.2779).
8. Under `prefill_time`, with three repetitions measuring 0.19, 0.20 and 0.21 requests in
   prefill (for example 1.47 req/s × a mean TTFT of 136 ms): φ = 0.1998 [0.1765,
   0.2263]. Input 0.1998 × 1,882,918 × 10⁶ / (1502.12 × 3600) = 69,581 micros
   ($0.0696), CI [φ_lo with the high input throughput, φ_hi with the low] = [$0.0584,
   $0.0830]; output 0.8002 × 1,882,918 × 10⁶ / (1506.53 × 3600) = 277,800 micros
   ($0.2778) [$0.2577, $0.2980]. Blended is unchanged at $0.1738.
9. The other columns, same throughputs: spot, 1,838,600 + 21,917.8 → 1,860,518 micros/h,
   $0.3430 per 1M output; as run, for a spot host that observed $1.70/h at launch,
   1,700,000 + 21,917.8 → 1,721,918 micros/h, $0.3175.

The budget guard would have accrued that spot host at a higher rate: ⌈1,700,000 × 1.25⌉
+ ⌈80,000 × 200 / 730⌉ = 2,125,000 + 21,918 = 2,146,918 micros ($2.1469/h). At the
`prices.yaml` spot price the planner estimates ⌈1,838,600 × 1.25⌉ + 21,918 = 2,320,168
micros ($2.3202/h), which is what `bench plan` shows for `qwen3-8b-vllm-vs-sglang`.

To check numbers like these, call the code directly:

```bash
uv run python -c "
from loom_bench.cost import cost_at_slo
from loom_bench.prices import load_prices
from loom_bench.stats import log_mean_ci
h = load_prices().replica_hourly_cost('aws', 'us-east-1', 'g6e.xlarge', storage_gb=200).per_hour
out, inp = log_mean_ci([1480, 1510, 1530]), log_mean_ci([1470, 1505, 1532])
print(h, cost_at_slo(h, input_tok_s=inp, output_tok_s=out).output_per_mtok)"
```

## 6. Competitiveness and margin

`bench competitiveness`, and the competitiveness section at the end of every `bench
report` leaderboard (also `leaderboard.competitiveness.csv`), take per registry model and
workload the on-demand cost at SLO of the board's quoted config, and compare it with the
model's planned price (`pricing` in `config/models.yaml`, micros per 1M tokens) and with
public list prices from `bench/competitors.yaml` (entered by hand; no competitor endpoint
is ever called, see [legal/competitor-benchmarking.md](legal/competitor-benchmarking.md)).

- **Quoted config** (`leaderboard.headline_row`): among configs with a cost at SLO that
  did not fail the quality gate, trusted first, then quality verified (the gate's
  reference or a gate pass), then leaderboard order. An untrusted config is quoted only
  when no trusted one has a cost, with its reason in the row's basis. The ranking is not
  changed by this.
- **Our cost**: $/1M input and $/1M output under the `prefill_time` split, and $/1M
  blended at the workload's mix.
- **Like with like**: every public price is also shown at each workload's token mix,
  in_price × input share + out_price × output share (`competitiveness.price_at_mix`), next
  to our blended cost, with our cost as a multiple of it. Input is compared with input and
  output with output. A competitor's disclosed quantization is shown and marked against
  ours: "like for like" when it matches, "not like for like" when it differs (DeepInfra's
  Llama 3.3 70B Turbo is fp8: like for like for the `llama-3.3-70b-instruct-fp8` row, not
  for the unquantized one).
- **Quantized entries**: providers list prices per model, not per checkpoint, so a
  registry entry with `base_model` (a quantization of another entry, e.g.
  `llama-3.3-70b-instruct-fp8` of `llama-3.3-70b-instruct`) is compared with its base
  model's listings (`Registry.market_model_id`), and the report says so. `base_model` is
  left out of the spec dump, so it does not change any config hash.
- **No cost at the declared SLO**: when an alternative-SLO analysis is run
  (`--alt-slo`), its figure is added as a separate row titled "at the ALTERNATIVE SLO, not
  the declared one".
- **Margin** per side = price − cost; "worst" = price − the cost CI high bound; also as
  a share of price. A side with no cost of its own has no margin.
- **Break-even** price per side, shown while no price is set: our cost, and the cost CI
  high bound (the lowest price that covers the cost with 95% confidence on run-to-run
  variation).
- **Flags** (`competitiveness.assess`):

| Flag | When |
|---|---|
| `cost_above_market` | our cost at SLO is above the lowest eligible public price: per side (input, output) and blended at the workload's mix (also reports the median) |
| `price_above_market` | our price is above the lowest eligible public price for that side (also reports the median) |
| `negative_margin` | our price is below our measured cost |
| `no_price_set` | the registry has `pricing: null` (both models today) |
| `no_cost_measurement` | no quoted config with a cost |
| `no_public_comparison` | no eligible public price |

Eligible means a listed price from a non-aggregator; `--include-aggregators` and
`--include-unverified` widen it. Example with the worked `all_output` cost above and a
hypothetical price of $0.40 per 1M output tokens: margin $0.0528 (13.2% of price), $0.0381
at the cost CI high bound; against the worked `prefill_time` output cost ($0.2778
[$0.2577, $0.2980]) the margin is $0.1222, $0.1020 at the CI high bound. For `qwen3-8b` no default-eligible price exists (the Fireworks entry is
`availability: unverified`, OpenRouter is an aggregator), so the flag is
`no_public_comparison`; with both options the market minimum is Fireworks' $0.20 and the
price is flagged `price_above_market`.

## Open items

- **GCP prices unverified**: the three `gcp/us-central1` instances are
  `verified: false` (third-party snapshot), so cost math refuses them.
- **Spot prices are indicative**: the spot column uses the `prices.yaml` spot average,
  not a price any run paid; the price paid is the as-run column, and spend is in
  `bench_spend`.
- **No committed prices**: no `prices.yaml` entry has `committed_1y_per_hour` yet, so
  that column is empty everywhere.
- **Data transfer** is not modelled (see above).
