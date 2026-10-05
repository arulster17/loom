# Cost model

How the Benchmark Lab turns a load sweep into "$ per 1M tokens at SLO", what the hourly
price contains, and how that cost is compared with our planned price and public list
prices.

Code: `money.py` (integer micro-dollar math), `prices.py` (price book), `slo.py`
(goodput), `cost.py` (prices at goodput), `report/analyze.py` (which hourly price),
`competitiveness.py` and `report/competitiveness.py` (margins and flags).

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
`rel_tol`).

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
   failed, goodput is "at least" the highest load, the result is flagged
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

where E = in_tok_s + r × out_tok_s. The weighted prices bill exactly H per hour at the
measured token mix. "n/a" is not $0: that side is assigned no cost, so there is nothing
to price or to compare a margin against (`MicrosRange.na_reason`). Every result also has:

- blended total: H × 10⁶ / ((in_tok_s + out_tok_s) × 3600);
- per 1,000 requests: H × 1000 / (request_rate × 3600).

Leaderboards rank by $/1M output tokens (by input under `all_input`).

## 3. Confidence intervals on cost

Cost falls as throughput rises, so the **low** cost bound comes from the **high**
throughput bound and vice versa; when a price depends on both input and output
throughput, both are taken at the same end of their CIs (`cost.cost_at_slo`). Throughput
intervals are `log_t` (positive bounds), so the cost interval is exactly the reciprocal
of the throughput interval, scaled. Exceptions:

- a repetition measured zero throughput: that metric falls back to an arithmetic
  interval clipped at 0, and the high cost bound is unbounded (`None`);
- one repetition: no interval at all, and the result is untrusted.

The interval is the run-to-run variation of throughput at the goodput load. It does not
include the uncertainty of where goodput lies between the passing load and the first
failing one; `rel_tol` bounds that bracket.

## 4. The hourly price H

Which hourly price is used depends on what is being computed. All of them start from
`bench/prices.yaml`.

### Reported cost (`bench report`, `bench competitiveness`, the table after `bench run`, the site)

`report.analyze.default_price_resolver` reads the run's provenance (cloud, region,
instance type, market) and looks the price up in `bench/prices.yaml` with no storage
(`replica_hourly_cost(..., storage_gb=0)`):

| Run market | H |
|---|---|
| `on_demand` | `on_demand_per_hour` |
| `spot` | `spot_per_hour` from `prices.yaml` (an indicative average across us-east-1 AZs, not the price the run paid); on-demand if the entry has no spot price |
| `local` (mock, local provider) | the `hourly_micros` recorded in the provenance: the mock's simulated `hourly_price`, or 0 for a local endpoint |

So the reported cost is the instance price only: no EBS volume, no spot safety
multiplier, no S3, Secrets Manager or Lambda. A price entry marked `verified: false`
makes the report refuse rather than use it (`UnverifiedPriceError`).

### Budget accrual (what the guard records as spend)

`providers/aws_ec2.py` (`_candidates`) gives each host an `hourly_micros`:

- spot: ⌈current spot price of the chosen AZ (`describe_spot_price_history`) × 1.25 ×
  10⁶⌉, where 1.25 is `AwsSettings.spot_price_multiplier`; an AZ with no spot price is
  accrued at the on-demand rate;
- on-demand: `on_demand_per_hour`;
- plus the root EBS volume: ⌈`per_gb_month` × volume GB / 730⌉, volume =
  max(`disk_gb`, `root_volume_gb` 200). 730 is `prices.HOURS_PER_MONTH`, the hours per
  month AWS uses to convert GB-month prices.

The multiplier keeps recorded spend above the real bill, since spot prices move during a
run. The planner estimates the same way from `prices.yaml` (`plan.host_price`). Each
`bench_spend` row records the basis (spot price and timestamp, multiplier, EBS rate).

`results/<experiment>/goodput.json`, written by the runner, is priced with this accrual
rate, so for AWS runs it is higher than the reported cost of the same result.

### Not included anywhere

- **Network**: load is generated on the GPU host itself, so benchmark traffic has no
  internet egress; image and weight downloads are inbound. `prices.yaml` accepts a
  per-region `data_transfer` block (`ingress_per_gb`, `egress_internet_per_gb`), but no
  region sets one and no cost uses it.
- **Instance-store NVMe** (weights cache) is part of the instance price.
- **Fixed costs**: the S3 bucket, Secrets Manager ($0.40 per secret per month), the
  reaper Lambda and its logs.

## 5. Worked example

Qwen3-8B on `g6e.xlarge` spot in us-east-1, `all_output`. H from `bench/prices.yaml`:
`spot_per_hour: 1_838_600` ($1.8386/h). The throughputs are illustrative (no GPU results
are published yet): three repetitions at the goodput load measured 1480, 1510 and 1530
output tok/s, 1470, 1505 and 1532 input tok/s, and 1.44, 1.47 and 1.49 req/s.

1. Output throughput, `log_t`: geometric mean 1506.53 tok/s, 95% CI [1445.19, 1570.46].
2. $/1M output tokens:
   1,838,600 × 10⁶ / (1506.53 × 3600) = 339,007 micros = **$0.3390**.
3. CI: low bound from the high throughput, 1,838,600 × 10⁶ / (1570.46 × 3600) = 325,205;
   high bound from the low throughput, 1,838,600 × 10⁶ / (1445.19 × 3600) = 353,394.
   Reported: $0.3390 [$0.3252, $0.3534] per 1M output tokens.
4. Blended total: input geometric mean 1502.12, so 1,838,600 × 10⁶ /
   ((1502.12 + 1506.53) × 3600) = 169,752 micros ($0.1698 per 1M tokens).
5. Per 1,000 requests: 1,838,600 × 1000 / (1.4665 × 3600) = 348,254 micros ($0.3483).
6. Under `weighted` with r = 4: E = 1502.12 + 4 × 1506.53 = 7528.22, input
   1,838,600 × 10⁶ / (7528.22 × 3600) = 67,841 micros ($0.0678), output 4 × that,
   computed exactly: 271,364 micros ($0.2714).

The budget guard would have accrued this host at a higher rate: at the `prices.yaml` spot
price, ⌈1,838,600 × 1.25⌉ + ⌈80,000 × 200 / 730⌉ = 2,298,250 + 21,918 = 2,320,168 micros
($2.3202/h), which is what `bench plan` shows for `qwen3-8b-vllm-vs-sglang`.

To check numbers like these, call the code directly:

```bash
uv run python -c "
from loom_bench.cost import cost_at_slo
from loom_bench.stats import log_mean_ci
out, inp = log_mean_ci([1480, 1510, 1530]), log_mean_ci([1470, 1505, 1532])
print(cost_at_slo(1_838_600, input_tok_s=inp, output_tok_s=out).output_per_mtok)"
```

## 6. Competitiveness and margin

`bench competitiveness` takes, per registry model and workload, the cost at SLO of the
top-ranked leaderboard config (trusted, not failing the quality gate), and compares it
with the model's planned price (`pricing` in `config/models.yaml`, micros per 1M tokens)
and with public list prices from `bench/competitors.yaml` (entered by hand; no
competitor endpoint is ever called, see
[legal/competitor-benchmarking.md](legal/competitor-benchmarking.md)).

- **Margin** per side = price − cost; "worst" = price − the cost CI high bound; also as
  a share of price. A side the allocation does not price has no margin.
- **Flags** (`competitiveness.assess`):

| Flag | When |
|---|---|
| `price_above_market` | our price is above the lowest eligible public price for that side (also reports the median) |
| `negative_margin` | our price is below our measured cost |
| `no_price_set` | the registry has `pricing: null` (both models today) |
| `no_cost_measurement` | no ranked result with a cost |
| `no_public_comparison` | no eligible public price |

Eligible means a listed price from a non-aggregator; `--include-aggregators` and
`--include-unverified` widen it. Example with the worked cost above and a hypothetical
price of $0.40 per 1M output tokens: margin $0.0610 (15.2% of price), $0.0466 at the cost
CI high bound. For `qwen3-8b` no default-eligible price exists (the Fireworks entry is
`availability: unverified`, OpenRouter is an aggregator), so the flag is
`no_public_comparison`; with both options the market minimum is Fireworks' $0.20 and the
price is flagged `price_above_market`.

## Open items

- **EBS price unverified**: `bench/prices.yaml` storage `per_gb_month: 80_000` is
  `verified: false`. The planner and the AWS provider use it anyway (with a note); a
  report that included storage would refuse it.
- **GCP prices unverified**: the three `gcp/us-central1` instances are
  `verified: false` (third-party snapshot), so cost math refuses them.
- **Spot prices are indicative**: reports use the `prices.yaml` spot average, not the
  price paid; the paid basis is only in `bench_spend` and `goodput.json`.
- **Data transfer** is not modelled (see above).
