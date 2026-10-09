# How to put a measured config and its price into the registry

After a config sweep (for Qwen3-8B: `qwen3-8b-config-sweep-runpod`, see
[benchmark-lab.md](../benchmark-lab.md#qwen3-8b-config-sweep-proposed)), the winning
serving config goes into `config/models.yaml` with a price derived from its measured cost
at the SLO. The margin is a business decision: this page lays out the options and the
arithmetic, it does not pick one.

## 1. Pick the winner from the report

```bash
uv run bench report -e <sweep experiment id> --out reports/<name>
uv run bench competitiveness -e <sweep experiment id>
```

The winner is the cheapest config, by $/1M blended at the SLO on the realistic workload
(chat-sharegpt), that is all of:

- **trusted**: no warning that untrusts it (run-to-run CV under 10% at goodput, three
  repetitions, a bracketed goodput). An untrusted figure is indicative only;
- **gate PASS** against the BF16 reference. INCONCLUSIVE is blocked, the same as FAIL
  (docs/quality-gate.md). REVIEW (divergence above its limit with every task passing) is
  reported, not blocking: read the reason before choosing it;
- **not contradicted** on fixed-1k-1k: it also meets the SLO there and is not the most
  expensive config on that board.

The summary's quoted config per workload already applies the first two rules
(`leaderboard.headline_row`). Note its goodput bracket: cost at SLO is computed at the
highest passing load, so a coarse bracket makes the cost conservative, not wrong.

## 2. Write the config into its registry row

The precision is always in the row name and BF16 stays the reference row
([PLAN.md](../PLAN.md), quality standard). So:

| Winner | Row to edit |
|---|---|
| `fp8`, `fp8-kv8` or `fp8-kv8-mbt1024` | `qwen3-8b-fp8`: copy the cell's `kv_cache_dtype` and `engine.args` from the sweep's variant; name every reduced precision in `display_name` (e.g. "Qwen3 8B FP8, FP8 KV cache") |
| `bf16-kv8` | a new row, e.g. `qwen3-8b-kv8` (BF16 weights, `kv_cache_dtype: fp8`, `base_model: qwen3-8b`), added as in [add-model.md](add-model.md) |
| `bf16` | `qwen3-8b` itself |

Everything else (checkpoint, revision, image digest, hardware, `max_context`) stays as
the sweep ran it, so the row is the config the numbers describe. `pricing` and `status`
are part of the spec dump, so a priced row's config hash differs from the swept cell's;
reports and the competitiveness view match prices to results by model id.

## 3. Choose the price

`pricing` is integer micro-dollars per 1M tokens: `input_per_mtok`, `output_per_mtok`
and `cached_input_per_mtok` (at most the input price). Take the cost from the
competitiveness view of the winner on chat-sharegpt: $/1M input and $/1M output under the
measured `prefill_time` split, with their 95% CI high bounds (the "break-even" columns),
and $/1M blended at the workload's mix ([cost-model.md](../cost-model.md), sections 2-3).

Two things the cost at SLO leaves out, whichever option you choose:

- **Utilization.** Cost at SLO assumes the replica serves at its goodput every second it
  is billed. A replica busy a fraction u of the time costs 1/u as much per token served
  (u = 0.5 doubles it). Scale-to-zero (`scaling.min_replicas: 0`) trims idle time but adds
  cold starts (299-337 s median for Qwen3-8B on RunPod, 565b8d3f).
- **Spread.** The CI high bound covers run-to-run variation in the lab, not other
  hardware, datacenters or traffic shapes. The chat workload is ShareGPT; your traffic's
  input:output mix moves the blended cost.

### Option A: cost-plus

    price_side = cost_side_CI_high / u x (1 + markup)

per side (input, output). It guarantees a margin on every token at utilization u or
better, whatever the market does. Worked with the one measured 8B cost so far, vLLM BF16
on fixed-1k-1k (565b8d3f: input $0.0639 [0.0402, 0.1015], output $0.2328 [0.1752,
0.3016] per 1M):

| u | markup | input $/1M | output $/1M | registry micros (in / out) |
|---|---|---|---|---|
| 100% | 0% (break-even) | 0.1015 | 0.3016 | 101_500 / 301_600 |
| 100% | 20% | 0.1218 | 0.3619 | 121_800 / 361_900 |
| 50% | 0% | 0.2030 | 0.6032 | 203_000 / 603_200 |
| 50% | 20% | 0.2436 | 0.7238 | 243_600 / 723_800 |

The sweep's FP8 cells are expected to cut these by roughly 2-3x (they read ~half the bytes
per decode step); use their measured numbers, not this projection.

### Option B: match the market

Price each side at the lowest public list price for the model (`bench/competitors.yaml`,
public prices only). For Qwen3-8B today: Fireworks $0.20 / $0.20 (its 4-16B size bucket,
`availability: unverified`) and OpenRouter $0.117 input / $0.455 output (an aggregator).
At chat-sharegpt's mix (~74% input tokens) both come to about $0.20-0.21 per 1M blended;
at 1k/1k, $0.20 and $0.29. Neither is default-eligible for the flags (unverified,
aggregator): pass `--include-unverified --include-aggregators` to compare. The margin is
whatever is left: `bench competitiveness` shows it per side and at the CI high bound, and
flags `negative_margin` when the price is below cost. Against the BF16 example above,
$0.20 input clears the input cost but $0.20 output does not clear $0.30 (break-even),
so BF16 at a market price would lose money on output tokens at any utilization.

### Option C: market, floored at cost-plus

    price_side = max(market_side, cost_side_CI_high / u x (1 + minimum markup))

Market-priced while the cost allows it, and never below the floor you accept. It is A
when the market is below your floor and B otherwise.

### Cached input

`cached_input_per_mtok` prices prompt tokens served from the prefix cache. The lab
captures caching (a cached prefix shortens TTFT, which lowers the measured input share on
shared-prefix), but no workload isolates a cached token's cost yet, and the 8B runs'
usage blocks did not report cached tokens. Options: no discount (equal to input), or a
fixed fraction of input as other providers publish (commonly 10-50%).

## 4. Check and commit

```bash
uv run pytest -q bench/tests/config/test_registry.py   # the row validates
uv run bench competitiveness -e <sweep experiment id>   # margins per side, flags
```

Look for `negative_margin` and `price_above_market` on the chat and 1k/1k rows. Commit
the registry change with the report directory (`reports/<name>/`) it was derived from,
and record in docs/PLAN.md the experiment id, the winner, the cost basis (point or CI
high, u) and the option chosen. `status` stays `preview` until the row is approved for
serving; only `enabled` rows must carry a price.
