# Quality gate

Loom never ships a cheaper config without measuring that it is as good. Any change that can
alter model outputs (quantization, engine or engine version, speculative decoding, KV-cache
dtype, context extension) runs the model's pinned eval suite on the current config (the
**baseline**) and the proposed one (the **candidate**), and the gate decides PASS, FAIL or
INCONCLUSIVE. FAIL blocks the change; INCONCLUSIVE blocks it too unless the suite says
otherwise.

Code: `bench/src/loom_bench/quality/`. Suites: `bench/evals/<model>.yaml`.

## What is measured

| Suite task | Kind | Source | Items | Score per item |
|---|---|---|---|---|
| `mmlu_pro` | `lm_eval` | TIGER-Lab/MMLU-Pro via lm-eval `mmlu_pro` | 150 per subject x 14 = 2100 | exact match on the extracted letter (`custom-extract`) |
| `gsm8k` | `lm_eval` | openai/gsm8k via lm-eval `gsm8k_cot_llama` | 1319 (full test) | exact match, `flexible-extract` |
| `ifeval` | `lm_eval` | google/IFEval via lm-eval `ifeval` | 541 (full) | prompt-level strict accuracy |
| `ruler_niah` | `lm_eval` | lm-eval RULER `niah_single_2` (synthetic) | 500 per length | string match, keyed by the doc's length |
| `needle` | `needle` | synthetic, seeded | lengths x depths x samples | the six-digit code appears in the reply |
| `code` | `code_exec` | openai/openai_humaneval (MIT) + google-research-datasets/mbpp test (CC-BY-4.0), pinned revisions | 164 + 500 | program + tests exit 0 in the sandbox |
| `tool_calling` | `tool_calling` | self-authored, `quality/data/tool_calling.yaml` (Apache-2.0) | 60 | exactly one call, right name, AST-matched arguments |
| `json_schema` | `json_schema` | self-authored, `quality/data/json_schema.yaml` (Apache-2.0) | 60 | reply parses and validates under strict `response_format` |
| `toy_arithmetic` | `toy_arithmetic` | generated, seeded | configurable | "The answer is N" is right; used with the mock backend |

All requests are greedy (temperature 0) with a fixed seed. Qwen3 runs in non-thinking mode
(`chat_template_kwargs: {enable_thinking: false}`, from the suite), which is forwarded to
native tasks and to lm-eval's chat requests.

Besides task scores, the gate looks at:

- **Logprob divergence** of the candidate from the BF16 reference (below).
- **Sanity rates** over all candidate outputs: empty, truncated (`finish_reason: length`
  where not expected), repetition loops, language drift (below).

## Decision rule

For every task, baseline and candidate scores are paired by item id (the gate refuses to
compare runs whose item ids or item content hashes differ). With n items,

- delta = mean(candidate) - mean(baseline);
- CI = 95% percentile bootstrap of the paired delta (`stats.paired_bootstrap_delta`, items
  resampled jointly, 10 000 resamples, seeded), widened to at least ±3/n around the delta.

The widening is the rule of three: n items cannot show that an effect rarer than about 3/n
is absent. Without it, two configs that agree on every item give a zero-width bootstrap CI
and would "prove" non-inferiority from ten items.

With threshold t (default 0.01 = 1 point absolute; per task in the suite):

| Condition (checked in order) | Verdict |
|---|---|
| n < min_samples (default 300) | INCONCLUSIVE |
| delta < -t, or CI upper bound < -t | FAIL |
| CI lower bound >= -t | PASS (non-inferiority shown) |
| otherwise | INCONCLUSIVE (needs more samples) |

The PASS rule is a one-sided non-inferiority test at 2.5%. The FAIL rule also fires on a
point drop beyond t even when the CI is wide: a measured drop larger than the allowed margin
is not shipped on the hope that it is noise.

Independently of the tasks:

- divergence FAILs when mean KL(ref || cand) > `max_kl` or top-1 agreement < `min_top1`;
- sanity FAILs when any rate exceeds its limit (defaults: empty 1%, truncated 2%,
  repetition 2%, language drift 2%).

The overall decision is the worst of all parts (FAIL > INCONCLUSIVE > PASS); `blocked` is
true on FAIL, and on INCONCLUSIVE while `inconclusive_blocks` is on (the default). Every
part carries a human-readable reason, e.g.

```
gate FAIL (blocked)
  arithmetic: delta -26.50 pts [-30.75 pts, -22.50 pts], n=400 is a drop of more than 1.00 pts
  json_schema: delta -23.33 pts [-35.00 pts, -13.33 pts], n=60 is a drop of more than 6.00 pts
  divergence: mean KL 2.0022 nats exceeds 0.0500; top-1 agreement 21.71% is below 95.00%
  sanity: 460 outputs: empty 0.00%, truncated 0.00%, repetition 0.00%, language_drift 0.00%
```

`GateDecision.details()` is the JSON stored in `bench_gate_decisions.details`.

## Sample sizes

For a pass/fail task, let d be the share of items whose outcome flips between baseline and
candidate (half each way when there is no real change). The paired delta's 95% CI
half-width is about 1.96 * sqrt(d / n); an unpaired comparison of two accuracies near p
would need 1.96 * sqrt(2p(1-p) / n), several times wider, which is why the gate pairs.

| n | d = 2% | d = 5% | floor 3/n | unpaired, p = 0.8 |
|---|---|---|---|---|
| 60 | 3.6 pts | 5.7 | 5.0 | 14.3 |
| 100 | 2.8 | 4.4 | 3.0 | 11.1 |
| 300 | 1.6 | 2.5 | 1.0 | 6.4 |
| 400 | 1.4 | 2.2 | 0.75 | 5.5 |
| 541 | 1.2 | 1.9 | 0.55 | 4.8 |
| 664 | 1.1 | 1.7 | 0.45 | 4.3 |
| 1000 | 0.9 | 1.4 | 0.30 | 3.5 |
| 1319 | 0.8 | 1.2 | 0.23 | 3.1 |
| 2100 | 0.6 | 1.0 | 0.14 | 2.4 |
| 5000 | 0.4 | 0.6 | 0.06 | 1.6 |

To decide a 1-point margin when ~2-5% of items flip takes roughly 1000-2000 items. The
suites use that where the source data allows (MMLU-Pro 2100, GSM8K 1319, RULER 1000) and
raise the task threshold where it does not: IFEval (541) and code (664) decide 2 points,
the native needle (400) 2 points, and the two 60-item pinned sets (tool calling, JSON
schema) 6 points, which catches broken parsers, templates and grammar backends rather
than subtle drift. Each suite file repeats this table next to the tasks.

## Logprob divergence

`quality/divergence.py`, against a reference endpoint serving the BF16 weights:

1. The reference greedily continues each of the 48 pinned prompts
   (`quality/data/divergence_prompts.yaml`) for `max_new_tokens` tokens.
2. Both endpoints score the identical text prompt + continuation with `/v1/completions`,
   `echo: true, logprobs: k, max_tokens: 1` (vLLM v0.30 and SGLang v0.5.21 both support
   echo with logprobs). Every continuation position is then conditioned on the same tokens
   on both sides (teacher forcing), so differences come from weights and kernels alone.
3. Per position: top-1 agreement (same argmax token) and approximate KL(ref || cand).
4. Per prompt: mean over its positions. Reported: the mean over prompts with a bootstrap CI
   over prompts (positions within a prompt are correlated).

**KL approximation.** Servers return only the top-k. Over U = union of both top-k token
sets plus an "other" bucket: for each distribution, tokens of U missing from its own top-k
get an equal share of its unlisted mass (1 - listed mass), capped at its smallest listed
probability since they ranked below it; the remaining unlisted mass goes to "other". Both
vectors are floored at 1e-10 and renormalised. With identical top-k sets this is the exact
KL of the coarsened distributions, a lower bound on the true KL (data-processing
inequality). When the sets differ it is an estimate that grows sharply when one side's
confident token is missing from the other's list, which is the failure that matters. Both
sides must use the same tokenizer; the measurement refuses otherwise.

The default limits (KL 0.05 nats, top-1 95%) are initial values; the first Phase 0 GPU
runs measure BF16 vs BF16 across engines to confirm they sit above engine noise.

## Sanity checks

`quality/sanity.py`, no heavy dependencies:

- **empty**: whitespace only.
- **truncated**: `finish_reason == "length"` where the task did not expect it. lm-eval
  outputs carry no finish reason and are not counted.
- **repetition**: the output ends in a loop (its tail is periodic with period <= 200
  characters, at least 4 repeats and 64 characters), or, for prose, more than half of its
  word 4-grams repeat.
- **language drift** (prose only): more than 20% of letters outside the target scripts
  (Latin by default), by Unicode character name.

Code and JSON outputs skip the n-gram and script checks.

## Running it

```python
from pathlib import Path
from loom_bench.quality.runner import gate_against_baseline, run_divergence, run_suite
from loom_bench.quality.suite import load_suite

suite = load_suite("qwen3-8b")
base = await run_suite(
    suite, "http://baseline:8000/v1", "qwen3-8b", workdir=Path("runs/base"), allow_code_exec=True
)
cand = await run_suite(
    suite, "http://candidate:8000/v1", "qwen3-8b", workdir=Path("runs/cand"), allow_code_exec=True
)
div = await run_divergence(
    suite, "http://bf16-reference:8000/v1", "http://candidate:8000/v1", "qwen3-8b"
)
decision = gate_against_baseline(base, cand, suite, div)
print(decision.summary())
```

`record_suite_result` writes one `bench_eval_runs` row per task (mean, t-interval, task
version, provenance including dataset source, revision and license) and `record_gate`
writes the decision to `bench_gate_decisions`.

- **lm-eval** is the optional `lmeval` extra (`uv sync --extra lmeval`); without it,
  `lm_eval` tasks fail with an error saying how to install it. The command line and the
  per-sample format were checked against lm_eval 0.4.13. RULER needs `transformers` (for its
  tokenizer) and lm-eval's humaneval needs `evaluate`. lm-eval loads its datasets from the
  Hub without a pinned revision; each item's `doc_hash` is stored as its content hash, so
  the gate refuses to compare runs whose items changed underneath it.
- **Code execution** runs model-written programs. The sandbox (`quality/sandbox.py`) gives
  each program a fresh `python -I -S` process group with an empty environment, a temporary
  working directory, rlimits on CPU time, file size, open files and process creation (and
  address space on Linux; macOS does not enforce it), and a wall-clock timeout. It is a
  resource sandbox, not a security boundary: programs run as the current user and can
  reach the network. Run code evals in a disposable container or VM without network
  egress or credentials. Nothing executes unless `allow_code_exec=True` is passed.
- The working directory keeps lm-eval's raw samples and log. Loom itself never logs
  prompts or outputs.

## Adding an eval task

1. Write a class with `name`, `version` and `async def run(self, ctx: EvalContext) ->
   TaskOutput` (the `EvalTask` protocol in `quality/tasks/base.py`); subclassing
   `ParamTask` gives YAML-validated parameters via a Pydantic `Params` model. Return one
   `ItemResult(item_id, score in [0, 1], content_hash, meta)` per item, with stable item
   ids and a hash of the item's content, plus the raw outputs as `Completion`s for the
   sanity checks. Score an item whose request fails after retries as 0 with the error in
   `meta` (`failed_item`): a config that cannot answer must not look as good as one that
   can.
2. Ship any fixed data under `quality/data/` with a `version` field and a license note;
   self-author it or use data whose license allows redistribution. Load third-party data
   from the Hub at a pinned revision, only when the task runs.
3. Register the kind in `quality/tasks/__init__.py`, add it to the suites with a sample
   size chosen from the table above, and bump `version` whenever items, prompts or scoring
   change (baselines must then be rerun).
4. Test the scorer on hand-written cases and the task against a fake endpoint
   (`httpx.MockTransport`) or the mock backend. Tests never touch the network.

## Acceptance test

`bench/tests/quality/test_gate_acceptance.py` runs the full path over HTTP against the
mock backend (`loom_bench.mock`): baseline = clean mock; "over-aggressive quantization" =
`degrade: 0.3` (30% of prompts get wrong answers or truncated JSON) plus heavy logprob
noise. The gate FAILs it on arithmetic, JSON validity and divergence and blocks it; an
identically configured second server PASSes with 400 items; 100 identical items are
INCONCLUSIVE (the ±3/n floor exceeds the 1-point margin), and 20 are INCONCLUSIVE by
`min_samples`. The decision and per-task results are persisted to SQLite and read back.
