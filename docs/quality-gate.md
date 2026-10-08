# Quality gate

Loom never ships a cheaper config without measuring that it is as good. Any change that can
alter model outputs (quantization, engine or engine version, speculative decoding, KV-cache
dtype, context extension) runs the model's pinned eval suite on the current config (the
**baseline**) and the proposed one (the **candidate**), and the gate decides PASS, REVIEW,
FAIL or INCONCLUSIVE. FAIL blocks the change; INCONCLUSIVE blocks it too unless the suite
says otherwise. REVIEW (every task passes, but the logprob divergence is above the limits
calibrated on the baseline's own numerical noise) is reported everywhere the decision
shows and does not block unless the suite says so.

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
| `json_schema` | `json_schema` | self-authored, `quality/data/json_schema.yaml` (Apache-2.0), data version 2 | 300 | reply parses and validates under strict `response_format` |
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

The verdicts never test against zero, but the reasons do. When the CI lies wholly below 0
the drop is real, and the reason says so whatever the verdict: a PASS reads "non-inferior
at t pts, but measurably lower (CI below 0)", an INCONCLUSIVE reads "a real drop (CI below
0), not shown to be within t pts". An INCONCLUSIVE therefore never hides a regression the
data already shows. The procedure is named by `GATE_METHOD`
(`paired-bootstrap-over-items/replicate-means/v2`), stored in every decision with each
side's replicate count.

### Replicated passes (engine nondeterminism)

Greedy decoding is not deterministic run to run on a serving engine: batch composition
changes reduction order, and a near-tie token can flip. Measured on Qwen3-8B / vLLM 0.30
IFEval (541 prompts, three independent runs on separate pods: 565b8d3f, 53f38c7b,
b1b904dc):

| Pair | Score A | Score B | Items that differ (A better, B better) |
|---|---|---|---|
| vLLM run 1 vs vLLM run 2 | 0.8244 | 0.8262 | 21 (10, 11) |
| vLLM run 1 vs vLLM run 3 | 0.8244 | 0.8207 | 22 (12, 10) |
| vLLM run 2 vs vLLM run 3 | 0.8262 | 0.8207 | 21 (12, 9) |
| vLLM run 3 vs SGLang | 0.8207 | 0.8078 | 21 (14, 7) |

32 of 541 items changed score across the three vLLM runs. So two runs of the same engine
disagree on as many items as vLLM and SGLang do, and with one pass per side that noise
alone gives the paired CI about ±1.5 pts: wider than half of IFEval's 2-point margin.
What separates the engines is the direction of the disagreements (14 vs 7 against SGLang,
in every pairing with a vLLM run), not their number.

`quality.replicates: R` runs the suite R times on each cell's engine instance (the
divergence half on the first pass only) and pools them (`pool_replicates`): each item's
score becomes its mean over the R passes, an unbiased estimate of the probability that
the config passes it. The gate is unchanged on those means. Writing an item's pass on run
r as Y_ir = p_i + e_ir, with p_i the config's pass probability and e_ir run noise, the
paired difference of means is

    D_i = (p_i^cand - p_i^base) + (ebar_i^cand - ebar_i^base),   Var(ebar_i) = Var(e_i) / R

so averaging divides the run-noise variance by R. The between-item spread of the true
differences, the thing non-inferiority generalises over, is untouched, and resampling items
with their pooled means is the standard bootstrap for a two-stage (item, pass) design. It
needs no new statistic, no new margin and no tuning, and it cannot average away a real
regression, which moves every pass the same way. `bench/tests/quality/test_replicates.py`
fails the build if a simulated -5 pt regression under this noise stops failing, at R = 1
and R = 3. Simulated with the measured noise, R = 3 cuts the IFEval CI half-width from
about 1.4 to 0.8 pts and R = 5 to 0.6.

Passes on one engine instance share its hardware. They sample the batching nondeterminism
above, not pod-to-pod hardware variation, which is a property of the deployment rather
than the engine. Each item's per-pass scores are kept in the samples
(`meta["replicate_scores"]`), so within-instance noise can be compared with the
cross-instance figures in the table.

Logprob divergence (below) is held to limits calibrated on the baseline's measured noise
floor, with self-KL taken at the upper and self top-1 agreement at the lower bound of
their bootstrap CIs:

- KL limit = max(`max_kl`, `noise_multiple` x self-KL);
- top-1 limit = 1 - max(1 - `min_top1`, `noise_multiple` x (1 - self top-1)), i.e. the
  allowed disagreement scales the same way;
- hard ceiling: `ceiling_kl`, `ceiling_top1`, whatever the floor.

| Divergence (checked in order) | Divergence verdict |
|---|---|
| asked for by the suite but failed (capture or scoring raised) | INCONCLUSIVE ("not measured: ...", with the error) |
| not measured | PASS ("not measured") |
| mean KL > `ceiling_kl` or top-1 < `ceiling_top1` | FAIL |
| within both calibrated limits | PASS |
| beyond a limit, and every task PASSes | REVIEW |
| beyond a limit, and any task FAILs or is INCONCLUSIVE | FAIL |

Without a self-divergence (a reference captured with plain `capture`, or older artefacts)
the absolute `max_kl` / `min_top1` are the limits and the reason says "uncalibrated".

| Suite default | Value | Why |
|---|---|---|
| `noise_multiple` | 5 | another engine's BF16 kernels perturb logits the way batch variance does (a different reduction order), so the gap is a small factor, not orders of magnitude |
| `max_kl`, `min_top1` | 0.05 nats, 95% | the limits never get stricter, so a bit-exact baseline (zero floor) does not flag last-ulp differences |
| `ceiling_kl` | 0.5 nats | the candidate gives the reference's confident token ~60% (e^-0.5) of its probability on average: a different model, not noise |
| `ceiling_top1` | 80% | one teacher-forced token in five changes; sound quantizations stay well above it, broken templates, RoPE scaling or weights do not |
| `review_blocks` | false | REVIEW asks a person to look; task scores already passed |

Sanity FAILs when any rate exceeds its limit (defaults: empty 1%, truncated 2%,
repetition 2%, language drift 2%).

The overall decision is the worst of all parts (FAIL > INCONCLUSIVE > REVIEW > PASS).
`blocked` is true on FAIL, on INCONCLUSIVE while `gate.inconclusive_blocks` is on (the
default), and on REVIEW only while `gate.review_blocks` is on (default off). `bench quality
gate` exits 7 when blocked and 0 otherwise, REVIEW included; `bench run` prints each gate
as BLOCKED, "allowed, needs review" or allowed. Every part carries a human-readable
reason, e.g.

```
gate FAIL (blocked)
  arithmetic: delta -26.50 pts [-30.75 pts, -22.50 pts], n=400 is a drop of more than 1.00 pts
  json_schema: delta -23.33 pts [-35.00 pts, -13.33 pts], n=60 is a drop of more than 6.00 pts
  divergence: mean KL 2.0022 nats is above the hard ceiling 0.5000; top-1 agreement 21.71% is below the hard ceiling 80.00%
  sanity: 460 outputs: empty 0.00%, truncated 0.00%, repetition 0.00%, language_drift 0.00%

gate REVIEW (needs review, not blocking)
  ...
  divergence: needs review: top-1 agreement 87.70% is below 95.00% while every task passed non-inferiority; limits KL 0.0500 nats, top-1 95.00%: the looser of the absolute limits and 5x the baseline's noise floor (self-KL <= 0.0000 nats, self top-1 >= 100.00%)
```

`GateDecision.details()` is the JSON stored in `bench_gate_decisions.details`; its
`divergence` carries the candidate's result, the baseline's `self_divergence` and the
`limits` applied (calibrated or not, the noise multiple, the self-KL and self top-1 bounds
used, the resulting limits and the ceiling). Reports and the site rank a REVIEW config
normally and badge it "needs review", with the divergence against its limits and the
noise floor in its recommendation and on its model page.

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
the native needle (400) 2 points, the 300-item JSON-schema set 3 points, and the 60-item
tool-calling set 6 points, which catches broken parsers and templates rather
than subtle drift. Each suite file repeats this table next to the tasks.

A pinned set's data version is part of its task version (`json_schema` 300 items is
`1+data.2`, the earlier 60 items `1+data.1`). The gate refuses to pair a baseline and a
candidate whose versions differ ("rerun the baseline"), so results from different item
sets never mix.

## Logprob divergence

`quality/divergence.py`, against a reference config serving the BF16 weights:

1. The reference greedily continues each of the 48 pinned prompts
   (`quality/data/divergence_prompts.yaml`) for `max_new_tokens` tokens.
   A suite can name its prompts by index instead (`divergence.prompt_ids`), and its
   `hard_prompts`: prompts whose continuations split characters across tokens on that
   model (Qwen3: 20 and 40). A limited suite (`quality.limit`, the smoke) keeps the hard
   prompts first, so the smoke meets what the real run meets.
2. Both endpoints score the identical text prompt + continuation with `/v1/completions`,
   `echo: true, logprobs: k, max_tokens: 1` (vLLM v0.30 and SGLang v0.5.21 both support
   echo with logprobs). Every continuation position is then conditioned on the same tokens
   on both sides (teacher forcing), so differences come from weights and kernels alone.
3. Per position: top-1 agreement (same argmax token) and approximate KL(ref || cand).
   Positions are the echoed tokens that start in the continuation; a server that reports
   no `text_offset` (SGLang: -1 for every token) is first brought to vLLM's form (below).
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

**Characters split across tokens.** Byte-level BPE tokenizers (Qwen3, Llama 3) can split
one character across tokens: Qwen3 splits " √" into the bytes `" \xe2\x88"` and `"\x9a"`.
Engines render such tokens differently (`loom_bench/detokenize.py`). vLLM renders the
first token as "" and the one that finishes the character as all of it (" √"), and
reports text offsets. SGLang renders each token on its own, a fragment as its raw bytes
in latin-1 (`" â\x88"`, `"\x9a"`), and reports every offset as -1. Before comparing, a
per-token response is converted to vLLM's form:
- its tokens are matched against the scored text's UTF-8 bytes, which fixes each token's
  bytes, lossy U+FFFD fragments included;
- it is re-rendered with vLLM's own rule (ported from v0.30.0), and its offsets are
  rebuilt the way vLLM computes them;
- top-k keys get the same treatment.

A token sequence that does not spell the text is rejected, and a different tokenization
still fails the position-by-position comparison. One case stays a guess. SGLang's form
cannot tell a latin-1 letter key from a lone fragment byte ("é" vs the byte 0xE9). Such a
key reads as a fragment when it holds a C1 control, finishes the previous tokens'
unfinished character, ends in an unfinished character of two or more bytes, or is a lone
3- or 4-byte lead byte after CJK or symbol text. Otherwise it reads as the letter it
shows. This only touches low-ranked alternatives; the echoed token's key is exact. Sweep
565b8d3f lost SGLang's eval to this before the fix. The real vLLM capture of prompts 20
and 40 from that sweep is a test fixture, scored against SGLang's rendering of the same
Qwen3 token ids, with KL 0 and top-1 1.0. A leading special token (BOS) that a server
counts into its offsets, although the text does not hold it, no longer shifts the
selected positions.

**A failed divergence keeps the scores.** The divergence half of an eval job (capture,
floor or score) runs after its tasks. If it raises, the job still returns the task scores,
with `divergence_error`. The runner then:
- records a `divergence_failed` event;
- keeps the task scores;
- leaves candidates unscored when the baseline's capture failed;
- gates with the divergence check INCONCLUSIVE ("not measured: candidate divergence
  failed: ..." or "... baseline divergence capture failed: ..."), blocked, while still
  deciding and reporting each task.

The experiment ends `completed` with "divergence failed in N of M quality evals" and
exit 8. `samples.json` keeps the error, so `bench quality gate` decides the same way.

**Noise floor.** Two healthy engines serving the same BF16 weights never agree bit for
bit: kernels sum in different orders, and an engine's own numerics change with batch
composition. Fixed limits would either let real damage through or block a healthy engine
on kernel noise, so each run measures the noise and the gate scales its limits to it.
After capturing the reference, the baseline's eval job (`divergence: capture_and_floor`)
scores the same endpoint against its own capture once more at the suite's
`floor_concurrency` (1, against the job's 16 at capture: every request alone, so batch
sizes, kernel tile choices and reduction orders differ). That self-divergence is pure
numerical noise of this model on this engine and hardware; it is stored with the
reference (`ReferenceLogprobs.self_divergence`), in the baseline's `samples.json`, and in
the `reference_captured` event, and the gate calibrates every candidate's limits on it.
It costs one more echo request per prompt (48) on the baseline only. It runs on the host
inside the eval job, so it works the same on the mock, local and aws_ec2 providers.

**Reference capture.** Steps 1-2 for the reference and steps 2-4 for the candidate are
separate calls, so the two engines never have to be up at the same time (one GPU host runs
one engine at a time):

- `capture_reference(client, prompts, top_k, max_new_tokens) -> ReferenceLogprobs`: the
  reference's continuation of each prompt and its top-k logprobs at every continuation
  position, as JSON;
- `score_against_reference(client, reference) -> DivergenceResult`: the candidate scores the
  same texts and is compared position by position with the stored reference.

`measure_divergence` (both endpoints up) is capture followed by score. In an experiment the
baseline variant's eval job captures the reference and its noise floor while its engine is
up; the runner stores both as `results/<experiment>/evals/<config hash>/reference.json`
with the config hash and full provenance, and passes the reference in every candidate's
eval job, which scores against it when that engine is up later. Prompts, top-k and the continuation length travel with the
reference, so a candidate is always scored on exactly what the reference saw.

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

`record_suite_result` writes one `bench_eval_runs` row per task (mean, t-interval clipped
to [0, 1] as scores are fractions, task
version, provenance including dataset source, revision and license) and `record_gate`
writes the decision to `bench_gate_decisions`.

Inside an experiment the same work is an `EvalJob` (`loom_bench.jobs`): the resolved suite,
the task subset, the endpoint as seen from where the job runs, request extras, the seed,
the code-execution opt-in, and `divergence: capture_and_floor` (baseline: capture, then the
noise-floor pass) or `score` with the stored reference (candidates); plain `capture` skips
the floor. `execute_eval_job` runs it in-process for the mock and local
providers; `bench quality job --in job.json --out result.json` runs it on a GPU host. The
`EvalJobResult` carries per-task ItemResults (scores and content hashes), task versions,
per-task seconds, the sanity rates and the capture (with its self-divergence) or the
divergence; model outputs stay where the job ran, except that the native tasks on
self-authored data keep a capped copy (`clip`, 2000 characters per string) of each
*failed* item's output in its `meta`: tool_calling the raw tool calls and text, the
expected call and the parameter that failed; json_schema the raw reply and the main schema
violation. A low score can then be explained from the samples (a quoted `"250"` for a
number parameter shows as such) without re-running the model. Each config's `samples.json` keeps its
divergence (with the config hash of the reference it was scored on) or its self-divergence,
so `bench quality gate` re-decides with the same calibrated limits; a divergence scored on
another baseline's reference is not reused.

**Subsets.** A suite may name subsets of its tasks (`subsets: {phase0: [...]}`); an
experiment picks one with `quality.subset`. Tasks in a subset keep their full pinned items,
so the half-widths in the sample-size table still apply to them and their results pair item
for item with a full-suite run. Tasks left out are simply not measured by that run. The
Phase 0 subset of both pinned suites is GSM8K, IFEval, tool calling and JSON schema; MMLU-Pro,
RULER, needle and code wait for a full-suite run.

**Planning.** Each task reports how many items it will score (`planned_items`); lm-eval
tasks without explicit `samples` cannot count their docs before the harness loads them,
so the suite states `items` for them and `bench plan` refuses a suite that does not. The
baseline's eval step also carries the noise-floor pass: prompts x `EVAL_FLOOR_PROMPT_S`
(1 s, one echo request alone on the engine) / `floor_concurrency`, 48 s with the pinned
suites.

- **lm-eval** is the optional `lmeval` extra of the `loom-bench` workspace package:
  `uv sync --all-packages --extra lmeval` from the repo root (plain `uv sync --extra
  lmeval` fails there: the root project has no extras). Without it, `lm_eval` tasks fail
  with an error saying how to install it. The command line and the
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
  egress or credentials. Nothing executes unless `allow_code_exec=True` is passed. On
  `aws_ec2` an eval job runs in the client container on a disposable host: uid 10001,
  denied the instance metadata service (no instance-role credentials), no secrets or AWS
  settings in its environment, a read-only virtualenv and only its job directory
  writable, removed after the job. It shares the host network (to reach the engine on
  loopback), so programs can reach the internet and the engine; that is why code tasks
  still need `quality.allow_code_exec: true`.
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
   can. `run_suite` then fails the whole task (`EvalTaskFailed`, naming the status and the
   first error) when every item errored, or when more than 10% of items were rejected
   with a non-retryable 4xx: the server refused the request shape (e.g. a missing engine
   flag), so the zeros would measure the config and two broken engines would gate 0 vs 0.
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

Two more cases run the experiment path (`capture_and_floor` on the baseline, `score` on
the candidate) with the mock's `logprob_jitter`, per-request logit noise that stands in for
batch-variant kernels. A jittery baseline against a candidate with the same jitter plus a
little static `logprob_noise` (another engine's kernels): its divergence is above the old
fixed 0.05 nats and 95% top-1 but within 5x the measured floor, and it PASSes (gated on the
absolute limits alone it would be REVIEW). A drifted candidate against a bit-exact baseline
(zero floor) with unchanged answers is REVIEW, not blocked.
`bench/tests/runner/test_quality_e2e.py` runs PASS, REVIEW and FAIL candidates through a
mock experiment, `bench quality gate` (exit 0 for REVIEW, 7 for FAIL) and the report.
