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
| `tool_calling_strict` | `tool_calling_strict` | the `tool_calling` items, tools sent with `strict: true` (below) | 60 | as `tool_calling` |
| `json_schema` | `json_schema` | self-authored, `quality/data/json_schema.yaml` (Apache-2.0), data version 2 | 300 | reply parses and validates under strict `response_format` |
| `toy_arithmetic` | `toy_arithmetic` | generated, seeded | configurable | "The answer is N" is right; used with the mock backend |

All requests are greedy (temperature 0) with a fixed seed. Qwen3 runs in non-thinking mode
(`chat_template_kwargs: {enable_thinking: false}`, from the suite), which is forwarded to
native tasks and to lm-eval's chat requests.

### Strict tool calling

`tool_calling_strict` runs the 60 `tool_calling` items with the same matching rules. Only
the request differs: every offered function carries `"strict": true`, and every object in
its `parameters` gets `additionalProperties: false` (`strict_parameters` in
`quality/tasks/tool_calling.py`). `tool_choice` stays `"auto"`, and `required` lists are
unchanged. OpenAI's strict style would also make every property required, with optional
ones nullable. Neither engine needs that, and it would change the task: the model would
have to write `null` for an argument it means to leave out.

Why it exists: Llama 3.3 70B scores 0.450 on `tool_calling` because it writes numbers as
strings (`"250"` for a number parameter), which the scorer rejects as a real client would.
With strict tools the engine is meant to constrain the arguments to the schema, so that
failure could not happen. On Llama 3.3 70B with vLLM 0.30 it does not: strict mode never
engaged (below, "What the 70B H100 run showed"). Both numbers are worth knowing.
**`tool_calling` stays the headline**: most clients send plain tools, so it measures what
they get. `tool_calling_strict` measures what a client that opts in gets, which for this
model and engine is the same thing. Reports list it right after `tool_calling` (task names
sort together). The model page notes that it is the strict variant and that
`tool_calling` is the headline, and the site's methodology says the same.

Version `strict.1+data.1` (the plain task is `1+data.1`). Each item's content hash covers
the strict tools, so the two tasks never pair in a gate, and a change to the strict
request bumps `strict.1`. Both pinned suites list it with the same 6-point margin and
`min_samples: 50` as `tool_calling`, since it has the same 60 items. It is gated like any
other task whenever a run measures it: a full-suite run, or a subset that names it. It is
not in the `phase0` subsets.

Engine behaviour (checked against vLLM v0.30.0 and SGLang v0.5.21 source):

- **vLLM** (`tool_parsers/structural_tag_registry.py`, `parser/abstract_parser.py`): under
  `tool_choice: "auto"`, a request with at least one `strict: true` tool gets an xgrammar
  structural tag when the tool parser has one and `VLLM_ENFORCE_STRICT_TOOL_CALLING` is on
  (the default). Without a strict tool, auto output is never constrained. The tag lets
  text through until the model starts a call. From then on, the name must be one of the
  offered tools and the arguments must match that tool's `parameters` schema. For
  `llama3_json` (Llama 3.3) that is xgrammar's builtin `llama` format, triggered by
  `{"name": ` and forcing `{"name": "<tool>", "parameters": <schema JSON>}`. That trigger
  is the only way in: a call written any other way (`{"type": "function", "name": ...}`,
  `{"name":"...` without the space, a newline after `{`) never fires it, stays free text,
  and the `llama3_json` parser, which accepts any JSON object with `name` and
  `parameters` or `arguments`, still returns it as a tool call. For `hermes`
  (Qwen3) it is vLLM's own format, triggered by `<tool_call>` and forcing
  `<tool_call>\n{"name": "<tool>", "arguments": <schema JSON>}\n</tool_call>`. vLLM's docs
  recommend the OpenAI strict-schema style "for best compatibility" but do not require
  it. A reply with no tool call is still allowed, and still scores 0.
- **SGLang** (`function_call/function_call_parser.py`): the same rule. Under `"auto"`, any
  `strict: true` tool (or `SGLANG_TOOL_STRICT_LEVEL` at `FUNCTION` or above) switches on
  a structural tag. The `qwen25` detector (Qwen3) has no xgrammar builtin format, so
  SGLang uses its legacy structural tag: triggered by `<tool_call>`, it forces
  `<tool_call>\n{"name":"<tool>", "arguments":` followed by the schema-constrained
  arguments and `}\n</tool_call>`. Only tools marked strict get their schema; the others
  get `{}`. SGLang's `llama3` detector triggers on `<|python_tag|>`, which Llama 3.3
  normally does not emit before a JSON call, so strict mode would rarely engage for the
  70B on SGLang. The 70B runs on vLLM only.

Both engines enforce the schema through their grammar backend (xgrammar by default, as for
`json_schema`). No extra engine flag is needed beyond the tool parser flags that
`tool_calling` already uses.

#### What the 70B H100 run showed (9f0853d7): strict mode never engaged

BF16 scored 0.450 on both tasks, FP8 0.450 and 0.433. The stored failures (each failed
item keeps its raw call) show the grammar constrained none of them:

- Every one of BF16's 33 strict failures carries a value the schema forbids, which the
  structural tag cannot emit once it is triggered: `"days_from_now": "30"` and
  `"party_size": "10"` for integers, `"amount": "40"` for a number, `"notify": "true"`
  for a boolean, `"dimensions": "[6, 9]"` for an array (FP8: 34 of 34).
- Keys come out of schema order, which xgrammar's JSON schema grammar (`any_order:
  false`) also forbids: BF16 strict m-currency-2 is
  `{"to_currency": "NZD", "amount": "40", "from_currency": "AUD"}` (schema order:
  amount, from_currency, to_currency), m-table-2 is
  `{"party_size": "10", "time": "20:45", "restaurant": "Sakura Garden"}`.
- 31 of the 33 calls both tasks failed are byte-identical between `tool_calling` and
  `tool_calling_strict`. The two that differ (s-reminder-1 `"Renew your passport"` vs
  `"renew my passport"`, m-currency-2's key order) differ because the prompt does: the
  chat template prints each tool's JSON, now with `"strict": true` and
  `"additionalProperties": false`. That also shows the strict flag reached vLLM.

Why, checked against vLLM v0.30.0 and xgrammar 0.2.7 (the image was built 2026-09-22;
0.2.3, vLLM's test pin, has the same trigger):

1. The request is right. `strict: true` survives vLLM's `FunctionDefinition`, the
   renderer's `preprocess_chat` calls the parser's `adjust_request` whenever
   `tool_choice` is not `"none"`, and `_apply_structural_tag` builds the tag for
   `"auto"` with a strict tool (`structural_tag_registry.get_model_structural_tag`) and
   sets it as the request's `structured_outputs`. Nothing in the path drops it.
2. The grammar works once triggered. Built exactly as vLLM builds it (`llama`, auto, the
   m-currency-2 tools) and compiled with the Llama 3.3 tokenizer, xgrammar rejects
   `{"name": "convert_currency", "parameters": {"to...` and `... {"amount": "` (the
   observed calls), with or without a leading `<|python_tag|>` (an ordinary token to
   xgrammar here), and accepts `... {"amount": 40, ...}`.
3. It also accepts the observed call written as `{"type": "function", "name":
   "convert_currency", "parameters": {...}}` or `{"name":"convert_currency",...}`: no
   trigger, so free text to the end.

So Llama 3.3 70B did not begin its calls with `{"name": `. Which prefix it used cannot be
read from these samples: `llama3_json` re-serialises the call (`json.dumps` of the
arguments), so only argument order and values survive. A sibling under the same chat
template behaves differently: Llama 3.1 8B (MLX 4-bit, greedy, all 60 strict prompts)
began 58 calls with `<|python_tag|>{"name": ` and 2 with `{"name": `, all on the
trigger. llama.cpp's Llama 3.x grammar (ggml-org/llama.cpp a83f528, "fix llama 3.x")
allows an optional `"type": "function",` before `name` and triggers on
`{"type": "function"`, `{"name":` and `{\n  "name":` as well, because Llama 3.x models
write all of these. The likeliest prefix for the 70B is `{"type": "function", ...`,
which mirrors how the template prints each tool (`{"type": "function", "function":
{...}}`).

Classification: engine behaviour, a mismatch between xgrammar's single Llama trigger and
what Llama 3.3 70B writes. It is not a harness bug: no request shape under
`tool_choice: "auto"` changes where the model starts its call. `tool_choice: "required"`
would engage the grammar (its `llama` format then forces the output to start with the tag,
so `{"type": ...` and `{"name":"` are rejected at the first token), but that is a
different request (the model may not answer without a call), not what a client opting
into strict tools sends, so the task keeps `"auto"`. The task's numbers are therefore
correct for what they claim: a client sending strict tools to Llama 3.3 70B on vLLM 0.30
gets no constraint and the same quoted numbers.

To settle the prefix, a failed `tool_calling_strict` item now also stores
`meta["unconstrained_text"]`: the same messages and tools re-sent once with
`tool_choice: "none"` and `skip_special_tokens: false`, so no parser and no structural
tag run and the reply is the model's own text, special tokens included. On vLLM the tools
stay in the prompt under `"none"` (`--exclude-tools-when-tool-choice-none` is off by
default), so with greedy decoding this is the reply the scored request got whenever the
grammar did not engage. (SGLang drops the tools under `"none"`, so there it answers a
different prompt.) It is never scored, and the task version is unchanged (`strict.1`):
items, prompts and scoring are the same.

The one item FP8 lost on the strict task (s-books-1: BF16 `{"max_results": 5}`, FP8
`{"max_results": "5"}`) is that same quoted-number habit tipping on a borderline item
under the strict prompt; the plain prompt gives a typed 5 on both. FP8 and BF16 differ
on 2 of 60 strict items (s-books-1, and m-table-2's key order); on the plain task they
agree on every verdict and on every one of the 33 failed calls, byte for byte, in line
with the 97.99% top-1 agreement.

#### What can decide a 60-item task

The strict verdict, -1.67 pts [-6.67, +3.33] against a 6-point margin, is the ±3/n
floor at work: with 60 items the CI is at least ±5 pts around the delta, so a pass needs
delta ≥ -1 pt, and one net lost item is -1.67. Run through `evaluate_task`:

| Lever | Strict verdict |
|---|---|
| as run: 60 items, FP8 loses 1 | INCONCLUSIVE, [-6.67, +3.33] |
| `replicates: 3`, the flip repeats every pass | INCONCLUSIVE, unchanged |
| `replicates: 3`, FP8 fails it in 2 of 3 passes | INCONCLUSIVE, [-6.11, +3.89] |
| `replicates: 3`, FP8 fails it in 1 of 3 passes | PASS, [-5.56, +4.44] |
| 67 items, 1 lost | PASS, [-5.97, +2.99] |
| 120 items, 2 lost (the observed rate) / 3 lost | PASS, [-4.17, +0.83] / [-5.83, 0.00] |
| 150 items, 3 lost | PASS, [-4.67, 0.00] |

Replicates only help if the flip is run noise, and tool calls here are deterministic:
in b03b3c52 no `tool_calling` item changed across three passes on either engine (IFEval:
19 and 32 did), and in 9f0853d7 BF16 and FP8, different weights, wrote byte-identical
calls on all 33 plain failures. The s-books-1 flip is a BF16-vs-FP8 difference and
would repeat, so replicates, or simply running the strict task again, buy nothing. More
items do: at 120+ items the task tolerates the observed rate with room to spare, and so
does `tool_calling`, whose zero-delta pass today sits on the same ±5-pt floor. Growing
`tool_calling.yaml` (data version 2, which also fixes the `from_currency`/`to_currency`
descriptions: the unquoted `ISO 4217 code, e.g. USD` in a YAML flow mapping parses as a
truncated description plus a stray `"e.g. USD": null` key, which the prompt prints) and
re-running both configs' quality suite on one 2x H100 pod is the lever. `bench plan` on a
quality-only twin of the H100 spec (`workloads: []`, `phase0-strict`): cold start 30.6
min, warm start 15.1, two evals of ~30 min each, 1.86 h, $14.88 at $8.0156/h; worst
case $18.04 at a 135-minute TTL. The 9f0853d7 timings (BF16 healthy after 16.5 min,
suites of 7.3 min BF16 and 4.7 min FP8, FP8 restart 7 min, plus eval setup) put the
likely cost near 40 minutes, ~$5.30.

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

Measured within one instance in b03b3c52 (three passes each): vLLM IFEval pass means were
0.8096, 0.8262 and 0.8170, with 32 items unstable and 18-27 disagreements per pair of
passes. SGLang's were 0.8059, 0.8096 and 0.8133, with 19 items unstable and 8-16
disagreements. So noise within one instance is as large as noise across pods. The
resulting gate:

| Task | delta (SGLang - vLLM), 3 passes each | Margin | Verdict |
|---|---|---|---|
| gsm8k (1319) | +0.03 pts [-0.35, +0.40] | 1 | PASS |
| ifeval (541) | -0.80 pts [-2.03, +0.37] | 2 | INCONCLUSIVE (lower bound 0.03 pts past the margin; no measurable drop) |
| json_schema (300) | -0.22 pts [-1.22, +0.78] | 3 | PASS |
| tool_calling (60) | +0.00 pts [-5.00, +5.00] | 6 | PASS |

The single-pass gate in b1b904dc read IFEval as -1.29 [-2.96, +0.37]; most of that drop
was run noise. A post-hoc pool of every stored pass (vLLM 6 across four pods, SGLang 4
across two) gives -1.16 [-2.26, -0.14]. It mixes instances, and the bootstrap over items
does not model instance-to-instance variance, so that CI is optimistic. The defensible
reading: on IFEval, SGLang is between equal and about 2 points below vLLM, with roughly 1
point most likely. That is not shown to be within the 2-point margin, so the gate stays
blocked.

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

**A baseline from an earlier experiment.** A candidate can be gated against a config that
another experiment already evaluated (`quality.baseline: {experiment, config_hash}`,
docs/benchmark-lab.md): its stored `samples.json`, `reference.json` and noise floor take the
baseline cell's place, so a 141 GB BF16 model is not served again to gate its FP8 row. The
candidate's eval job scores divergence on the stored reference (the prompts, top-k and
continuation length travel with it), and `bench plan` refuses a stored baseline that cannot
pair. When the two sides ran different task lists (a task added since the baseline ran),
the gate pairs the tasks both ran and lists the others in `GateDecision.ungated` with a
reason line "`<task>: not gated: the baseline did not run it`"; their scores are still
recorded. Sanity rates cover every candidate output, ungated tasks included. Within one
experiment both sides run the same tasks and a mismatch is still an error.

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
