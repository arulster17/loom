# Speculative decoding: what it is, when it pays, and what Loom measured

Status, 2026-11-04: the Qwen3-8B study **ran** on AWS (experiment 53e4c107, results
below; report in [reports/8b-specdec-v1](../../reports/8b-specdec-v1/leaderboard.md)).
The 70B findings are from runs stored earlier (cf4d1614, 55102ddb).

## The short version (how to explain it in an interview)

A language model writes one token at a time, and on a GPU each of those steps is slow for
a boring reason: the GPU has to read every weight of the model out of memory just to
produce one token. The arithmetic is cheap; moving the weights is what takes the time.

Speculative decoding cheats that. Something cheap guesses the next few tokens (the
**draft**). The big model then checks all of the guesses in **one** step (the
**verify**), which costs about the same as producing one token, because the weights are
read once either way. Every guess it agrees with is a token you got for free; at the
first wrong guess it keeps its own token instead and throws the rest away. The output is
the same as the big model would have written on its own. You trade spare compute for
fewer slow, memory-bound steps.

It helps most when the GPU is serving few requests at once (there is spare compute to
burn on checking guesses) and fades as the batch grows (the GPU is already busy, so
checking wrong guesses starts to cost real time). Where the guesses come from is the
other half of the story: **n-gram** lookup copies them from text already in the prompt
(free, great when the answer repeats the input, useless otherwise); **EAGLE3** uses a
small trained network that reads the big model's internal state (costs a little, works
on ordinary chat).

## What we measured on Qwen3-8B (53e4c107, 2026-11-04)

**In plain words.** On one L40S, EAGLE3 made chat replies stream about twice as fast per
token at every load we tried, and n-gram did almost nothing for chat. On real code edits,
where the reply copies most of the prompt, both made replies stream about three times as
fast, n-gram slightly more. We never found the load where speculation stops helping:
up to the highest load we ran, the speculating engines kept fewer requests in flight
(each one finishes sooner), so they never got busy enough for the extra checking to
cost them. Quality looks unchanged, but the gate could not formally prove it on one
task (IFEval), so neither speculative config is ranked or priced yet. The handful of
chat replies EAGLE3 could not speed up were almost all in Chinese, Japanese, Arabic or
German, whose tokens its 32,000-token guessing vocabulary mostly lacks.

**The run.** `qwen3-8b-specdec-aws` as experiment 53e4c107: one AWS g6e.xlarge on-demand
host (1x L40S) in us-west-2, vLLM v0.30.0, Qwen3-8B BF16 non-thinking; 63 of 63 load runs
completed, 3 replicated eval passes per row, two gates. It took 5.1 h (01:56 to 07:03 UTC)
of its 8 h TTL, planned at 7.46 h, and cost $9.62 recorded (planned $14.04, cap $15.25).
Its smoke, 75ba8277, passed every path for $1.51. No request was cancelled at the drain
timeout at any load, so every speed figure below is over complete requests. Report:
`bench report -e 53e4c107 --alt-slo tpot_ms.p95=20 --out reports/8b-specdec-v1`.

**Speed: time per output token (TPOT p50), and the speed-up against no speculation at
the same load** (geometric means over 3 repetitions; CIs in the report):

| Workload | Load | bf16 | bf16-ngram | bf16-eagle3 | Requests in flight: bf16 / ngram / eagle3 |
|---|---|---|---|---|---|
| chat | 0.25 req/s | 24.0 ms | 22.5 ms (1.07x) | 12.0 ms (2.00x) | 1.7 / 1.6 / 0.9 |
| chat | 1 req/s | 27.1 ms | 26.7 ms (1.01x) | 13.8 ms (1.96x) | 8.3 / 7.6 / 4.4 |
| chat | 2 req/s | 34.0 ms | 33.1 ms (1.03x) | 16.4 ms (2.07x) | 19.9 / 18.6 / 10.2 |
| chat | 3 req/s | 46.5 ms | 43.1 ms (1.08x) | 20.1 ms (2.31x) | 40.3 / 36.0 / 18.6 |
| code-edit | 0.5 req/s | 23.7 ms | 7.7 ms (3.09x) | 8.8 ms (2.70x) | 2.8 / 1.0 / 1.1 |
| code-edit | 2 req/s | 25.0 ms | 8.2 ms (3.06x) | 9.3 ms (2.70x) | 11.1 / 4.0 / 4.3 |
| code-edit | 4 req/s | 27.1 ms | 9.0 ms (3.02x) | 10.1 ms (2.69x) | 22.4 / 8.2 / 8.7 |

("Requests in flight" is vLLM's running-request gauge, averaged over the window.)

**Acceptance** (vLLM's counters; nearly flat across loads, ranges below):

| Workload | Method | Tokens per verify step that drafted | Draft tokens kept | Tokens per streamed chunk |
|---|---|---|---|---|
| chat | n-gram | 1.84-1.97 | 28-32% | 1.23-1.26 |
| chat | EAGLE3 | 2.29-2.34 | 43-45% | 2.29-2.32 |
| code-edit | n-gram | 3.61-3.65 | 87-88% | 3.09-3.14 |
| code-edit | EAGLE3 | 3.06 | 69% | 3.03-3.05 |

n-gram's two numbers differ because it drafts only when it finds the last few tokens
earlier in the text: on chat that happened on a minority of steps, so the per-chunk
figure (every step) is 1.25 although a step that did draft gained ~1.9 tokens. On code
edits the match is almost always there, and its drafts are kept 88% of the time, more
than EAGLE3's 69%: copying the input is exactly what an n-gram lookup does.

**Why the speed-up grew with load instead of fading.** The textbook expectation (section
"Why it helps at low batch") is a crossover at high batch. It did not appear here, and
the in-flight column says why: at the same arrival rate, a request that streams twice as
fast is in the engine half as long (Little's law), so EAGLE3 at 3 req/s ran 18.6 requests
at once against bf16's 40.3. bf16's own steps slowed with its larger batch (24 to 46.5 ms
per token), while EAGLE3's verify step stayed near bf16's low-load step (ITL 27.9-38.1 ms
against bf16's 23.8-38.0). The largest batch any speculating row reached was ~19
requests, short of where verifying 4 tokens per request would cost more than it saves. So
the crossover, if it exists on this card, lies beyond these loads; finding it would take
chat at 4-6 req/s (where bf16 is already past its SLO).

**Cost at the SLO** (TTFT p95 1 s, TPOT p95 50 ms; $1.8829/h). On chat, bf16 and n-gram
hold the SLO up to 2 req/s (both fail at 3, TPOT p95 55-56 ms), while EAGLE3 held it at every
tested load, 3 req/s included (TPOT p95 27.2 ms): $0.1336 per 1M blended against
$0.2047, about 35% less. That is an upper bound for EAGLE3 (its goodput was not
bracketed), and none of it is ranked: both gates are inconclusive and the chat cells are
untrusted (throughput run-to-run CV 10.8% for bf16 and n-gram, TTFT p95 CV 13.2% for
EAGLE3). On code edits every row met the SLO at every load, so all three tie at ≥4 req/s
($0.2783 per 1M blended): the 50 ms target is too loose for speculation to change the
cost there.

**At a tight interactive SLO (TPOT p95 20 ms; not the declared SLO).** bf16 cannot serve
either workload at all (its TPOT p95 is 24-25 ms even at the lowest load). On code edits
both speculative rows meet 20 ms at every load up to 4 req/s ($0.2783 per 1M blended,
unranked). On chat, EAGLE3 met it at 1 req/s (TPOT p95 17.7 ms [16.3, 19.3]) but the
view reports no goodput: its 0.25 req/s point failed on the CI's upper bound (15.7 ms
[11.5, 21.6], from only 33-36 requests per repetition), and loads above a failing one
never count. The report's sentence that this "points to per-request latency, not load"
does not fit that case: it is a thin sample at the lowest load.

**Quality: the gates are inconclusive on IFEval only, and the outputs match bf16 as
closely as bf16 matches itself.**

| Check | bf16-ngram vs bf16 | bf16-eagle3 vs bf16 | Margin |
|---|---|---|---|
| gsm8k (1,319) | +0.08 pts [-0.40, +0.56] pass | +0.05 [-0.28, +0.40] pass | 1 pt |
| ifeval (541) | -0.55 pts [-2.03, +0.92] inconclusive | -0.49 [-2.16, +1.11] inconclusive | 2 pts |
| json_schema (300) | -0.33 [-1.33, +0.67] pass | +0.11 [-0.89, +1.11] pass | 3 pts |
| tool_calling, strict (135 each) | 0.00 [-2.22, +2.22] pass | 0.00 [-2.22, +2.22] pass | 6 pts |
| divergence (48 prompts, 3,068 positions) | KL 0.00053, top-1 99.15% pass | KL 0.00048, top-1 99.48% pass | |
| sanity (7,284 outputs) | pass | pass | |

- The divergence check feeds bf16's own greedy continuations to each engine and compares
  the next-token probabilities. bf16 scored against its own capture gives KL 0.00051 and
  top-1 99.22%; n-gram and EAGLE3 land at the same values. At the level of individual
  token probabilities, speculation changed nothing measurable.
- IFEval misses its 2-point margin by 0.03 pts (n-gram) and 0.16 pts (EAGLE3) on the
  confidence interval's lower end; neither point delta is close (-0.55, -0.49). bf16's
  three IFEval passes themselves scored 0.8207, 0.8207 and 0.8262 (21 items changed on
  the third pass), against 0.8170 on every n-gram pass and 0.8189, 0.8170, 0.8170 for
  EAGLE3. So half the measured gap is bf16's third pass running high.
- Item by item, 12 IFEval items flipped consistently between bf16 and n-gram (4 in
  n-gram's favour, 8 against) and 15 for EAGLE3 (5 for, 10 against). With so few items
  that split is not distinguishable from a coin (two-sided binomial p ≈ 0.4 and 0.3), and
  vLLM's own reruns disagree on ~21 IFEval items ([quality-gate.md](../quality-gate.md),
  "Replicated passes"). It is not evidence of a drop; it is also not proof there is none.
- What settles it is a tighter IFEval interval, not more passes: the run-noise part is
  already averaged over 3 passes a side, and what remains is item-level flips. More
  IFEval-like items (IFEval has 541) or a second environment would; neither was run. The
  eval outputs themselves are not stored (only scores), so a byte-for-byte comparison of
  answers is not possible from this run.

**What the draft missed** (`bench draft-misses 53e4c107 --cell bf16-eagle3`, per
workload, in `reports/8b-specdec-v1/draft-misses/`):

- Chat: 10 of 2,865 requests got 1.2 tokens per chunk or fewer. Their output was a median
  69% outside the head's 32,000-token draft vocabulary, against 3.3% for every other
  request. Their outside tokens are Chinese, Japanese, Arabic, Persian and German words;
  one is a 6-token reply with nothing outside (too short to draft). Across all chat
  requests the share outside the vocabulary and tokens per chunk move against each other
  (Spearman -0.30): median 2.30 tokens per chunk under 5% outside, 2.17 at 5-20%, 1.55
  at 20-50%, 1.18 above 50%.
- Code edits: no request fell to 1.2; the lowest, 1.67-2.00, have rare identifiers and
  licence headers outside the vocabulary (`.basicConfig`, `Copyright`, ...); the
  correlation is stronger (Spearman -0.60).
- So the draft vocabulary does explain the low-acceptance requests on this model. It is
  the hypothesis the 70B section below left open; the 70B itself was never re-run with
  stored text.

**Where the data is weak.**
- Every goodput except chat bf16/n-gram is a lower bound (no tested load failed), so the
  costs for those rows are upper bounds.
- The chat cells are "untrusted" (run-to-run CV above 10% on throughput or TTFT p95);
  three repetitions of 180 s at these loads.
- Both gates are inconclusive, which blocks ranking and pricing under the quality
  standard: nothing here is a shippable row yet.
- One host, one region, one day; the 0.25 req/s points have only ~33-36 requests per
  repetition.
- n-gram's lower TTFT on code edits (46-59 ms against bf16's 75-87, CIs separated) is
  measured but not explained; plausibly fewer decode steps competing with prefill, not
  tested.

## Terms (each defined once)

- **Token**: the unit a model reads and writes, roughly three quarters of a word.
- **Decode step**: one forward pass of the model that produces the next token for every
  request in the batch.
- **Batch**: the requests the engine advances together in one step. vLLM forms it
  continuously: requests join and leave between steps.
- **Memory-bound / compute-bound**: a step is memory-bound when reading the weights from
  GPU memory takes longer than the arithmetic. Decode at small batch is memory-bound:
  Qwen3-8B's 16.4 GB of BF16 weights over an L40S's 864 GB/s is at least 19 ms per step
  whether the batch holds 1 request or 20.
- **Draft tokens (k)**: how many tokens are guessed per step. Every config here uses 3.
- **Verify step**: the target (big) model's forward pass over the last real token plus
  the k drafts, for each request. It yields 1 to k + 1 tokens.
- **Acceptance**: a draft token is kept if it equals what the target model would have
  picked (with greedy decoding: its most likely token). Drafts are checked in order and
  checking stops at the first miss.
- **Acceptance rate**: accepted draft tokens / drafted tokens.
- **Accepted per step (mean acceptance length)**: 1 + accepted / drafts, the tokens a
  request gains per verify step, counting the target's own token. vLLM's log line and
  Loom's reports use this definition.
- **TTFT**: time to first token (queueing plus processing the prompt, "prefill").
- **TPOT**: time per output token after the first, per request:
  (end - first token) / (tokens - 1).
- **ITL**: inter-token latency, the gap between streamed chunks. vLLM streams once per
  step, so under speculation ITL is the time of one verify step, not of one token.
- **Goodput at the SLO**: the highest load at which the latency targets still hold (Loom
  uses TTFT p95 under 1 s and TPOT p95 under 50 ms). **$/1M at SLO** prices the pod's
  hour over the tokens it serves at that load.
- **TP=4, all-reduce, P2P**: tensor parallelism splits each layer across 4 GPUs, which
  must then sum their partial results (an all-reduce) twice per layer. P2P is GPUs
  talking directly; without it, that traffic goes through host memory.

## Why it helps at low batch and fades at high batch

A decode step costs roughly a fixed part (read every weight once) plus a part that
grows with the tokens in the step (arithmetic, plus reading each request's cached
attention keys and values). At batch 1 the fixed part is nearly all of it, so verifying
1 + 3 tokens costs about one step and every accepted draft is nearly free.

If each draft token is accepted with probability a (given the earlier ones were), a step
yields on average

    tokens per step = 1 + a + a^2 + ... + a^k

which is 2.53 at a = 0.7 and k = 3, 2.18 at a = 0.6, 1.62 at a = 0.4. The speed-up per
token is that number divided by how much longer a verify step takes than a plain one.

As the batch grows, the per-token part of the step takes over: the GPU is now doing real
arithmetic for every request, and a verify step asks for k + 1 times as much of it, much
of it on drafts that will be thrown away. The ratio "verify step / plain step" climbs from
about 1 toward k + 1, and once it passes the tokens gained per step, speculation is a net
loss. The draft also takes GPU memory (its weights and its own cache), which leaves less
room for the batch. So speculation is a latency tool: it makes each request faster when
the GPU has slack, and it is the wrong tool at the batch sizes that minimise cost per
token. Where exactly the crossover sits depends on the GPU, the model and how
predictable the text is, which is what the study below measures.

## n-gram vs EAGLE3

| | n-gram (prompt lookup) | EAGLE3 |
|---|---|---|
| Where drafts come from | the last few tokens are looked up in the prompt and earlier output; whatever followed the match last time is proposed | a one-layer network trained to predict the target's next tokens from its hidden states (three of its layers) |
| Cost | no weights; a CPU lookup per step | ~2 GB of weights (Qwen3-8B head), up to k small forward passes per step, its own KV cache |
| Drafts every step? | only when a match is found; otherwise the step is a plain one | yes |
| Works on | text that repeats its input: code edits, summaries that quote, RAG answers, templated output | ordinary chat and reasoning, in the distribution it was trained on |
| Fails on | novel text (nothing to copy) | text unlike its training data; tokens outside its draft vocabulary |

vLLM counts a "draft" only when a request proposed at least one token, so n-gram's
accepted-per-step figure leaves out the steps where it found no match. The client-side
"tokens per chunk" (below) includes them.

## What Loom measures

Each load run scrapes vLLM's `/metrics` during the window (`metrics/prometheus.py`). For
speculation, vLLM v0.30.0 exposes four counters, checked against
`vllm/v1/spec_decode/metrics.py` at the v0.30.0 tag (it registers them only when the
engine runs with a speculative config):

| Counter | Meaning |
|---|---|
| `vllm:spec_decode_num_drafts_total` | request-steps that proposed at least one token |
| `vllm:spec_decode_num_draft_tokens_total` | tokens proposed |
| `vllm:spec_decode_num_accepted_tokens_total` | tokens kept |
| `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="i"}` | drafts whose token at position i was kept |

The run summary stores the window's increases and, from them, the acceptance rate,
accepted per step and per-position acceptance (`server.spec_*`). A speculative launch
whose counters are missing lists `spec_decode` in `server.missing`, so a silent scrape
failure shows. Independently of the server, the client records **tokens per chunk** for
every request (completion tokens over streamed chunks): the per-request view the server's
averages hide. A request that no draft helps sits at 1.0, and that is what sets a p95.

`bench report` adds, on any board where a config speculates, a "Speculative decoding by
load" table: per load and config, accepted per step, acceptance by position, tokens per
chunk (mean, lowest request), TPOT p50/p95, ITL p50 and the TPOT p50 speed-up against the
board's config without speculation, with the load at which the speed-up ends.
`leaderboard.csv` gets `spec_mean_accepted_length` and `spec_acceptance_rate` at the
goodput point. The mock backend speculates too (`spec_tokens`, `spec_acceptance`,
`spec_draft_rate`, `spec_overhead_ms`), exposes the same counters and streams one chunk
per step, so all of this is tested offline, including the low-batch win and high-batch
loss.

**What the draft missed.** A variant with `store_output_text: true` keeps each load
request's generated text in its Parquet row (`output_text`; the config hash does not
move). `bench draft-misses <experiment> --cell <eagle3 row> [--workload W] [--out DIR]`
then reads those runs, takes the head (repo, revision) from their speculative config,
fetches its draft vocabulary (the `d2t` tensor, by ranged requests: ~256 KB, not the
2 GB of weights) and the target's tokenizer, and lists per request: tokens per chunk, the
share of output tokens outside the draft vocabulary (and which ones), the share of
distinct tokens and the most common token's share (a loop). One sentence compares the
requests at or below 1.2 tokens per chunk with the rest; `--out` writes
`draft_misses.md/json`. For the pinned Qwen3-8B head, the 32,000-token draft vocabulary
covers a Python snippet completely and misses 80% of the tokens of a Chinese sentence
(checked 2026-11-02), so the vocabulary is a real way for a request to get no draft
accepted.

## The Qwen3-8B study: design (ran as 53e4c107 on AWS; results above)

`bench/experiments/qwen3-8b-specdec-runpod.yaml`: one RunPod Secure 1x L40S pod, vLLM
v0.30.0, Qwen3-8B BF16, non-thinking mode.

| Row | Speculation |
|---|---|
| `bf16` | none: the registry row, reference and gate baseline (same config hash as 7a8237d0's bf16) |
| `bf16-ngram` | n-gram, 3 draft tokens, match windows of 2 to 4 tokens |
| `bf16-eagle3` | EAGLE3, 3 draft tokens, `RedHatAI/Qwen3-8B-speculator.eagle3` |

**The draft head.** Candidates on Hugging Face (2026-11-01):

| Repo | License | Format | Notes |
|---|---|---|---|
| `RedHatAI/Qwen3-8B-speculator.eagle3` (**chosen**) | Apache-2.0 | `speculators` (vLLM loads it natively) | trained on ShareGPT and UltraChat with thinking disabled, as Loom serves Qwen3; card gives acceptance lengths 2.13-2.48 at k = 3; ~29k downloads; same publisher and format as the 70B head 55102ddb ran on vLLM 0.30, and already pinned in `runpod-smoke-h200` |
| `Tengyunw/qwen3_8b_eagle3` | MIT (card) | EAGLE `LlamaForCausalLMEagle3` | SGLang-oriented; no license file beyond the card |
| `AngelSlim/Qwen3-8B_eagle3` | Apache-2.0 (Tencent AngelSlim terms) | EAGLE | no vLLM usage on the card |
| `RedHatAI/Qwen3-8B-Thinking-speculator.eagle3` | Apache-2.0 | `speculators` | trained for thinking mode, which Loom does not serve |

Pinned at commit `08610ffa01dd9f16731fe8f627b85905b6aa51c4` (the repo's head; 2.04 GB).
vLLM v0.30.0's `speculative_config` takes `method: eagle3`, `model`, `revision` and
`num_speculative_tokens`; its `speculators` loader maps this config (`speculators_model_type:
eagle3`, `draft_vocab_size: 32000`). Like every RedHatAI EAGLE3 head, its draft vocabulary is
the 32,000 most common tokens, not Qwen3's 151,936: a token outside it can never be
drafted.

**Workloads and loads.** Fixed open-loop rates, every row at every rate, so speed-ups are
read at equal load: chat-sharegpt (the leading, realistic workload) at 0.25, 1, 2 and
3 req/s (bf16's goodput there was 2.213 req/s in 7a8237d0, so 3 is past its knee), and
real code edits at 0.5, 2 and 4 req/s, the case n-gram is built for. Windows 180 s after
30 s of warmup, 3 repetitions, the usual SLO. The EAGLE3 row stores its output text.

**The copy-heavy workload: CommitPackFT** (`bench/workloads/code-edit-commitpackft.yaml`,
profile kind `edit_dataset`). Each request is a real GitHub commit: the Python file
before it and the commit message, asking for the complete updated file; the file after
the commit sets the output length (`ignore_eos`, as every workload here). An edit keeps
most of the file, so the reply copies long spans of the prompt: in 1,495 sampled edits
the new file shares a median 87% of its characters with the old one (mean 65%). Prompts
average ~250 tokens and replies ~225 (Qwen3 tokenizer; capped at 512 in the specs so the
30 s drain covers a passing request).

| Candidate | License | Why (not) |
|---|---|---|
| `bigcode/commitpackft`, `data/python/data.jsonl` (**chosen**) | MIT (card); each record keeps its repository's license | 56,025 real commits with before/after files and an instruction-like message; plain JSON lines (136 MB) the pod downloads and checks itself; large enough that no prompt repeats within a run |
| `nuprl/CanItEdit` | MIT | real before/after edits, but only ~105 problems: prompts would repeat across a run and hit the prefix cache |
| `likaixin/InstructCoder` | none declared on the card | code-edit instructions, but no license to rely on |
| SQuAD-style extractive QA | CC BY-SA 4.0 | answers are a few tokens: nothing for decode-side speculation to win |

Pinned at revision `fc56fe33c030c6daa414c2b112c932b8eed085e6` (the repo's head on
2026-11-02), file sha256 `d167da37e1058371c48e057cd8815d03700c867dd8bcf58e61420d4dcd288d73` (its Git LFS oid,
checked on download), 135,858,935 bytes. The profile keeps only MIT, Apache-2.0,
BSD-2/3-Clause, ISC, Unlicense and CC0 records (50,152 of 56,025), skips edits that
change nothing, and uses the code only as request load (lengths and latency are
published, never the code). It replaces the templated `code-completion` stand-in in both
specs; its lighter prompts move its rates up (0.5, 2, 4 instead of 0.25, 1, 2 req/s).

**Quality.** Both speculative rows are gated in-run against bf16 on phase0-strict with
three replicated passes, divergence included, under the standard rule: a row ships only
on a gate pass, the technique stays in its name, BF16 stays the reference. Speculation is
lossless in principle (with greedy decoding a draft is kept only if it is the target's own
choice), but the verify pass runs the target on 1 + k tokens per request, a different
kernel shape, so a near-tie can flip as it does under batching noise. The 70B EAGLE3 run
measured exactly that: KL 0.00022 against BF16, below either config's noise floor, with
11 of 48 continuations flipping at near-ties (benchmark-lab.md, "70B tuning").

**Quote** (`bench plan` against a copy of the results DB, 2026-11-02; $96.57 billable
spent, $153.18 left under the $249.75 overall cap):

| Spec | Plan | Estimate | Worst case (TTL) | Cap |
|---|---|---|---|---|
| `runpod-smoke-specdec` | 1.49 h of a 100 min TTL | $1.64 | $1.83 | $2.00 |
| `qwen3-8b-specdec-runpod` | 7.39 h of an 8 h TTL (RunPod's limit; 7.6% margin) | $8.14 | $8.81 | $9.00 |

The real code edits add 4 minutes of plan to the templated code they replace (7.32 h
before): the 136 MB dataset's download and a 10 s allowance per run for reading it
(measured: 1.0 s for 500 requests on a laptop). Past RunPod runs cost 30-80% of plan
(eval passes plan ~1.6x long), so expect about $5-6. The planner does not count the draft
head's 2 GB download (~15 s). For a wider margin: dropping chat's 3 req/s point saves
~41 min of plan, but it is the only point past bf16's knee; or the EAGLE3 row can go on
its own pod (`host_group`), one more cold start (~12 min, ~$0.25) for two pods well
inside the TTL.

**On AWS instead** (2026-11-02, proposed, not approved). RunPod Secure 1x L40S was out of
stock all day, so `aws-smoke-specdec` and `qwen3-8b-specdec-aws` run the same study on one
AWS g6e.xlarge on-demand host (the same L40S; 4 vCPU, 32 GiB; the account's 4-vCPU quota
fits exactly one). Same rows, workloads, loads, windows and quality section; only the
provider block and budget differ (`bench/tests/aws/test_aws_perf_track.py`). The host is
$1.8829/h accrued against the pod's $1.10, and EC2 starts are slower, so:

| Spec | Plan | Estimate | Worst case (TTL) | Cap |
|---|---|---|---|---|
| `aws-smoke-specdec` | 1.56 h of a 115 min TTL | $2.94 | $3.61 | $3.75 |
| `qwen3-8b-specdec-aws` | 7.46 h of an 8 h TTL (the aws_ec2 limit; 6.7% margin) | $14.04 | $15.06 | $15.25 |

The AWS runs 95cde129 and fccf9ed3 came in at 72-74% of plan, so expect about $10-11.
The host fetches the EAGLE3 head into its instance-store HF cache on the warm restart and
vLLM loads it offline; the CommitPackFT file lands under `/opt/dlami/nvme/loom-data` and
the client container reads it read-only (both run end to end on the host simulator). The
same margin options apply: the EAGLE3 row on its own host (`host_group`) would add one
cold start (~20 min, ~$0.60), and teardown now waits for each instance to terminate, so
the quota's one host is free before the next.

**The smoke** (`runpod-smoke-specdec`) has the same provider, variants (so the same config
hashes and launches), workload profiles and overrides, repetitions, SLO and quality
section, with 10 s windows and tasks capped at 4 items. It runs once what no earlier run
has: vLLM's n-gram proposer, EAGLE3 on Qwen3-8B on an L40S with its head fetched on a warm
restart, the spec-decode counters in the run summary (check `server.missing` has no
`spec_decode` and the two speculative rows show accepted per step), replicated passes and
divergence scoring against a speculating engine, the CommitPackFT download and edit
prompts on a pod, and stored output text on the EAGLE3 row (check its Parquet rows have
`output_text`, then run `bench draft-misses <smoke id> --cell bf16-eagle3`). Chat at
12 req/s with full-length replies overloads every row, so each speculating engine has
requests cancelled at the drain timeout, as the run's 3 req/s point may.

What the smoke cannot show: the crossover itself and steady-state acceptance (10 s
windows, a handful of requests per point); code edits at 0.5 and 4 req/s (it runs 2);
whether 7.39 h fits the 8 h TTL on the day (measured runs came in at 60-75% of plan); and
a draft-miss finding, which needs the run's hundreds of requests.
`bench/tests/runpod/test_runpod_specdec.py` runs it end to end on the pod simulator,
with an engine that speculates when its launch says so, including the stored text and
`bench draft-misses` (with the head's vocabulary and the tokenizer faked).

**What it should answer**: the TPOT speed-up of each method at each load (and where it
falls below 1); acceptance by method and workload; whether speculation changes goodput
and $/1M at the declared 50 ms SLO, and at a tight interactive one (20 ms TPOT p95),
where a latency tool should be worth more; and what the EAGLE3 draft fails to predict.

**After the run** (no GPU; the 20 ms view was approved on 2026-11-02):

```sh
ID=<experiment id>; OUT=reports/8b-specdec-v1
uv run bench report -e $ID --alt-slo tpot_ms.p95=20 --out $OUT
uv run bench draft-misses $ID --cell bf16-eagle3 --workload chat-sharegpt --out $OUT/draft-misses/chat
uv run bench draft-misses $ID --cell bf16-eagle3 --workload code-edit --out $OUT/draft-misses/code-edit
```

The first writes the declared-SLO leaderboard (with the "Speculative decoding by load"
tables) and next to it `leaderboard.alt-slo.*`: the same runs re-judged at TPOT p95
20 ms, labelled as not the declared SLO (benchmark-lab.md, "Alternative SLO view").
The other two write the draft-miss tables per workload.

## Why EAGLE3 gave no gain on Llama 3.3 70B over 4x L40S

The claim to check: "EAGLE3 on the 70B over 4x L40S (55102ddb, `vllm-tp4-eagle3`) gave
about no gain." Compared with the same model without it (cf4d1614, `vllm-tp4`), same
pod shape, image, workloads, seeded requests and windows (both runs and their numbers:
[benchmark-lab.md, "70B tuning"](../benchmark-lab.md#70b-tuning); `bench report
-e cf4d1614 -e 55102ddb` rebuilds their leaderboard from the results store):

**It did speed up the typical request, about 2x.** From the stored per-request rows,
EAGLE3 delivered 2.31-2.38 tokens per streamed chunk on average at every load, against
1.00-1.01 without it. Median TPOT:

| Workload | Load (req/s) | Step time, no spec / EAGLE3 (ms) | Tokens per step, EAGLE3 | Median TPOT, no spec / EAGLE3 (ms) |
|---|---|---|---|---|
| fixed-1k-1k | 0.0625 | 61.0 / 73.8 | 2.36 | 60.2 / 31.2 |
| fixed-1k-1k | 0.125 | 66.0 / 78.5 | 2.38 | 65.4 / 32.8 |
| fixed-1k-1k | 0.25 | 79.2 / 98.4 | 2.36 | 78.5 / 42.2 |
| shared-prefix | 0.125 | 60.1 / 70.1 | 2.34 | 60.1 / 29.0 |
| shared-prefix | 0.25 | 60.4 / 70.1 | 2.31 | 60.4 / 30.2 |
| shared-prefix | 0.5 | 73.8 / 86.1 | 2.33 | 73.8 / 36.9 |

(Step time is the median gap between streamed chunks; per-request medians over the three
repetitions. The leaderboard's TPOT p50 rows say the same: 31.0 vs 61.3 ms at
0.0625 req/s, CIs disjoint.)

**It gained nothing on what Loom prices: goodput at the SLO was zero both ways**, so
neither config has a $/1M at the SLO, and the open-loop throughput is the offered load
either way. Each workload failed for its own reason:

1. **fixed-1k-1k fails TPOT p95, because a verify step costs more than a plain one and
   some requests accept nothing.** On fixed-1k-1k a verify step took 12.5-19.2 ms (19-24%)
   longer than a plain step (73.8 vs 61.0 ms at the lowest load). A request whose drafts
   are all rejected therefore runs *slower* than without speculation: one token per
   74-105 ms instead of 60-79. And 15 of the 213 measured fixed-1k-1k requests (7.0%)
   accepted **no draft at all**: exactly 1.00 tokens per chunk over all 1,023 decode
   steps (at every load, in every repetition). Seven percent is more than the 5% a p95
   can hide: per repetition, TPOT p95 ranged from 36.7 to 84.2 ms while the median sat
   near 31-43 ms, and the SLO verdict takes the upper end of the p95's confidence
   interval across repetitions, which was above 50 ms at every load. At 0.25 req/s
   the verify step grows to 98 ms, so a request needs more than 2 tokens per step to
   meet 50 ms, and 26 of 120 missed.
2. **shared-prefix fails TTFT p95, which speculation does not touch.** Its TPOT passed
   (p95 34-43 ms at 0.125 and 0.25 req/s; no request over 50 ms). But time to first
   token sat at ~1.0 s at every load on both configs: prefill on 4 L40S without P2P sends
   every layer's all-reduce (32 MB for a 2k-token prompt) through host memory, measured at
   2.76 ms each, 441 ms per 2k-token prompt (benchmark-lab.md, "The measured floors").
   Speculation only changes decoding.

**The decode floor, and a correction.** A plain decode step here takes ~60 ms at low
load: at least 41 ms is reading each GPU's 35 GB weight shard at 864 GB/s, and about
4.4 ms is the 160 small all-reduces through host memory (27.6 µs each, measured). So the
host-memory all-reduce dominates *prefill* (TTFT), not decode; decode is dominated by
weight reads, which is exactly the regime where speculation should help, and it did for
the median. What it cannot do is lower the floor for a request the draft cannot predict:
that request pays the full verify step per token. EAGLE3 also cut the KV cache from 71k
to 41k tokens (the draft's weights and cache).

**Why exactly zero acceptance on 7% of requests: not settled.** Random content alone
would still accept the odd draft, so "exactly 1.00 over 1,023 steps" points to something
systematic. The leading hypothesis is the draft vocabulary: the head can only propose the
32,000 most common Llama 3 tokens (of 128,256), and a greedy continuation of a
random-word prompt that drifts into rare tokens (or a loop of one) can never be drafted.
Those load runs did not store output text, so this is unconfirmed for the 70B. On
Qwen3-8B (53e4c107, stored text) the same kind of head's low-acceptance requests were
indeed the ones whose output left its draft vocabulary (median 69% of tokens outside,
against 3.3% for the rest; section "What we measured on Qwen3-8B"), which makes the
hypothesis likely for the 70B too, not proven. Not known either: how the extra 10-19 ms of a
verify step (16-24%, both workloads) splits between the head's three sequential forward passes (each with its own
all-reduces at TP=4) and the 4x larger verify batch; it was not profiled.

**In one sentence:** EAGLE3 halved the median time per token, but on this pod a request
the draft cannot predict gets slower, 7% of the synthetic requests were such requests,
and the other workload was limited by prefill communication that speculation cannot
reach, so the SLO-based goodput, and with it the cost, did not move.

## Open questions

- Settling the two IFEval gates (53e4c107): more IFEval-like items or a second host, not
  more passes; until then neither speculative row ranks.
- The load where speculation stops paying on an L40S: not reached at chat 3 req/s or
  code edits 4 req/s (the speculating rows never ran more than ~19 requests at once).
- Why n-gram lowers TTFT on code edits (measured, not explained).
- CommitPackFT covers one kind of copy-heavy text (code edits). RAG over pasted passages
  with long extractive answers would be the other; no permissive dataset with long enough
  answers was chosen here.
- Settling the 70B's zero-acceptance requests needs a 70B EAGLE3 load run with
  `store_output_text`; the Qwen3-8B run answers the question for that model only.
- SGLang's speculative-decoding metrics are not mapped (unverified at v0.5.21); only vLLM
  rows get acceptance figures.
