# Speculative decoding: what it is, when it pays, and what Loom measured

Status, 2026-11-01: the Qwen3-8B study is **proposed, not run** (specs, quote and smoke
below). The 70B findings are from runs already stored (cf4d1614, 55102ddb).

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

## The Qwen3-8B study (proposed, not run)

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
code-completion with 256-token continuations at 0.25, 1 and 2 req/s, the case where
n-gram should shine: its prompts are 1.5k tokens of templated Python (ten line templates,
sixteen identifiers), so a continuation repeats spans of the prompt. **No profile in the
repo has real edits or quoted text** (a code-edit or RAG workload with a real dataset is
the missing piece); this templated code is the closest, and it flatters n-gram. Windows
180 s after 30 s of warmup, 3 repetitions, the usual SLO.

**Quality.** Both speculative rows are gated in-run against bf16 on phase0-strict with
three replicated passes, divergence included, under the standard rule: a row ships only
on a gate pass, the technique stays in its name, BF16 stays the reference. Speculation is
lossless in principle (with greedy decoding a draft is kept only if it is the target's own
choice), but the verify pass runs the target on 1 + k tokens per request, a different
kernel shape, so a near-tie can flip as it does under batching noise. The 70B EAGLE3 run
measured exactly that: KL 0.00022 against BF16, below either config's noise floor, with
11 of 48 continuations flipping at near-ties (benchmark-lab.md, "70B tuning").

**Quote** (`bench plan` against a copy of the results DB, 2026-11-01; $96.57 billable
spent, $53.18 left under the overall cap):

| Spec | Plan | Estimate | Worst case (TTL) | Cap |
|---|---|---|---|---|
| `runpod-smoke-specdec` | 1.47 h of a 100 min TTL | $1.62 | $1.83 | $2.00 |
| `qwen3-8b-specdec-runpod` | 7.32 h of an 8 h TTL (RunPod's limit) | $8.06 | $8.81 | $9.00 |

Past RunPod runs cost 30-80% of plan (eval passes plan ~1.6x long), so expect about
$5-6 for the run. The planner does not count the draft head's 2 GB download (~15 s). The
caps together ($11) fit what is left even if all of option B ($40.70 of caps) runs first.

**The smoke** (`runpod-smoke-specdec`) has the same provider, variants (so the same config
hashes and launches), workload profiles and overrides, repetitions, SLO and quality
section, with 10 s windows and tasks capped at 4 items. It runs once what no earlier run
has: vLLM's n-gram proposer, EAGLE3 on Qwen3-8B on an L40S with its head fetched on a warm
restart, the spec-decode counters in the run summary (check `server.missing` has no
`spec_decode` and the two speculative rows show accepted per step), replicated passes and
divergence scoring against a speculating engine, and 256-token code continuations on a
pod. Chat at 12 req/s with full-length replies overloads every row, so each speculating
engine has requests cancelled at the drain timeout, as the run's 3 req/s point may. What
it cannot show: the crossover itself (10 s windows), or whether the 8 h TTL is enough
(the run is planned at 7.32 h; measured runs have come in at 60-75% of plan).
`bench/tests/runpod/test_runpod_specdec.py` runs it end to end on the pod simulator,
with an engine that speculates when its launch says so.

**What it should answer**: the TPOT speed-up of each method at each load (and where it
falls below 1); acceptance by method and workload; whether speculation changes goodput
and $/1M at the declared 50 ms SLO; and `bench report --alt-slo tpot_ms.p95=20` (or
similar, re-judged for free from the same runs) for a tight interactive SLO, where a
latency tool should be worth more.

## Why EAGLE3 gave no gain on Llama 3.3 70B over 4x L40S

The claim to check: "EAGLE3 on the 70B over 4x L40S (55102ddb, `vllm-tp4-eagle3`) gave
about no gain." Compared with the same model without it (cf4d1614, `vllm-tp4`), same
pod shape, image, workloads, seeded requests and windows
([reports/70b-final-v2](../../reports/70b-final-v2/leaderboard.md)):

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
Load runs do not store output text, so this is unconfirmed; the next EAGLE3 run's
tokens-per-chunk minimum will show whether chat content does it too, and storing outputs
for one EAGLE3 load run would settle it. Not known either: how the extra 10-19 ms of a
verify step (16-24%, both workloads) splits between the head's three sequential forward passes (each with its own
all-reduces at TP=4) and the 4x larger verify batch; it was not profiled.

**In one sentence:** EAGLE3 halved the median time per token, but on this pod a request
the draft cannot predict gets slower, 7% of the synthetic requests were such requests,
and the other workload was limited by prefill communication that speculation cannot
reach, so the SLO-based goodput, and with it the cost, did not move.

## Open questions

- A real code-edit or RAG workload (outputs that quote the prompt) would be the fair test
  for n-gram; the templated code profile is a stand-in.
- Whether to store output text on one EAGLE3 load run to settle the zero-acceptance
  requests (needs a small runner option; `keep_output` exists on the load job but is not
  exposed in the experiment spec).
- SGLang's speculative-decoding metrics are not mapped (unverified at v0.5.21); only vLLM
  rows get acceptance figures.
