# How to add a workload profile

A workload profile is the shape of the requests: lengths, endpoint, prompt content and
its source. Adding one of the existing kinds is a configuration change: one YAML file in
`bench/workloads/`, validated by `loom_bench.workloads.profiles`. A new *kind* (a new way
to build prompts) is code; see the end of this page.

Load (rate or concurrency, arrivals, duration, warmup) is not part of the profile; it is
set per experiment.

## 1. Pick a kind

| `kind` | Endpoint | Fields (besides the common ones) |
|---|---|---|
| `synthetic` | completions or chat | `input_len`, `output_len`, `range_ratio` (lengths uniform in [len·(1−r), len·(1+r)], 0 ≤ r < 1), `ignore_eos` |
| `shared_prefix` | chat | `input_len`, `output_len`, `range_ratio`, `prefix_share` (0-0.95 of the prompt shared), `num_prefix_groups`, `ignore_eos` |
| `chat_dataset` | chat | `path` (ShareGPT format), `max_input_len`, `min_output_len`, `max_output_len`, `ignore_eos` |
| `long_context_needle` | completions or chat | `context_len` (≥ 128), `depths` (0-1), `output_len` |
| `code_completion` | completions | `input_len`, `output_len`, `range_ratio`, `ignore_eos` |
| `long_generation` | completions or chat | `input_len` (≥ 32), `output_len`, `range_ratio`, `ignore_eos` |
| `trace` | completions or chat | `path`, `format` (e.g. `azure`), `max_rows`, `max_input_len`, `max_output_len`, `ignore_eos` |

Common fields: `name`, `description`, `content` (`synthetic` or `realistic`), `dataset`
(provenance, below), `seed` (default 0), `temperature` (default 0). Unknown keys are
errors.

`content` is published with every result: random text understates prefix caching and
speculative decoding, so say `synthetic` unless the prompts are real text.

## 2. Write the file

`bench/workloads/<name>.yaml`, with `name` equal to the file name. Example:

```yaml
name: fixed-2k-256
description: >-
  RAG-style 2k/256 shape. Random-word prompts with ignore_eos, so output length is
  exact and comparable across engines.
kind: synthetic
content: synthetic
endpoint: completions
input_len: 2048
output_len: 256
range_ratio: 0.0
ignore_eos: true
seed: 0
```

Keep `input_len + output_len` within the smallest `max_context` you will run it against
(see `fixed-32k-1k.yaml`, which leaves 32 tokens of slack for special tokens).

**Datasets.** Profiles that read files (`chat_dataset`, `trace`) never ship the data.
Use `${LOOM_DATA_DIR}/...` in `path` (`~` and environment variables are expanded) and
record where it comes from and its license:

```yaml
dataset:
  name: <dataset and file>
  source: <URL>
  revision: null        # pin the commit you downloaded before publishing results
  license: <license, or what must be verified>
  license_url: <URL>
```

Check the license allows benchmarking and publishing derived numbers before any result
is published.

## 3. Rules that reject mistakes

- `range_ratio` must be in [0, 1); `prefix_share` at most 0.95, with room left for a
  unique suffix after the prefix.
- `shared_prefix` and `chat_dataset` only use the chat endpoint, `code_completion` only
  completions.
- Experiment `overrides` are deep-merged into the profile and re-validated, so an
  override cannot produce an invalid profile either.
- `bench plan` resolves every workload of an experiment and fails on a missing or
  invalid profile.

## 4. Update the pinned test

`bench/tests/workloads/test_profiles.py` lists every shipped profile in `SHIPPED`
and asserts the directory matches it. Add the new name.

## 5. Prove it

```bash
# Validates; shows the resolved fields
uv run python -c "from loom_bench.workloads import load_profile; print(load_profile('fixed-2k-256'))"

# The requests it generates (simple tokenizer, no download)
uv run python -c "
from loom_bench.workloads import load_profile, build_requests
from loom_bench.tokenize import SimpleTokenizer
for r in build_requests(load_profile('fixed-2k-256'), SimpleTokenizer(), 2):
    print(r.request_id, r.endpoint, r.expected_prompt_tokens, r.max_tokens, sorted(r.payload))"

uv run pytest -q bench/tests/workloads
```

Then use it in an experiment on the mock backend: copy
`bench/experiments/mock-smoke.yaml`, replace a `profile:` with the new name (drop the
`overrides` or adapt them), and run `uv run bench run <file>`.

## Using it in an experiment

```yaml
workloads:
  - profile: fixed-2k-256          # a name in bench/workloads/, or a YAML path
    label: rag-2k                  # optional; defaults to the profile name; unique per experiment
    overrides: {output_len: 128}   # optional
    load: {mode: open_loop, values: [1, 2, 4], duration_s: 180, warmup_s: 30}
```

The workload name stored with each run is the label (or profile name); provenance also
records a hash of the resolved profile and its `content` kind.

Limits to know:

- Open-loop experiments sweep one rate, so their `arrival.kind` must be `poisson`,
  `gamma`, `constant`, `diurnal` or `trace`. `ramp` and `onoff_burst` arrivals have no
  single rate and cannot be selected from an experiment.
- A `trace` profile replays its token lengths; pair it with `trace` arrivals on the same
  file to replay its arrival times too:

  ```yaml
  - profile: trace-azure-code
    load:
      mode: open_loop
      values: [2, 4, 8]       # mean req/s over the run
      duration_s: 300
      arrival: {kind: trace, path: "${LOOM_DATA_DIR}/azure/AzureLLMInferenceTrace_code.csv", format: azure}
  ```

  At load value r the run replays the trace's first r x `duration_s` arrivals,
  stretched or compressed in time so their mean rate is r and their gaps keep their
  proportions (`arrivals.trace_offsets_at_rate`); request i gets trace row i's lengths.
  `time_scale` and `rate` come from the load value, so the arrival must not set them.
  The trace needs more rows than the highest r x `duration_s`.
- The wrapped tools (`vllm_bench`, `sglang_bench`) support only some kinds; see
  [load-generators.md](../load-generators.md#what-each-wrapper-supports).

## A new kind is code

A new kind needs a profile class in `workloads/profiles.py` (added to the
`WorkloadProfile` union), a request builder registered in `_BUILDERS` in
`workloads/generate.py`, a look at `plan.profile_shape` (the planner's length
assumption), the wrapper mappings in `loadgen/vllm_bench.py` and `loadgen/sglang_bench.py`
(or an `UnsupportedByTool` for it), and tests in `bench/tests/workloads/`.
