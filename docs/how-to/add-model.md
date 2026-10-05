# How to add a model

Adding a model is a configuration change: one entry in `config/models.yaml`, validated by
`loom_bench.registry` on every load. No code or test changes, as long as the engine
(vLLM or SGLang) and the GPU type are already supported.

## 1. Collect the facts

| Field | Where it comes from |
|---|---|
| `hf.repo` | Hugging Face repo, `org/name` |
| `hf.revision` | The commit sha you reviewed (40 hex characters). Branches and tags are rejected |
| `hf.size_bytes` | Sum of the `*.safetensors` files at that revision; the planner uses it for download and load time |
| `hf.license`, `hf.gated` | Model card. `gated: manual` (license accepted per account, like Llama), `auto`, or `false` |
| `hf.quant_method` | `quantization_config.quant_method` in the checkpoint's `config.json`; `null` for BF16/FP16 weights |
| `hf.trust_remote_code` | `false` unless the model needs custom code (see step 3) |
| `engine.image` | Engine image pinned by digest, `repo@sha256:<64 hex>`. Tags are rejected |
| `max_context` | The native context you will serve (no rope scaling unless you enable it) |

Revision, size and quantization in one call (`huggingface_hub` is already a dependency;
gated repos need `HF_TOKEN` in the environment):

```bash
uv run python -c "
from huggingface_hub import model_info
i = model_info('Qwen/Qwen3-8B', revision='main', files_metadata=True)
print(i.sha)
print(sum(s.size for s in i.siblings if s.rfilename.endswith('.safetensors')))
print((i.config or {}).get('quantization_config'))"
```

Image digest (multi-arch index digest, as used for the existing entries):

```bash
docker buildx imagetools inspect vllm/vllm-openai:v0.30.0 --format '{{json .Manifest.Digest}}'
```

## 2. Add the entry

Append to `models:` in `config/models.yaml`. Every field below is required unless it has
a default in `registry.py`:

```yaml
  - id: my-model                    # lowercase, digits, '.', '-'; the public API id
    display_name: My Model
    hf:
      repo: org/My-Model
      revision: <40-char commit sha>
      license: apache-2.0
      gated: false
      size_bytes: <bytes of *.safetensors>
      quant_method: null
      trust_remote_code: false
    engine:
      name: vllm                    # vllm | sglang
      version: "0.30.0"
      image: vllm/vllm-openai@sha256:<64 hex>
      args: {}                      # extra engine flags, e.g. {max_num_seqs: 256}
      chat_template_kwargs: {}      # sent with chat requests in benchmarks and evals
    hardware:
      gpu: L40S                     # must match the instance's GPU in bench/prices.yaml
      gpus_per_replica: 1
      nodes_per_replica: 1
      instance_types: {aws: g6e.xlarge}
    parallelism: {tp: 1, pp: 1, ep: 1}
    max_context: 32768
    quantization: none
    kv_cache_dtype: auto
    pricing: null                   # micros per 1M tokens; required once status is enabled
    scaling: {min_replicas: 0, max_replicas: 1}
    clouds: [aws]
    capabilities: {tools: true, json_schema: true, vision: false, reasoning: false}
    routing_tier: 1
    status: preview                 # enabled | preview | disabled
```

## 3. Rules that reject mistakes

Load-time validation (`registry.py`, tested in `bench/tests/config/test_registry.py`):

| Rule | Error mentions |
|---|---|
| `hf.revision` is a 40-hex commit sha | `revision` |
| `engine.image` is `repo@sha256:<64 hex>` (no tag) | `image` |
| `engine.name` is `vllm` or `sglang`; `engine.version` is `x.y.z` | `name`, `version` |
| `gpus_per_replica × nodes_per_replica == tp × pp`; `ep` divides `tp × pp` | `must equal tp * pp`, `ep` |
| `nodes_per_replica` is 1 (multi-node is not built) | `reserved` |
| `quantization` is servable from `hf.quant_method` (table in [PLAN.md](../PLAN.md#model-registry-schema-configmodelsyaml)) | `cannot be served from` |
| `trust_remote_code: true` needs `trust_remote_code_review: {reviewer, date, notes}` | `trust_remote_code_review` |
| every cloud in `clouds` has an `instance_types` entry | `instance_types missing` |
| `status: enabled` needs `pricing`; prices are integers; `cached_input_per_mtok ≤ input_per_mtok` | `requires pricing`, `cached_input` |
| unique `id`s, no unknown keys, no duplicate YAML keys | `duplicate`, the key name |

Rendering (`engines.render_launch`, `bench/tests/aws/test_engines.py`): `engine.args` may
not set flags that come from registry fields (model, revision, tokenizer, served model
name, tp/pp, max length, quantization, KV-cache dtype, host, port, download dir,
trust-remote-code).

Planning (`bench plan`): the instance type must have a price in `bench/prices.yaml`, its
GPU must equal `hardware.gpu` and its GPU count must cover `gpus_per_replica`; otherwise
the plan is refused.

**`trust_remote_code`.** It lets the checkpoint run arbitrary Python inside the engine.
Before setting it, read the repo's `*.py` files at the pinned revision, then record the
review:

```yaml
      trust_remote_code: true
      trust_remote_code_review:
        reviewer: <name>
        date: 2026-10-05
        notes: Reviewed modeling_x.py and configuration_x.py at <sha>; no network or file access.
```

A new revision needs a new review.

## 4. What the tests check for you

No test lists the shipped models. The suite checks properties of whatever
`config/models.yaml` holds, so a new entry is covered as soon as it is added:

- `test_registry.py::test_every_shipped_model_loads_and_can_be_launched`: it loads, and
  `render_launch` builds its engine command (repo, revision, image, GPUs);
- `test_prices.py::test_every_registry_instance_type_has_a_price`: its instance type has
  a verified AWS price with the right GPU and count;
- the site tests expect one page per registry model, so re-export the committed snapshot
  (step 6).

## 5. Prove it

```bash
# Loads and validates
uv run python -c "from loom_bench.registry import load_registry; print([m.id for m in load_registry().models])"

# The exact engine command line
uv run python -c "
from loom_bench.registry import load_registry
from loom_bench.engines import render_launch
print(' '.join(render_launch(load_registry().get('my-model')).args))"

# Config, engine and suite tests
uv run pytest -q bench/tests/config bench/tests/aws/test_engines.py bench/tests/quality/test_suite.py
```

Then run it end to end on the mock backend and plan it on AWS. Save as
`/tmp/my-model-mock.yaml`:

```yaml
name: my-model-mock
description: Plumbing check for my-model on the mock backend.
model: my-model
provider: {kind: mock, hourly_price: "$1.00", time_scale: 0.01}
variants:
  - name: default
workloads:
  - profile: fixed-128-128
    overrides: {input_len: 64, output_len: 32}
    load: {mode: closed_loop, values: [1, 4], num_requests: 12, warmup_requests: 2}
repetitions: 2
slo: {ttft_ms: {p95: 1000}, tpot_ms: {p95: 50}, max_error_rate: 0.01}
budget: {max_spend: "$1", ttl_minutes: 10}
```

```bash
uv run bench run /tmp/my-model-mock.yaml --out /tmp/loom-results
```

For the real run, copy `bench/experiments/qwen3-8b-vllm-vs-sglang.yaml`, change `model`
and the workloads, and check `uv run bench plan <file>` is accepted (exit 0) before
following [runbook.md](../runbook.md). While iterating, `LOOM_MODELS_YAML=<file>` points
every command at a scratch copy of the registry.

## 6. Afterwards

- **Quality suite.** The gate needs a pinned suite for the model:
  `bench/evals/<model id>.yaml` (copy `bench/evals/qwen3-8b.yaml`). `test_suite.py`
  requires the `lm_eval`, `needle`, `code_exec`, `tool_calling` and `json_schema` task
  kinds, a `divergence` section, context lengths within `max_context`, and
  `chat_template_kwargs` equal to the registry's.
- **Competitor prices.** Add public list prices for the model to `bench/competitors.yaml`
  (`model_id` is the registry id) if you want the competitiveness view to compare it.
- **Site.** The committed snapshot lists models; re-export it (`bench site export`) so the
  new model gets its "first runs pending" page.
- **Gated models.** The HF account behind `loom/hf-token` must have accepted the license.
