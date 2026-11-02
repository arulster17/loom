"""The Qwen3-8B speculative decoding study (proposed 2026-11-01, not approved): no
speculation vs n-gram vs EAGLE3 on one RunPod Secure 1x L40S pod, every speculative row
gated against BF16 in-run, and its smoke."""

import json

import pytest

from loom_bench.budget import caps_for, load_budget
from loom_bench.engines import draft_weights
from loom_bench.experiment import EXPERIMENTS_DIR, expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.plan import build_plan
from loom_bench.prices import load_prices
from loom_bench.registry import load_registry

from .test_plan import _eval_paths

REGISTRY = load_registry()
PRICES = load_prices()
BUDGET = load_budget()
REAL = EXPERIMENTS_DIR / "qwen3-8b-specdec-runpod.yaml"
SMOKE = EXPERIMENTS_DIR / "runpod-smoke-specdec.yaml"
SWEEP = EXPERIMENTS_DIR / "qwen3-8b-config-sweep-runpod.yaml"
H200_SMOKE = EXPERIMENTS_DIR / "runpod-smoke-h200.yaml"
EAGLE3 = ("RedHatAI/Qwen3-8B-speculator.eagle3", "08610ffa01dd9f16731fe8f627b85905b6aa51c4")
NAMES = ["bf16", "bf16-ngram", "bf16-eagle3"]


def _plan(path):
    exp = load_experiment(path)
    return build_plan(
        exp, expand(exp, REGISTRY), prices=PRICES, caps=caps_for(exp.budget.max_spend, BUDGET, 0)
    )


def _spec_config(cell) -> dict | None:
    args = cell.launch.args
    if "--speculative-config" not in args:
        return None
    return json.loads(args[args.index("--speculative-config") + 1])


@pytest.mark.parametrize("path", [REAL, SMOKE], ids=lambda p: p.stem)
def test_the_specs_are_marked_proposed_not_approved(path):
    text = path.read_text()
    assert text.startswith("# PROPOSED 2026-11-01, NOT APPROVED")
    assert "APPROVED 20" not in text.split("\nname:")[0].replace("NOT APPROVED", "")


def test_three_rows_on_one_pod_differing_only_by_the_speculative_config():
    cells = expand(load_experiment(REAL), REGISTRY)
    assert [c.variant for c in cells] == NAMES
    assert len({c.host_key for c in cells}) == 1  # warm restarts onto n-gram, then EAGLE3
    bf16, ngram, eagle = cells
    for c in (ngram, eagle):
        i = c.launch.args.index("--speculative-config")
        assert c.launch.args[:i] + c.launch.args[i + 2 :] == bf16.launch.args
        assert c.spec.hf == bf16.spec.hf and c.launch.image == bf16.launch.image
    assert _spec_config(bf16) is None
    assert _spec_config(ngram) == {
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    }
    assert _spec_config(eagle) == {
        "method": "eagle3",
        "model": EAGLE3[0],
        "revision": EAGLE3[1],
        "num_speculative_tokens": 3,
    }
    # The same draft length on both methods; only EAGLE3 loads draft weights.
    assert (
        _spec_config(ngram)["num_speculative_tokens"]
        == _spec_config(eagle)["num_speculative_tokens"]
    )
    assert draft_weights("vllm", ngram.launch.args) == []
    assert draft_weights("vllm", eagle.launch.args) == [EAGLE3]
    # The technique is in the row name; the reference row is the registry's BF16 row.
    assert len({c.config_hash for c in cells}) == 3


def test_the_reference_is_the_registry_bf16_row_as_7a8237d0_ran_it():
    (bf16, *_) = expand(load_experiment(REAL), REGISTRY)
    sweep_bf16 = expand(load_experiment(SWEEP), REGISTRY)[0]
    assert sweep_bf16.variant == "bf16" and bf16.config_hash == sweep_bf16.config_hash
    # Qwen3 runs non-thinking, as the EAGLE3 head was trained.
    assert bf16.spec.engine.chat_template_kwargs == {"enable_thinking": False}


def test_the_eagle3_head_is_the_one_runpod_smoke_h200_pins():
    h200 = expand(load_experiment(H200_SMOKE), REGISTRY)[1]
    assert draft_weights("vllm", h200.launch.args) == [EAGLE3]


def test_every_speculative_row_is_gated_against_bf16_in_run_on_three_passes():
    q = load_experiment(REAL).quality
    assert q is not None and q.baseline_variant == "bf16" and q.baseline is None
    assert (q.suite, q.subset, q.replicates, q.limit) == ("qwen3-8b", "phase0-strict", 3, None)


def test_fixed_loads_on_chat_and_templated_code():
    exp = load_experiment(REAL)
    chat, code = exp.workloads
    assert chat.profile == "chat-sharegpt" and chat.load.search is None
    assert chat.load.values == [0.25, 1, 2, 3]  # bf16's 7a8237d0 knee is 2.213 req/s
    assert code.profile == "code-completion" and code.label == "code-completion-256"
    assert code.overrides == {"output_len": 256} and code.load.values == [0.25, 1, 2]
    for w in (chat, code):
        assert w.load.mode.value == "open_loop"
        assert (w.load.duration_s, w.load.warmup_s) == (180, 30)
    assert exp.repetitions == 3
    assert exp.slo.model_dump(exclude_none=True) == {
        "ttft_ms": {"p95": 1000.0},
        "tpot_ms": {"p95": 50.0},
        "max_error_rate": 0.01,
    }


def test_the_quote_fits_the_cap_to_ttl():
    plan = _plan(REAL)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$9")
    assert plan.total_micros < plan.ttl_worst_micros <= plan.caps.effective
    assert plan.n_runs_max == 3 * (4 + 3) * 3
    (host,) = plan.hosts
    kinds = [s.kind for s in host.steps]
    assert kinds.count("cold_start") == 1 and kinds.count("warm_start") == 2
    assert kinds.count("eval") == 3
    assert host.seconds < host.ttl_s
    # Under the ~$10 target with the worst case too.
    assert plan.ttl_worst_micros < parse_usd("$10")


# --- the smoke ---------------------------------------------------------------------


def test_the_smoke_runs_every_path_of_the_real_run():
    smoke, real = load_experiment(SMOKE), load_experiment(REAL)
    assert smoke.smoke and not real.smoke
    assert smoke.model == real.model and smoke.provider == real.provider
    assert smoke.variants == real.variants  # same launches, same config hashes
    s_cells, r_cells = expand(smoke, REGISTRY), expand(real, REGISTRY)
    assert [c.config_hash for c in s_cells] == [c.config_hash for c in r_cells]
    assert len({c.host_key for c in s_cells}) == 1
    assert [(w.profile, w.label, w.overrides, w.load.mode) for w in smoke.workloads] == [
        (w.profile, w.label, w.overrides, w.load.mode) for w in real.workloads
    ]
    assert all(w.load.search is None for w in smoke.workloads)  # fixed loads, as the run
    assert smoke.repetitions == real.repetitions and smoke.slo == real.slo
    sq, rq = smoke.quality, real.quality
    assert sq is not None and rq is not None
    assert (sq.suite, sq.subset, sq.baseline_variant, sq.replicates) == (
        rq.suite,
        rq.subset,
        rq.baseline_variant,
        rq.replicates,
    )
    assert sq.limit == 4 and rq.limit is None
    assert _eval_paths(real) <= _eval_paths(smoke)
    # A chat point past every row's knee: each speculating engine has requests cancelled
    # at the drain timeout, as the run's highest point may.
    (s_chat, _), (r_chat, _) = smoke.workloads, real.workloads
    assert max(s_chat.load.values) > 3 * max(r_chat.load.values)
    assert s_chat.overrides == r_chat.overrides == {}  # full-length replies


def test_the_smoke_fits_its_cap_to_ttl():
    plan = _plan(SMOKE)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$2")
    assert plan.total_micros < plan.ttl_worst_micros <= plan.caps.effective
    (host,) = plan.hosts
    kinds = [s.kind for s in host.steps]
    assert kinds.count("cold_start") == 1 and kinds.count("warm_start") == 2
    assert kinds.count("eval") == 3
