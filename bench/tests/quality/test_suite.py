import pytest

from loom_bench.quality.suite import EVALS_DIR, Suite, load_suite
from loom_bench.registry import load_registry

SUITES = sorted(p.stem for p in EVALS_DIR.glob("*.yaml"))


def test_pinned_suites_exist():
    assert {"qwen3-8b", "llama-3.3-70b-instruct"} <= set(SUITES)


@pytest.mark.parametrize("name", SUITES)
def test_pinned_suite_is_valid_and_fits_the_model(name):
    suite = load_suite(name)
    model = load_registry().get(suite.model)
    assert suite.suite == name
    kinds = {t.kind for t in suite.tasks}
    assert {"lm_eval", "needle", "code_exec", "tool_calling", "json_schema"} <= kinds
    for t in suite.tasks:
        if t.kind == "needle":
            assert max(t.params["context_tokens"]) <= model.max_context
        if t.kind == "lm_eval" and "max_seq_lengths" in t.params.get("metadata", {}):
            assert max(t.params["metadata"]["max_seq_lengths"]) <= model.max_context
    assert suite.divergence is not None
    expected = model.engine.chat_template_kwargs
    assert suite.chat_template_kwargs == expected


def test_policy_from_suite():
    suite = load_suite("qwen3-8b")
    policy = suite.policy()
    assert policy.threshold == 0.01 and policy.min_samples == 300
    assert policy.threshold_for("mmlu_pro") == 0.01
    assert policy.threshold_for("ifeval") == 0.02
    assert policy.min_samples_for("json_schema") == 50
    assert policy.max_kl == suite.divergence.max_kl
    assert policy.seed == suite.seed
    assert suite.extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


def _suite(**over):
    doc = {
        "suite": "t",
        "model": "m",
        "tasks": [{"name": "a", "kind": "toy_arithmetic", "params": {"n": 10}}],
    }
    doc.update(over)
    return Suite.model_validate(doc)


def test_suite_validation():
    assert _suite().extra_body() == {}
    assert _suite().policy().max_kl is None
    with pytest.raises(ValueError, match="duplicate task names"):
        _suite(tasks=[{"name": "a", "kind": "toy_arithmetic"}] * 2)
    with pytest.raises(ValueError, match="unknown eval task kind"):
        _suite(tasks=[{"name": "a", "kind": "nope"}])
    with pytest.raises(ValueError):
        _suite(tasks=[{"name": "a", "kind": "toy_arithmetic", "params": {"bogus": 1}}])
    with pytest.raises(ValueError):
        _suite(gate={"threshold": 1.5})
    with pytest.raises(ValueError):
        _suite(extra_key=1)
