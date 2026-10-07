import pytest

from loom_bench.quality.suite import EVALS_DIR, DivergenceSpec, Suite, load_suite
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


@pytest.mark.parametrize("name", SUITES)
def test_pinned_native_tasks_can_decide_their_margin(name):
    """A pinned native set must hold at least its min_samples, and its rule-of-three floor
    (the gate never claims less than 3/n) must sit below its margin, or the task is
    inconclusive by construction."""
    suite = load_suite(name)
    policy = suite.policy()
    for t in suite.tasks:
        if t.kind not in ("json_schema", "tool_calling"):
            continue
        n = t.planned_items()
        assert n is not None and n >= policy.min_samples_for(t.name), t.name
        assert 3 / n < policy.threshold_for(t.name), t.name


# lm-eval tasks whose modules import packages that only an lm-eval extra installs. GPU hosts
# install the `lmeval` extra from uv.lock, so a missing extra only fails on the host (the
# first RunPod sweep lost its eval to `No module named 'langdetect'`).
LMEVAL_TASK_EXTRAS = {
    "ifeval": ("ifeval", {"langdetect", "immutabledict", "nltk"}),
    "niah_": ("ruler", {"wonderwords", "nltk"}),
}


def _lmeval_pin_extras() -> set[str]:
    import re
    import tomllib

    from loom_bench.registry import REPO_ROOT

    project = tomllib.loads((REPO_ROOT / "bench" / "pyproject.toml").read_text())
    pins = project["project"]["optional-dependencies"]["lmeval"]
    (pin,) = [p for p in pins if p.startswith("lm-eval")]
    match = re.match(r"lm-eval\[([^\]]*)\]", pin)
    return {e.strip() for e in match.group(1).split(",")} if match else set()


def _locked_packages() -> set[str]:
    import tomllib

    from loom_bench.registry import REPO_ROOT

    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    return {p["name"] for p in lock["package"]}


@pytest.mark.parametrize("name", SUITES)
def test_lmeval_tasks_have_their_extras_locked(name):
    extras, locked = _lmeval_pin_extras(), _locked_packages()
    for t in load_suite(name).tasks:
        if t.kind != "lm_eval":
            continue
        for task in t.params["tasks"]:
            for prefix, (extra, packages) in LMEVAL_TASK_EXTRAS.items():
                if task.startswith(prefix):
                    assert extra in extras, f"{name}/{task} needs lm-eval[{extra}]"
                    assert packages <= locked, f"{name}/{task}: {packages - locked} not in uv.lock"


def test_policy_from_suite():
    suite = load_suite("qwen3-8b")
    policy = suite.policy()
    assert policy.threshold == 0.01 and policy.min_samples == 300
    assert policy.threshold_for("mmlu_pro") == 0.01
    assert policy.threshold_for("ifeval") == 0.02
    assert policy.min_samples_for("json_schema") == 250
    assert policy.threshold_for("json_schema") == 0.03
    assert policy.max_kl == suite.divergence.max_kl
    assert (policy.noise_multiple, policy.ceiling_kl, policy.ceiling_top1) == (5.0, 0.5, 0.8)
    assert suite.divergence.floor_concurrency == 1 and not policy.review_blocks
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
    with pytest.raises(ValueError, match="ceiling_kl must be above max_kl"):
        _suite(divergence={"max_kl": 0.6})
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


GSM8K = {
    "name": "gsm8k",
    "kind": "lm_eval",
    "params": {"tasks": ["gsm8k"], "metric": "exact_match"},
}


def test_planned_items_per_kind():
    tasks = [
        {"name": "a", "kind": "toy_arithmetic", "params": {"n": 10}},
        {
            "name": "n",
            "kind": "needle",
            "params": {"context_tokens": [4096], "samples_per_cell": 3},
        },
        {"name": "c", "kind": "code_exec", "params": {"datasets": ["humaneval", "mbpp"]}},
        {"name": "c2", "kind": "code_exec", "params": {"datasets": ["mbpp"], "limit": 20}},
        {"name": "j", "kind": "json_schema", "params": {"limit": 7}},
        {"name": "t", "kind": "tool_calling"},
        GSM8K,
        {**GSM8K, "name": "g2", "items": 1319},
        {**GSM8K, "name": "g3", "params": {**GSM8K["params"], "samples": {"gsm8k": [3, 1, 1]}}},
    ]
    got = {t.name: t.planned_items() for t in _suite(tasks=tasks).tasks}
    assert got == {
        "a": 10,
        "n": 15,
        "c": 664,
        "c2": 20,
        "j": 7,
        "t": 60,
        "gsm8k": None,
        "g2": 1319,
        "g3": 2,
    }


def test_items_must_agree_with_what_params_give():
    with pytest.raises(ValueError, match="items 11 but its params give 10"):
        _suite(tasks=[{"name": "a", "kind": "toy_arithmetic", "params": {"n": 10}, "items": 11}])


def test_subsets_select_tasks_in_suite_order():
    tasks = [
        {"name": "a", "kind": "toy_arithmetic"},
        {"name": "b", "kind": "json_schema"},
        {"name": "c", "kind": "tool_calling"},
    ]
    s = _suite(tasks=tasks, subsets={"quick": ["c", "a"]})
    assert [t.name for t in s.select("quick")] == ["a", "c"]
    assert [t.name for t in s.select(None)] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="no subset 'slow'"):
        s.select("slow")
    with pytest.raises(ValueError, match="unknown or repeated"):
        _suite(tasks=tasks, subsets={"x": ["a", "zzz"]})
    with pytest.raises(ValueError, match="unknown or repeated"):
        _suite(tasks=tasks, subsets={"x": ["a", "a"]})


def test_limited_caps_every_kind_and_keeps_everything_else():
    tasks = [
        {"name": "a", "kind": "toy_arithmetic", "params": {"n": 10}},
        {
            "name": "n",
            "kind": "needle",
            "params": {"context_tokens": [4096], "samples_per_cell": 3},
        },
        {"name": "c", "kind": "code_exec", "params": {"datasets": ["humaneval", "mbpp"]}},
        {"name": "j", "kind": "json_schema", "params": {"limit": 2}},
        {"name": "t", "kind": "tool_calling"},
        {**GSM8K, "items": 1319},
        {**GSM8K, "name": "mmlu", "items": 2100, "params": {**GSM8K["params"], "limit": 150}},
        {**GSM8K, "name": "g3", "params": {**GSM8K["params"], "samples": {"gsm8k": [3, 1, 4, 5]}}},
    ]
    full = _suite(tasks=tasks, divergence={"prompts": 48}, subsets={"q": ["a", "t"]})
    small = full.limited(4)
    assert small.item_limit == 4 and full.item_limit is None
    got = {t.name: (t.planned_items(), t.params) for t in small.tasks}
    assert got["a"][0] == 4
    assert got["n"] == (5, {"context_tokens": [4096], "samples_per_cell": 1})  # 5 depths
    assert got["c"][0] == 8  # 4 per dataset
    assert got["j"][0] == 2  # an existing smaller limit stays
    assert got["t"][0] == 4
    assert got["gsm8k"] == (4, {**GSM8K["params"], "limit": 4})
    assert got["mmlu"][0] == 56  # 14 subjects x 4: the planner count scales with the limit
    assert got["g3"][0] == 4 and got["g3"][1]["samples"] == {"gsm8k": [3, 1, 4, 5]}
    assert small.divergence is not None and small.divergence.prompt_ids == [0, 1, 2, 3]
    tiny = full.limited(1).divergence
    assert tiny is not None and tiny.prompt_ids == [0, 1]  # divergence needs two prompts
    assert (small.gate, small.subsets, small.seed) == (full.gate, full.subsets, full.seed)
    assert [t.kind for t in small.tasks] == [t.kind for t in full.tasks]


def test_limited_divergence_keeps_the_hard_prompts_first():
    tasks = [{"name": "a", "kind": "toy_arithmetic", "params": {"n": 10}}]
    full = _suite(tasks=tasks, divergence={"hard_prompts": [20, 40]})
    div = full.limited(4).divergence
    assert div is not None and div.prompt_ids == [20, 40, 0, 1] and div.prompts is None
    assert div.hard_prompts == [20, 40]
    # a hard prompt outside the spec's own prompts is not added
    first = _suite(tasks=tasks, divergence={"prompts": 12, "hard_prompts": [5, 40]})
    assert first.limited(3).divergence.prompt_ids == [5, 0, 1]  # type: ignore[union-attr]


def test_divergence_prompt_selection():
    pinned = [f"p{i}" for i in range(10)]
    spec = DivergenceSpec(prompt_ids=[3, 1])
    assert spec.select(pinned) == ["p3", "p1"]
    assert DivergenceSpec(prompts=2).select(pinned) == ["p0", "p1"]
    assert DivergenceSpec().select(pinned) == pinned
    with pytest.raises(ValueError, match="not both"):
        DivergenceSpec(prompts=2, prompt_ids=[0, 1])
    with pytest.raises(ValueError, match="distinct"):
        DivergenceSpec(prompt_ids=[1, 1])
    with pytest.raises(ValueError, match="out of range"):
        DivergenceSpec(prompt_ids=[0, 10]).select(pinned)


def test_the_qwen3_suite_names_its_byte_split_prompts():
    div = load_suite("qwen3-8b").divergence
    assert div is not None and div.hard_prompts == [20, 40]
    assert load_suite("qwen3-8b").limited(4).divergence.prompt_ids[:2] == [20, 40]  # type: ignore[union-attr]


@pytest.mark.parametrize("name", ["qwen3-8b", "llama-3.3-70b-instruct"])
def test_shipped_suites_can_run_at_smoke_scale(name):
    small = load_suite(name).limited(4)
    assert all((t.planned_items() or t.items or 0) <= 60 for t in small.tasks)
