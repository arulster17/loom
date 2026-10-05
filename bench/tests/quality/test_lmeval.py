import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from loom_bench.quality.client import EvalClient
from loom_bench.quality.tasks import build_task
from loom_bench.quality.tasks.base import EvalContext, OutputKind
from loom_bench.quality.tasks.lmeval import (
    LmEvalError,
    LmEvalNotInstalled,
    LmEvalParams,
    build_command,
    ensure_installed,
    parse_results,
    parse_samples,
)

FIXTURES = Path(__file__).parent / "fixtures" / "lm_eval"
RUN = FIXTURES / "Qwen__Qwen3-8B"


def files(task):
    return list(RUN.glob(f"samples_{task}_*.jsonl"))


def opt(argv, flag):
    return argv[argv.index(flag) + 1]


def test_chat_command():
    p = LmEvalParams(
        tasks=("gsm8k",),
        limit=500,
        num_fewshot=5,
        metric="exact_match",
        filter="flexible-extract",
        gen_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        num_concurrent=32,
    )
    argv = build_command(
        p,
        base_url="http://h:8000/v1/",
        model="qwen3-8b",
        seed=1234,
        output_path=Path("/o"),
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "top_k": 1},
    )
    assert argv[:5] == [sys.executable, "-m", "lm_eval", "run", "--model"]
    assert opt(argv, "--model") == "local-chat-completions"
    assert json.loads(opt(argv, "--model_args")) == {
        "base_url": "http://h:8000/v1/chat/completions",
        "max_retries": 3,
        "model": "qwen3-8b",
        "num_concurrent": 32,
        "seed": 1234,
    }
    assert opt(argv, "--tasks") == "gsm8k"
    assert opt(argv, "--limit") == "500"
    assert opt(argv, "--num_fewshot") == "5"
    assert opt(argv, "--seed") == "1234"
    assert opt(argv, "--output_path") == "/o"
    # Task gen_kwargs override the client's extra body.
    assert json.loads(opt(argv, "--gen_kwargs")) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 1,
    }
    assert "--log_samples" in argv and "--apply_chat_template" in argv
    assert "--samples" not in argv and "--confirm_run_unsafe_code" not in argv


def test_completions_command_with_samples_and_metadata():
    p = LmEvalParams(
        tasks=("niah_single_2", "niah_multikey_1"),
        api="completions",
        apply_chat_template=False,
        samples={"niah_single_2": [9, 3, 3, 0]},
        metric_from_doc="max_length",
        metadata={"max_seq_lengths": [4096, 16384], "tokenizer": "Qwen/Qwen3-8B"},
        model_args={"tokenized_requests": False, "tokenizer_backend": None},
    )
    argv = build_command(
        p,
        base_url="http://h/v1",
        model="m",
        seed=0,
        output_path=Path("/o"),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    assert opt(argv, "--model") == "local-completions"
    assert "--gen_kwargs" not in argv
    margs = json.loads(opt(argv, "--model_args"))
    assert margs["base_url"] == "http://h/v1/completions"
    assert margs["tokenized_requests"] is False and margs["tokenizer_backend"] is None
    assert opt(argv, "--tasks") == "niah_single_2,niah_multikey_1"
    # lm_eval maps logged doc ids through the index list as given, so it must be sorted.
    assert json.loads(opt(argv, "--samples")) == {"niah_single_2": [0, 3, 9]}
    assert json.loads(opt(argv, "--metadata"))["max_seq_lengths"] == [4096, 16384]
    assert "--limit" not in argv and "--apply_chat_template" not in argv


def test_unsafe_code_needs_permission():
    p = LmEvalParams(tasks=("humaneval",), api="completions", metric="pass_at_k", unsafe_code=True)
    with pytest.raises(LmEvalError, match="allow_code_exec"):
        build_command(p, base_url="http://h/v1", model="m", seed=0, output_path=Path("/o"))
    argv = build_command(
        p, base_url="http://h/v1", model="m", seed=0, output_path=Path("/o"), allow_code_exec=True
    )
    assert "--confirm_run_unsafe_code" in argv


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"metric": "a", "metric_from_doc": "b"}, "exactly one"),
        ({}, "exactly one"),
        ({"metric": "a", "limit": 5, "samples": {"t": [1]}}, "mutually exclusive"),
        ({"metric": "a", "apply_chat_template": False}, "apply_chat_template"),
        ({"metric": "a", "model_args": {"base_url": "x"}}, "may not set"),
    ],
)
def test_param_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LmEvalParams(tasks=("t",), **kwargs)


def test_parse_gsm8k_needs_a_filter():
    with pytest.raises(LmEvalError, match="set `filter`"):
        parse_samples(files("gsm8k"), LmEvalParams(tasks=("gsm8k",), metric="exact_match"))
    strict = LmEvalParams(tasks=("gsm8k",), metric="exact_match", filter="strict-match")
    items, comps = parse_samples(files("gsm8k"), strict)
    assert [(i.item_id, i.score) for i in items] == [
        ("gsm8k/0", 1.0),
        ("gsm8k/1", 0.0),
        ("gsm8k/2", 1.0),
    ]
    assert all(len(i.content_hash) == 64 for i in items)
    assert comps[0].text == "Let me compute. #### 24" and comps[0].finish_reason is None
    flex = strict.model_copy(update={"filter": "flexible-extract"})
    assert [i.score for i in parse_samples(files("gsm8k"), flex)[0]] == [1.0, 1.0, 0.0]
    with pytest.raises(LmEvalError, match="not logged"):
        parse_samples(files("gsm8k"), strict.model_copy(update={"filter": "nope"}))


def test_parse_ifeval_bool_and_list_metrics():
    prompt = LmEvalParams(tasks=("ifeval",), metric="prompt_level_strict_acc")
    assert [i.score for i in parse_samples(files("ifeval"), prompt)[0]] == [1.0, 0.0, 1.0]
    inst = prompt.model_copy(update={"metric": "inst_level_strict_acc"})
    assert [i.score for i in parse_samples(files("ifeval"), inst)[0]] == pytest.approx(
        [1.0, 1 / 3, 1.0]
    )


def test_parse_ruler_metric_keyed_by_doc_length():
    p = LmEvalParams(
        tasks=("niah_single_2",), metric_from_doc="max_length", output_kind=OutputKind.TEXT
    )
    items, _ = parse_samples(files("niah_single_2"), p)
    assert [i.score for i in items] == [1.0, 0.0, 1.0, 0.5]
    with pytest.raises(LmEvalError, match="no metric"):
        parse_samples(files("niah_single_2"), LmEvalParams(tasks=("x",), metric="exact_match"))
    with pytest.raises(LmEvalError, match=r"outside \[0, 1\]"):
        parse_samples(files("niah_single_2"), LmEvalParams(tasks=("x",), metric="4096"))


def test_parse_results():
    info = parse_results(RUN.glob("results_*.json"))
    assert info["lm_eval_version"] == "0.4.13"
    assert info["task_versions"] == {"gsm8k": 3.0, "ifeval": 4.0, "niah_single_2": 1.0}
    assert info["n_samples"]["gsm8k"]["effective"] == 3


def test_missing_lm_eval_gives_clear_error(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(LmEvalNotInstalled, match="--extra lmeval"):
        ensure_installed()


FAKE_MAIN = textwrap.dedent(
    """
    import json, os, shutil, sys
    out = sys.argv[sys.argv.index("--output_path") + 1]
    shutil.copytree(os.environ["FAKE_LM_EVAL_FIXTURE"], os.path.join(out, "Qwen__Qwen3-8B"))
    with open(os.path.join(out, "call.json"), "w") as f:
        json.dump({"argv": sys.argv[1:], "key": os.environ.get("OPENAI_API_KEY")}, f)
    """
)


async def test_run_end_to_end_with_fake_harness(tmp_path, monkeypatch):
    pkg = tmp_path / "site" / "lm_eval"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_MAIN)
    fixture = tmp_path / "fixture"
    shutil.copytree(RUN, fixture)
    for f in fixture.glob("samples_*.jsonl"):
        if "gsm8k" not in f.name:
            f.unlink()
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "site"))
    monkeypatch.setenv("FAKE_LM_EVAL_FIXTURE", str(fixture))

    task = build_task(
        "lm_eval",
        "gsm8k",
        {"tasks": ["gsm8k"], "limit": 3, "metric": "exact_match", "filter": "strict-match"},
    )
    work = tmp_path / "work"
    async with EvalClient("http://h/v1", "qwen3-8b", api_key="sk-test", seed=7) as client:
        out = await task.run(EvalContext(client=client, workdir=work))
        with pytest.raises(LmEvalError, match="not empty"):
            await task.run(EvalContext(client=client, workdir=work))
    assert [i.score for i in out.items] == [1.0, 0.0, 1.0]
    call = json.loads((work / "lm_eval-gsm8k" / "call.json").read_text())
    assert call["key"] == "sk-test"
    assert "sk-test" not in " ".join(call["argv"])
    assert call["argv"][call["argv"].index("--seed") + 1] == "7"
    info = out.provenance["lm_eval"]
    assert info["lm_eval_version"] == "0.4.13"
    assert out.version == "lm_eval=0.4.13;gsm8k=3.0;ifeval=4.0;niah_single_2=1.0"
