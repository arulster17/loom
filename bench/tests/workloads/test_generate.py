import json

import pytest

from loom_bench.tokenize import SimpleTokenizer
from loom_bench.workloads import build_requests, load_profile, parse_profile

TOK = SimpleTokenizer()
BASE = {"name": "t", "description": "d", "content": "synthetic"}


def prompt_tokens(req):
    if req.endpoint == "completions":
        return TOK.count(req.payload["prompt"])
    return sum(TOK.count(m["content"]) for m in req.payload["messages"])


def test_synthetic_exact_lengths_and_payload():
    p = load_profile("fixed-128-128")
    reqs = build_requests(p, TOK, 20)
    assert len({r.request_id for r in reqs}) == 20
    for r in reqs:
        assert r.endpoint == "completions"
        assert TOK.count(r.payload["prompt"]) == r.expected_prompt_tokens == 128
        assert r.max_tokens == r.payload["max_tokens"] == 128
        assert r.payload["ignore_eos"] is True and r.payload["temperature"] == 0.0
        assert "stream" not in r.payload and "model" not in r.payload


def test_synthetic_range_ratio_and_chat_endpoint():
    p = parse_profile(
        {**BASE, "kind": "synthetic", "endpoint": "chat", "input_len": 100, "output_len": 50,
         "range_ratio": 0.2, "ignore_eos": False}
    )  # fmt: skip
    reqs = build_requests(p, TOK, 200)
    ins = [r.expected_prompt_tokens for r in reqs]
    outs = [r.max_tokens for r in reqs]
    assert min(ins) >= 80 and max(ins) <= 120 and len(set(ins)) > 10
    assert min(outs) >= 40 and max(outs) <= 60
    for r in reqs:
        assert r.payload["messages"][0]["role"] == "user"
        assert prompt_tokens(r) == r.expected_prompt_tokens
        assert "ignore_eos" not in r.payload


def test_deterministic_by_seed_and_stable_prefix():
    p = load_profile("code-completion")
    a = build_requests(p, TOK, 10)
    assert [r.payload for r in a] == [r.payload for r in build_requests(p, TOK, 10)]
    assert [r.payload for r in build_requests(p, TOK, 4)] == [r.payload for r in a[:4]]
    other = build_requests(p, TOK, 10, rng_seed=1)
    assert other[0].payload != a[0].payload
    assert other[0].request_id == "code-completion-s1-000000"
    assert a[0].request_id == "code-completion-s0-000000"


@pytest.mark.parametrize("share", [0.0, 0.3, 0.95])
def test_shared_prefix_honours_share_and_groups(share):
    p = load_profile("shared-prefix", {"prefix_share": share, "num_prefix_groups": 3})
    reqs = build_requests(p, TOK, 60)
    prefix_len = round(2048 * share)
    by_group: dict[int, set[str]] = {}
    for r in reqs:
        system, user = r.payload["messages"]
        assert system["role"] == "system" and user["role"] == "user"
        assert TOK.count(system["content"]) == prefix_len == r.meta["prefix_tokens"]
        assert TOK.count(system["content"]) + TOK.count(user["content"]) == 2048
        by_group.setdefault(r.meta["prefix_group"], set()).add(system["content"])
    assert set(by_group) == {0, 1, 2}
    assert all(len(prefixes) == 1 for prefixes in by_group.values())
    if share > 0:
        assert len(set().union(*by_group.values())) == 3
    users = [r.payload["messages"][1]["content"] for r in reqs]
    assert len(set(users)) == len(users)


def test_needle_placement_and_answer():
    p = load_profile("long-context-needle", {"context_len": 2000, "depths": [0.0, 0.25, 1.0]})
    reqs = build_requests(p, TOK, 6)
    assert [r.meta["depth"] for r in reqs] == [0.0, 0.25, 1.0, 0.0, 0.25, 1.0]
    for r in reqs:
        text = r.payload["messages"][0]["content"]
        assert TOK.count(text) == r.expected_prompt_tokens == 2000
        answer = r.meta["expected_answer"]
        needle_at = text.index(f" is {answer}.")
        start = text.rindex("The secret code", 0, needle_at)
        question = text[text.rindex("\n\n") :]
        haystack_tokens = TOK.count(text) - TOK.count(question)
        fraction = TOK.count(text[:start]) / haystack_tokens
        assert fraction == pytest.approx(r.meta["depth"], abs=0.02)
        key = text[start:needle_at].removeprefix("The secret code for the ")
        assert f"secret code for the {key}?" in question
        assert answer not in question
        assert r.max_tokens == 32 and "ignore_eos" not in r.payload


def write_sharegpt(path, jsonl=False):
    convs = [
        {"id": "a", "conversations": [
            {"from": "human", "value": "What is two plus two?"},
            {"from": "gpt", "value": "It is four, as basic arithmetic shows."},
            {"from": "human", "value": "And three plus three?"},
            {"from": "gpt", "value": "That is six."},
        ]},
        {"id": "b", "conversations": [
            {"from": "system", "value": "Be terse."},
            {"from": "human", "value": "Name a color."},
            {"from": "gpt", "value": " ".join(["blue"] * 50)},
            {"from": "human", "value": "Thanks"},
        ]},
        {"id": "bad-order", "conversations": [
            {"from": "gpt", "value": "Hello there friend, how are you?"},
            {"from": "human", "value": "Fine thanks a lot."},
        ]},
        {"id": "too-short", "conversations": [
            {"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hey"},
        ]},
        {"id": "empty", "conversations": []},
    ]  # fmt: skip
    if jsonl:
        path.write_text("\n".join(json.dumps(c) for c in convs) + "\n")
    else:
        path.write_text(json.dumps(convs))


@pytest.mark.parametrize("jsonl", [False, True])
def test_chat_dataset_multi_turn_and_reference_lengths(tmp_path, monkeypatch, jsonl):
    f = tmp_path / ("sg.jsonl" if jsonl else "sg.json")
    write_sharegpt(f, jsonl)
    monkeypatch.setenv("LOOM_DATA_DIR", str(tmp_path))
    p = load_profile("chat-sharegpt", {"path": f"$LOOM_DATA_DIR/{f.name}", "max_output_len": 20})
    reqs = build_requests(p, TOK, 2)
    by_turns = {r.meta["turns"]: r for r in reqs}
    assert [m["role"] for m in by_turns[3].payload["messages"]] == ["user", "assistant", "user"]
    assert by_turns[3].max_tokens == TOK.count("That is six.")
    sys_req = by_turns[2]
    assert [m["role"] for m in sys_req.payload["messages"]] == ["system", "user"]
    assert sys_req.max_tokens == 20  # 50-token reference capped
    assert all(r.expected_prompt_tokens == prompt_tokens(r) for r in reqs)
    with pytest.raises(ValueError, match="only 2 usable conversations"):
        build_requests(p, TOK, 3)


def test_chat_dataset_missing_file():
    p = load_profile("chat-sharegpt", {"path": "/nonexistent/sharegpt.json"})
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        build_requests(p, TOK, 1)


def test_code_completion_shape():
    reqs = build_requests(load_profile("code-completion"), TOK, 30)
    for r in reqs:
        assert r.endpoint == "completions"
        assert TOK.count(r.payload["prompt"]) == r.expected_prompt_tokens
        assert 1029 <= r.expected_prompt_tokens <= 2042
        assert 32 <= r.max_tokens <= 63
        assert r.payload["prompt"].startswith("# ") and "def " in r.payload["prompt"]


def test_long_generation_shape():
    reqs = build_requests(load_profile("long-generation"), TOK, 5)
    for r in reqs:
        text = r.payload["messages"][0]["content"]
        assert text.startswith("Write a long, detailed essay")
        assert TOK.count(text) == r.expected_prompt_tokens == 256
        assert r.max_tokens == 2048 and r.payload["ignore_eos"] is True


def test_trace_lengths_and_clamps(tmp_path):
    f = tmp_path / "azure.csv"
    f.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n"
        "2023-11-16 18:15:46.000000,40000,10\n"
        "2023-11-16 18:15:47.000000,120,0\n"
        "2023-11-16 18:15:48.000000,300,5000\n"
    )
    p = load_profile("trace-azure-code", {"path": str(f)})
    reqs = build_requests(p, TOK, 3)
    assert [r.expected_prompt_tokens for r in reqs] == [30000, 120, 300]
    assert [r.max_tokens for r in reqs] == [10, 1, 2048]
    assert [r.meta["trace_offset_s"] for r in reqs] == [0.0, 1.0, 2.0]
    assert all(TOK.count(r.payload["prompt"]) == r.expected_prompt_tokens for r in reqs)
    with pytest.raises(ValueError, match="3 rows"):
        build_requests(p, TOK, 4)


def test_every_generative_shipped_profile_builds():
    for name in ["fixed-1k-1k", "fixed-8k-1k", "fixed-32k-1k", "shared-prefix",
                 "long-context-needle", "code-completion", "long-generation"]:  # fmt: skip
        reqs = build_requests(load_profile(name), TOK, 2)
        assert all(prompt_tokens(r) == r.expected_prompt_tokens for r in reqs)
