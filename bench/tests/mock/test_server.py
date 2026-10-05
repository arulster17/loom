import json
import random
import re
import time

import httpx
import jsonschema
import pytest
from prometheus_client.parser import text_string_to_metric_families

from loom_bench.mock.content import render_chat_prompt
from loom_bench.tokenize import SimpleTokenizer

TOK = SimpleTokenizer()
MODEL = "mock-model"


def chat(client: httpx.Client, content: str, **params) -> httpx.Response:
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], **params}
    return client.post("/v1/chat/completions", json=body)


def stream_events(client: httpx.Client, path: str, body: dict) -> list[dict | str]:
    events: list[dict | str] = []
    with client.stream("POST", path, json={**body, "stream": True}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line:
                continue
            assert line.startswith("data: ")
            data = line.removeprefix("data: ")
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def metrics(client: httpx.Client) -> dict[str, float]:
    samples = {}
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for s in family.samples:
            labels = ",".join(f"{k}={v}" for k, v in sorted(s.labels.items()) if k != "model_name")
            samples[f"{s.name}{{{labels}}}" if labels else s.name] = s.value
    return samples


def assert_openai_error(response: httpx.Response, status: int) -> dict:
    assert response.status_code == status
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["message"]
    return error


# ---------------------------------------------------------------- basics


def test_health_version_models(client):
    assert client.get("/health").status_code == 200
    assert "version" in client.get("/version").json()
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert [m["id"] for m in models["data"]] == [MODEL]


def test_custom_model_ids(make_client):
    c = make_client(models=["a", "b"])
    assert [m["id"] for m in c.get("/v1/models").json()["data"]] == ["a", "b"]
    assert chat(c, "hi", model="b", max_tokens=2).status_code == 200


def test_chat_usage_matches_tokenizer(client):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Tell me about batching."},
    ]
    body = {"model": MODEL, "messages": messages, "max_tokens": 50}
    data = client.post("/v1/chat/completions", json=body).json()
    usage = data["usage"]
    text = data["choices"][0]["message"]["content"]
    assert data["object"] == "chat.completion"
    assert usage["prompt_tokens"] == TOK.count(render_chat_prompt(messages, None))
    assert usage["completion_tokens"] == TOK.count(text)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0


def test_chat_streaming_one_token_per_chunk_and_usage(client):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a story."}],
        "max_tokens": 40,
        "stream_options": {"include_usage": True},
    }
    events = stream_events(client, "/v1/chat/completions", body)
    assert events[-1] == "[DONE]"
    usage_chunk = events[-2]
    assert usage_chunk["choices"] == []
    chunks = events[:-2]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    content = [c["choices"][0]["delta"]["content"] for c in chunks[1:]]
    assert all(TOK.count(piece) == 1 for piece in content)
    finish = [c["choices"][0]["finish_reason"] for c in chunks]
    assert finish[-1] in ("stop", "length") and set(finish[:-1]) == {None}
    usage = usage_chunk["usage"]
    assert usage["completion_tokens"] == len(content) == TOK.count("".join(content))

    plain = client.post("/v1/chat/completions", json=body).json()
    assert plain["choices"][0]["message"]["content"] == "".join(content)
    assert plain["usage"] == usage


def test_streaming_without_usage_option(client):
    events = stream_events(
        client, "/v1/completions", {"model": MODEL, "prompt": "Once upon", "max_tokens": 5}
    )
    assert events[-1] == "[DONE]"
    assert all("usage" not in e for e in events[:-1])
    assert len(events) == 6


def test_completions_usage_and_streaming(client):
    prompt = "The cluster scheduler"
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 30}
    data = client.post("/v1/completions", json=body).json()
    choice = data["choices"][0]
    assert data["object"] == "text_completion"
    assert choice["text"].startswith(" ")
    assert data["usage"]["prompt_tokens"] == TOK.count(prompt)
    assert data["usage"]["completion_tokens"] == TOK.count(choice["text"])

    events = stream_events(
        client, "/v1/completions", {**body, "stream_options": {"include_usage": True}}
    )
    texts = [e["choices"][0]["text"] for e in events[:-2]]
    assert "".join(texts) == choice["text"]
    assert events[-2]["usage"] == data["usage"]


def test_completions_prompt_list(client):
    prompts = ["What is 2 + 2?", "What is 3 * 4?"]
    data = client.post(
        "/v1/completions", json={"model": MODEL, "prompt": prompts, "max_tokens": 10}
    ).json()
    assert [c["index"] for c in data["choices"]] == [0, 1]
    assert "The answer is 4." in data["choices"][0]["text"]
    assert "The answer is 12." in data["choices"][1]["text"]
    assert data["usage"]["prompt_tokens"] == sum(TOK.count(p) for p in prompts)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_length_controls(client, path):
    def body(**params):
        if path == "/v1/completions":
            return {"model": MODEL, "prompt": "Hello", **params}
        return {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], **params}

    data = client.post(path, json=body(max_tokens=300, ignore_eos=True)).json()
    assert data["usage"]["completion_tokens"] == 300
    assert data["choices"][0]["finish_reason"] == "length"

    data = client.post(path, json=body(max_tokens=3)).json()
    assert data["usage"]["completion_tokens"] <= 3

    data = client.post(path, json=body(max_tokens=500, min_tokens=400)).json()
    assert data["usage"]["completion_tokens"] >= 400


def test_natural_length_finishes_with_stop(client):
    reasons = {
        chat(client, f"prompt {i}", max_tokens=4000).json()["choices"][0]["finish_reason"]
        for i in range(5)
    }
    assert reasons == {"stop"}


def test_output_is_deterministic_and_seeded(client, make_client):
    first = chat(client, "Describe a kernel.", max_tokens=30).json()["choices"][0]["message"]
    again = chat(client, "Describe a kernel.", max_tokens=30).json()["choices"][0]["message"]
    other = make_client(seed=7)
    reseeded = chat(other, "Describe a kernel.", max_tokens=30).json()["choices"][0]["message"]
    assert first == again
    assert first != reseeded


# ---------------------------------------------------------------- errors


def test_errors_are_openai_shaped(client):
    error = assert_openai_error(chat(client, "hi", model="nope"), 404)
    assert (error["param"], error["code"]) == ("model", "model_not_found")

    error = assert_openai_error(chat(client, "hi", max_tokens=40_000), 400)
    assert error["code"] == "context_length_exceeded"

    assert_openai_error(client.post("/v1/chat/completions", json={"model": MODEL}), 400)
    assert_openai_error(chat(client, "hi", max_tokens="many"), 400)
    assert_openai_error(chat(client, "hi", n=2), 400)
    assert_openai_error(chat(client, "hi", logprobs=True, top_logprobs=50), 400)
    assert_openai_error(client.post("/v1/completions", content=b"{nope"), 400)
    assert_openai_error(client.get("/v1/nothing"), 404)


def test_context_length_counts_prompt(make_client):
    c = make_client(max_model_len=64, kv_capacity_tokens=1024)
    long_prompt = TOK.random_text(60, random.Random(0))
    assert chat(c, long_prompt).status_code == 400  # template alone exceeds 64 tokens
    data = c.post("/v1/completions", json={"model": MODEL, "prompt": long_prompt, "max_tokens": 4})
    assert data.status_code == 200
    error = assert_openai_error(
        c.post("/v1/completions", json={"model": MODEL, "prompt": long_prompt, "max_tokens": 5}),
        400,
    )
    assert "65 tokens" in error["message"]


# ---------------------------------------------------------------- quality knobs

QUESTIONS = [(a, op, b) for a in range(3, 60, 7) for op in "+-*" for b in (4, 11)]


def _answer(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


def _answered(client: httpx.Client, a: int, op: str, b: int) -> int:
    content = f"Solve carefully. What is {a} {op} {b}?"
    text = chat(client, content, max_tokens=20).json()["choices"][0]["message"]["content"]
    match = re.search(r"The answer is (-?\d+)\.", text)
    assert match, text
    return int(match.group(1))


def test_arithmetic_correct_without_degrade(client):
    for a, op, b in QUESTIONS:
        assert _answered(client, a, op, b) == _answer(a, op, b)


def test_arithmetic_degrade(make_client):
    degraded = make_client(degrade=0.3)
    wrong = [_answered(degraded, a, op, b) != _answer(a, op, b) for a, op, b in QUESTIONS]
    assert 0.1 < sum(wrong) / len(wrong) < 0.5
    again = [_answered(degraded, a, op, b) != _answer(a, op, b) for a, op, b in QUESTIONS]
    assert wrong == again


def test_chat_logprobs_shape(client):
    params = {"max_tokens": 8, "logprobs": True, "top_logprobs": 5}
    data = chat(client, "Hello", **params).json()
    content = data["choices"][0]["logprobs"]["content"]
    assert len(content) == data["usage"]["completion_tokens"]
    assert "".join(e["token"] for e in content) == data["choices"][0]["message"]["content"]
    for entry in content:
        assert entry["logprob"] <= 0
        assert len(entry["top_logprobs"]) == 5
        top = [t["logprob"] for t in entry["top_logprobs"]]
        assert top == sorted(top, reverse=True)
        assert entry["bytes"] == list(entry["token"].encode())

    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], **params}
    events = stream_events(client, "/v1/chat/completions", body)
    streamed = [e["choices"][0]["logprobs"]["content"][0] for e in events[1:-1]]
    assert streamed == content


def test_completions_echo_logprobs(client):
    prompt = "The quick brown fox"
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 3, "logprobs": 2, "echo": True}
    data = client.post("/v1/completions", json=body).json()
    choice = data["choices"][0]
    lp = choice["logprobs"]
    n = data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    assert choice["text"].startswith(prompt)
    assert lp["tokens"][:4] == TOK.pieces(prompt)
    assert "".join(lp["tokens"]) == choice["text"]
    assert len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"]) == n
    assert lp["token_logprobs"][0] is None and lp["top_logprobs"][0] is None
    for token, logprob, top in zip(
        lp["tokens"][1:], lp["token_logprobs"][1:], lp["top_logprobs"][1:], strict=True
    ):
        assert top[token] == logprob
        assert 2 <= len(top) <= 3  # top-2 plus the sampled token
    assert [choice["text"][o] for o in lp["text_offset"]] == [t[0] for t in lp["tokens"]]

    events = stream_events(client, "/v1/completions", body)
    streamed_tokens = [t for e in events[:-1] for t in e["choices"][0]["logprobs"]["tokens"]]
    streamed_lps = [x for e in events[:-1] for x in e["choices"][0]["logprobs"]["token_logprobs"]]
    assert streamed_tokens == lp["tokens"]
    assert streamed_lps == lp["token_logprobs"]


def test_logprob_noise_changes_distribution(client, make_client):
    noisy = make_client(logprob_noise=1.0)
    body = {"model": MODEL, "prompt": "Noise check", "max_tokens": 5, "logprobs": 5}
    clean_lp = client.post("/v1/completions", json=body).json()["choices"][0]["logprobs"]
    noisy_lp = noisy.post("/v1/completions", json=body).json()["choices"][0]["logprobs"]
    assert clean_lp["tokens"] == noisy_lp["tokens"]
    assert clean_lp["token_logprobs"] != noisy_lp["token_logprobs"]


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 12},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "score": {"type": "number"},
        "active": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "level": {"enum": ["low", "mid", "high"]},
        "address": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    "required": ["name", "age", "level"],
    "additionalProperties": False,
}


def _json_schema_format(schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "person", "schema": schema}}


def test_json_schema_output_is_valid(client):
    for i in range(10):
        data = chat(
            client, f"Person {i}", max_tokens=400, response_format=_json_schema_format(SCHEMA)
        ).json()
        jsonschema.validate(json.loads(data["choices"][0]["message"]["content"]), SCHEMA)
        assert data["choices"][0]["finish_reason"] == "stop"

    data = chat(client, "Any JSON", response_format={"type": "json_object"}).json()
    assert isinstance(json.loads(data["choices"][0]["message"]["content"]), dict)


def test_json_degrade_breaks_validity(make_client):
    degraded = make_client(degrade=0.5)
    valid = []
    for i in range(30):
        data = chat(
            degraded, f"Person {i}", max_tokens=400, response_format=_json_schema_format(SCHEMA)
        ).json()
        try:
            jsonschema.validate(json.loads(data["choices"][0]["message"]["content"]), SCHEMA)
            valid.append(True)
        except (json.JSONDecodeError, jsonschema.ValidationError):
            valid.append(False)
    assert 0.2 < valid.count(False) / len(valid) < 0.8


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"enum": ["C", "F"]}},
            "required": ["city", "unit"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}},
    },
}


def test_tool_calls(client):
    tools = [WEATHER_TOOL, TIME_TOOL]
    data = chat(client, "Weather in Paris?", tools=tools, max_tokens=200).json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None
    (call,) = message["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    args = json.loads(call["function"]["arguments"])
    jsonschema.validate(args, WEATHER_TOOL["function"]["parameters"])
    assert data["usage"]["completion_tokens"] == TOK.count(call["function"]["arguments"])

    named = {"type": "function", "function": {"name": "get_time"}}
    data = chat(client, "Time?", tools=tools, tool_choice=named, max_tokens=200).json()
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_time"

    data = chat(client, "No tools", tools=tools, tool_choice="none", max_tokens=5).json()
    assert data["choices"][0]["message"]["content"]


def test_tool_call_streaming(client):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Weather in Oslo?"}],
        "tools": [WEATHER_TOOL],
        "max_tokens": 200,
    }
    events = stream_events(client, "/v1/chat/completions", body)
    deltas = [e["choices"][0]["delta"] for e in events[:-1]]
    header = deltas[0]["tool_calls"][0]
    assert header["function"]["name"] == "get_weather" and header["id"].startswith("call_")
    args = "".join(d["tool_calls"][0]["function"]["arguments"] for d in deltas)
    jsonschema.validate(json.loads(args), WEATHER_TOOL["function"]["parameters"])
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------- metrics and caching


def test_metrics_counters_move_and_prefix_cache(make_client):
    c = make_client()
    before = metrics(c)
    assert before["vllm:num_requests_running"] == 0
    assert before["vllm:kv_cache_usage_perc"] == 0
    assert before["vllm:request_queue_time_seconds_count"] == 0

    prompt = TOK.random_text(300, random.Random(1))
    first = c.post("/v1/completions", json={"model": MODEL, "prompt": prompt, "max_tokens": 10})
    second = c.post("/v1/completions", json={"model": MODEL, "prompt": prompt, "max_tokens": 10})
    assert first.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert second.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 288

    after = metrics(c)
    generated = sum(r.json()["usage"]["completion_tokens"] for r in (first, second))
    assert after["vllm:generation_tokens_total"] == generated
    assert after["vllm:prompt_tokens_total"] == 600
    assert after["vllm:prefix_cache_queries_total"] == 600
    assert after["vllm:prefix_cache_hits_total"] == 288
    assert after["vllm:request_success_total{finished_reason=length}"] == 2
    assert after["vllm:request_queue_time_seconds_count"] == 2
    assert after["vllm:request_queue_time_seconds_bucket{le=+Inf}"] == 2
    assert after["vllm:num_preemptions_total"] == 0
    assert after["vllm:num_requests_waiting"] == 0


def test_client_disconnect_aborts_request(make_client):
    c = make_client(time_scale=0.05)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "long"}],
        "max_tokens": 2000,
        "ignore_eos": True,
        "stream": True,
    }
    with c.stream("POST", "/v1/chat/completions", json=body) as response:
        lines = response.iter_lines()
        next(lines)
        assert metrics(c)["vllm:num_requests_running"] == 1
    deadline = time.monotonic() + 5
    while metrics(c)["vllm:num_requests_running"] != 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert metrics(c)["vllm:kv_cache_usage_perc"] == 0


# ---------------------------------------------------------------- faults


def test_error_rate(make_client):
    c = make_client(error_rate=1.0)
    responses = [chat(c, "hi") for _ in range(20)]
    assert {r.status_code for r in responses} == {500, 503}
    for r in responses:
        assert assert_openai_error(r, r.status_code)["type"] == "server_error"
    assert metrics(c)["vllm:generation_tokens_total"] == 0


def test_abort_rate_drops_stream(make_client):
    c = make_client(abort_rate=1.0)
    body = {"model": MODEL, "prompt": "hi", "max_tokens": 20, "ignore_eos": True, "stream": True}
    for _ in range(3):
        with (
            pytest.raises(httpx.RemoteProtocolError),
            c.stream("POST", "/v1/completions", json=body) as response,
        ):
            for _ in response.iter_lines():
                pass


def test_startup_delay(make_client):
    c = make_client(startup_delay_s=0.5, time_scale=1.0)
    assert_openai_error(c.get("/health"), 503)
    assert_openai_error(chat(c, "hi"), 503)
    deadline = time.monotonic() + 5
    while c.get("/health").status_code != 200:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert chat(c, "hi", max_tokens=2).status_code == 200
