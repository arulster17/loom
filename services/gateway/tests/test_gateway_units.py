"""Unit tests: request preparation, SSE rewriting, replica choice, headers, logs. No
servers needed."""

from __future__ import annotations

import json
import logging

import pytest

from loom_gateway.logs import JsonFormatter
from loom_gateway.metrics import GatewayMetrics
from loom_gateway.proxy import _engine_error, cost_headers
from loom_gateway.request_body import Prepared, parse_json_object, prepare
from loom_gateway.routing import Router
from loom_gateway.settings import GatewaySettings
from loom_gateway.sse import ChunkRewriter, client_usage, sse_payloads, token_counts
from loom_platform.errors import ApiError
from loom_platform.pool import PoolState, Replica
from loom_platform.pricing import Price, cost_micros, max_cost_micros, prompt_tokens_upper_bound

QWEN = Price(77_000, 77_000, 368_000)
CTX = 32768
CAP = 16384


def prep(endpoint="chat", **data):
    data.setdefault("model", "m")
    if endpoint == "chat":
        data.setdefault("messages", [{"role": "user", "content": "hi"}])
    body = json.dumps(data).encode()
    return prepare(
        endpoint,
        json.loads(body),
        body_bytes=len(body),
        model_id="m",
        price=QWEN,
        max_context=CTX,
        max_tokens_cap=CAP,
    )


def upstream(p) -> dict:
    return json.loads(p.upstream_body)


# ------------------------------------------------------------------ request preparation


def test_max_tokens_defaults_to_the_cap_and_is_reserved():
    p = prep()
    assert p.max_tokens == CAP and upstream(p)["max_tokens"] == CAP
    upper = prompt_tokens_upper_bound(
        len(json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})), CTX
    )
    assert p.prompt_upper == upper
    assert p.max_cost_micros == max_cost_micros(QWEN, upper, CAP, CTX)
    assert p.tpm_estimate == upper


def test_max_tokens_rules():
    assert prep(max_tokens=5).max_tokens == 5
    p = prep(max_completion_tokens=7, max_tokens=9)
    assert p.max_tokens == 7
    assert upstream(p)["max_completion_tokens"] == 7 and upstream(p)["max_tokens"] == 7
    assert "max_completion_tokens" not in upstream(prep(max_tokens=3))
    for bad in ({"max_tokens": CAP + 1}, {"max_completion_tokens": CAP + 1}, {"max_tokens": 0},
                {"max_tokens": True}, {"max_tokens": "5"}):  # fmt: skip
        with pytest.raises(ApiError) as e:
            prep(**bad)
        assert e.value.status == 400


def test_completions_default_and_prompt_counts():
    p = prep("completions", prompt="once")
    assert p.max_tokens == 16 and upstream(p)["max_tokens"] == 16
    p = prep("completions", prompt=["a", "b", "c"], max_tokens=10, n=2)
    assert p.sequences == 6
    assert p.max_cost_micros == 6 * max_cost_micros(QWEN, p.prompt_upper, 10, CTX)
    assert prep("completions", prompt=[1, 2, 3]).sequences == 1
    assert prep("completions", prompt=[[1, 2], [3]]).sequences == 2
    for bad in ([], [1, "a"], None, 5):
        with pytest.raises(ApiError):
            prep("completions", prompt=bad)
    with pytest.raises(ApiError):
        prep("completions", prompt=["a", "b"], stream=True)


def test_stream_options_are_forced_on_streams_and_dropped_otherwise():
    p = prep(stream=True)
    assert upstream(p)["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert not p.client_include_usage and not p.client_continuous_usage
    p = prep(stream=True, stream_options={"include_usage": True})
    assert p.client_include_usage and not p.client_continuous_usage
    p = prep(stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True})
    assert p.client_continuous_usage
    p = prep(stream_options={"include_usage": True})
    assert "stream_options" not in upstream(p) and not p.client_include_usage


def test_pass_through_fields_are_untouched():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    rf = {"type": "json_object"}
    p = prep(tools=tools, tool_choice="auto", response_format=rf, temperature=0.2, top_p=0.9,
             seed=7, logprobs=True, top_logprobs=2, stop=["\n"], user="u")  # fmt: skip
    up = upstream(p)
    assert up["tools"] == tools and up["response_format"] == rf
    for k, v in {"temperature": 0.2, "top_p": 0.9, "seed": 7, "stop": ["\n"]}.items():
        assert up[k] == v


@pytest.mark.parametrize(
    "bad",
    [
        {"messages": []},
        {"messages": "hi"},
        {"messages": ["hi"]},
        {"n": 0},
        {"n": 17},
        {"best_of": 3},
        {"use_beam_search": True},
        {"stream": "true"},
        {"stream_options": "x"},
    ],
)
def test_invalid_requests(bad):
    with pytest.raises(ApiError) as e:
        prep(**bad)
    assert e.value.status == 400 and e.value.type == "invalid_request_error"


def test_parse_json_object():
    assert parse_json_object(b'{"a": 1}') == {"a": 1}
    for bad in (b"", b"[1]", b"{", b"\xff\xfe"):
        with pytest.raises(ApiError):
            parse_json_object(bad)


def test_unicode_body_bound_holds():
    """The prompt bound counts UTF-8 bytes of the body as sent, not characters."""
    text = "日本語" * 100
    data = {"model": "m", "messages": [{"role": "user", "content": text}]}
    body = json.dumps(data, ensure_ascii=False).encode()
    p = prepare("chat", data, body_bytes=len(body), model_id="m", price=QWEN,
                max_context=CTX, max_tokens_cap=CAP)  # fmt: skip
    assert p.prompt_upper >= len(text.encode()) + 1024


def test_template_rendered_json_is_bounded_by_the_context():
    """`tools` reach the prompt through the template's `tojson`, which re-serialises
    them: `9e15` (4 bytes) comes out as `9000000000000000.0` (18 tokens on Qwen3, one per
    digit), far beyond body bytes + 1024. The bound for such requests is the context."""
    enum = ",".join(["9e15"] * 1000)
    raw = (
        '{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function",'
        '"function":{"name":"f","parameters":{"type":"number","enum":[' + enum + "]}}}]}"
    )
    data = json.loads(raw)
    rendered = json.dumps(data["tools"], ensure_ascii=False)  # what tojson renders
    assert len(rendered.encode()) > len(raw) + 1024  # the body-size bound would break

    def bound(d: dict) -> Prepared:
        body = json.dumps(d).encode()
        return prepare("chat", d, body_bytes=len(raw) if d is data else len(body),
                       model_id="m", price=QWEN, max_context=CTX, max_tokens_cap=CAP)  # fmt: skip

    p = bound(data)
    assert p.prompt_upper == CTX >= len(rendered.encode())
    assert p.max_cost_micros == max_cost_micros(QWEN, CTX, CAP, CTX)
    assert p.tpm_estimate == len(raw) + 1024  # rate limits keep the body-size estimate
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [call]}]
    assert bound({"model": "m", "messages": msgs}).prompt_upper == CTX
    for field in ("documents", "chat_template", "chat_template_kwargs", "functions"):
        assert bound({"model": "m", "messages": msgs[:1], field: [{"a": 1}]}).prompt_upper == CTX
    # Plain text and an empty tools list keep the tight bound.
    assert bound({"model": "m", "messages": msgs[:1], "tools": []}).prompt_upper < CTX


# ------------------------------------------------------------------ SSE


async def _collect(chunks):
    async def gen():
        for c in chunks:
            yield c

    return [p async for p in sse_payloads(gen())]


async def test_sse_payloads_split_across_reads():
    stream = b'data: {"a":1}\n\ndata: {"b":2}\r\n\r\n: comment\n\ndata: [DONE]\n\n'
    pieces = [stream[i : i + 5] for i in range(0, len(stream), 5)]
    assert await _collect(pieces) == [b'{"a":1}', b'{"b":2}', b"[DONE]"]
    assert await _collect([b"data:x\ndata:y\n\ndata: tail"]) == [b"x\ny", b"tail"]


def _chunk(content="x", usage=None, choices=True):
    obj = {"id": "c", "choices": [{"index": 0, "delta": {"content": content}}] if choices else []}
    if usage is not None:
        obj["usage"] = usage
    return json.dumps(obj).encode()


USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
FINAL = {**USAGE, "prompt_tokens_details": {"cached_tokens": 4}}


def _cost(u):
    return cost_micros(QWEN, *token_counts(u))


def test_rewriter_without_client_usage():
    rw = ChunkRewriter(include_usage=False, continuous_usage=False, cost_of=_cost)
    p = rw.process(_chunk(usage=USAGE))
    assert p.usage == USAGE and p.content
    assert "usage" not in json.loads(p.out[6:])
    final = rw.process(_chunk(usage=FINAL, choices=False))
    assert final.out is None and final.usage == FINAL
    done = rw.process(b"[DONE]")
    assert done.done and done.out == b"data: [DONE]\n\n"


def test_rewriter_with_include_usage():
    rw = ChunkRewriter(include_usage=True, continuous_usage=False, cost_of=_cost)
    p = rw.process(_chunk(usage=USAGE))
    assert json.loads(p.out[6:])["usage"] is None
    final = json.loads(rw.process(_chunk(usage=FINAL, choices=False)).out[6:])
    assert final["usage"]["cost_micros"] == cost_micros(QWEN, 10, 2, 4)
    assert final["usage"]["prompt_tokens_details"]["cached_tokens"] == 4


def test_rewriter_continuous_passes_raw_bytes():
    rw = ChunkRewriter(include_usage=True, continuous_usage=True, cost_of=_cost)
    raw = _chunk(usage=USAGE)
    assert rw.process(raw).out == b"data: " + raw + b"\n\n"


def test_rewriter_error_and_odd_payloads():
    rw = ChunkRewriter(include_usage=False, continuous_usage=False, cost_of=_cost)
    err = rw.process(b'{"error": {"message": "boom"}}')
    assert err.error and err.out is not None
    assert rw.process(b"not json").out == b"data: not json\n\n"
    assert rw.process(_chunk(usage=None)).usage is None


def test_token_counts_and_client_usage():
    assert token_counts(None) == (0, 0, 0)
    assert token_counts({"prompt_tokens": 5, "completion_tokens": 1,
                         "prompt_tokens_details": {"cached_tokens": 9}}) == (5, 1, 5)  # fmt: skip
    assert token_counts({"prompt_tokens": -1, "completion_tokens": True}) == (0, 0, 0)
    u = client_usage({"prompt_tokens": 5, "completion_tokens": 1}, 42)
    assert u["total_tokens"] == 6 and u["prompt_tokens_details"] == {"cached_tokens": 0}
    assert u["cost_micros"] == 42


# ------------------------------------------------------------------ routing, headers, errors


def _router():
    return Router(None, GatewaySettings(replica_penalty_s=60), GatewayMetrics())  # type: ignore[arg-type]


def test_replica_choice_least_inflight_then_order_then_penalty():
    r = _router()
    a, b, c = (Replica(x, f"https://{x}", "s", "t") for x in "abc")
    pool = PoolState("m", "ready", 0, (a, b, c))
    assert r.pick(pool, set()) == a  # ties go to the first listed
    r.acquire(a)
    assert r.pick(pool, set()) == b
    r.acquire(b)
    assert r.pick(pool, set()) == c
    r.release(a, failed=True)  # a: 0 in flight but penalised
    r.acquire(c)
    assert r.pick(pool, set()) == b  # b and c have 1 each; b listed first
    assert r.pick(pool, {"b", "c"}) == a  # a penalised one is still used as a last resort
    assert r.pick(pool, {"a", "b", "c"}) is None


def test_cold_start_estimates():
    r = _router()
    assert r.cold_start_estimate("qwen3-8b") == 240
    assert r.cold_start_estimate("llama-3.3-70b-instruct-fp8") == 450
    assert r.cold_start_estimate("unknown") == 300


def test_cost_headers():
    assert cost_headers(261) == {"X-Loom-Cost-Micros": "261", "X-Loom-Cost-USD": "0.000261"}
    assert cost_headers(12_345_678)["X-Loom-Cost-USD"] == "12.345678"
    assert cost_headers(0)["X-Loom-Cost-USD"] == "0.000000"


def test_engine_error_mapping():
    e = _engine_error(
        400, b'{"error": {"message": "maximum context length is 32768", "param": "messages"}}'
    )
    assert (e.status, e.code, e.param) == (400, "context_length_exceeded", "messages")
    e = _engine_error(422, b"garbage")
    assert e.status == 400 and e.code == "invalid_request"
    e = _engine_error(400, b'{"error": "plain"}')
    assert e.message == "plain"


def test_settings_cors_list():
    s = GatewaySettings(cors_origins="http://a, http://b ,")
    assert s.cors_origin_list == ["http://a", "http://b"]
    with pytest.raises(ValueError):
        GatewaySettings(auth_local_ttl_s=2)  # would break the 5 s revocation bound


# ------------------------------------------------------------------ logs


def test_json_log_lines():
    rec = logging.LogRecord(
        "loom_gateway.request", logging.INFO, __file__, 1, "request", None, None
    )
    rec.fields = {"request_id": "r1", "cost_micros": 5}
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "request" and out["request_id"] == "r1" and out["cost_micros"] == 5
    assert out["level"] == "info" and out["ts"].endswith("+00:00")


# ------------------------------------------------------------------ LOOM_ENV


PROD_OK = {
    "env": "prod",
    "database_url": "postgresql://loom:s3cret@db:5432/loom_platform",
    "pod_ca_cert": "/certs/ca.crt",
    "gateway_client_cert": "/certs/gateway.crt",
    "gateway_client_key": "/certs/gateway.key",
    "cors_origins": "https://loom.example",
}


@pytest.mark.parametrize(
    ("unset", "problem"),
    [
        ("pod_ca_cert", "LOOM_POD_CA_CERT"),
        ("gateway_client_cert", "LOOM_GATEWAY_CLIENT_CERT"),
        ("gateway_client_key", "LOOM_GATEWAY_CLIENT_CERT"),
        ("cors_origins", "LOOM_CORS_ORIGINS"),
        ("database_url", "LOOM_DATABASE_URL"),
    ],
)
def test_gateway_refuses_dev_defaults_outside_dev(unset, problem):
    assert GatewaySettings(**PROD_OK).production_problems() == []
    kw = {k: v for k, v in PROD_OK.items() if k != unset}
    [found] = GatewaySettings(**kw).production_problems()
    assert problem in found
    assert GatewaySettings(**{**kw, "env": "dev"}).production_problems() == []


def test_gateway_main_exits_before_serving_on_dev_defaults(monkeypatch, capsys):
    import loom_gateway.__main__ as entry

    for k in ("LOOM_ENV", "LOOM_POD_CA_CERT", "LOOM_DATABASE_URL", "LOOM_CORS_ORIGINS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(entry, "create_app", lambda s: pytest.fail("started"))
    with pytest.raises(SystemExit) as e:
        entry.main()
    assert e.value.code == 2
    assert "LOOM_POD_CA_CERT" in capsys.readouterr().err


def test_router_never_picks_a_plain_http_replica_outside_dev():
    http, https = Replica("h", "http://h:8000", "s", "t"), Replica("s", "https://s", "s", "t")
    pool = PoolState("m", "ready", 0, (http, https))
    prod = Router(None, GatewaySettings(**PROD_OK), GatewayMetrics())  # type: ignore[arg-type]
    assert prod.pick(pool, set()) == https
    assert prod.pick(pool, {"s"}) is None
    dev = Router(None, GatewaySettings(env="dev"), GatewayMetrics())  # type: ignore[arg-type]
    assert dev.pick(pool, set()) == http
