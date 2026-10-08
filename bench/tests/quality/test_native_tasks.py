import json

import httpx
import pytest

from loom_bench.quality.client import EvalClient, ToolCall
from loom_bench.quality.tasks import TASKS, build_task
from loom_bench.quality.tasks.base import SAMPLE_CHARS, EvalContext, clip
from loom_bench.quality.tasks.json_schema import failure_meta as json_failure_meta
from loom_bench.quality.tasks.json_schema import load_items, score_reply
from loom_bench.quality.tasks.needle import NeedleParams, found, generate_items
from loom_bench.quality.tasks.tool_calling import (
    ToolCallingStrictTask,
    ToolCallingTask,
    ToolItem,
    check_call,
    load_data,
    match_call,
    strict_parameters,
    values_equal,
)


def fake_client(handler):
    return EvalClient("http://t/v1", "m", transport=httpx.MockTransport(handler), max_retries=0)


def reply(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}]})


def test_registry_knows_native_tasks():
    assert {"toy_arithmetic", "json_schema", "tool_calling", "needle"} <= set(TASKS)
    assert TASKS["tool_calling_strict"] == ToolCallingStrictTask.from_params
    with pytest.raises(ValueError, match="unknown eval task kind"):
        build_task("nope", "x")
    with pytest.raises(ValueError):
        build_task("needle", "x", {"bogus": 1})


# ---------------------------------------------------------------- json_schema

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["n"],
    "properties": {"n": {"type": "integer", "minimum": 0}},
}


def test_json_score_reply():
    assert score_reply('{"n": 3}', SCHEMA) == (1.0, "valid")
    assert score_reply('{"n": -1}', SCHEMA) == (0.0, "schema_violation")
    assert score_reply('{"n": 3, "x": 1}', SCHEMA) == (0.0, "schema_violation")
    assert score_reply('{"n": 3', SCHEMA) == (0.0, "invalid_json")
    assert score_reply("", SCHEMA) == (0.0, "invalid_json")


def test_json_pinned_set_is_valid():
    items, version = load_items()
    assert len(items) >= 50 and version >= 1


# The 60 items of data version 1, in order. Version 2 appended to them; changing or
# reordering these would silently change what older results measured.
JSON_V1_IDS = (
    "person-basic book-record weather-report todo-list recipe-card color-rgb sentiment "
    "address meeting product planet-facts flight quiz-question invoice-line tags bug-report "
    "coordinates movie-review employee unit-conversion nested-company http-response "
    "chess-move workout language-detect semver shopping-cart timezone git-commit animal "
    "matrix event-ticket country support-ticket grade-book rgb-palette date-parts "
    "server-config math-answer poem-meta inventory user-profile train-schedule "
    "classification-multi key-value optional-fields tree nutrition rating-breakdown "
    "anyof-contact boolean-flags sql-intent dice-roll citation parcel-status geometry "
    "playlist thermostat word-stats access-policy"
).split()


def _keywords(schema):
    if isinstance(schema, dict):
        for key, value in schema.items():
            yield key
            if key == "properties":
                for sub in value.values():
                    yield from _keywords(sub)
            else:
                yield from _keywords(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _keywords(value)


def test_json_set_v2_keeps_v1_and_grows_to_300():
    items, version = load_items()
    assert version == 2 and len(items) == 300
    assert [i.item_id for i in items[:60]] == JSON_V1_IDS
    # Neither vLLM's nor SGLang's grammar backend implements uniqueItems (HTTP 400 on both
    # in b1b904dc): only the two v1 items keep it, well under the eval error guard's 10%.
    unique = {i.item_id for i in items if "uniqueItems" in set(_keywords(i.schema))}
    assert unique == {"tags", "classification-multi"}
    # Every required key is declared (YAML 1.1 reads a bare `on` as True, which this catches).
    for item in items:
        for node in _objects(item.schema):
            assert set(node.get("required", [])) <= set(node.get("properties", {})), item.item_id


def _objects(schema):
    if isinstance(schema, dict):
        if "properties" in schema:
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


async def test_json_task_requests_schema_and_scores(tmp_path):
    items, _ = load_items()
    first = items[0]
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["response_format"])
        if body["response_format"]["json_schema"]["name"] == first.item_id:
            return reply('{"name": "Mara Quist", "age": 41}')
        return reply("not json")

    async with fake_client(handler) as c:
        out = await build_task("json_schema", "js", {"limit": 5}).run(
            EvalContext(client=c, workdir=tmp_path)
        )
    assert [r.score for r in out.items] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert seen[0]["type"] == "json_schema" and seen[0]["json_schema"]["strict"] is True
    assert out.provenance["dataset"]["license"] == "Apache-2.0"
    assert out.version == "1+data.2"
    # A failed item keeps its raw reply so the score can be explained; a passing one does not.
    assert "output" not in out.items[0].meta
    assert out.items[1].meta == {"result": "invalid_json", "output": "not json"}


def test_json_failure_meta_names_the_violation():
    meta = json_failure_meta('{"n": -1}', SCHEMA, "schema_violation")
    assert meta["output"] == '{"n": -1}'
    assert meta["violation"].startswith("n: -1 is less than the minimum of 0")
    assert json_failure_meta('{"n": 3', SCHEMA, "invalid_json") == {"output": '{"n": 3'}
    assert "error" not in meta  # the request-error guard keys on `error`


def test_clip_caps_stored_outputs():
    assert clip("abc", 5) == "abc"
    assert clip("a" * 12, 5) == "aaaaa…[+7 chars]"
    assert len(clip("x" * 10_000)) < SAMPLE_CHARS + 20


# ---------------------------------------------------------------- tool_calling


def test_values_equal_type_coercion():
    assert values_equal(3, 3, {"type": "integer"})
    assert values_equal(3.0, 3, {"type": "integer"})
    assert not values_equal(3.5, 3, {"type": "integer"})
    assert not values_equal("3", 3, {"type": "integer"})
    assert not values_equal(True, 1, {"type": "integer"})
    assert values_equal(250, 250.0, {"type": "number"})
    assert values_equal(64.2, 64.2, {"type": "number"})
    assert not values_equal("64.2", 64.2, {"type": "number"})
    assert not values_equal(False, 0, {"type": "number"})
    assert values_equal("  New   York ", "new york", {"type": "string"})
    assert not values_equal("NewYork", "new york", {"type": "string"})
    assert values_equal(True, True, {"type": "boolean"})
    assert not values_equal("true", True, {"type": "boolean"})
    assert not values_equal(1, True, {"type": "boolean"})
    arr = {"type": "array", "items": {"type": "number"}}
    assert values_equal([8, 3.0], [8, 3], arr)
    assert not values_equal([3, 8], [8, 3], arr)
    assert not values_equal([8], [8, 3], arr)
    obj = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert values_equal({"a": 2.0}, {"a": 2}, obj)
    assert not values_equal({"a": 2, "b": 1}, {"a": 2}, obj)


TIP = load_data().functions["calculate_tip"]
TIP_ITEM = ToolItem(
    item_id="t",
    category="simple",
    query="q",
    functions=("calculate_tip",),
    expected_name="calculate_tip",
    expected_args={"bill_amount": [64.2], "tip_percent": [18], "split": [None, 1]},
)


@pytest.mark.parametrize(
    ("calls", "ok", "why"),
    [
        ([("calculate_tip", '{"bill_amount": 64.2, "tip_percent": 18}')], True, "match"),
        ([("calculate_tip", '{"bill_amount": 64.2, "tip_percent": 18.0, "split": 1}')], True, ""),
        ([("calculate_tip", '{"bill_amount": 64.2, "tip_percent": 18, "split": 2}')], False, ""),
        ([("calculate_tip", '{"bill_amount": "64.2", "tip_percent": 18}')], False, "wrong_value"),
        ([("calculate_tip", '{"bill_amount": 64.2}')], False, "missing_required_argument"),
        ([("calculate_tip", '{"bill_amount": 64.2, "tip_percent": 18, "x": 1}')], False, ""),
        ([("calculate_tip", "{bill_amount: 64.2")], False, "arguments_not_json"),
        ([("calculate_tip", "[1, 2]")], False, "arguments_not_object"),
        ([("get_weather", '{"city": "Porto"}')], False, "wrong_function"),
        ([], False, "expected 1 tool call, got 0"),
        ([("calculate_tip", '{"bill_amount": 64.2, "tip_percent": 18}')] * 2, False, ""),
    ],
)
def test_match_call(calls, ok, why):
    got_ok, got_why = match_call(tuple(ToolCall(n, a) for n, a in calls), TIP_ITEM, TIP)
    assert got_ok is ok
    if why:
        assert got_why == why


def test_match_call_requires_listed_value_when_omitted():
    item = ToolItem(
        "t", "simple", "q", ("calculate_tip",), "calculate_tip",
        {"bill_amount": [10], "tip_percent": [15], "split": [2]},
    )  # fmt: skip
    ok, why = match_call(
        (ToolCall("calculate_tip", '{"bill_amount": 10, "tip_percent": 15}'),), item, TIP
    )
    assert not ok and why == "missing_argument"


def test_tool_pinned_set_is_consistent():
    data = load_data()
    assert len(data.items) >= 50
    assert {i.category for i in data.items} == {"simple", "multiple"}
    assert all(len(i.functions) > 1 for i in data.items if i.category == "multiple")


async def test_tool_task_sends_tools_and_scores(tmp_path):
    data = load_data()
    target = data.items[0]  # s-weather-1

    def handler(request):
        body = json.loads(request.content)
        assert body["tool_choice"] == "auto"
        names = [t["function"]["name"] for t in body["tools"]]
        if body["messages"][0]["content"] == target.query:
            args = json.dumps({"city": "porto", "unit": "celsius"})
        else:
            args = "{}"
        call = {"id": "c", "type": "function", "function": {"name": names[0], "arguments": args}}
        return reply(None, [call], "tool_calls")

    async with fake_client(handler) as c:
        out = await build_task("tool_calling", "tc", {"categories": ["simple"]}).run(
            EvalContext(client=c, workdir=tmp_path)
        )
    by_id = {r.item_id: r for r in out.items}
    assert len(out.items) == sum(i.category == "simple" for i in data.items)
    assert by_id[target.item_id].score == 1.0
    assert sum(r.score for r in out.items) == 1.0
    assert by_id["s-timer-2"].meta["result"] == "missing_required_argument"
    assert by_id["s-timer-2"].meta["argument"] == "minutes"
    assert "output" not in by_id[target.item_id].meta


def test_check_call_names_the_failing_argument():
    def check(args):
        return check_call((ToolCall("calculate_tip", args),), TIP_ITEM, TIP)

    assert check('{"bill_amount": "64.2", "tip_percent": 18}').argument == "bill_amount"
    assert check('{"bill_amount": 64.2, "tip_percent": 18, "x": 1}').argument == "x"
    assert check('{"bill_amount": 64.2}').argument == "tip_percent"
    assert check('{"bill_amount": 64.2, "tip_percent": 18}').argument is None


async def test_tool_task_stores_failed_output_and_expectation(tmp_path):
    """A quoted number (the 70B run's suspected failure) is visible in the stored sample."""
    data = load_data()
    item = next(i for i in data.items if i.item_id == "s-currency-1")
    raw = '{"amount": "250", "from_currency": "USD", "to_currency": "EUR"}'

    def handler(request):
        body = json.loads(request.content)
        args = raw if body["messages"][0]["content"] == item.query else "{}"
        call = {"id": "c", "type": "function", "function": {"name": item.expected_name,
                "arguments": args}}  # fmt: skip
        return reply(None, [call], "tool_calls")

    async with fake_client(handler) as c:
        out = await build_task("tool_calling", "tc", {"categories": ["simple"]}).run(
            EvalContext(client=c, workdir=tmp_path)
        )
    meta = next(r for r in out.items if r.item_id == item.item_id).meta
    assert meta["result"] == "wrong_value" and meta["argument"] == "amount"
    assert meta["output"] == {
        "tool_calls": [{"name": "convert_currency", "arguments": raw}],
        "n_tool_calls": 1,
        "text": "",
    }
    assert meta["expected"] == {
        "name": "convert_currency",
        "arguments": {"amount": [250], "from_currency": ["USD"], "to_currency": ["EUR"]},
    }
    assert "error" not in meta  # stored outputs never trip the request-error guard


# ---------------------------------------------------------------- tool_calling_strict


def test_strict_parameters_closes_every_object():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "object", "properties": {"c": {"type": "string"}}},
            "d": {"type": "array", "items": {"type": "object", "properties": {}}},
        },
        "required": ["a"],
    }
    out = strict_parameters(schema)
    assert out["additionalProperties"] is False
    assert out["properties"]["b"]["additionalProperties"] is False
    assert out["properties"]["d"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in out["properties"]["a"]
    assert out["required"] == ["a"]  # required lists are not widened
    assert "additionalProperties" not in schema  # the pinned schema is not mutated


def test_strict_tools_are_the_plain_tools_plus_strict_mode():
    data = load_data()
    for item in data.items:
        plain, strict = data.tools_for(item), data.tools_for(item, strict=True)
        assert len(plain) == len(strict)
        for p, t in zip(plain, strict, strict=True):
            assert "strict" not in p["function"]  # the headline task sends plain tools
            assert "additionalProperties" not in p["function"]["parameters"]
            assert t["type"] == "function" and t["function"]["strict"] is True
            params = t["function"]["parameters"]
            assert params["additionalProperties"] is False
            assert params["properties"] == p["function"]["parameters"]["properties"]
            assert params.get("required") == p["function"]["parameters"].get("required")
            rest = {k: v for k, v in t["function"].items() if k not in ("strict", "parameters")}
            assert rest == {k: v for k, v in p["function"].items() if k != "parameters"}


def _tool_handler(seen, answers):
    """Answers each query with its scripted call (else empty arguments); records bodies."""

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        query = body["messages"][0]["content"]
        name, args = answers.get(query, (body["tools"][0]["function"]["name"], "{}"))
        call = {"id": "c", "type": "function", "function": {"name": name, "arguments": args}}
        return reply(None, [call], "tool_calls")

    return handler


async def test_strict_task_sends_strict_tools_and_scores_like_tool_calling(tmp_path):
    data = load_data()
    by_id = {i.item_id: i for i in data.items}
    # A right call, a quoted number (what Llama 3.3 sends unconstrained) and a missing
    # required argument: both tasks must score each one the same way.
    answers = {
        by_id["s-weather-1"].query: ("get_weather", '{"city": "porto", "unit": "celsius"}'),
        by_id["s-currency-1"].query: (
            "convert_currency",
            '{"amount": "250", "from_currency": "USD", "to_currency": "EUR"}',
        ),
    }
    outs, bodies = {}, {}
    for kind in ("tool_calling", "tool_calling_strict"):
        seen: list[dict] = []
        async with fake_client(_tool_handler(seen, answers)) as c:
            outs[kind] = await build_task(kind, kind, {"categories": ["simple"]}).run(
                EvalContext(client=c, workdir=tmp_path)
            )
        bodies[kind] = seen

    for body in bodies["tool_calling_strict"]:
        assert body["tool_choice"] == "auto"
        for tool in body["tools"]:
            assert tool["function"]["strict"] is True
            assert tool["function"]["parameters"]["additionalProperties"] is False
    for body in bodies["tool_calling"]:
        assert all("strict" not in t["function"] for t in body["tools"])

    plain, strict = outs["tool_calling"], outs["tool_calling_strict"]
    assert [r.item_id for r in plain.items] == [r.item_id for r in strict.items]
    for p, t in zip(plain.items, strict.items, strict=True):
        assert (p.score, p.meta) == (t.score, t.meta)
        assert p.content_hash != t.content_hash  # the strict request is other content
    got = {r.item_id: r for r in strict.items}
    assert got["s-weather-1"].score == 1.0
    assert got["s-currency-1"].meta["result"] == "wrong_value"
    assert got["s-timer-2"].meta["result"] == "missing_required_argument"
    assert sum(r.score for r in strict.items) == 1.0


async def test_strict_task_is_versioned_apart_from_tool_calling(tmp_path):
    data = load_data()
    assert ToolCallingTask.version == "1"  # the headline task's version is unchanged
    outs = {}
    for kind in ("tool_calling", "tool_calling_strict"):
        async with fake_client(_tool_handler([], {})) as c:
            outs[kind] = await build_task(kind, kind, {"limit": 1}).run(
                EvalContext(client=c, workdir=tmp_path)
            )
    assert outs["tool_calling"].version == f"1+data.{data.version}"
    assert outs["tool_calling_strict"].version == f"strict.1+data.{data.version}"
    assert "tools" not in outs["tool_calling"].provenance
    assert outs["tool_calling_strict"].provenance["tools"] == (
        "strict: true, additionalProperties: false"
    )
    plain, strict = outs["tool_calling"], outs["tool_calling_strict"]
    assert strict.provenance["dataset"] == plain.provenance["dataset"]
    assert build_task("tool_calling_strict", "x", {}).planned_items() == len(data.items)


# ---------------------------------------------------------------- needle


def test_needle_items_deterministic_and_sized():
    p = NeedleParams(context_tokens=(1024, 4096), depths=(0.0, 0.5, 1.0), samples_per_cell=2)
    a, b = generate_items(p), generate_items(p)
    assert a == b and len(a) == 12
    for item in a:
        assert len(item.prompt) <= item.context_tokens * p.chars_per_token
        assert len(item.prompt) > 0.9 * item.context_tokens * p.chars_per_token
        assert item.prompt.count(item.code) == 1
        assert f"code for the {item.key} vault is {item.code}." in item.prompt
    start = a[0].prompt
    needle = f"The access code for the {a[0].key} vault"
    assert start.index(needle) < 100
    end = next(i for i in a if i.depth == 1.0)
    assert end.prompt.index(f"The access code for the {end.key} vault") > 0.9 * len(end.prompt)


def test_needle_found():
    assert found("The code is 004213.", "004213")
    assert found("004213", "004213")
    assert not found("10042131", "004213")
    assert not found("I don't know", "004213")


async def test_needle_task_scores(tmp_path):
    p = {"context_tokens": [512], "depths": [0.5], "samples_per_cell": 4}
    items = generate_items(NeedleParams(**p))
    codes = {i.prompt: i.code for i in items}

    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        code = codes[prompt]
        return reply(code if int(code) % 2 == 0 else "000000x")

    async with fake_client(handler) as c:
        out = await build_task("needle", "nd", p).run(EvalContext(client=c, workdir=tmp_path))
    assert [r.score for r in out.items] == [float(int(i.code) % 2 == 0) for i in items]
