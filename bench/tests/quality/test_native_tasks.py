import json

import httpx
import pytest

from loom_bench.quality.client import EvalClient, ToolCall
from loom_bench.quality.tasks import TASKS, build_task
from loom_bench.quality.tasks.base import EvalContext
from loom_bench.quality.tasks.json_schema import load_items, score_reply
from loom_bench.quality.tasks.needle import NeedleParams, found, generate_items
from loom_bench.quality.tasks.tool_calling import ToolItem, load_data, match_call, values_equal


def fake_client(handler):
    return EvalClient("http://t/v1", "m", transport=httpx.MockTransport(handler), max_retries=0)


def reply(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}]})


def test_registry_knows_native_tasks():
    assert {"toy_arithmetic", "json_schema", "tool_calling", "needle"} <= set(TASKS)
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
    assert out.version == "1+data.1"


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
