import json
import re

import httpx

from loom_bench.quality.client import EvalClient
from loom_bench.quality.tasks import build_task
from loom_bench.quality.tasks.arithmetic import ArithmeticParams, generate_items, parse_answer
from loom_bench.quality.tasks.base import EvalContext


def test_parse_answer():
    assert parse_answer("The answer is 42.") == 42
    assert parse_answer("Let me think. the answer is -7") == -7
    assert parse_answer("The answer is 3. No wait, the answer is 5.") == 5
    assert parse_answer("42") is None


def test_items_are_deterministic():
    p = ArithmeticParams(n=50, seed=3)
    a, b = generate_items(p), generate_items(p)
    assert a == b
    assert generate_items(ArithmeticParams(n=50, seed=4)) != a
    assert len({i.item_id for i in a}) == 50
    item = a[0]
    assert item.question == f"What is {item.a} {item.op} {item.b}?"


async def test_scores_against_fake_server(tmp_path):
    def handler(request):
        content = json.loads(request.content)["messages"][0]["content"]
        a, op, b = re.match(r"What is (\d+) (.) (\d+)\?", content).groups()
        value = eval(f"{a}{op}{b}")
        if int(a) % 2:
            value += 1  # odd first operand -> wrong answer
        if int(a) < 100:
            return httpx.Response(500)
        message = {"role": "assistant", "content": f"The answer is {value}."}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    task = build_task("toy_arithmetic", "arith", {"n": 60, "seed": 1})
    client = EvalClient("http://t/v1", "m", transport=httpx.MockTransport(handler), max_retries=0)
    async with client:
        out = await task.run(EvalContext(client=client, workdir=tmp_path))
    items = generate_items(ArithmeticParams(n=60, seed=1))
    by_id = {r.item_id: r for r in out.items}
    failed = [i for i in items if i.a < 100]
    assert failed, "fixture should exercise failed requests"
    for item in items:
        result = by_id[item.item_id]
        if item.a < 100:
            assert result.score == 0.0 and "HTTP 500" in result.meta["error"]
        else:
            assert result.score == float(item.a % 2 == 0)
    assert len(out.completions) == len(items) - len(failed)
    assert all(r.content_hash for r in out.items)
