import json
import sys
import time

import httpx
import pytest

from loom_bench.quality.client import EvalClient
from loom_bench.quality.sandbox import CodeExecDisabled, SandboxLimits, run_python
from loom_bench.quality.tasks.base import EvalContext
from loom_bench.quality.tasks.code_exec import (
    CodeExecParams,
    CodeExecTask,
    build_program,
    extract_code,
    humaneval_problems,
    mbpp_problems,
)

FAST = SandboxLimits(timeout_s=2.0, cpu_s=2)

# Tiny inline fixtures in the shape of the HF rows; no network.
HUMANEVAL_ROWS = [
    {
        "task_id": "Fixture/0",
        "prompt": 'from typing import List\n\n\ndef total(xs: List[int]) -> int:\n    """Sum."""\n',
        "test": (
            "def check(candidate):\n"
            "    assert candidate([1, 2, 3]) == 6\n"
            "    assert candidate([]) == 0\n"
        ),
        "entry_point": "total",
    },
    {
        "task_id": "Fixture/1",
        "prompt": 'def double(x: int) -> int:\n    """Twice x."""\n',
        "test": "def check(candidate):\n    assert candidate(4) == 8\n",
        "entry_point": "double",
    },
]
MBPP_ROWS = [
    {
        "task_id": 7,
        "text": "Write a function to reverse a string.",
        "test_list": ["assert rev('abc') == 'cba'", "assert rev('') == ''"],
        "test_setup_code": "",
    }
]


def test_refuses_unless_enabled():
    with pytest.raises(CodeExecDisabled):
        run_python("print(1)", allow_code_exec=False)


def test_passes_and_fails():
    assert run_python("assert 1 + 1 == 2", allow_code_exec=True, limits=FAST).status == "passed"
    res = run_python("assert 1 + 1 == 3", allow_code_exec=True, limits=FAST)
    assert res.status == "failed" and res.error_type == "AssertionError"
    res = run_python("def f(:\n  pass", allow_code_exec=True, limits=FAST)
    assert res.status == "failed" and res.error_type == "SyntaxError"


def test_timeout_is_enforced():
    t0 = time.monotonic()
    res = run_python(
        "import time\nwhile True:\n    time.sleep(0.01)",
        allow_code_exec=True,
        limits=SandboxLimits(timeout_s=0.5, cpu_s=5),
    )
    assert res.status == "timeout"
    assert time.monotonic() - t0 < 5


def test_cpu_limit_is_enforced():
    res = run_python(
        "while True:\n    pass", allow_code_exec=True, limits=SandboxLimits(timeout_s=20, cpu_s=1)
    )
    assert res.status == "timeout"
    assert res.duration_s < 10


def test_empty_env_and_temp_cwd():
    program = (
        "import os, tempfile\n"
        "assert os.environ.get('HOME') is None and os.environ.get('PATH') is None\n"
        "assert 'loom-sandbox-' in os.getcwd()\n"
        "assert os.listdir('.') == ['program.py']\n"
    )
    assert run_python(program, allow_code_exec=True, limits=FAST).status == "passed"


def test_file_size_limit():
    program = "open('big', 'wb').write(b'x' * (3 * 2**20))"
    limits = SandboxLimits(timeout_s=5, cpu_s=5, max_file_mb=1)
    assert run_python(program, allow_code_exec=True, limits=limits).status == "failed"


def test_no_subprocesses():
    program = "import os\npid = os.fork()\nif pid == 0:\n    os._exit(0)\n"
    res = run_python(program, allow_code_exec=True, limits=FAST)
    assert res.status == "failed" and res.error_type in {"BlockingIOError", "OSError"}


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only enforced on Linux")
def test_memory_limit_on_linux():
    program = "x = bytearray(512 * 2**20)"
    limits = SandboxLimits(timeout_s=5, cpu_s=5, memory_mb=256)
    res = run_python(program, allow_code_exec=True, limits=limits)
    assert res.status == "failed" and res.error_type == "MemoryError"


def test_extract_code():
    assert extract_code("Here:\n```python\ndef f():\n    return 1\n```\nDone") == (
        "def f():\n    return 1\n"
    )
    assert extract_code("```\nx = 1\n```") == "x = 1\n"
    assert extract_code("x = 2") == "x = 2"


def test_programs_from_fixture_rows_run():
    he = humaneval_problems(HUMANEVAL_ROWS)
    full_function = "```python\ndef total(xs):\n    return sum(xs)\n```"
    body_only = "    return sum(xs)\n"
    for reply in (full_function, body_only):
        prog = build_program(he[0], reply)
        assert run_python(prog, allow_code_exec=True, limits=FAST).status == "passed"
    wrong = build_program(he[0], "```python\ndef total(xs):\n    return 0\n```")
    assert run_python(wrong, allow_code_exec=True, limits=FAST).status == "failed"
    mb = mbpp_problems(MBPP_ROWS)[0]
    assert mb.item_id == "mbpp/7" and "assert rev('abc')" in mb.prompt
    prog = build_program(mb, "```python\ndef rev(s):\n    return s[::-1]\n```")
    assert run_python(prog, allow_code_exec=True, limits=FAST).status == "passed"


def _fake_client():
    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        code = "def total(xs):\n    return sum(xs)" if "total" in prompt else "def double(x): 0"
        message = {"role": "assistant", "content": f"```python\n{code}\n```"}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    return EvalClient("http://t/v1", "m", transport=httpx.MockTransport(handler), max_retries=0)


async def test_task_requires_explicit_enable(tmp_path):
    task = CodeExecTask(
        "he", CodeExecParams(dataset="humaneval"), humaneval_problems(HUMANEVAL_ROWS)
    )
    async with _fake_client() as c:
        with pytest.raises(CodeExecDisabled):
            await task.run(EvalContext(client=c, workdir=tmp_path))


async def test_task_scores_fixture(tmp_path):
    params = CodeExecParams(dataset="humaneval", sandbox=FAST)
    task = CodeExecTask("he", params, humaneval_problems(HUMANEVAL_ROWS))
    async with _fake_client() as c:
        out = await task.run(EvalContext(client=c, workdir=tmp_path, allow_code_exec=True))
    assert [(r.item_id, r.score) for r in out.items] == [("Fixture/0", 1.0), ("Fixture/1", 0.0)]
    assert out.items[1].meta["status"] == "failed"
    assert out.provenance["dataset"]["license"] == "MIT"
    assert out.provenance["dataset"]["revision"] == "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
