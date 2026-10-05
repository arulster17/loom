"""Stands in for `vllm bench serve` / `python -m sglang.benchmark.serving` in tests.

Run as `python fake_bench_tool.py <tool argv...>` (the tests' `command_prefix`).
Copies the fixture named by FAKE_BENCH_FIXTURE to the path after
`--result-filename` or `--output-file`, and writes its argv, OPENAI_API_KEY and
the `--dataset-path` contents to FAKE_BENCH_LOG. FAKE_BENCH_EXIT=<code> makes it
print a traceback to stderr and exit with that code instead.
"""

import json
import os
import shutil
import sys


def main() -> int:
    argv = sys.argv[1:]
    dataset = None
    if "--dataset-path" in argv:
        with open(argv[argv.index("--dataset-path") + 1]) as f:
            dataset = f.read()
    if log := os.environ.get("FAKE_BENCH_LOG"):
        with open(log, "w") as f:
            json.dump(
                {"argv": argv, "api_key": os.environ.get("OPENAI_API_KEY"), "dataset": dataset}, f
            )
    print("Starting main benchmark run...")
    if code := int(os.environ.get("FAKE_BENCH_EXIT", "0")):
        print("Traceback (most recent call last):", file=sys.stderr)
        print("ValueError: Initial test run failed", file=sys.stderr)
        return code
    flag = "--result-filename" if "--result-filename" in argv else "--output-file"
    shutil.copyfile(os.environ["FAKE_BENCH_FIXTURE"], argv[argv.index(flag) + 1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
