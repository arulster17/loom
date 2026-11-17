"""Find tests with mixed results across repeated runs (plan R15).

Reads pytest JUnit XML files (one repeated run from pytest-repeat, or several runs of the
same suite), groups every execution by test, and prints a Markdown report. A test is
flaky when at least one execution passed and at least one failed (a failure or an error);
skips count as neither. Exit status: 1 when any test is flaky, else 2 when a file could
not be read, else 0.

    python3 scripts/flaky_report.py --repeat-count 2 flake-*.xml    # CI job flake-check
    python3 scripts/flaky_report.py junit-1.xml junit-2.xml ...      # workflow flake-hunt

`--repeat-count N` strips pytest-repeat's `[i-N]` / `-i-N]` suffix from test names so the
copies of a test count as one test (across all the files given). Stdlib only: it runs on
the runner's own python3.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Test:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return self.passed + self.failed


def strip_repeat(name: str, count: int | None) -> str:
    """`t[3-20]` -> `t`, `t[a-3-20]` -> `t[a]` (only for the given repeat count). Also drops
    the `@<group>` that xdist's `--dist loadgroup` appends to `xdist_group` tests."""
    name = re.sub(r"@[^\[\]@]*$", "", name)
    if count is None:
        return name
    name = re.sub(rf"\[\d+-{count}\]$", "", name)
    return re.sub(rf"-\d+-{count}\]$", "]", name)


def first_line(text: str, limit: int = 200) -> str:
    line = next((s.strip() for s in text.splitlines() if s.strip()), "")
    line = line.replace("`", "'").replace("|", "/")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def collect(paths: list[Path], repeat_count: int | None) -> tuple[dict[str, Test], list[str]]:
    tests: dict[str, Test] = defaultdict(Test)
    problems: list[str] = []
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as e:
            problems.append(f"{path}: unreadable ({e})")
            continue
        for case in root.iter("testcase"):
            name = case.get("name")
            if not name:  # a test still running when the run was interrupted
                continue
            key = f"{case.get('classname', '')}::{strip_repeat(name, repeat_count)}"
            test = tests[key]
            bad = [c for c in case if c.tag in ("failure", "error")]
            if bad:
                test.failed += 1
                if len(test.messages) < 3:
                    test.messages.append(first_line(bad[0].get("message") or bad[0].text or ""))
            elif any(c.tag == "skipped" for c in case):
                test.skipped += 1
            else:
                test.passed += 1
    return dict(tests), problems


def report(
    tests: dict[str, Test], problems: list[str], expect_runs: int | None, title: str
) -> tuple[str, list[str]]:
    flaky = sorted(k for k, t in tests.items() if t.passed and t.failed)
    always = sorted(k for k, t in tests.items() if t.failed and not t.passed)
    ran = [t.runs for t in tests.values() if t.runs]
    out = [f"## {title}", ""]
    for p in problems:
        out.append(f"- :warning: {p}")
    if not ran:
        out.append("No test executions found.")
        return "\n".join(out) + "\n", flaky
    out.append(
        f"{len(ran)} tests, {sum(ran)} executions (each test ran {min(ran)} to {max(ran)}"
        f" times). **{len(flaky)} flaky** (passed and failed), {len(always)} failed every run."
    )
    if expect_runs is not None and min(ran) < expect_runs:
        short = sum(1 for r in ran if r < expect_runs)
        out.append(
            f"\n:warning: {short} tests ran fewer than {expect_runs} times (the run hit its"
            " time limit): a pass here is weaker evidence for them."
        )
    for heading, keys in (("Flaky: passed and failed", flaky), ("Failed every run", always)):
        if not keys:
            continue
        out += ["", f"### {heading}", ""]
        for k in keys:
            t = tests[k]
            out.append(f"- `{k}`: {t.passed} passed, {t.failed} failed")
            out += [f"  - `{m}`" for m in t.messages if m]
    return "\n".join(out) + "\n", flaky


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("junit", nargs="+", type=Path, help="pytest --junitxml files")
    p.add_argument("--repeat-count", type=int, help="pytest-repeat's --count, to merge copies")
    p.add_argument("--expect-runs", type=int, help="warn about tests that ran fewer times")
    p.add_argument("--title", default="Flaky tests")
    args = p.parse_args(argv)
    tests, problems = collect(args.junit, args.repeat_count)
    text, flaky = report(tests, problems, args.expect_runs, args.title)
    sys.stdout.write(text)
    if flaky:
        return 1
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
