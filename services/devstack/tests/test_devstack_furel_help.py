"""`loom-deploy acceptance` refuses a help option after `--` (FU2; M3 spec §6.12 "Help after
`--`", §13 A12).

`loom-m3-acceptance` prints its help and exits 0 without writing a report when its argparse
reads a help option, so `loom-deploy acceptance` refuses (exit 2) a run in which any word
after the first `--` is `--help` or begins with `-h`, before the tmpfs and marker checks:
nothing runs (no docker call), nothing is logged or written, and an older `report.json` and
`host.json` stay. The message names the word and points at `loom-m3-acceptance --help`.
Words argparse does not read as help pass through to the container.

Against the fakes of `devstack_m3_kit.py` (Linux only, like the other deploy tests).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from devstack_m3_kit import Box, deploy_root_base, mount_fstype, runs, site

from loom_devstack import deploy
from loom_devstack import m3_acceptance as m3

SHA_A = "4d" * 20
SHA_B = "5e" * 20
POINTER = "loom-m3-acceptance --help"


def assert_exit(proc: Any, code: int) -> None:
    """`loom-deploy` ran (an import error also exits 1) and exited with `code`."""
    assert "No module named" not in proc.stderr, ran(proc)
    assert proc.returncode == code, ran(proc)


def ran(proc: Any) -> str:
    out, err = proc.stdout[-3000:], proc.stderr[-3000:]
    return f"exit {proc.returncode}\n--- stdout\n{out}\n--- stderr\n{err}"


@pytest.fixture
def box() -> Iterator[Box]:
    b = Box(deploy_root_base(), site_doc=site())
    yield b
    b.close()


def applied(box: Box) -> None:
    box.publish(SHA_A)
    proc = box.deploy("apply", "--release", SHA_A)
    assert proc.returncode == 0, f"applying release A first: {ran(proc)}"


def acceptance(box: Box, *words: str) -> Any:
    return box.deploy("acceptance", "--p4-release", SHA_B, "--", *words)


def names_word(text: str, word: str) -> bool:
    """`word` appears as a word of its own, apart from the `loom-m3-acceptance --help`
    pointer."""
    rest = text.replace(POINTER, "")
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", rest) is not None


def old_files(box: Box) -> dict[Path, bytes]:
    acc = box.var_lib / "acceptance"
    acc.mkdir(parents=True, exist_ok=True)
    files = {
        acc / "report.json": b'{"items": [], "note": "an older run"}\n',
        acc / "host.json": b'{"m3-p": {"ok": true, "detail": "an older run"}}\n',
    }
    for p, data in files.items():
        p.write_bytes(data)
    return files


@pytest.mark.parametrize(
    "words, word",
    [
        pytest.param(["-h"], "-h", id="-h"),
        pytest.param(["--help"], "--help", id="--help"),
        pytest.param(["-hh"], "-hh", id="-hh"),
        pytest.param(["-hx"], "-hx", id="-hx"),
        pytest.param(["--only", "m3-0", "--help"], "--help", id="only-m3-0-then-help"),
        pytest.param(["--only", "m3-p", "-h"], "-h", id="only-m3-p-then-h"),
        pytest.param(["--external", "-h"], "-h", id="external-then-h"),
        pytest.param(["--", "-h"], "-h", id="second-dashdash-then-h"),
    ],
)
def test_help_after_dashdash_is_refused(box: Box, words: list[str], word: str) -> None:
    applied(box)
    files = old_files(box)
    n_log, n_tl = len(box.log_records()), len(box.timeline())
    proc = acceptance(box, *words)
    assert_exit(proc, 2)
    assert names_word(proc.stderr, word), f"the refusal does not name {word}: {ran(proc)}"
    assert POINTER in proc.stderr, ran(proc)
    docker = [r["argv"] for r in box.timeline()[n_tl:] if r["tool"] == "docker"]
    assert not docker, f"a refused acceptance called docker: {docker}"
    assert box.log_records()[n_log:] == [], "a refused acceptance logged a record"
    for p, data in files.items():
        assert p.is_file() and p.read_bytes() == data, f"{p.name} changed"


@pytest.mark.parametrize("case", ["no-release", "not-tmpfs", "marker-set"])
def test_help_is_refused_before_the_box_is_checked(tmp_path: Path, case: str) -> None:
    """The help check comes before the tmpfs, marker and `release.json` checks."""
    if case == "not-tmpfs":
        root = tmp_path / "root"
        root.mkdir()
        if mount_fstype(root) == "tmpfs":
            pytest.skip("tmp_path is on a tmpfs here; the case needs a disk directory")
        b = Box(deploy_root_base(), root=root, site_doc=site())
    else:
        b = Box(deploy_root_base(), site_doc=site())
    try:
        if case == "marker-set":
            applied(b)
            doc = {"release": SHA_B, "only": None, "started_at": "2026-11-20T00:00:00+00:00"}
            (b.var_lib / "apply-in-progress.json").write_text(json.dumps(doc) + "\n")
        proc = acceptance(b, "--help")
        assert_exit(proc, 2)
        assert POINTER in proc.stderr, ran(proc)
        for other in ("no current release", "tmpfs", "apply-in-progress"):
            assert other not in proc.stderr, f"{other!r} came before the help check: {ran(proc)}"
    finally:
        b.close()


WORDS = [
    "-h", "--help", "-hh", "-hhh", "-hx", "-hH", "-xh", "-h=x", "-h=h", "-hh=x", "--help=x",
    "--he", "--hel", "-H", "--HELP", "--help-me", "--only=-h",
]  # fmt: skip


def driver_reads(word: str, capsys: pytest.CaptureFixture[str]) -> str:
    """How the driver's own parser reads `word`: "help", "usage" (an error) or "parsed"."""
    try:
        m3.parser().parse_args(["--domain", "x", word])
    except SystemExit as e:
        out = capsys.readouterr().out
        if e.code == 0:
            assert "loom-m3-acceptance" in out, f"{word}: exit 0 without the help text"
            return "help"
        return "usage"
    capsys.readouterr()
    return "parsed"


def test_the_help_words_agree_with_the_driver(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every word the driver reads as help is refused; every word refused this way is one
    the driver would not parse normally (help or a usage error)."""
    b = Box(deploy_root_base(), site_doc=site())
    try:
        monkeypatch.setenv("LOOM_DEPLOY_ROOT", str(b.root))
        monkeypatch.setenv("PATH", f"{b.bin}:/usr/bin:/bin")
        monkeypatch.delenv("LOOM_DEPLOY_PROJECT", raising=False)
        helps, refused = [], []
        for w in WORDS:
            how = driver_reads(w, capsys)
            if how == "help":
                helps.append(w)
            try:
                code = deploy.main(["acceptance", "--p4-release", SHA_B, "--", w])
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
            err = capsys.readouterr().err
            if code == 2 and POINTER in err:
                refused.append(w)
                assert how != "parsed", f"{w} is refused, but the driver parses it normally"
        assert {"-h", "--help"} <= set(helps), f"the driver's help words: {helps}"
        missing = [w for w in helps if w not in refused]
        assert not missing, f"help words the driver reads as help, not refused: {missing}"
    finally:
        b.close()


@pytest.mark.parametrize(
    "words",
    [
        pytest.param(["--only", "m3-0"], id="only-m3-0"),
        pytest.param(["--help-me"], id="--help-me"),
        pytest.param(["--only=-h"], id="only=-h"),
    ],
)
def test_words_that_are_not_help_still_run(box: Box, words: list[str]) -> None:
    applied(box)
    n_tl = len(box.timeline())
    proc = acceptance(box, *words)
    assert_exit(proc, 0)
    acc = [rest for _, svc, rest in runs(box.timeline()[n_tl:]) if svc == "acceptance"]
    assert len(acc) == 1, f"acceptance runs: {acc}"
    assert acc[0][-len(words) :] == words, f"the words after -- are not passed: {acc[0]}"
