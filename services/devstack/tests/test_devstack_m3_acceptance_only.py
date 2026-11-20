"""`loom-m3-acceptance`'s item selection (M3 spec §6.12 "Item selection", §10; user
2026-11-19): `--only <items>` and `--only=<items>`, comma-separated, spaces around an item
ignored; given more than once, the last one wins; no `--only` runs every item of the mode
(m3-p included in box mode); `--external` runs m3-h only. No option name may be
abbreviated, and an empty or unknown item is a usage error (exit 2).

The selection is read through `items(args)`, the function `main` uses, which
`loom-deploy acceptance`'s tests also call to decide whether m3-p's host half should run.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from loom_devstack import m3_acceptance as m3

DOMAIN = ["--domain", "example.test"]


def selection(*argv: str) -> set[str]:
    return set(m3.items(m3.parser().parse_args([*DOMAIN, *argv])))


def usage_error(*argv: str) -> None:
    """`main` stops with a usage error (exit 2) before it reaches any service."""
    err = io.StringIO()
    with pytest.raises(SystemExit) as exit_, contextlib.redirect_stderr(err):
        m3.main([*DOMAIN, *argv])
    assert exit_.value.code == 2, f"exit {exit_.value.code}: {err.getvalue()}"


def test_no_only_runs_every_box_item() -> None:
    got = selection()
    assert got == set(m3.BOX_ITEMS)
    assert "m3-p" in got and "m3-h" not in got


def test_external_runs_only_m3_h() -> None:
    assert selection("--external") == {"m3-h"}
    assert selection("--external", "--only", "m3-h") == {"m3-h"}


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        (["--only", "m3-0", "--only", "m3-p"], {"m3-p"}),
        (["--only", "m3-p", "--only", "m3-0"], {"m3-0"}),
        (["--only=m3-0", "--only", "m3-p"], {"m3-p"}),
        (["--only", "m3-p", "--only=m3-0"], {"m3-0"}),
        (["--only", "m3-0,m3-b", "--only", "m3-i,m3-p"], {"m3-i", "m3-p"}),
    ],
)
def test_the_last_only_wins(argv: list[str], want: set[str]) -> None:
    assert selection(*argv) == want


def test_only_equals_form() -> None:
    assert selection("--only=m3-0,m3-p") == {"m3-0", "m3-p"}
    assert selection("--only=m3-x") == {"m3-x"}


def test_spaces_around_items_are_ignored() -> None:
    assert selection("--only", " m3-0 , m3-p ") == {"m3-0", "m3-p"}
    assert selection("--only= m3-d") == {"m3-d"}


def test_one_item() -> None:
    assert selection("--only", "m3-p") == {"m3-p"}


@pytest.mark.parametrize(
    "argv",
    [
        ["--onl", "m3-0"],
        ["--onl=m3-0"],
        ["--on", "m3-p"],
        ["--ext"],
        ["--only", "m3-0", "--rel", "a" * 40],
    ],
)
def test_abbreviated_option_names_are_usage_errors(argv: list[str]) -> None:
    usage_error(*argv)


def test_an_abbreviated_domain_is_not_taken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOOM_DOMAIN", raising=False)
    err = io.StringIO()
    with pytest.raises(SystemExit) as exit_, contextlib.redirect_stderr(err):
        m3.main(["--dom", "example.test", "--only", "m3-0"])
    assert exit_.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--only", ""],
        ["--only="],
        ["--only", " "],
        ["--only", ","],
        ["--only", "m3-0,"],
        ["--only", "m3-0,,m3-p"],
        ["--only", "m3-0", "--only", ""],
    ],
)
def test_an_empty_item_is_a_usage_error(argv: list[str]) -> None:
    usage_error(*argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--only", "m3-z"],
        ["--only", "m3-0,m3-z"],
        ["--only", "m3-a"],  # the runbook's, by hand
        ["--only", "M3-0"],
        ["--only", "m3-h"],  # external only
        ["--external", "--only", "m3-p"],  # m3-p is not an external item
        ["--only", "m3-z", "--only", "m3-0", "--only", "m3-y"],
    ],
)
def test_an_unknown_item_is_a_usage_error(argv: list[str]) -> None:
    usage_error(*argv)


def test_an_earlier_bad_only_is_overridden_by_the_last() -> None:
    """The last `--only` wins outright: an earlier one is not read at all."""
    assert selection("--only", "m3-z", "--only", "m3-0") == {"m3-0"}
