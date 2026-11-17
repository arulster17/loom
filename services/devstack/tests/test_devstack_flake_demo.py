"""TEMPORARY: proves the flake-check job catches a flaky test. Removed in the next commit."""

import random


def test_fails_about_one_run_in_three() -> None:
    assert random.random() >= 1 / 3


def test_always_passes() -> None:
    assert True
