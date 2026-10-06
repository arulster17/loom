"""Suite-wide guards.

This machine may export a real RunPod account key (e.g. from a shell profile) or
hold one in the macOS Keychain. No test may ever reach the real RunPod API or
spend money, so every test starts with the key unset and the Keychain lookup off.
"""

from collections.abc import Iterator

import pytest

from loom_bench.providers import runpod_api


@pytest.fixture(autouse=True)
def _no_real_runpod_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(runpod_api.KEY_ENV, raising=False)
    monkeypatch.setattr(runpod_api, "_keychain_lookup", lambda: None)
    yield
