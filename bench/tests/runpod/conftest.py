from collections.abc import Iterator

import pytest
from moto import mock_aws

REGION = "us-east-1"
BUCKET = "loom-bench-test"


@pytest.fixture(autouse=True)
def fake_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


@pytest.fixture
def moto_aws() -> Iterator[None]:
    with mock_aws():
        yield
