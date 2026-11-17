"""The bench RunPod pieces the pool manager reuses, with their M2 additions
(docs/phase1-m2-spec.md §10, §14 step 2): `RunpodApi(user_agent=...)`, `public_port`,
and `generate_keypair(comment=...)`. Each keeps its old default."""

import shutil
import stat
from pathlib import Path

import pytest

from loom_bench.providers import runpod_api
from loom_bench.providers.runpod_api import public_ssh
from loom_bench.providers.runpod_fake import FakeRunpod, create_response_fixture
from loom_bench.providers.runpod_ssh import generate_keypair

POD = {"publicIp": "203.0.113.7", "portMappings": {"22": 28813, "8443": 28814}}


def public_port(pod: dict, port: int) -> tuple[str, int] | None:
    """`runpod_api.public_port`, looked up per test so each fails on its own until it
    exists."""
    return runpod_api.public_port(pod, port)


needs_ssh_keygen = pytest.mark.skipif(
    shutil.which("ssh-keygen") is None, reason="ssh-keygen not installed"
)


# ------------------------------------------------------------------ user agent


def test_user_agent_defaults_to_loom_bench() -> None:
    rp = FakeRunpod()
    rp.api().list_pods()
    assert [r.headers["User-Agent"] for r in rp.requests] == ["loom-bench"]


def test_user_agent_is_sent_on_every_call() -> None:
    rp = FakeRunpod()
    pod = rp.add_pod()
    rp.rest_delete_status = 403  # terminate falls back to GraphQL: both APIs are called
    api = rp.api(user_agent="loom-poolmgr")
    api.list_pods()
    api.get_pod(pod["id"])
    api.terminate_pod(pod["id"])
    assert {r.url.host for r in rp.requests} == {"rest.runpod.test", "api.runpod.test"}
    assert [r.headers["User-Agent"] for r in rp.requests] == ["loom-poolmgr"] * len(rp.requests)
    # Still authenticated as before.
    assert all(r.headers["Authorization"].startswith("Bearer ") for r in rp.requests)


# ------------------------------------------------------------------ public_port


def test_public_port_finds_any_mapped_port() -> None:
    assert public_port(POD, 8443) == ("203.0.113.7", 28814)
    assert public_port(POD, 22) == ("203.0.113.7", 28813)


def test_public_port_generalises_public_ssh() -> None:
    for pod in (
        POD,
        {"publicIp": "203.0.113.7", "portMappings": {"8443": 1}},
        {"publicIp": "", "portMappings": {"22": 1}},
        create_response_fixture(),
    ):
        assert public_port(pod, 22) == public_ssh(pod)


@pytest.mark.parametrize(
    "pod",
    [
        {"publicIp": "203.0.113.7", "portMappings": {"22": 28813}},  # 8443 not mapped yet
        {"publicIp": "", "portMappings": {"8443": 28814}},  # no IP yet
        {"publicIp": None, "portMappings": {"8443": 28814}},
        {"publicIp": "203.0.113.7", "portMappings": {}},
        {"publicIp": "203.0.113.7", "portMappings": None},
        {},
    ],
)
def test_public_port_is_none_until_runpod_assigns_it(pod: dict) -> None:
    assert public_port(pod, 8443) is None


def test_public_port_on_the_create_response_is_none() -> None:
    """RunPod's create answer has no IP or mappings yet."""
    assert public_port(create_response_fixture(), 8443) is None


# ------------------------------------------------------------------ generate_keypair


@needs_ssh_keygen
def test_generate_keypair_comment(tmp_path: Path) -> None:
    key, pub = generate_keypair(tmp_path / "keys", comment="loom-poolmgr")
    assert pub.startswith("ssh-ed25519 ") and pub.endswith(" loom-poolmgr")
    assert " loom-bench" not in pub
    assert stat.S_IMODE((tmp_path / "keys").stat().st_mode) == 0o700
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert key.with_suffix(".pub").read_text().strip() == pub
    with pytest.raises(FileExistsError):
        generate_keypair(tmp_path / "keys", comment="loom-poolmgr")


@needs_ssh_keygen
def test_generate_keypair_default_comment_is_unchanged(tmp_path: Path) -> None:
    _, pub = generate_keypair(tmp_path / "keys")
    assert pub.endswith(" loom-bench")
