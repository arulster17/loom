"""The availability-gated stock wait (`RunpodProvider.provision`).

On 2026-11-02 every Secure 1x L40S create was refused for ~11 h (c1470fb9, 627c9916 and
the profiling and spec-decode smokes) while `gpuTypes.lowestPrice` kept reporting "Low"
stock; RunPod's per-datacenter `dataCenters.gpuAvailability` was the signal that matched
the refusals. After a stock refusal the provider now reads that signal every
`availability_poll_s` and sends a create only when an allowed datacenter shows stock,
aimed at those datacenters; a failing query falls back to blind creates on the backoff
schedule, and `capacity_wait_s` stays the deadline."""

import logging
from typing import Any

import boto3
import pytest

from loom_bench.experiment import RunpodProviderSpec
from loom_bench.providers.runpod_api import RunpodApiError

from .fakes import FakeRunpod
from .test_runpod_provider import BUCKET, NO_STOCK, REGION, FakeTime, provider, request, settings


@pytest.fixture
def s3(moto_aws: None) -> Any:
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    return client


@pytest.fixture
def fake() -> FakeRunpod:
    return FakeRunpod(gets_before_ssh=0)


def bodies(fake: FakeRunpod) -> list[dict]:
    return fake.bodies("POST", "/v1/pods")


async def test_a_create_goes_out_when_stock_appears(fake, s3, tmp_path, caplog) -> None:
    fake.fail[("POST", "/v1/pods")] = [NO_STOCK]  # the first, blind create
    fake.stock = [[], [], ["US-TX-3", "OC-AU-1"]]
    t = FakeTime()
    p = provider(fake, s3, tmp_path, clock=t.clock, sleep=t.sleep)
    with caplog.at_level(logging.WARNING, logger="loom_bench.providers.runpod"):
        host = await p.provision(request())
    sent = bodies(fake)
    assert len(sent) == 2  # no create while no datacenter shows stock
    assert "dataCenterIds" not in sent[0]
    assert sent[1]["dataCenterIds"] == ["US-TX-3", "OC-AU-1"]
    assert t.slept == [30.0, 30.0]  # polled every availability_poll_s
    assert fake.stock_queries == 3
    assert host.host_id in fake.pods
    text = caplog.text
    assert "none in any allowed datacenter (waited 0 s)" in text
    assert "US-TX-3, OC-AU-1 (waited 60 s)" in text


async def test_stock_outside_the_specs_datacenters_is_ignored(fake, s3, tmp_path) -> None:
    fake.fail[("POST", "/v1/pods")] = [NO_STOCK]
    fake.stock = [["US-TX-3"], ["US-TX-3", "EUR-IS-2"]]
    t = FakeTime()
    spec = RunpodProviderSpec(kind="runpod", data_center_ids=["EUR-IS-2", "EU-NL-1"])
    p = provider(fake, s3, tmp_path, spec=spec, clock=t.clock, sleep=t.sleep)
    await p.provision(request())
    sent = bodies(fake)
    assert sent[0]["dataCenterIds"] == ["EUR-IS-2", "EU-NL-1"]  # the spec's own
    assert sent[1]["dataCenterIds"] == ["EUR-IS-2"]  # stocked and allowed
    assert t.slept == [30.0]


async def test_a_refusal_despite_listed_stock_waits_the_backoff(fake, s3, tmp_path) -> None:
    # The listing can lag: a create aimed at a stocked datacenter may still be refused.
    # The same datacenters are not retried before the backoff; a new one is at once.
    fake.fail[("POST", "/v1/pods")] = [NO_STOCK, NO_STOCK]
    fake.stock = [["US-TX-3"], ["US-TX-3"], ["US-TX-3", "US-TX-4"]]
    t = FakeTime()
    p = provider(fake, s3, tmp_path, clock=t.clock, sleep=t.sleep)
    await p.provision(request())
    sent = bodies(fake)
    assert [b.get("dataCenterIds") for b in sent] == [None, ["US-TX-3"], ["US-TX-3", "US-TX-4"]]
    # First refusal (blind) -> stock seen -> create at once; refused in US-TX-3 -> one
    # poll interval while only US-TX-3 shows stock (its 60 s backoff is not over) ->
    # US-TX-4 appears at 30 s and is tried at once, before the backoff ends.
    assert t.slept == [30.0]


async def test_listed_stock_that_keeps_refusing_is_retried_on_the_backoff(
    fake, s3, tmp_path
) -> None:
    fake.fail[("POST", "/v1/pods")] = [NO_STOCK, NO_STOCK, NO_STOCK]
    fake.stock = [["US-TX-3"]]  # listed throughout, refused twice
    t = FakeTime()
    p = provider(fake, s3, tmp_path, clock=t.clock, sleep=t.sleep)
    await p.provision(request())
    # blind refused at 0; aimed at US-TX-3 at once, refused (backoff 60): polls at 0, 30,
    # create at 60, refused (backoff 120): polls 60..150, create at 180, created.
    assert t.slept == [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    assert len(bodies(fake)) == 4


async def test_a_failing_availability_query_falls_back_to_blind_creates(
    fake, s3, tmp_path, caplog
) -> None:
    fake.fail[("POST", "/v1/pods")] = [NO_STOCK, NO_STOCK]
    fake.stock = ["error"]
    t = FakeTime()
    p = provider(fake, s3, tmp_path, clock=t.clock, sleep=t.sleep)
    with caplog.at_level(logging.WARNING, logger="loom_bench.providers.runpod"):
        host = await p.provision(request())
    assert t.slept == [30.0, 60.0]  # the blind backoff schedule
    assert all("dataCenterIds" not in b for b in bodies(fake))
    assert host.host_id in fake.pods
    assert caplog.text.count("availability query failed") == 1


async def test_the_deadline_still_ends_the_wait_with_nothing_created(fake, s3, tmp_path) -> None:
    fake.fail[("POST", "/v1/pods")] = [NO_STOCK] * 10
    fake.stock = [[]]  # never any stock
    t = FakeTime()
    p = provider(
        fake, s3, tmp_path, clock=t.clock, sleep=t.sleep, settings=settings(capacity_wait_s=100)
    )
    with pytest.raises(RunpodApiError, match="no instances currently available"):
        await p.provision(request())
    assert sum(t.slept) == 100.0
    assert t.slept == [30.0, 30.0, 30.0, 10.0]
    # The first create and one last at the deadline; none while no stock was listed.
    assert len(bodies(fake)) == 2
    assert fake.pods == {}  # nothing exists, so nothing is billed


async def test_the_poll_interval_is_a_setting(fake, s3, tmp_path) -> None:
    fake.fail[("POST", "/v1/pods")] = [NO_STOCK]
    fake.stock = [[], ["US-TX-4"]]
    t = FakeTime()
    p = provider(
        fake, s3, tmp_path, clock=t.clock, sleep=t.sleep,
        settings=settings(availability_poll_s=7.5),
    )  # fmt: skip
    await p.provision(request())
    assert t.slept == [7.5]


def test_gpu_stock_reads_per_datacenter_availability_best_first() -> None:
    fake = FakeRunpod()
    fake.stock = [["US-TX-4", "EUR-IS-2"]]
    with fake.api() as api:
        assert api.gpu_stock("NVIDIA L40S") == ["US-TX-4", "EUR-IS-2"]
        assert api.gpu_stock("NVIDIA L40") == []
    fake.stock = ["error"]
    with fake.api() as api, pytest.raises(RunpodApiError, match="upstream timeout"):
        api.gpu_stock("NVIDIA L40S")
