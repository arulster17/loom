"""EC2 calls that fail on the network, not with an answer (aws_ec2.NETWORK_ERRORS).

014218c9 (qwen3-8b-profile-aws, 2026-11-03) lost the whole experiment, during its capacity
wait, to one RunInstances `ReadTimeoutError` (read timeout=60). Such a call may or may not
have launched an instance. The provider now resends the identical request (same
ClientToken: EC2 then returns the instance it launched, never a second one) on the backoff,
at most NETWORK_RETRIES times in a row; when it gives up it terminates anything that token
launched. Teardown resends terminate (idempotent) and keeps polling through describe
timeouts, so a network error there never leaves an instance running.

Moto has neither ClientToken idempotency nor the `client-token` describe filter, so
`TokenEc2` adds both the way EC2 documents them."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from botocore.exceptions import (
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from loom_bench.providers.aws_ec2 import NETWORK_RETRIES, TERMINATE_RETRY_S, Ec2Unreachable
from loom_bench.providers.base import Host

from .fakes import client_error
from .test_aws_ec2 import AWS_LOG, IIC, FakeTime, aws_env, instances, provider, request

EC2 = "https://ec2.us-east-1.amazonaws.com"


@pytest.fixture
def aws(moto_aws: None) -> dict[str, Any]:
    return aws_env()


def read_timeout() -> ReadTimeoutError:
    return ReadTimeoutError(endpoint_url=EC2)


class TokenEc2:
    """Moto EC2 with ClientToken idempotency and the `client-token` filter. `script` is
    what each RunInstances does in turn: "ok", "launch+timeout" (launches, then the reply
    is lost), "timeout" (lost before EC2 acted), or an error code; then "ok"."""

    def __init__(self, inner: Any, script: list[str]) -> None:
        self._inner = inner
        self.script = list(script)
        self.by_token: dict[str, str] = {}
        self.calls: list[str] = []  # the ClientToken of each RunInstances
        self.terminate_failures = 0
        self.describe_failures = 0
        self.subnet_failures: list[Exception] = []
        # Every reply lost from now on, launches and idempotent returns alike.
        self.lose_replies = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _reply(self, iid: str) -> dict[str, Any]:
        inst = self._inner.describe_instances(InstanceIds=[iid])["Reservations"][0]
        return {"Instances": inst["Instances"]}

    def run_instances(self, **kw: Any) -> Any:
        token = kw["ClientToken"]
        self.calls.append(token)
        if token in self.by_token:  # EC2: the same token returns the same instance
            if self.lose_replies:
                raise read_timeout()
            return self._reply(self.by_token[token])
        step = self.script.pop(0) if self.script else "ok"
        if step == "timeout":
            raise read_timeout()
        if step not in ("ok", "launch+timeout"):
            raise client_error(step)
        out = self._inner.run_instances(**kw)
        self.by_token[token] = out["Instances"][0]["InstanceId"]
        if step == "launch+timeout" or self.lose_replies:
            raise read_timeout()
        return out

    def describe_subnets(self, **kw: Any) -> Any:
        if self.subnet_failures:
            raise self.subnet_failures.pop(0)
        return self._inner.describe_subnets(**kw)

    def describe_instances(self, **kw: Any) -> Any:
        if self.describe_failures:
            self.describe_failures -= 1
            raise ConnectionClosedError(endpoint_url=EC2)
        filters = kw.get("Filters") or []
        tokens = [f["Values"] for f in filters if f["Name"] == "client-token"]
        if tokens:
            ids = [self.by_token[t] for t in tokens[0] if t in self.by_token]
            if not ids:
                return {"Reservations": []}
            return self._inner.describe_instances(InstanceIds=ids)
        return self._inner.describe_instances(**kw)

    def terminate_instances(self, **kw: Any) -> Any:
        if self.terminate_failures:
            self.terminate_failures -= 1
            raise ReadTimeoutError(endpoint_url=EC2)
        return self._inner.terminate_instances(**kw)


def running(aws: dict[str, Any]) -> list[str]:
    return [i["InstanceId"] for i in instances(aws["ec2"]) if i["State"]["Name"] != "terminated"]


def make(aws: dict[str, Any], script: list[str]) -> tuple[Any, FakeTime, TokenEc2]:
    t = FakeTime()
    ec2 = TokenEc2(aws["ec2"], script)
    p = provider(aws, ec2=ec2, clock=t.clock)
    p.sleep = t.sleep
    return p, t, ec2


async def test_a_launch_whose_reply_timed_out_is_adopted_not_doubled(
    aws: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=AWS_LOG)
    p, t, ec2 = make(aws, ["launch+timeout"])
    host = await p.provision(request())
    (iid,) = running(aws)
    assert host.host_id == iid  # the instance the lost call launched, adopted
    assert ec2.calls[0] == ec2.calls[1] and len(ec2.calls) == 2  # resent with its token
    assert t.slept == [30]
    assert host.launched_at == t.start  # its billing and TTL are the original call's
    text = caplog.text
    assert "ReadTimeoutError" in text and f"ClientToken {ec2.calls[0]}" in text
    assert "by resending the unanswered RunInstances" in text


async def test_a_call_lost_before_ec2_acted_launches_once_on_the_resend(aws: dict[str, Any]):
    p, t, ec2 = make(aws, ["timeout"])
    host = await p.provision(request())
    assert running(aws) == [host.host_id]
    assert ec2.calls[0] == ec2.calls[1] and t.slept == [30]


async def test_a_resend_refused_for_capacity_moves_on_to_the_other_azs(aws: dict[str, Any]):
    # The first AZ's call was lost, its resend refused (so nothing launched there): the
    # sweep goes on with the next AZ under a fresh token.
    p, _, ec2 = make(aws, ["timeout", IIC])
    host = await p.provision(request())
    assert running(aws) == [host.host_id]
    assert host.info["az"] == "us-east-1b"
    assert ec2.calls[0] == ec2.calls[1] != ec2.calls[2]


async def test_endless_network_errors_give_up_and_leave_nothing_running(
    aws: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=AWS_LOG)
    p, t, ec2 = make(aws, ["launch+timeout"])
    ec2.lose_replies = True  # the first call launched; no reply gets through after it
    with pytest.raises(Ec2Unreachable) as err:
        await p.provision(request())
    assert NETWORK_RETRIES == 4
    assert t.slept == [30, 60, 120, 240]  # then one check, at once
    assert running(aws) == []  # the instance the first lost call launched is terminated
    assert "terminated i-" in str(err.value) and "failed on the network 5 times" in str(err.value)


async def test_a_lost_call_that_launched_nothing_says_so_when_giving_up(aws: dict[str, Any]):
    p, _, _ = make(aws, ["timeout"] * 10)
    with pytest.raises(Ec2Unreachable, match="launched nothing"):
        await p.provision(request())
    assert running(aws) == []


@pytest.mark.parametrize(
    "error",
    [
        ReadTimeoutError(endpoint_url=EC2),
        ConnectTimeoutError(endpoint_url=EC2),
        EndpointConnectionError(endpoint_url=EC2),
        ConnectionClosedError(endpoint_url=EC2),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_network_errors_elsewhere_in_a_sweep_are_retried(aws: dict[str, Any], error):
    # Not RunInstances: the sweep's subnet lookup failed before any launch.
    p, t, ec2 = make(aws, [])
    ec2.subnet_failures = [error]
    host = await p.provision(request())
    assert running(aws) == [host.host_id] and t.slept == [30]
    assert len(ec2.calls) == 1  # no RunInstances was pending, so nothing to resend


async def test_network_errors_do_not_eat_the_capacity_backoff_forever(aws: dict[str, Any]):
    # A capacity refusal resets the network count: errors must be NETWORK_RETRIES in a row.
    script = ["timeout", IIC, IIC] * 2 + ["ok"]
    p, _, _ = make(aws, script)
    host = await p.provision(request())
    assert running(aws) == [host.host_id]


# --- teardown -------------------------------------------------------------------------


async def launched(p: Any) -> Host:
    return await p.provision(request())


async def test_teardown_resends_terminate_after_network_errors(aws: dict[str, Any]) -> None:
    p, t, ec2 = make(aws, [])
    host = await launched(p)
    ec2.terminate_failures = 2
    ec2.describe_failures = 1
    await p.teardown(host)
    assert running(aws) == []
    assert t.slept[:2] == list(TERMINATE_RETRY_S[:2])


async def test_teardown_that_cannot_reach_ec2_raises_rather_than_pretend(
    aws: dict[str, Any],
) -> None:
    p, t, ec2 = make(aws, [])
    host = await launched(p)
    ec2.terminate_failures = len(TERMINATE_RETRY_S) + 1
    with pytest.raises(ReadTimeoutError):
        await p.teardown(host)
    assert t.slept == list(TERMINATE_RETRY_S)
    assert running(aws) == [host.host_id]  # the runner reports it; TTL and reaper remain


def test_fake_time_starts_where_the_aws_tests_do() -> None:
    assert FakeTime().start == datetime(2026, 11, 2, 22, 20, tzinfo=UTC)
    assert timedelta(seconds=sum(TERMINATE_RETRY_S)) < timedelta(minutes=3)
