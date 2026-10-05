import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.cost import NO_LOCAL_PRICE
from loom_bench.experiment import Experiment, LocalProviderSpec, mock_launch
from loom_bench.jobexec import execute_load_job
from loom_bench.jobs import LoadJob, LoadJobResult, TokenizerSpec
from loom_bench.mock.config import MockConfig
from loom_bench.provenance import PriceBasis, Provenance
from loom_bench.providers.base import HostRequest
from loom_bench.providers.local import LocalProvider
from loom_bench.providers.mock import MockProvider, live_host_ids
from loom_bench.records import LoadMode, Market
from loom_bench.registry import load_registry
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.leaderboard import RowStatus, build_leaderboard
from loom_bench.runner import run_experiment
from loom_bench.store import repo
from loom_bench.store.db import session_scope

from .conftest import mock_doc, mock_experiment

SPEC = load_registry().get("qwen3-8b")
FAST = {"time_scale": 0.01, "models": [SPEC.id]}
WORKLOAD = {
    "name": "tiny",
    "description": "tiny",
    "kind": "synthetic",
    "content": "synthetic",
    "endpoint": "chat",
    "input_len": 32,
    "output_len": 8,
}


def _job(endpoint, **over) -> LoadJob:
    base = dict(
        run_id="r1",
        base_url=endpoint.base_url,
        metrics_url=endpoint.metrics_url,
        engine=endpoint.engine,
        served_model=endpoint.served_model,
        workload=WORKLOAD,
        tokenizer=TokenizerSpec(kind="simple"),
        mode=LoadMode.CLOSED_LOOP,
        load_value=2,
        num_requests=6,
        warmup_requests=2,
        scrape_interval_s=0.05,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return LoadJob(**{**base, **over})


@pytest.fixture
async def mock_host():
    provider = MockProvider(hourly_micros=1_000_000)
    host = await provider.provision(HostRequest(ttl_s=600))
    try:
        yield provider, host
    finally:
        await provider.teardown(host)


async def test_mock_cold_then_warm_start(mock_host):
    provider, host = mock_host
    cold = await provider.start_engine(
        host, mock_launch(SPEC, MockConfig(**FAST, startup_delay_s=10)), warm=False
    )
    assert list(cold.start_stages) == [
        "instance_running",
        "server_listening",
        "engine_healthy",
        "first_token",
    ]
    assert cold.start_stages["engine_healthy"] >= 0.1  # 10 simulated s at time_scale 0.01
    assert cold.system["engine"] == "mock" and not cold.warm

    await provider.stop_engine(host)
    warm = await provider.start_engine(
        host, mock_launch(SPEC, MockConfig(**FAST, max_num_seqs=4)), warm=True
    )
    assert warm.warm and "instance_running" not in warm.start_stages
    assert warm.base_url != cold.base_url
    async with httpx.AsyncClient() as http:
        with pytest.raises(httpx.ConnectError):
            await http.get(cold.base_url + "/models")


async def test_execute_load_job_closed_and_open_loop(mock_host):
    provider, host = mock_host
    endpoint = await provider.start_engine(host, mock_launch(SPEC, MockConfig(**FAST)), warm=False)

    closed = await provider.run_job(host, _job(endpoint))
    records = closed.request_records()
    assert len(records) == 8 and sum(r.warmup for r in records) == 2
    assert all(r.ok for r in records)
    assert len(closed.scrapes) >= 2 and "vllm:num_requests_running" in closed.scrapes[-1][1]
    assert closed.meta["loadgen"] == "native"

    open_loop = await execute_load_job(
        _job(
            endpoint,
            mode=LoadMode.OPEN_LOOP,
            load_value=20,
            num_requests=None,
            warmup_requests=0,
            arrival={"kind": "constant", "rate": 20},
            duration_s=0.5,
            warmup_s=0.1,
        )
    )
    recs = open_loop.request_records()
    assert len(recs) == 10 and sum(r.warmup for r in recs) == 2
    assert all(r.scheduled_at_s is not None for r in recs)


async def test_job_json_round_trips_through_bench_job_run(mock_host, tmp_path):
    provider, host = mock_host
    endpoint = await provider.start_engine(host, mock_launch(SPEC, MockConfig(**FAST)), warm=False)
    job_path, out_path = tmp_path / "job.json", tmp_path / "result.json"
    job_path.write_text(_job(endpoint).model_dump_json())
    # CliRunner drives asyncio.run, which needs a thread without a running loop.
    import asyncio

    result = await asyncio.to_thread(
        CliRunner().invoke, app, ["job", "run", "--in", str(job_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    parsed = LoadJobResult.model_validate_json(out_path.read_text())
    assert len(parsed.records) == 8


async def test_mock_reap_only_touches_expired_hosts():
    provider = MockProvider()
    keep = await provider.provision(HostRequest(ttl_s=600))
    expired = await provider.provision(HostRequest(ttl_s=0))
    await provider.start_engine(expired, mock_launch(SPEC, MockConfig(**FAST)), warm=False)
    try:
        reaped = await provider.reap(datetime.now(UTC) + timedelta(seconds=1))
        assert reaped == [expired.host_id]
        assert keep.host_id in live_host_ids() and expired.host_id not in live_host_ids()
    finally:
        await provider.teardown(keep)


async def test_local_provider_checks_the_served_model(mock_host):
    provider, host = mock_host
    endpoint = await provider.start_engine(host, mock_launch(SPEC, MockConfig(**FAST)), warm=False)
    settings = LocalProviderSpec(
        kind="local",
        base_url=endpoint.base_url,
        metrics_url=endpoint.metrics_url,
        engine="mock",
        served_model=SPEC.id,
        tokenizer="simple",
    )
    local = LocalProvider(settings)
    lhost = await local.provision(HostRequest(ttl_s=60))
    assert lhost.hourly_micros == 0 and lhost.request.market.value == "local"
    assert lhost.as_run_micros is None and lhost.price_basis is None  # unpriced, not $0
    ep = await local.start_engine(lhost, mock_launch(SPEC, MockConfig(**FAST)), warm=False)
    assert ep.system["engine_version"]
    wrong = LocalProvider(settings.model_copy(update={"served_model": "nope"}))
    with pytest.raises(RuntimeError, match="does not serve"):
        await wrong.start_engine(lhost, mock_launch(SPEC, MockConfig(**FAST)), warm=False)


def _local_doc(base_url: str, metrics_url: str | None, **over: Any) -> dict[str, Any]:
    provider = {
        "kind": "local",
        "base_url": base_url,
        "metrics_url": metrics_url,
        "engine": "mock",
        "served_model": SPEC.id,
        "tokenizer": "simple",
        **over,
    }
    return mock_doc(provider=provider)


@pytest.mark.parametrize("price", ["$0", "$-1"])
def test_declared_hourly_price_must_be_positive(price):
    with pytest.raises(ValidationError, match="hourly_price"):
        Experiment.model_validate(_local_doc("http://127.0.0.1:1/v1", None, hourly_price=price))
    with pytest.raises(ValidationError, match="hourly_price"):
        mock_experiment(provider={"kind": "mock", "hourly_price": price})
    assert mock_experiment(provider={"kind": "mock"}).provider.hourly_price is None


async def test_local_results_are_unpriced_unless_the_experiment_sets_a_price(mock_host, ctx):
    provider, host = mock_host
    endpoint = await provider.start_engine(host, mock_launch(SPEC, MockConfig(**FAST)), warm=False)
    ranked = {}
    for name, over in (("unpriced", {}), ("priced", {"hourly_price": "$2"})):
        exp = Experiment.model_validate(
            {**_local_doc(endpoint.base_url, endpoint.metrics_url, **over), "name": name}
        )
        outcome = await run_experiment(exp, ctx)
        with session_scope(ctx.db_url) as s:
            runs = repo.list_runs(s, experiment_id=outcome.experiment_id)
        prov = Provenance.model_validate(runs[0].provenance)
        (result,) = analyze_runs(
            runs,
            slo=exp.slo,
            allocation=exp.cost_allocation.allocation(),
            price_resolver=default_price_resolver(ctx.prices),
        )
        ranked[name] = (prov, result, build_leaderboard([result]).boards[0].rows[0])
        assert outcome.spent_micros == 0  # a local endpoint is never billed

    prov, result, row = ranked["unpriced"]
    assert prov.hourly_micros is None and prov.price_basis is None
    assert result.cost is None and result.as_run_cost is None
    assert row.status is RowStatus.NO_COST and row.rank is None
    assert row.recommendation == f"No cost at SLO: {NO_LOCAL_PRICE}"

    prov, result, row = ranked["priced"]
    assert prov.hourly_micros == 2_000_000
    assert prov.price_basis == PriceBasis(market=Market.LOCAL, source="experiment")
    assert result.prices.on_demand == result.prices.as_run == 2_000_000
    assert result.cost is not None and row.status is not RowStatus.NO_COST


def test_mock_launch_carries_the_whole_config():
    cfg = MockConfig(**FAST, max_num_seqs=3)
    launch = mock_launch(SPEC, cfg)
    assert json.loads(launch.args[1])["max_num_seqs"] == 3
