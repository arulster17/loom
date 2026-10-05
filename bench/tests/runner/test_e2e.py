"""Acceptance: the whole Lab end to end on the mock backend, through the CLI."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from loom_bench.cli import EXIT_MISMATCH, app
from loom_bench.experiment import load_experiment, mock_launch
from loom_bench.metrics.summary import RunSummary
from loom_bench.mock.config import MockConfig
from loom_bench.money import parse_usd
from loom_bench.provenance import GitInfo, Provenance
from loom_bench.providers.base import HostRequest, SpotInterrupted
from loom_bench.providers.mock import MockProvider, live_host_ids
from loom_bench.registry import load_registry
from loom_bench.runner import EXIT_BUDGET_ABORT, RUN_INTERRUPTED, run_experiment
from loom_bench.store import repo
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.models import (
    BenchColdStart,
    BenchEvalRun,
    BenchExperiment,
    BenchGateDecision,
    BenchResource,
    BenchRun,
    BenchSpend,
)
from loom_bench.store.parquet import read_requests

from .conftest import ABORT, SMOKE, mock_experiment, write_yaml

pytestmark = pytest.mark.timeout(240)


def invoke(*args: str):
    return CliRunner().invoke(app, [str(a) for a in args])


@dataclass
class SmokeRun:
    db: str
    out: Path
    exit_code: int
    output: str


@pytest.fixture(scope="module")
def smoke(tmp_path_factory) -> SmokeRun:
    tmp = tmp_path_factory.mktemp("smoke")
    db = f"sqlite:///{tmp / 'loom.db'}"
    result = invoke("run", SMOKE, "--db", db, "--out", tmp / "results")
    return SmokeRun(db, tmp / "results", result.exit_code, result.output)


def _experiment(db: str) -> BenchExperiment:
    with session_scope(db) as s:
        return s.scalars(select(BenchExperiment)).one()


def test_smoke_completes(smoke):
    assert smoke.exit_code == 0, smoke.output
    exp = _experiment(smoke.db)
    assert exp.status == "completed" and exp.abort_reason is None
    assert exp.git_sha is not None
    assert 0 < exp.spent_micros < parse_usd("$0.10")


def test_smoke_runs_have_complete_provenance_and_summaries(smoke):
    with session_scope(smoke.db) as s:
        runs = list(s.scalars(select(BenchRun)))
    assert len(runs) == 24  # 2 variants x 2 workloads x 3 loads x 2 reps
    assert {r.status for r in runs} == {"completed"}
    assert {r.cell_key for r in runs} == {"baseline", "small-batch"}
    assert {(r.load_mode, r.repetition) for r in runs} >= {("open_loop", 0), ("closed_loop", 1)}
    for run in runs:
        prov = Provenance.model_validate(run.provenance)
        assert prov.config_hash == run.config_hash
        known = [
            prov.git.sha,
            prov.bench_version,
            prov.loadgen.name,
            prov.loadgen.version,
            prov.engine.name,
            prov.engine.version,
            prov.engine.args,
            prov.model.repo,
            prov.model.revision,
            prov.model.quantization,
            prov.hardware.gpu_type,
            prov.hardware.gpu_count,
            prov.parallelism.tp,
            prov.market,
            prov.hourly_micros,
            prov.workload.name,
            prov.workload.profile_hash,
            prov.workload.content,
            prov.load.mode,
            prov.load.value,
            prov.load.seed,
            prov.repetition,
            prov.host.python,
        ]
        assert None not in known, prov
        assert prov.hourly_micros == 1_000_000
        summary = RunSummary.model_validate(run.summary)
        assert summary.n_total > 0 and summary.ttft_ms.p95 is not None
        assert summary.server is not None and summary.server.n_scrapes >= 2
        if run.load_mode == "closed_loop":
            assert summary.n_warmup_excluded == 2  # warmup_requests
        records = read_requests(run.requests_uri)
        assert len(records) == summary.n_total + summary.n_warmup_excluded
        assert Path(run.requests_uri).parent.joinpath("provenance.json").is_file()


def test_smoke_hosts_cold_warm_and_teardown(smoke):
    with session_scope(smoke.db) as s:
        resources = list(s.scalars(select(BenchResource)))
        starts = list(s.scalars(select(BenchColdStart).order_by(BenchColdStart.created_at)))
        spend = list(s.scalars(select(BenchSpend)))
        hashes = {r.config_hash for r in s.scalars(select(BenchRun))}
    assert len(resources) == 1  # both configs on one host
    assert resources[0].terminated_by == "runner" and resources[0].terminated_at is not None
    assert [c.kind for c in starts] == ["cold", "warm"]
    assert {c.config_hash for c in starts} == hashes
    assert starts[0].stages["engine_healthy"] >= 0.2  # startup_delay 20 s x time_scale 0.01
    assert spend and all(sp.basis["simulated"] for sp in spend)
    assert not live_host_ids()


def test_smoke_quality_suite_and_gate(smoke):
    with session_scope(smoke.db) as s:
        evals = list(s.scalars(select(BenchEvalRun)))
        gates = list(s.scalars(select(BenchGateDecision)))
    assert {e.task for e in evals} == {"arithmetic", "json_schema"}
    assert len({e.config_hash for e in evals}) == 2
    assert len(gates) == 1 and gates[0].decision == "pass"
    assert "quality gate small-batch vs baseline: pass" in smoke.output

    base, cand = gates[0].baseline_config_hash, gates[0].candidate_config_hash
    result = invoke("quality", "gate", "--baseline", base, "--candidate", cand, "--db", smoke.db)
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output.upper()


def test_smoke_reports_export_and_self_compare(smoke, tmp_path):
    report = invoke("report", "--db", smoke.db, "--out", tmp_path)
    assert report.exit_code == 0, report.output
    assert (tmp_path / "leaderboard.md").is_file()
    export = invoke("export", "csv", "--out", tmp_path / "runs.csv", "--db", smoke.db)
    assert export.exit_code == 0 and "wrote 24 runs" in export.output
    exp_id = str(_experiment(smoke.db).id)
    compare = invoke("compare", exp_id, exp_id, "--db", smoke.db)
    assert compare.exit_code == 0, compare.output


def test_smoke_site_export_build_and_waitlist(smoke, tmp_path):
    exp_id = str(_experiment(smoke.db).id)
    export = invoke("site", "export", "-e", exp_id, "--out", tmp_path / "data", "--db", smoke.db)
    assert export.exit_code == 0, export.output
    build = invoke("site", "build", "--data", tmp_path / "data", "--out", tmp_path / "_build")
    assert build.exit_code == 0, build.output
    assert (tmp_path / "_build" / "index.html").is_file()

    docs = tmp_path / "waitlist.md"
    docs.write_text("# Waitlist\n\n| Date | Count | Source |\n|---|---|---|\n")
    count = invoke("waitlist", "count", "--db", smoke.db, "--docs", docs)
    assert count.exit_code == 0, count.output
    assert "0 signups" in count.output
    assert "| 0 | waitlist_signups table |" in docs.read_text()


def test_hard_budget_abort(tmp_path):
    db = f"sqlite:///{tmp_path / 'loom.db'}"
    result = invoke("run", ABORT, "--db", db, "--out", tmp_path / "results")
    assert result.exit_code == EXIT_BUDGET_ABORT, result.output
    spec = load_experiment(ABORT)
    exp = _experiment(db)
    assert exp.status == "aborted"
    assert exp.abort_reason.startswith("budget cap reached")
    cap = spec.budget.max_spend
    # Worst-case overshoot: one accrual interval at the host's price, plus teardown.
    per_second = spec.provider.hourly_price / 3600
    slack_s = spec.budget.accrual_interval_s + 1.0
    assert cap <= exp.spent_micros <= cap + per_second * slack_s
    with session_scope(db) as s:
        resources = list(s.scalars(select(BenchResource)))
        runs = list(s.scalars(select(BenchRun)))
    assert resources and all(r.terminated_by == "runner" for r in resources)
    assert [r.status for r in runs] == ["aborted"]  # the in-flight run was cancelled
    assert not live_host_ids()


def test_reaper_removes_expired_resources_of_a_dead_runner(tmp_path):
    db = f"sqlite:///{tmp_path / 'loom.db'}"
    upgrade(db)
    spec = load_registry().get("qwen3-8b")
    provider = MockProvider()

    async def orphan():
        host = await provider.provision(HostRequest(ttl_s=0))
        endpoint = await provider.start_engine(
            host, mock_launch(spec, MockConfig(time_scale=0.01, models=[spec.id])), warm=False
        )
        return host, endpoint

    host, endpoint = asyncio.run(orphan())
    past = datetime.now(UTC) - timedelta(minutes=1)
    with session_scope(db) as s:
        exp = repo.create_experiment(
            s,
            name="died",
            spec={"provider": {"kind": "mock"}},
            git=GitInfo(),
            budget_micros=None,
        )
        for rid in (host.host_id, "mock-from-a-dead-process"):
            repo.record_resource(
                s,
                provider="mock",
                resource_type="mock_server",
                resource_id=rid,
                experiment_id=exp.id,
                ttl_at=past,
            )
        repo.record_resource(
            s,
            provider="mock",
            resource_type="mock_server",
            resource_id="still-alive",
            ttl_at=datetime.now(UTC) + timedelta(hours=1),
        )

    dry = invoke("reap", "--dry-run", "--db", db)
    assert dry.exit_code == 0 and "would" in dry.output
    assert host.host_id in live_host_ids()

    result = invoke("reap", "--db", db)
    assert result.exit_code == 0, result.output
    assert host.host_id not in live_host_ids()
    with pytest.raises(httpx.ConnectError):
        httpx.get(endpoint.base_url + "/models")
    with session_scope(db) as s:
        rows = {r.resource_id: r for r in s.scalars(select(BenchResource))}
    assert rows[host.host_id].terminated_by == "reaper"
    assert rows["mock-from-a-dead-process"].terminated_by == "reaper"
    assert rows["still-alive"].terminated_at is None


def _reproducible_doc() -> dict:
    # Slow simulated steps (20 ms real) so latencies dwarf HTTP and scheduler jitter.
    return {
        "provider": {"kind": "mock", "hourly_price": "$1", "time_scale": 0.5, "step_base_ms": 40},
        "workloads": [
            {
                "profile": "fixed-128-128",
                "overrides": {"input_len": 64, "output_len": 16},
                "load": {
                    "mode": "closed_loop",
                    "values": [2],
                    "num_requests": 6,
                    "warmup_requests": 1,
                    "scrape_interval_s": 0.05,
                },
            }
        ],
    }


def test_reproduce_a_run_from_its_provenance(ctx, db, tmp_path):
    exp = mock_experiment(**_reproducible_doc())
    outcome = asyncio.run(run_experiment(exp, ctx))
    assert outcome.status.value == "completed"
    run_id = outcome.run_ids[1]

    result = invoke("reproduce", run_id, "--db", db, "--out", tmp_path / "repro")
    assert result.exit_code == 0, result.output
    assert "reproduced within normal variance" in result.output
    with session_scope(db) as s:
        exps = {e.name: e for e in s.scalars(select(BenchExperiment))}
        repro = exps["unit--reproduce"]
        new = list(s.scalars(select(BenchRun).where(BenchRun.experiment_id == repro.id)))
        old = repo.get_run(s, run_id)
    assert repro.status == "completed"
    assert len(new) == 1
    assert new[0].config_hash == old.config_hash
    assert new[0].provenance["load"] == old.provenance["load"]
    assert new[0].repetition == old.repetition

    prov_file = ctx.out_dir / str(outcome.experiment_id) / "runs" / str(run_id) / "provenance.json"
    by_file = invoke("reproduce", prov_file, "--db", db, "--out", tmp_path / "repro2")
    assert by_file.exit_code == 0, by_file.output


def test_reproduce_reports_a_real_difference(ctx, db, tmp_path):
    outcome = asyncio.run(run_experiment(mock_experiment(**_reproducible_doc()), ctx))
    run_id = outcome.run_ids[0]
    with session_scope(db) as s:  # pretend the original was 3x faster
        run = repo.get_run(s, run_id)
        doc = dict(run.summary)
        doc["throughput"] = {k: v * 3 for k, v in doc["throughput"].items()}
        run.summary = doc
    result = invoke(
        "reproduce", run_id, "--db", db, "--out", tmp_path / "repro", "--tolerance", "0.1"
    )
    assert result.exit_code == EXIT_MISMATCH, result.output


class FlakySpot(MockProvider):
    """Reclaims the first host on its first job."""

    def __init__(self) -> None:
        super().__init__(hourly_micros=1_000_000)
        self.reclaimed: str | None = None

    async def run_job(self, host, job):
        if self.reclaimed is None:
            self.reclaimed = host.host_id
            raise SpotInterrupted(
                host.host_id,
                state="terminated",
                reason_code="Server.SpotInstanceTermination",
                reason_message="reclaimed",
                detected_at=datetime.now(UTC),
                seconds_since_launch=1.0,
            )
        return await super().run_job(host, job)


def test_spot_interruption_is_recorded_and_the_cell_retried_on_a_new_host(ctx, db):
    provider = FlakySpot()
    ctx.provider = provider
    outcome = asyncio.run(run_experiment(mock_experiment(**_reproducible_doc()), ctx))
    assert outcome.status.value == "completed", outcome.reason
    with session_scope(db) as s:
        runs = list(s.scalars(select(BenchRun)))
        resources = list(s.scalars(select(BenchResource)))
    assert [r.status for r in runs].count(RUN_INTERRUPTED) == 1
    assert sum(r.status == "completed" for r in runs) == 2
    assert len(resources) == 2 and all(r.terminated_at for r in resources)
    assert any(e["kind"] == "spot_interruption" for e in outcome.events)


def test_second_spot_interruption_fails_the_experiment(ctx, db):
    class AlwaysReclaimed(FlakySpot):
        async def run_job(self, host, job):
            self.reclaimed = None
            return await super().run_job(host, job)

    ctx.provider = AlwaysReclaimed()
    outcome = asyncio.run(run_experiment(mock_experiment(**_reproducible_doc()), ctx))
    assert outcome.status.value == "failed"
    assert "SpotInterrupted" in outcome.reason
    assert not live_host_ids()


def test_overall_cap_refusal_uses_prior_spend_through_the_cli(db, tmp_path):
    with session_scope(db) as s:
        prior = repo.create_experiment(
            s,
            name="earlier-aws",
            spec={"provider": {"kind": "aws_ec2"}},
            git=GitInfo(),
            budget_micros=None,
        )
        s.get(BenchExperiment, prior.id).spent_micros = parse_usd("$149.999")
    doc = {**mock_experiment().model_dump(mode="json"), "name": "after"}
    path = write_yaml(tmp_path / "after.yaml", doc)
    result = invoke("run", path, "--db", db, "--out", tmp_path)
    assert result.exit_code == 3, result.output
    assert "$0.00 already spent" not in result.output
    with session_scope(db) as s:
        assert (
            s.scalars(select(BenchExperiment).where(BenchExperiment.name == "after")).first()
            is None
        )
