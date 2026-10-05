import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from loom_bench.provenance import EngineInfo, GitInfo, Provenance, build_provenance
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import (
    BenchExperiment,
    BenchSpend,
    ExperimentStatus,
    TerminatedBy,
    WaitlistSignup,
)

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
GIT = GitInfo(sha="a" * 40, dirty=False, branch="main")


class Summary(BaseModel):
    ttft_p50_s: float
    goodput_rps: float


def _experiment(session, **kw):
    args = dict(name="qwen3-8b-l40s", spec={"model": "qwen3-8b"}, git=GIT, budget_micros=50_000_000)
    args.update(kw)
    return repo.create_experiment(session, **args)


def _prov(rep: int = 0) -> Provenance:
    return build_provenance(
        {"engine": "vllm", "max_num_seqs": 256},
        created_at=T0,
        engine=EngineInfo(name="vllm", version="0.11.0"),
        repetition=rep,
    )


def test_experiment_round_trip(db_url):
    with session_scope(db_url) as s:
        exp_id = _experiment(s, created_at=T0).id
    with session_scope(db_url) as s:
        exp = s.get(BenchExperiment, exp_id)
        assert exp.spec == {"model": "qwen3-8b"}
        assert exp.spec_hash and exp.git_sha == "a" * 40 and exp.git_dirty is False
        assert exp.status == "planned" and exp.spent_micros == 0
        assert exp.created_at == T0 and exp.created_at.tzinfo is not None
        assert exp.created_at.utcoffset() == timedelta(0)


def test_status_transitions(session):
    exp = _experiment(session)
    repo.update_experiment_status(session, exp.id, ExperimentStatus.RUNNING)
    assert exp.finished_at is None
    repo.update_experiment_status(session, exp.id, "aborted", abort_reason="budget", at=T0)
    assert exp.status == "aborted" and exp.abort_reason == "budget" and exp.finished_at == T0
    with pytest.raises(ValueError):
        repo.update_experiment_status(session, exp.id, "running")
    with pytest.raises(ValueError):
        repo.update_experiment_status(session, exp.id, "bogus")
    with pytest.raises(LookupError):
        repo.update_experiment_status(session, uuid.uuid4(), "running")


def test_add_spend_keeps_running_total(db_url):
    with session_scope(db_url) as s:
        exp_id = _experiment(s).id
        basis = {"price_per_hour_micros": 1_861_000, "seconds": 600, "market": "spot"}
        assert repo.add_spend(s, exp_id, 310_167, basis=basis, resource_id="i-1") == 310_167
        assert repo.add_spend(s, exp_id, 1_000, basis=basis) == 311_167
        with pytest.raises(TypeError):
            repo.add_spend(s, exp_id, 1.5, basis=basis)
        with pytest.raises(ValueError):
            repo.add_spend(s, exp_id, -1, basis=basis)
        with pytest.raises(LookupError):
            repo.add_spend(s, uuid.uuid4(), 1, basis=basis)
    with session_scope(db_url) as s:
        assert s.get(BenchExperiment, exp_id).spent_micros == 311_167
        rows = s.scalars(select(BenchSpend).order_by(BenchSpend.amount_micros)).all()
        assert [r.amount_micros for r in rows] == [1_000, 310_167]
        assert rows[1].basis["market"] == "spot" and rows[1].resource_id == "i-1"


def test_run_round_trip_with_models(db_url):
    p = _prov()
    with session_scope(db_url) as s:
        exp_id = _experiment(s).id
        run = repo.record_run(
            s,
            experiment_id=exp_id,
            config_hash=p.config_hash,
            provenance=p,
            summary=Summary(ttft_p50_s=0.125, goodput_rps=12.5),
            status="completed",
            cell_key="vllm/l40s/chat",
            workload="chat",
            load_mode="open_loop",
            load_value=4.0,
            repetition=0,
            requests_uri="results/run.parquet",
            started_at=T0,
            finished_at=T0 + timedelta(minutes=5),
        )
        run_id = run.id
    with session_scope(db_url) as s:
        got = repo.get_run(s, run_id)
        assert Provenance.model_validate(got.provenance) == p
        assert got.summary == {"ttft_p50_s": 0.125, "goodput_rps": 12.5}
        assert got.load_mode == "open_loop" and got.load_value == 4.0
        assert got.finished_at - got.started_at == timedelta(minutes=5)
        assert repo.get_run(s, uuid.uuid4()) is None


def test_record_run_takes_the_callers_run_id(session):
    exp = _experiment(session)
    p = _prov()
    rid = uuid.uuid4()
    run = repo.record_run(
        session,
        experiment_id=exp.id,
        config_hash=p.config_hash,
        provenance=p,
        status="ok",
        run_id=rid,
    )
    assert run.id == rid
    assert repo.get_run(session, rid) is run


def test_record_run_validates(session):
    exp = _experiment(session)
    p = _prov()
    with pytest.raises(ValueError):
        repo.record_run(
            session, experiment_id=exp.id, config_hash="other", provenance=p, status="ok"
        )
    with pytest.raises(ValueError):
        repo.record_run(
            session,
            experiment_id=exp.id,
            config_hash=p.config_hash,
            provenance=p,
            status="ok",
            load_mode="sideways",
        )


def test_foreign_keys_enforced(db_url):
    with pytest.raises(IntegrityError), session_scope(db_url) as s:
        repo.record_run(s, experiment_id=uuid.uuid4(), config_hash="h", provenance={}, status="ok")


def test_check_constraint_enforced(db_url):
    with pytest.raises(IntegrityError), session_scope(db_url) as s:
        s.add(BenchExperiment(name="x", spec={}, spec_hash="h", status="bogus"))
        s.flush()


def test_naive_datetimes_rejected(db_url):
    with pytest.raises(Exception, match="naive"), session_scope(db_url) as s:
        _experiment(s, created_at=datetime(2026, 1, 1))


def test_list_runs_filters_and_orders(session):
    a, b = _experiment(session), _experiment(session, name="other")
    for rep, (exp, cfg) in enumerate([(a, "h1"), (a, "h2"), (b, "h1")]):
        repo.record_run(
            session,
            experiment_id=exp.id,
            config_hash=cfg,
            provenance={},
            status="completed",
            repetition=rep,
            started_at=T0 - timedelta(minutes=rep),
        )
    assert [r.repetition for r in repo.list_runs(session, experiment_id=a.id)] == [1, 0]
    assert [r.repetition for r in repo.list_runs(session, config_hash="h1")] == [2, 0]
    assert len(repo.list_runs(session, experiment_id=a.id, config_hash="h1")) == 1
    assert len(repo.list_runs(session)) == 3


def test_cold_start_eval_and_gate(session):
    exp = _experiment(session)
    cs = repo.record_cold_start(
        session,
        experiment_id=exp.id,
        kind="cold",
        stages={"provision_s": 41.0, "pull_s": 90.5, "load_s": 30.25},
        total_s=161.75,
        resource_id="i-1",
    )
    assert cs.stages["pull_s"] == 90.5
    ev = repo.record_eval_run(
        session,
        experiment_id=exp.id,
        config_hash="h",
        task="gsm8k",
        task_version="1.0",
        n=500,
        score=0.81,
        ci_low=0.78,
        ci_high=0.84,
        provenance=_prov(),
    )
    assert ev.provenance["engine"]["name"] == "vllm"
    gate = repo.record_gate_decision(
        session,
        experiment_id=exp.id,
        baseline_config_hash="base",
        candidate_config_hash="cand",
        decision="fail",
        details={"gsm8k": {"delta": -0.07, "margin": -0.02}},
    )
    assert gate.decision == "fail"
    with pytest.raises(ValueError):
        repo.record_cold_start(
            session, experiment_id=exp.id, kind="lukewarm", stages={}, total_s=0.0
        )
    with pytest.raises(ValueError):
        repo.record_gate_decision(
            session,
            experiment_id=exp.id,
            baseline_config_hash="b",
            candidate_config_hash="c",
            decision="maybe",
            details={},
        )


def test_expired_resources_and_termination(session):
    exp = _experiment(session)
    for rid, ttl in [("i-old", -60), ("i-new", 60), ("i-done", -120)]:
        repo.record_resource(
            session,
            provider="aws_ec2",
            resource_type="instance",
            resource_id=rid,
            region="us-east-1",
            experiment_id=exp.id,
            tags={"loom:experiment": str(exp.id)},
            ttl_at=T0 + timedelta(minutes=ttl),
        )
    repo.mark_terminated(session, provider="aws_ec2", resource_id="i-done", by="runner", at=T0)

    expired = repo.list_expired_resources(session, T0)
    assert [r.resource_id for r in expired] == ["i-old"]
    assert repo.list_expired_resources(session, T0 + timedelta(hours=2))[1].resource_id == "i-new"

    first = repo.mark_terminated(
        session, provider="aws_ec2", resource_id="i-old", by=TerminatedBy.REAPER, at=T0
    )
    again = repo.mark_terminated(
        session, provider="aws_ec2", resource_id="i-old", by="self-ttl", at=T0 + timedelta(1)
    )
    assert again.terminated_by == first.terminated_by == "reaper"
    assert again.terminated_at == T0
    assert repo.list_expired_resources(session, T0) == []
    with pytest.raises(LookupError):
        repo.mark_terminated(session, provider="aws_ec2", resource_id="i-nope", by="reaper")


def test_resource_ids_unique_per_provider(db_url):
    with pytest.raises(IntegrityError), session_scope(db_url) as s:
        for _ in range(2):
            repo.record_resource(
                s, provider="aws_ec2", resource_type="instance", resource_id="i-1", ttl_at=T0
            )


def test_waitlist_is_idempotent_and_case_insensitive(db_url):
    with session_scope(db_url) as s:
        first, created = repo.add_waitlist_signup(s, "Ada@Example.com", source="site")
        assert created
        again, created = repo.add_waitlist_signup(s, "  ada@example.COM ", source="other")
        assert not created and again.id == first.id
    with session_scope(db_url) as s:
        rows = s.scalars(select(WaitlistSignup)).all()
        assert [(r.email, r.source) for r in rows] == [("Ada@Example.com", "site")]
        with pytest.raises(ValueError):
            repo.add_waitlist_signup(s, "not-an-email")


def test_waitlist_concurrent_insert_returns_existing(db_url, monkeypatch):
    with session_scope(db_url) as s:
        winner, _ = repo.add_waitlist_signup(s, "ada@example.com")
    lookups = iter([None])
    real_find = repo._find_signup
    monkeypatch.setattr(repo, "_find_signup", lambda s, e: next(lookups, real_find(s, e)))
    with session_scope(db_url) as s:
        got, created = repo.add_waitlist_signup(s, "ADA@example.com")
        assert not created and got.id == winner.id


def test_waitlist_unique_index_is_case_insensitive(db_url):
    with session_scope(db_url) as s:
        repo.add_waitlist_signup(s, "ada@example.com")
    with pytest.raises(IntegrityError), session_scope(db_url) as s:
        s.add(WaitlistSignup(email="ADA@example.com", created_at=T0))
        s.flush()
    with session_scope(db_url) as s:
        assert s.scalar(select(func.count()).select_from(WaitlistSignup)) == 1
