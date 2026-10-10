import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from site_helpers import EXPORT_GIT, GENERATED_AT, SLO, SPEC, export, make_runs, store_runs

from loom_bench.cost import CostAllocation
from loom_bench.prices import load_prices
from loom_bench.provenance import GitInfo, config_hash
from loom_bench.registry import load_registry
from loom_bench.report import analyze_runs, default_price_resolver, with_quality
from loom_bench.site import SiteConfig, load_site_config, load_snapshot
from loom_bench.site.snapshot import check_pinned, model_slug
from loom_bench.slo import Slo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchEvalRun, BenchGateDecision
from loom_bench.store.repo import create_experiment, list_runs, update_experiment_status

REGISTERED = [m.id for m in load_registry().models]


def test_export_round_trips(populated, tmp_path):
    manifest = export(populated.url, tmp_path)
    snap = load_snapshot(tmp_path)

    with session_scope(populated.url) as s:
        runs = list_runs(s, experiment_id=populated.experiment_id)
        expected = with_quality(
            analyze_runs(
                runs,
                slo=SLO,
                allocation=CostAllocation.all_output(),
                price_resolver=default_price_resolver(load_prices()),
            ),
            s.query(BenchEvalRun).all(),
            s.query(BenchGateDecision).all(),
        )
        stored = {str(r.id): r.provenance for r in runs}

    assert snap.manifest == manifest
    assert manifest.generated_at == GENERATED_AT
    assert manifest.git == EXPORT_GIT
    assert manifest.experiment_ids == [str(populated.experiment_id)]
    assert manifest.slo == SLO
    assert manifest.allocation == "all_output"
    assert manifest.run_count == 36
    assert not manifest.empty

    qwen = next(m for m in snap.models if m.model_id == "qwen3-8b")
    assert sorted(qwen.results, key=lambda r: r.config_hash) == sorted(
        expected, key=lambda r: r.config_hash
    )
    assert qwen.registry is not None and qwen.registry["hf"]["repo"] == "Qwen/Qwen3-8B"
    assert set(qwen.cold_starts) == {populated.hashes["vllm-bf16"]}
    assert {r.name: r.quality.gate for r in qwen.results} == {
        "vllm-bf16": "baseline",
        "sglang-bf16": "review",
        "vllm-awq": "fail",
    }
    llama = next(m for m in snap.models if m.model_id == "llama-3.3-70b-instruct")
    assert llama.results == []

    assert snap.experiments[0].spec == SPEC
    assert snap.experiments[0].status == "completed"
    assert snap.price_book == load_prices()
    assert snap.competitiveness is not None
    assert {r.model_id for r in snap.competitiveness.rows} == set(REGISTERED)

    files = sorted((tmp_path / "provenance").glob("*.json"))
    assert {f.stem for f in files} == {run_id for r in qwen.results for run_id in r.run_ids}
    for f in files:
        doc = json.loads(f.read_text())
        assert doc["provenance"] == stored[f.stem]
        assert doc["provenance_digest"] == config_hash(doc["provenance"])
        assert doc["run"]["summary"]["n_total"] > 0


def test_export_of_empty_store_is_valid_and_empty(empty_db, tmp_path):
    manifest = export(empty_db, tmp_path)
    snap = load_snapshot(tmp_path)
    assert manifest.experiment_ids == []
    assert manifest.run_count == 0
    assert manifest.slo is None and manifest.allocation is None
    assert manifest.empty
    assert [m.model_id for m in snap.models] == REGISTERED  # every model gets a file
    assert all(m.results == [] for m in snap.models)
    assert not (tmp_path / "provenance").exists()


def test_manifest_alone_loads(empty_db, tmp_path):
    export(empty_db, tmp_path / "full")
    (tmp_path / "bare").mkdir()
    manifest = json.loads((tmp_path / "full" / "manifest.json").read_text())
    manifest["models"] = []
    (tmp_path / "bare" / "manifest.json").write_text(json.dumps(manifest))
    snap = load_snapshot(tmp_path / "bare")
    assert snap.models == [] and snap.competitiveness is None and snap.price_book is None


def test_unknown_schema_version_is_refused(empty_db, tmp_path):
    export(empty_db, tmp_path)
    path = tmp_path / "manifest.json"
    doc = json.loads(path.read_text())
    doc["schema_version"] = 99
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema 99"):
        load_snapshot(tmp_path)


def _experiment(url, name, status, created, runs=(), spec=None):
    with session_scope(url) as s:
        exp = create_experiment(
            s,
            name=name,
            spec=spec if spec is not None else SPEC,
            git=GitInfo(),
            budget_micros=None,
            created_at=created,
        )
        store_runs(s, exp.id, runs)
        if status != "planned":
            update_experiment_status(s, exp.id, status)
        return exp.id


def test_export_takes_exactly_the_named_experiments(empty_db, tmp_path):
    """No "newest of each name" default: that picked the 70B EAGLE3 sweep over the run
    meant for publishing. A later experiment is exported only when it is named."""
    runs = make_runs("vllm-bf16", reps=(1, 2), loads=(2.0, 4.0, 6.0))
    old = _experiment(empty_db, "a", "completed", datetime(2026, 9, 1, tzinfo=UTC), runs)
    new = _experiment(empty_db, "a", "completed", datetime(2026, 9, 2, tzinfo=UTC))

    explicit = export(empty_db, tmp_path, [old])
    assert explicit.experiment_ids == [str(old)]
    assert explicit.run_count == 6
    assert export(empty_db, tmp_path, [str(old)[:8], str(old)]).experiment_ids == [str(old)]
    assert export(empty_db, tmp_path, [new, old]).experiment_ids == [str(new), str(old)]
    nothing = export(empty_db, tmp_path, [])
    assert nothing.experiment_ids == [] and nothing.empty
    with pytest.raises(TypeError, match="list of experiment ids"):
        export(empty_db, tmp_path, "latest")


def test_check_pinned_accepts_only_the_pinned_experiments(empty_db, tmp_path):
    a = _experiment(empty_db, "a", "completed", datetime(2026, 9, 1, tzinfo=UTC))
    b = _experiment(empty_db, "b", "completed", datetime(2026, 9, 2, tzinfo=UTC))
    manifest = export(empty_db, tmp_path, [a])
    check_pinned(manifest, [a])
    with pytest.raises(ValueError, match=f"in the snapshot but not pinned: {a}"):
        check_pinned(manifest, [])
    with pytest.raises(ValueError, match=f"pinned but not in the snapshot: {b}"):
        check_pinned(manifest, [a, b])
    check_pinned(export(empty_db, tmp_path, []), [])  # the committed empty snapshot


def test_publish_config_takes_full_ids_only():
    full = "565b8d3f-b521-4e3e-91e5-ee07ee02b94d"
    cfg = SiteConfig.model_validate({"publish": {"experiments": [full]}})
    assert [str(i) for i in cfg.publish.experiments] == [full]
    assert SiteConfig().publish.experiments == []
    with pytest.raises(ValidationError, match="full experiment ids"):
        SiteConfig.model_validate({"publish": {"experiments": ["565b8d3f"]}})
    with pytest.raises(ValidationError, match="twice"):
        SiteConfig.model_validate({"publish": {"experiments": [full, full.upper()]}})
    assert load_site_config().publish.experiments == []  # nothing pinned in the repo yet


def test_reexport_replaces_previous_files(populated, empty_db, tmp_path):
    export(populated.url, tmp_path)
    (tmp_path / "README.txt").write_text("kept")
    export(empty_db, tmp_path)
    assert not (tmp_path / "provenance").exists()
    assert not (tmp_path / "experiments").exists()
    assert (tmp_path / "README.txt").read_text() == "kept"


def test_disagreeing_slos_need_an_explicit_slo(empty_db, tmp_path):
    runs = make_runs("vllm-bf16", reps=(1, 2), loads=(2.0, 4.0))
    other = {"slo": {"ttft_ms": {"p95": 1000}, "max_error_rate": 0.01}}
    _experiment(empty_db, "a", "completed", datetime(2026, 9, 1, tzinfo=UTC), runs, spec=other)
    with pytest.raises(ValueError, match="disagree on the SLO"):
        export(empty_db, tmp_path)
    slo = Slo.model_validate(other["slo"])
    assert export(empty_db, tmp_path, slo=slo).slo == slo


def test_unknown_experiment_is_an_error(empty_db, tmp_path):
    with pytest.raises(LookupError):
        export(empty_db, tmp_path, ["00000000-0000-0000-0000-000000000009"])


def test_unregistered_model_gets_a_slug():
    assert model_slug("Org/Some_Model-7B") == "org-some-model-7b"
