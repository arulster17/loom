"""Results snapshot: the JSON the public site is built from, committed under `site/data/`.

Layout of a snapshot directory:

    manifest.json            generated_at, git, bench version, experiments, SLO, model index
    models/<model_id>.json   registry entry, ConfigResults and cold starts for one model
    competitiveness.json     CompetitivenessReport: our cost at SLO, price, list prices
    prices.json              the price book the costs were computed with
    experiments/<id>.json    each experiment's record and spec
    provenance/<run_id>.json every run a result references: metadata, summary, provenance

Every model in the registry gets a file, with no results until it is benchmarked. A
`provenance/*.json` file carries the record exactly as stored, so its digest can be
checked: sha256 of its canonical JSON equals `provenance_digest`.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel
from sqlalchemy.orm import Session

from loom_bench import __version__
from loom_bench.cost import CostAllocation
from loom_bench.prices import Competitors, PriceBook, load_competitors, load_prices
from loom_bench.provenance import GitInfo, canonical_json, git_info
from loom_bench.registry import REPO_ROOT, Registry, load_registry
from loom_bench.report.analyze import (
    ColdStartStat,
    ConfigResult,
    analyze_runs,
    cold_starts_by_config,
    default_price_resolver,
    provenance_digest,
    with_quality,
)
from loom_bench.report.competitiveness import CompetitivenessReport, build_competitiveness
from loom_bench.slo import Slo
from loom_bench.store import repo
from loom_bench.store.models import BenchExperiment, BenchRun

# 2: Estimate.method (log-scale CIs), Methodology.ci_methods. 3: price columns
# (ConfigResult.prices and spot, committed_1y and as_run costs; PriceSource.as_run).
SCHEMA_VERSION = 3

MANIFEST = "manifest.json"
COMPETITIVENESS = "competitiveness.json"
PRICES = "prices.json"
MODELS_DIR = "models"
EXPERIMENTS_DIR = "experiments"
PROVENANCE_DIR = "provenance"


class ModelIndexEntry(BaseModel):
    id: str
    display_name: str
    repo: str | None
    configs: int  # ConfigResults in the model file


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    generated_at: AwareDatetime
    git: GitInfo
    bench_version: str
    experiment_ids: list[str]
    slo: Slo | None
    allocation: str | None
    confidence: float
    run_count: int
    models: list[ModelIndexEntry]

    @property
    def empty(self) -> bool:
        return not any(m.configs for m in self.models)


class ModelSnapshot(BaseModel):
    model_id: str
    display_name: str
    repo: str | None
    registry: dict[str, Any] | None  # the config/models.yaml entry; None if not registered
    results: list[ConfigResult]
    cold_starts: dict[str, ColdStartStat]


class ExperimentRecord(BaseModel):
    id: str
    name: str
    status: str
    spec: dict[str, Any]
    spec_hash: str
    git_sha: str | None
    git_dirty: bool | None
    budget_micros: int | None
    spent_micros: int
    abort_reason: str | None
    created_at: AwareDatetime
    finished_at: AwareDatetime | None


class RunRecord(BaseModel):
    id: str
    experiment_id: str
    cell_key: str | None
    config_hash: str
    workload: str | None
    load_mode: str | None
    load_value: float | None
    repetition: int | None
    status: str
    requests_uri: str | None
    started_at: AwareDatetime | None
    finished_at: AwareDatetime | None
    summary: dict[str, Any] | None


class ProvenanceFile(BaseModel):
    run: RunRecord
    provenance_digest: str
    provenance: dict[str, Any]


class Snapshot(BaseModel):
    manifest: Manifest
    models: list[ModelSnapshot]
    competitiveness: CompetitivenessReport | None
    price_book: PriceBook | None
    experiments: list[ExperimentRecord]

    @property
    def results(self) -> list[ConfigResult]:
        return [r for m in self.models for r in m.results]


def model_slug(repo: str) -> str:
    """File-safe id for a model that is not in the registry, from its HF repo."""
    return re.sub(r"[^a-z0-9.]+", "-", repo.lower()).strip("-.") or "unknown"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _dump(model: BaseModel) -> Any:
    return model.model_dump(mode="json")


def _experiments(
    session: Session, experiment_ids: Sequence[uuid.UUID | str]
) -> list[BenchExperiment]:
    if isinstance(experiment_ids, str):  # "latest" used to pick newest-of-each-name
        raise TypeError("experiment_ids must be a list of experiment ids")
    out: list[BenchExperiment] = []
    for value in experiment_ids:
        found = session.get(BenchExperiment, repo.resolve_experiment_id(session, value))
        assert found is not None
        if found not in out:
            out.append(found)
    return out


def _slo(
    slo: Slo | None, experiments: Iterable[BenchExperiment], runs: Iterable[BenchRun]
) -> Slo | None:
    """The SLO the runs were measured against: given, else the one every run and spec agrees on."""
    if slo is not None:
        return slo
    found: dict[str, Slo] = {}
    docs = [exp.spec.get("slo") for exp in experiments]
    docs += [run.summary.get("slo") for run in runs if run.summary]
    for doc in docs:
        if isinstance(doc, dict):
            parsed = Slo.model_validate(doc)
            found[canonical_json(parsed)] = parsed
    if len(found) > 1:
        raise ValueError(
            "experiments disagree on the SLO; pass slo= or export them as separate snapshots"
        )
    return next(iter(found.values()), None)


def _experiment_record(exp: BenchExperiment) -> ExperimentRecord:
    return ExperimentRecord(
        id=str(exp.id),
        name=exp.name,
        status=exp.status,
        spec=exp.spec,
        spec_hash=exp.spec_hash,
        git_sha=exp.git_sha,
        git_dirty=exp.git_dirty,
        budget_micros=exp.budget_micros,
        spent_micros=exp.spent_micros,
        abort_reason=exp.abort_reason,
        created_at=exp.created_at,
        finished_at=exp.finished_at,
    )


def _provenance_file(run: BenchRun) -> ProvenanceFile:
    return ProvenanceFile(
        run=RunRecord(
            id=str(run.id),
            experiment_id=str(run.experiment_id),
            cell_key=run.cell_key,
            config_hash=run.config_hash,
            workload=run.workload,
            load_mode=run.load_mode,
            load_value=run.load_value,
            repetition=run.repetition,
            status=run.status,
            requests_uri=run.requests_uri,
            started_at=run.started_at,
            finished_at=run.finished_at,
            summary=run.summary,
        ),
        provenance_digest=provenance_digest(run.provenance),
        provenance=run.provenance,
    )


def _model_snapshots(
    results: list[ConfigResult],
    cold_starts: dict[str, ColdStartStat],
    registry: Registry,
) -> list[ModelSnapshot]:
    by_repo: dict[str | None, list[ConfigResult]] = {}
    for r in results:
        by_repo.setdefault(r.model_repo, []).append(r)

    def colds(members: list[ConfigResult]) -> dict[str, ColdStartStat]:
        hashes = {r.config_hash for r in members}
        return {h: c for h, c in cold_starts.items() if h in hashes}

    out = []
    for spec in registry.models:
        members = by_repo.pop(spec.hf.repo, [])
        out.append(
            ModelSnapshot(
                model_id=spec.id,
                display_name=spec.display_name,
                repo=spec.hf.repo,
                registry=_dump(spec),
                results=members,
                cold_starts=colds(members),
            )
        )
    for hf_repo, members in sorted(by_repo.items(), key=lambda kv: kv[0] or ""):
        out.append(
            ModelSnapshot(
                model_id=model_slug(hf_repo or "unknown"),
                display_name=hf_repo or "Unknown model",
                repo=hf_repo,
                registry=None,
                results=members,
                cold_starts=colds(members),
            )
        )
    ids = [m.model_id for m in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"model ids collide in the snapshot: {ids}")
    return out


def _clear(out: Path) -> None:
    """Remove a previous snapshot's files, leaving anything else in `out` alone."""
    for name in (MODELS_DIR, EXPERIMENTS_DIR, PROVENANCE_DIR):
        if (out / name).is_dir():
            shutil.rmtree(out / name)
    for name in (MANIFEST, COMPETITIVENESS, PRICES):
        (out / name).unlink(missing_ok=True)


def export_snapshot(
    session: Session,
    out_dir: str | Path,
    experiment_ids: Sequence[uuid.UUID | str],
    *,
    slo: Slo | None = None,
    allocation: CostAllocation | None = None,
    confidence: float = 0.95,
    price_book: PriceBook | None = None,
    registry: Registry | None = None,
    competitors: Competitors | None = None,
    git: GitInfo | None = None,
    generated_at: datetime | None = None,
) -> Manifest:
    """Write a results snapshot of `experiment_ids` to `out_dir`, replacing a previous one.

    The experiments are always named (full ids or unique prefixes); there is no "newest
    of each name" default, which once picked a later tuning sweep over the run meant
    for publishing. `bench site export` passes the ids pinned in `site/config.yaml`
    (`publish.experiments`) unless given `-e`. An empty list writes a snapshot with no
    results.
    The SLO defaults to the one recorded in the experiment specs and run summaries (they
    must agree); cost allocation defaults to all_output.
    Price book, registry and competitors default to the files in the repository.
    """
    out = Path(out_dir)
    price_book = price_book or load_prices()
    registry = registry or load_registry()
    competitors = competitors or load_competitors()
    allocation = allocation or CostAllocation.all_output()

    experiments = _experiments(session, experiment_ids)
    exp_ids = [e.id for e in experiments]
    runs = [run for i in exp_ids for run in repo.list_runs(session, experiment_id=i)]
    slo = _slo(slo, experiments, runs)

    results: list[ConfigResult] = []
    if runs:
        if slo is None:
            raise ValueError("no SLO recorded for these runs; pass slo=")
        results = analyze_runs(
            runs,
            slo=slo,
            allocation=allocation,
            price_resolver=default_price_resolver(price_book),
            confidence=confidence,
        )
        results = with_quality(
            results,
            repo.list_eval_runs(session, exp_ids),
            repo.list_gate_decisions(session, exp_ids),
        )
    cold_starts = cold_starts_by_config(repo.list_cold_starts(session, exp_ids), results)
    models = _model_snapshots(results, cold_starts, registry)
    competitiveness = build_competitiveness(results, registry, competitors, price_book=price_book)

    referenced = {run_id for r in results for run_id in r.run_ids}
    published = sorted((r for r in runs if str(r.id) in referenced), key=lambda r: str(r.id))

    manifest = Manifest(
        generated_at=generated_at or datetime.now(UTC),
        git=git or git_info(REPO_ROOT),
        bench_version=__version__,
        experiment_ids=[str(i) for i in exp_ids],
        slo=slo if results else None,
        allocation=allocation.describe() if results else None,
        confidence=confidence,
        run_count=len(published),
        models=[
            ModelIndexEntry(
                id=m.model_id, display_name=m.display_name, repo=m.repo, configs=len(m.results)
            )
            for m in models
        ],
    )

    out.mkdir(parents=True, exist_ok=True)
    _clear(out)
    for m in models:
        _write_json(out / MODELS_DIR / f"{m.model_id}.json", _dump(m))
    for exp in experiments:
        _write_json(out / EXPERIMENTS_DIR / f"{exp.id}.json", _dump(_experiment_record(exp)))
    for run in published:
        _write_json(out / PROVENANCE_DIR / f"{run.id}.json", _dump(_provenance_file(run)))
    _write_json(out / COMPETITIVENESS, _dump(competitiveness))
    _write_json(out / PRICES, _dump(price_book))
    _write_json(out / MANIFEST, _dump(manifest))
    return manifest


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_pinned(manifest: Manifest, pinned: Sequence[uuid.UUID]) -> None:
    """Raise ValueError unless the snapshot holds exactly the experiments pinned in the
    site config (`publish.experiments`), so what deploys is what was reviewed there."""
    have = {uuid.UUID(i) for i in manifest.experiment_ids}
    want = set(pinned)
    if have == want:
        return
    parts = []
    if extra := sorted(str(i) for i in have - want):
        parts.append("in the snapshot but not pinned: " + ", ".join(extra))
    if missing := sorted(str(i) for i in want - have):
        parts.append("pinned but not in the snapshot: " + ", ".join(missing))
    raise ValueError(
        "the snapshot does not hold the experiments pinned in the site config "
        f"(publish.experiments); {'; '.join(parts)}. Pin them, or re-export with "
        "`bench site export` and no -e"
    )


def load_snapshot(snapshot_dir: str | Path) -> Snapshot:
    """Read a snapshot. Only `manifest.json` is required; missing parts load as empty."""
    root = Path(snapshot_dir)
    manifest = Manifest.model_validate(_read(root / MANIFEST))
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"snapshot schema {manifest.schema_version} is not supported (expected "
            f"{SCHEMA_VERSION}); re-export it"
        )
    models = [
        ModelSnapshot.model_validate(_read(root / MODELS_DIR / f"{m.id}.json"))
        for m in manifest.models
    ]
    comp_path, prices_path = root / COMPETITIVENESS, root / PRICES
    return Snapshot(
        manifest=manifest,
        models=models,
        competitiveness=(
            CompetitivenessReport.model_validate(_read(comp_path)) if comp_path.exists() else None
        ),
        price_book=PriceBook.model_validate(_read(prices_path)) if prices_path.exists() else None,
        experiments=[
            ExperimentRecord.model_validate(_read(root / EXPERIMENTS_DIR / f"{i}.json"))
            for i in manifest.experiment_ids
        ],
    )
