"""Static results site from a snapshot: plain HTML, one stylesheet, no external assets.

Pages: index (headline table, waitlist), one page per model (leaderboards with
CIs, config YAML, provenance links), methodology, pricing transparency, harness.
The snapshot is copied to `data/` in the output so every provenance link resolves.
Links are relative, so the site works from any base path (e.g. GitHub Pages).
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, cast

import yaml
from jinja2 import ChoiceLoader, Environment, PackageLoader

from loom_bench.cost import CostAllocation, cost_at_slo
from loom_bench.experiment import AwsEc2ProviderSpec
from loom_bench.money import SECONDS_PER_HOUR, TOKENS_PER_MTOK, Micros
from loom_bench.prices import BlockStorage, InstanceType, PriceBook, UnverifiedPriceError
from loom_bench.records import LoadMode, Market
from loom_bench.registry import Cloud, ModelSpec
from loom_bench.report.analyze import (
    CLOUD_NAMES,
    ConfigLabel,
    ConfigResult,
    LoadPoint,
    gate_label,
    name_configs,
    reference_label,
)
from loom_bench.report.competitiveness import (
    CompetitivenessRow,
    comparison_scope,
    margin_text,
    price_text,
)
from loom_bench.report.format import jinja_env, load_mode_label
from loom_bench.report.leaderboard import (
    Leaderboard,
    LeaderboardRow,
    RowStatus,
    build_leaderboard,
    cold_text,
    headline_basis,
    headline_row,
    no_headline_reason,
    price_header,
    quality_text,
    ranking_cost,
    split_ranges,
    unranked_reason,
)
from loom_bench.report.methodology import (
    ALLOCATION_TEXT,
    SPLIT_TEXT,
    ConfigProvenance,
    Methodology,
    methodology,
    storage_name,
)
from loom_bench.site.config import SiteConfig, load_site_config
from loom_bench.site.snapshot import ModelSnapshot, Snapshot, load_snapshot
from loom_bench.stats import Estimate

DATA_DIR = "data"
STYLESHEET = "style.css"
LEGAL_NOTE = "docs/legal/competitor-benchmarking.md"
# Throughput for the illustrative cost example when no result is published yet.
EXAMPLE_OUTPUT_TOK_S = 1000.0


@dataclass(frozen=True)
class Page:
    key: str  # nav key
    path: str  # output path relative to the site root
    title: str


NAV = (
    Page("results", "index.html", "Results"),
    Page("methodology", "methodology.html", "Methodology"),
    Page("pricing", "pricing.html", "Pricing"),
    Page("harness", "harness.html", "Harness"),
)


def slug(*parts: object) -> str:
    text = "-".join(str(p) for p in parts if p is not None)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def to_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def short_sha(sha: str | None) -> str:
    return sha[:12] if sha else "not recorded"


@cache
def _env() -> Environment:
    """The report environment (shared filters and globals) plus site-only helpers."""
    env = jinja_env(
        ChoiceLoader(
            [
                PackageLoader("loom_bench.site", "templates"),
                PackageLoader("loom_bench.report", "templates"),
            ]
        )
    )
    env.filters.update(yaml=to_yaml, short_sha=short_sha, sentence=sentence)
    env.globals.update(
        cold_text=cold_text,
        gate_label=gate_label,
        quality_text=quality_text,
        unranked_reason=unranked_reason,
        split_ranges=split_ranges,
        margin_text=margin_text,
        price_text=price_text,
        price_header=price_header,
        ranking_cost=ranking_cost,
        slug=slug,
        nav=NAV,
    )
    return env


def stylesheet() -> str:
    """The report stylesheet (colour tokens, tables, badges) followed by the site layout."""
    report_css = resources.files("loom_bench.report").joinpath("templates/style.css")
    site_css = resources.files("loom_bench.site").joinpath("templates/site.css")
    return report_css.read_text(encoding="utf-8") + "\n" + site_css.read_text(encoding="utf-8")


# ---- view models ---------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedConfig:
    engine: str
    image: str
    hardware: str
    parallelism: str
    quantization: str
    max_context: int
    license: str
    status: str

    @classmethod
    def of(cls, spec: ModelSpec) -> PlannedConfig:
        hw, par = spec.hardware, spec.parallelism
        instances = ", ".join(f"{c} {t}" for c, t in hw.instance_types.model_dump().items() if t)
        return cls(
            engine=f"{spec.engine.name} {spec.engine.version}",
            image=spec.engine.image,
            hardware=f"{hw.gpus_per_replica}×{hw.gpu} per replica ({instances})",
            parallelism=f"TP{par.tp} · PP{par.pp} · EP{par.ep}",
            quantization=ConfigLabel(
                engine=None,
                engine_version=None,
                quantization=spec.quantization,
                tp=None,
                gpu_type=None,
                gpu_count=None,
                instance_type=None,
                cloud=None,
                region=None,
                market=None,
                engine_args={},
                kv_cache_dtype=spec.kv_cache_dtype,
            ).precision_text,
            max_context=spec.max_context,
            license=spec.hf.license,
            status=spec.status,
        )


@dataclass(frozen=True)
class ConfigView:
    row: LeaderboardRow
    provenance: ConfigProvenance
    anchor: str
    config_yaml: str
    met: Mapping[float, bool]  # load -> SLO met

    @property
    def result(self) -> ConfigResult:
        return self.row.result

    def runs(self) -> list[tuple[LoadPoint, str, int | None]]:
        return [
            (p, run_id, rep)
            for p in self.result.points
            for run_id, rep in zip(p.run_ids, p.repetitions, strict=True)
        ]


@dataclass(frozen=True)
class BoardView:
    board: Leaderboard
    anchor: str
    configs: list[ConfigView]

    @property
    def head(self) -> LeaderboardRow | None:
        """The quoted config (`leaderboard.headline_row`): the same on the index, this
        model page and the pricing page."""
        return headline_row(self.board.rows)

    @property
    def basis(self) -> str:
        """Why the quoted config is quoted, or why nothing is."""
        head = self.head
        if head is None:
            return no_headline_reason(self.board.rows)
        return headline_basis(head)

    @property
    def rank_allocation(self) -> str:
        return self.board.allocation

    @property
    def title(self) -> str:
        b = self.board
        content = f" · {b.content.value} content" if b.content else ""
        return f"{b.workload} · {b.load_mode.value.replace('_', ' ')}{content}"


@dataclass(frozen=True)
class ModelView:
    snap: ModelSnapshot
    spec: ModelSpec | None
    planned: PlannedConfig | None
    boards: list[BoardView]
    methodology: Methodology

    @property
    def path(self) -> str:
        return f"models/{self.snap.model_id}.html"

    @property
    def revision(self) -> str | None:
        if self.spec is not None:
            return self.spec.hf.revision
        revisions = {r.model_revision for r in self.snap.results if r.model_revision}
        return revisions.pop() if len(revisions) == 1 else None


@dataclass(frozen=True)
class HeadlineRow:
    model: ModelView
    board: BoardView | None
    # The quoted config: rank 1, or (nothing ranked) an untrusted config of verified
    # quality shown as a comparison point. Never a config whose gate failed or is
    # inconclusive.
    quoted: LeaderboardRow | None

    @property
    def best(self) -> LeaderboardRow | None:
        """The best config: only a ranked one is ever called that."""
        return self.quoted if self.quoted is not None and self.quoted.rank is not None else None


@dataclass(frozen=True)
class WorkedExample:
    measured: bool  # False: illustrative throughput, not a measurement
    source: str  # where the inputs come from
    hourly_micros: Micros
    output_tok_s: float
    output_per_mtok: Micros | None
    instance: str
    # The measured prefill-time split of the same result (absent when not measured).
    share: float | None = None  # φ, the share of replica time charged to input
    input_tok_s: float | None = None
    split_input_per_mtok: Micros | None = None
    split_output_per_mtok: Micros | None = None

    @property
    def has_split(self) -> bool:
        return (
            self.share is not None
            and self.input_tok_s is not None
            and self.split_input_per_mtok is not None
            and self.split_output_per_mtok is not None
        )


@dataclass(frozen=True)
class PriceRow:
    cloud: str
    region: str
    instance: str
    entry: InstanceType
    storage: BlockStorage | None


def _model_view(m: ModelSnapshot, price_book: PriceBook | None) -> ModelView:
    spec = ModelSpec.model_validate(m.registry) if m.registry else None
    report = build_leaderboard(m.results, cold_starts=m.cold_starts, price_book=price_book)
    provenance = {c.key: c for c in report.methodology.configs}
    boards = []
    for b in report.boards:
        anchor = slug(b.workload, b.load_mode.value)
        configs = [
            ConfigView(
                row=row,
                provenance=provenance[row.result.key],
                anchor=slug(anchor, row.result.name, row.result.config_hash[:8]),
                config_yaml=to_yaml(row.result.provenance.get("config") or {}),
                met={p.load: p.met for p in row.result.goodput.points},
            )
            for row in b.rows
        ]
        boards.append(BoardView(board=b, anchor=anchor, configs=configs))
    return ModelView(
        snap=m,
        spec=spec,
        planned=PlannedConfig.of(spec) if spec else None,
        boards=boards,
        methodology=report.methodology,
    )


def _headline(models: Sequence[ModelView]) -> list[HeadlineRow]:
    rows = []
    for m in models:
        if not m.boards:
            rows.append(HeadlineRow(model=m, board=None, quoted=None))
        for b in m.boards:
            rows.append(HeadlineRow(model=m, board=b, quoted=b.head))
    return rows


def _worked_example(
    models: Sequence[ModelView], price_book: PriceBook | None
) -> WorkedExample | None:
    for m in models:
        for b in m.boards:
            for c in b.configs:
                r = c.result
                tok_s = r.goodput.output_tok_s
                if (
                    c.row.status is RowStatus.RANKED
                    and r.cost is not None
                    and r.allocation == "all_output"
                    and tok_s is not None
                ):
                    split = r.split_cost
                    tin = r.goodput.input_tok_s
                    share = split.input_time_share if split else None
                    return WorkedExample(
                        measured=True,
                        source=f"{m.snap.display_name}, {b.title}, config {r.name}",
                        hourly_micros=r.cost.hourly_micros,
                        output_tok_s=tok_s.mean,
                        output_per_mtok=r.cost.output_per_mtok.value,
                        instance=c.provenance.location,
                        share=share.mean if share else None,
                        input_tok_s=tin.mean if tin else None,
                        split_input_per_mtok=split.input_per_mtok.value if split else None,
                        split_output_per_mtok=split.output_per_mtok.value if split else None,
                    )
    if price_book is None:
        return None
    for m in models:
        if m.spec is None or m.spec.hardware.instance_types.aws is None:
            continue
        region = next(iter(price_book.clouds.get("aws", {})), None)
        if region is None:
            continue
        instance = m.spec.hardware.instance_types.aws
        storage_gb = AwsEc2ProviderSpec.model_fields["disk_gb"].default
        try:
            hourly = price_book.replica_hourly_cost(
                "aws", region, instance, Market.ON_DEMAND, storage_gb
            ).per_hour
        except (KeyError, UnverifiedPriceError):
            continue
        point = Estimate(mean=EXAMPLE_OUTPUT_TOK_S, lo=None, hi=None, n=1, std=None)
        cost = cost_at_slo(
            hourly, input_tok_s=point, output_tok_s=point, allocation=CostAllocation.all_output()
        )
        return WorkedExample(
            measured=False,
            source=(
                f"the {m.snap.display_name} replica's instance at its on-demand price, plus "
                f"the default {storage_gb} GB storage volume"
            ),
            hourly_micros=hourly,
            output_tok_s=EXAMPLE_OUTPUT_TOK_S,
            output_per_mtok=cost.output_per_mtok.value,
            instance=f"aws / {region} · {instance} · on-demand",
        )
    return None


def _price_rows(snapshot: Snapshot, models: Sequence[ModelView]) -> list[PriceRow]:
    """Price-book entries for the instances the registry plans and the results used."""
    book = snapshot.price_book
    if book is None:
        return []
    wanted: set[tuple[str, str, str]] = set()
    for r in snapshot.results:
        prov = r.provenance
        instance = (prov.get("hardware") or {}).get("instance_type")
        if prov.get("cloud") and prov.get("region") and instance:
            wanted.add((prov["cloud"], prov["region"], instance))
    for m in models:
        if m.spec is None:
            continue
        for cloud, instance in m.spec.hardware.instance_types.model_dump().items():
            regions = book.clouds.get(cast(Cloud, cloud), {})
            if instance:
                wanted.update((cloud, region, instance) for region in regions)
    rows = []
    for cloud, region, instance in sorted(wanted):
        try:
            entry = book.instance(cast(Cloud, cloud), region, instance)
        except KeyError:
            continue
        storage = book.region(cast(Cloud, cloud), region).storage
        rows.append(
            PriceRow(cloud=cloud, region=region, instance=instance, entry=entry, storage=storage)
        )
    return rows


def _storage_text(price_rows: Sequence[PriceRow]) -> str:
    """Which storage volume each cloud's replicas pay for, e.g. "an EBS gp3 volume on AWS,
    a container disk on RunPod"."""
    clouds = dict.fromkeys(p.cloud for p in price_rows if p.storage is not None)
    return ", ".join(f"{storage_name(c)} on {CLOUD_NAMES.get(c, c)}" for c in clouds)


@dataclass(frozen=True)
class PricingRow:
    """One pricing-page row: the snapshot's competitiveness row and the board it quotes."""

    comp: CompetitivenessRow
    board: BoardView | None

    @property
    def quoted(self) -> str | None:
        return self.comp.best_config

    @property
    def basis(self) -> str:
        if self.board is not None:
            return self.board.basis
        return self.comp.basis or ""


def _pricing_by_model(
    snapshot: Snapshot, models: Sequence[ModelView]
) -> dict[str, list[PricingRow]]:
    """Competitiveness rows per model, each checked against its leaderboard board: the
    pricing page quotes the same config as the index and the model page (one selection
    rule, `leaderboard.headline_row`), or the build refuses the snapshot."""
    boards = {
        (m.snap.model_id, b.board.workload, b.board.load_mode): b for m in models for b in m.boards
    }
    by_model: dict[str, list[PricingRow]] = defaultdict(list)
    if snapshot.competitiveness is None:
        return {}
    for r in snapshot.competitiveness.rows:
        board = (
            boards.get((r.model_id, r.workload, r.load_mode))
            if r.workload is not None and r.load_mode is not None
            else None
        )
        if board is not None and not r.alternative_slo:
            head = board.head
            want = head.result.config_hash if head is not None else None
            if r.best_config_hash != want:
                raise ValueError(
                    f"pricing data for {r.model_id} {r.workload} quotes "
                    f"{r.best_config or 'no config'}, but the leaderboard quotes "
                    f"{head.result.name if head else 'no config'}; the snapshot was exported "
                    "with another selection rule: re-export it with `bench site export`"
                )
            r = r.model_copy(update={"best_config": head.result.name if head else None})
        by_model[r.model_id].append(PricingRow(comp=r, board=board))
    return dict(by_model)


# ---- build ---------------------------------------------------------------------------


def _prepare_out(out: Path, snapshot_dir: Path) -> None:
    if out.resolve() == snapshot_dir.resolve() or snapshot_dir.resolve().is_relative_to(
        out.resolve()
    ):
        raise ValueError(f"output {out} would overwrite the snapshot {snapshot_dir}")
    if out.exists():
        if any(out.iterdir()) and not (out / "index.html").exists():
            raise ValueError(f"{out} is not empty and is not a previous site build")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def _named(snapshot: Snapshot) -> Snapshot:
    """Name every config once across the snapshot (`analyze.name_configs`) and the gate
    references by what they are, so every page uses the same, unique labels. A snapshot
    exported by this version already carries them; an older one gets them here."""
    results = name_configs(snapshot.results)
    labels = {r.config_hash: reference_label(r) for r in results}
    fixed = []
    for r in results:
        q = r.quality
        if q is not None and q.baseline_label is None and q.baseline_config_hash in labels:
            label = labels[q.baseline_config_hash]
            r = r.model_copy(update={"quality": q.model_copy(update={"baseline_label": label})})
        fixed.append(r)
    models, start = [], 0  # `snapshot.results` is every model's results in order
    for m in snapshot.models:
        end = start + len(m.results)
        models.append(m.model_copy(update={"results": fixed[start:end]}))
        start = end
    return snapshot.model_copy(update={"models": models})


def build_site(
    snapshot_dir: str | Path,
    out_dir: str | Path,
    config: SiteConfig | str | Path | None = None,
) -> list[Path]:
    """Render the site for the snapshot in `snapshot_dir` into `out_dir` (replaced).

    `config` is a SiteConfig, a path to `site/config.yaml`, or None for the repo's.
    Returns the HTML pages written.
    """
    snapshot_path, out = Path(snapshot_dir), Path(out_dir)
    cfg = config if isinstance(config, SiteConfig) else load_site_config(config)
    snapshot = _named(load_snapshot(snapshot_path))
    _prepare_out(out, snapshot_path)
    shutil.copytree(snapshot_path, out / DATA_DIR)
    (out / STYLESHEET).write_text(stylesheet(), encoding="utf-8")

    models = [_model_view(m, snapshot.price_book) for m in snapshot.models]
    manifest = snapshot.manifest
    price_rows = _price_rows(snapshot, models)
    common: dict[str, Any] = {
        "config": cfg,
        "manifest": manifest,
        "snapshot": snapshot,
        "models": models,
        "legal_url": f"{cfg.repo}/blob/main/{LEGAL_NOTE}",
        "registry_url": f"{cfg.repo}/blob/main/config/models.yaml",
    }
    all_results = snapshot.results
    pages: list[tuple[str, str, dict[str, Any]]] = [
        ("index.html", "index.html.j2", {"page": "results", "headline": _headline(models)}),
        (
            "methodology.html",
            "methodology.html.j2",
            {
                "page": "methodology",
                "method": methodology(all_results, snapshot.price_book),
                "example": _worked_example(models, snapshot.price_book),
                "price_rows": price_rows,
                "storage_text": _storage_text(price_rows),
                "allocation_text": ALLOCATION_TEXT,
                "split_text": SPLIT_TEXT,
                "seconds_per_hour": SECONDS_PER_HOUR,
                "tokens_per_mtok": TOKENS_PER_MTOK,
                "open_loop": load_mode_label(LoadMode.OPEN_LOOP),
                "closed_loop": load_mode_label(LoadMode.CLOSED_LOOP),
            },
        ),
        (
            "pricing.html",
            "pricing.html.j2",
            {
                "page": "pricing",
                "comp": snapshot.competitiveness,
                "by_model": _pricing_by_model(snapshot, models),
                "scope": (
                    comparison_scope(snapshot.competitiveness) if snapshot.competitiveness else ""
                ),
            },
        ),
        ("harness.html", "harness.html.j2", {"page": "harness"}),
    ]
    pages += [(m.path, "model.html.j2", {"page": "model", "model": m}) for m in models]

    env = _env()
    written = []
    for path, template, context in pages:
        root = "../" * path.count("/")
        html = env.get_template(template).render(**common, **context, root=root)
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        written.append(target)
    return written
