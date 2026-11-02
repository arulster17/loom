"""Stored runs against the roofline.

For each completed run this rebuilds, from its per-request Parquet rows, what the
engine was doing through the measurement window: how many sequences were decoding
(the batch) and how many tokens their KV cache held, every `grid_s` seconds. The
roofline floor of a decode step at that batch and cache size, weighted by tokens
produced, is what the measured inter-token latency (ITL) is compared against:

- the ITL median is a typical decode step (most steps carry no prefill), so
  `floor / ITL p50` is the share of the memory-bandwidth roofline a step achieves;
- the ITL mean also pays for prefill: chunked prefill mixes prompt chunks into
  decode steps, and those steps run at FLOP speed. The prompt tokens prefilled in
  the window give the least GPU time prefill takes (`prefill_share`), so the mean's
  floor is the step floor / (1 - prefill_share).

The database is opened read-only and nothing is written to it.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from loom_bench.profile.roofline import ServingSetup, decode_step, prefill
from loom_bench.registry import ModelSpec

GRID_S = 0.1


@dataclass(frozen=True)
class RunRow:
    """One stored run, as the roofline needs it."""

    run_id: str
    experiment_id: str
    cell: str
    workload: str
    load_mode: str
    load_value: float
    repetition: int
    provenance: dict[str, Any]
    summary: dict[str, Any]
    requests_uri: str | None


def _connect(db_url: str) -> sqlite3.Connection:
    """A read-only connection; only SQLite result stores are supported here."""
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        raise ValueError("bench roofline reads a SQLite results DB (sqlite:///path), read-only")
    path = Path(db_url[len(prefix) :])
    if not path.exists():
        raise ValueError(f"no results DB at {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _hex(experiment_id: str) -> str:
    return experiment_id.replace("-", "").lower()


def load_runs(db_url: str, experiments: Sequence[str]) -> list[RunRow]:
    """Completed runs of the experiments named by id or unique hex prefix (4+)."""
    con = _connect(db_url)
    try:
        ids = [r[0] for r in con.execute("select id from bench_experiments")]
        wanted: list[str] = []
        for ref in experiments:
            ref_hex = _hex(ref)
            hits = [i for i in ids if _hex(i).startswith(ref_hex)]
            if len(ref_hex) < 4 or len(hits) != 1:
                raise ValueError(f"experiment {ref!r} matches {len(hits)} experiments")
            wanted.append(hits[0])
        marks = ",".join("?" * len(wanted))
        rows = con.execute(
            "select id, experiment_id, cell_key, workload, load_mode, load_value, repetition,"
            " provenance, summary, requests_uri from bench_runs"
            f" where status = 'completed' and experiment_id in ({marks})",
            wanted,
        ).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        prov = json.loads(r[7]) if isinstance(r[7], str) else r[7]
        summ = json.loads(r[8]) if isinstance(r[8], str) else (r[8] or {})
        out.append(
            RunRow(
                run_id=str(uuid.UUID(hex=_hex(r[0]))),
                experiment_id=str(uuid.UUID(hex=_hex(r[1]))),
                cell=r[2] or "",
                workload=r[3] or "",
                load_mode=r[4] or "",
                load_value=float(r[5]),
                repetition=int(r[6] or 0),
                provenance=prov,
                summary=summ,
                requests_uri=r[9],
            )
        )
    return out


def setup_for(run: RunRow) -> ServingSetup:
    config = run.provenance["config"]
    spec = ModelSpec.model_validate(config["model"])
    env = dict(config.get("launch", {}).get("env") or {})
    return ServingSetup.from_spec(spec, env=env, label=run.cell)


def speculative(run: RunRow) -> bool:
    args = run.provenance["config"]["model"]["engine"].get("args") or {}
    return any("speculative" in k for k in args)


@dataclass(frozen=True)
class Requests:
    """The columns of one run's requests.parquet the reconstruction reads."""

    sent: np.ndarray
    first: np.ndarray
    finished: np.ndarray
    prompt: np.ndarray
    cached: np.ndarray
    completion: np.ndarray
    warmup: np.ndarray
    itl: list[np.ndarray]

    @classmethod
    def read(cls, path: str | Path) -> Requests:
        t = pq.read_table(
            path,
            columns=[
                "status",
                "sent_at_s",
                "first_token_at_s",
                "finished_at_s",
                "itl_s",
                "prompt_tokens",
                "cached_prompt_tokens",
                "completion_tokens",
                "warmup",
            ],
        ).to_pydict()
        keep = [
            i
            for i, s in enumerate(t["status"])
            if s == "ok"
            and t["first_token_at_s"][i] is not None
            and t["finished_at_s"][i] is not None
        ]

        def col(name: str, default: float = 0.0) -> np.ndarray:
            return np.array(
                [default if t[name][i] is None else t[name][i] for i in keep], dtype=float
            )

        return cls(
            sent=col("sent_at_s"),
            first=col("first_token_at_s"),
            finished=col("finished_at_s"),
            prompt=col("prompt_tokens"),
            cached=col("cached_prompt_tokens"),
            completion=col("completion_tokens"),
            warmup=np.array([bool(t["warmup"][i]) for i in keep]),
            itl=[np.asarray(t["itl_s"][i], dtype=float) for i in keep],
        )


@dataclass
class Observation:
    """One run reconstructed and set against its floor."""

    run: RunRow
    window: tuple[float, float]
    batch_mean: float  # token-weighted: the batch the average token was produced in
    context_mean: float  # cached tokens per decoding sequence, token-weighted
    floor_ms: float  # token-weighted decode-step floor
    floor_parts: dict[str, float]  # weights / kv / compute / comm, token-weighted
    prefill_share: float  # least share of the window prefill takes (FLOPs + TP all-reduce)
    itl_p50_ms: float
    itl_mean_ms: float
    tokens: int
    prompt_mean: float
    ttft_p50_ms: float | None
    kv_tokens_mean: float = 0.0  # time-weighted cached tokens of decoding sequences
    kv_tokens_max: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def floor_with_prefill_ms(self) -> float:
        return self.floor_ms / max(1e-9, 1.0 - self.prefill_share)

    @property
    def achieved_step(self) -> float:
        """Share of the roofline a typical decode step reaches (floor / ITL p50)."""
        return self.floor_ms / self.itl_p50_ms

    @property
    def achieved_mean(self) -> float:
        """Same for the average token, prefill interference included."""
        return self.floor_with_prefill_ms / self.itl_mean_ms


def window_of(req: Requests, window_s: float | None) -> tuple[float, float]:
    measured = req.sent[~req.warmup]
    if measured.size == 0:
        raise ValueError("no measured requests")
    t0 = float(measured.min())
    t1 = t0 + window_s if window_s else float(measured.max())
    return t0, t1


def observe(run: RunRow, req: Requests, setup: ServingSetup, grid_s: float = GRID_S) -> Observation:
    t0, t1 = window_of(req, run.summary.get("window_s"))
    grid = np.arange(t0, t1, grid_s)
    first, fin = req.first[None, :], req.finished[None, :]
    g = grid[:, None]
    active = (first <= g) & (g < fin)
    dur = np.maximum(req.finished - req.first, 1e-9)[None, :]
    generated = np.clip((g - first) / dur, 0.0, 1.0) * req.completion[None, :]
    batch = active.sum(axis=1).astype(float)
    kv = np.where(active, req.prompt[None, :] + generated, 0.0).sum(axis=1)
    floors = [decode_step(setup, b, k) for b, k in zip(batch, kv, strict=True)]
    weight = batch.sum()
    if weight <= 0:
        raise ValueError("no sequence was decoding in the window")

    def tw(values: Iterable[float]) -> float:
        return float(np.dot(batch, np.fromiter(values, dtype=float)) / weight)

    floor_ms = tw(f.total_ms for f in floors)
    parts = {
        "weights_ms": tw(f.weights_ms for f in floors),
        "kv_ms": tw(f.kv_ms for f in floors),
        "compute_ms": tw(f.compute_ms for f in floors),
        "comm_ms": tw(f.comm_ms for f in floors),
    }
    # Prefill in the window: every prompt whose first token came inside it. A cached
    # prefix is not recomputed; when the engine did not report cached tokens per
    # request (vLLM by default), its own prefix-cache hit rate stands in for them.
    inside = (req.first >= t0) & (req.first < t1)
    hit = (run.summary.get("server") or {}).get("prefix_cache_hit_rate") or 0.0
    reported = bool(np.any(req.cached > 0))
    prefill_ms = 0.0
    for p, c in zip(req.prompt[inside], req.cached[inside], strict=True):
        fresh = max(int(p - c if reported else p * (1.0 - hit)), 1)
        floor = prefill(setup, fresh)
        prefill_ms += floor.compute_ms + floor.comm_ms
    share = min(prefill_ms / ((t1 - t0) * 1000), 0.95)
    # Measured inter-token gaps of tokens that arrived inside the window.
    gaps = []
    for f0, itl in zip(req.first, req.itl, strict=True):
        if itl.size == 0:
            continue
        times = f0 + np.cumsum(itl)
        gaps.append(itl[(times >= t0) & (times < t1)])
    all_gaps = np.concatenate(gaps) if gaps else np.array([])
    if all_gaps.size == 0:
        raise ValueError("no tokens in the window")
    ttft = run.summary.get("ttft_ms") or {}
    return Observation(
        run=run,
        window=(t0, t1),
        batch_mean=float(np.dot(batch, batch) / weight),
        context_mean=float(np.sum(kv) / weight),
        floor_ms=floor_ms,
        floor_parts=parts,
        prefill_share=share,
        itl_p50_ms=float(np.median(all_gaps) * 1000),
        itl_mean_ms=float(all_gaps.mean() * 1000),
        tokens=int(all_gaps.size),
        prompt_mean=float(req.prompt[~req.warmup].mean()),
        ttft_p50_ms=ttft.get("p50"),
        kv_tokens_mean=float(kv.mean()),
        kv_tokens_max=float(kv.max()),
    )


def resolve_requests(uri: str | None, roots: Sequence[Path]) -> Path | None:
    """The run's Parquet file: as stored, else under one of `roots` (a run recorded
    with a relative path, or a moved checkout): the stored path's tail is tried from
    its longest."""
    if uri is None:
        return None
    p = Path(uri)
    if p.exists():
        return p
    for root in roots:
        for i in range(len(p.parts)):
            cand = root.joinpath(*p.parts[i:])
            if cand.exists():
                return cand
    return None


def default_roots(db_url: str) -> list[Path]:
    """Where relative `requests_uri`s are looked up: next to the DB, and its parent
    (the checkout `results/` sits in)."""
    if not db_url.startswith("sqlite:///"):
        return []
    db = Path(db_url[len("sqlite:///") :]).resolve()
    return [db.parent.parent, db.parent]


@dataclass
class PointSummary:
    """Repetitions of one load point averaged."""

    experiment: str
    cell: str
    workload: str
    load_mode: str
    load_value: float
    setup: ServingSetup
    speculative: bool
    reps: int
    batch: float
    context: float
    floor_ms: float
    floor_parts: dict[str, float]
    prefill_share: float
    itl_p50_ms: float
    itl_mean_ms: float
    tpot_p50_ms: float | None
    tpot_p95_ms: float | None
    ttft_p50_ms: float | None
    prompt_mean: float
    run_ids: list[str]
    ttft_p95_ms: float | None = None
    kv_usage: float | None = None  # engine-reported KV-cache usage, window mean
    running_mean: float | None = None  # engine-reported running sequences, time mean
    preemptions: float | None = None
    kv_capacity_tokens: float | None = None  # cached tokens / KV usage: the cache's size
    slo: dict[str, Any] = field(default_factory=dict)

    def failing(self) -> list[str]:
        """SLO targets the repetitions' mean exceeds at this point (a point can also fail
        on its CI upper bound alone; then this is empty)."""
        out = []
        for metric in ("ttft_ms", "tpot_ms"):
            target = (self.slo.get(metric) or {}).get("p95")
            value = self.ttft_p95_ms if metric == "ttft_ms" else self.tpot_p95_ms
            if target is not None and value is not None and value > target:
                out.append(f"{metric[:-3].upper()} p95 {value:.0f} > {target:.0f} ms")
        return out

    @property
    def achieved_step(self) -> float:
        return self.floor_ms / self.itl_p50_ms

    @property
    def floor_with_prefill_ms(self) -> float:
        return self.floor_ms / max(1e-9, 1.0 - self.prefill_share)

    @property
    def achieved_mean(self) -> float:
        return self.floor_with_prefill_ms / self.itl_mean_ms

    @property
    def overhead_ms(self) -> float:
        """Typical step time beyond the floor: what profiling has to account for."""
        return self.itl_p50_ms - self.floor_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "cell": self.cell,
            "workload": self.workload,
            "load_mode": self.load_mode,
            "load_value": self.load_value,
            "setup": self.setup.describe(),
            "speculative": self.speculative,
            "reps": self.reps,
            "batch": self.batch,
            "context": self.context,
            "floor_ms": self.floor_ms,
            "floor_parts": self.floor_parts,
            "prefill_share": self.prefill_share,
            "floor_with_prefill_ms": self.floor_with_prefill_ms,
            "itl_p50_ms": self.itl_p50_ms,
            "itl_mean_ms": self.itl_mean_ms,
            "achieved_step": self.achieved_step,
            "achieved_mean": self.achieved_mean,
            "overhead_ms": self.overhead_ms,
            "tpot_p50_ms": self.tpot_p50_ms,
            "tpot_p95_ms": self.tpot_p95_ms,
            "ttft_p50_ms": self.ttft_p50_ms,
            "prompt_mean": self.prompt_mean,
            "ttft_p95_ms": self.ttft_p95_ms,
            "kv_usage": self.kv_usage,
            "running_mean": self.running_mean,
            "preemptions": self.preemptions,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "run_ids": self.run_ids,
        }


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else None


def _server(o: Observation, key: str) -> float | None:
    v = (o.run.summary.get("server") or {}).get(key)
    return float(v) if isinstance(v, int | float) else None


# Below this KV usage the cached-tokens / usage ratio is too noisy to size the cache.
MIN_KV_USAGE_FOR_CAPACITY = 0.15
MAX_PREFIX_HIT_FOR_CAPACITY = 0.05


def _capacity(o: Observation) -> float | None:
    """The KV cache's size in tokens: the most tokens decoding sequences held in the
    window over the fullest the engine reported its cache. Peaks, not means: the
    engine's gauge is averaged over the whole job (warmup and drain included), the
    tokens over the window only. Prompts still prefilling and block rounding also
    fill the cache, so this reads low."""
    usage = _server(o, "kv_usage_max")
    if usage is None or usage < MIN_KV_USAGE_FOR_CAPACITY:
        return None
    # A preempted sequence still looks like it is decoding from the client's side
    # while its cache blocks are freed, so a run with preemptions overcounts.
    if _server(o, "preemptions"):
        return None
    # Shared prefixes share cache blocks, so the cache holds fewer blocks than the
    # sequences' summed contexts: such runs would overstate its size.
    if (_server(o, "prefix_cache_hit_rate") or 0.0) > MAX_PREFIX_HIT_FOR_CAPACITY:
        return None
    return o.kv_tokens_max / usage


def summarize_points(observations: Sequence[Observation]) -> list[PointSummary]:
    groups: dict[tuple[str, str, str, str, float], list[Observation]] = {}
    for o in observations:
        r = o.run
        groups.setdefault(
            (r.experiment_id, r.cell, r.workload, r.load_mode, r.load_value), []
        ).append(o)
    out = []
    for (exp, cell, wl, mode, value), obs in sorted(groups.items()):
        run0 = obs[0].run
        parts = {k: float(np.mean([o.floor_parts[k] for o in obs])) for k in obs[0].floor_parts}
        out.append(
            PointSummary(
                experiment=exp,
                cell=cell,
                workload=wl,
                load_mode=mode,
                load_value=value,
                setup=setup_for(run0),
                speculative=speculative(run0),
                reps=len(obs),
                batch=float(np.mean([o.batch_mean for o in obs])),
                context=float(np.mean([o.context_mean for o in obs])),
                floor_ms=float(np.mean([o.floor_ms for o in obs])),
                floor_parts=parts,
                prefill_share=float(np.mean([o.prefill_share for o in obs])),
                itl_p50_ms=float(np.mean([o.itl_p50_ms for o in obs])),
                itl_mean_ms=float(np.mean([o.itl_mean_ms for o in obs])),
                tpot_p50_ms=_mean([(o.run.summary.get("tpot_ms") or {}).get("p50") for o in obs]),
                tpot_p95_ms=_mean([(o.run.summary.get("tpot_ms") or {}).get("p95") for o in obs]),
                ttft_p50_ms=_mean([o.ttft_p50_ms for o in obs]),
                prompt_mean=float(np.mean([o.prompt_mean for o in obs])),
                run_ids=sorted(o.run.run_id for o in obs),
                ttft_p95_ms=_mean([(o.run.summary.get("ttft_ms") or {}).get("p95") for o in obs]),
                kv_usage=_mean([_server(o, "kv_usage_mean") for o in obs]),
                running_mean=_mean([_server(o, "running_mean") for o in obs]),
                preemptions=_mean([_server(o, "preemptions") for o in obs]),
                kv_capacity_tokens=_mean([_capacity(o) for o in obs]),
                slo=dict(run0.summary.get("slo") or {}),
            )
        )
    return out


@dataclass(frozen=True)
class StepFit:
    """ITL p50 = fixed + per_seq x batch, fitted over one cell's load points.

    `fixed_ms` is what a step costs however few sequences run (its floor: the weight
    read, `weights_floor_ms`); `per_seq_ms` is what each extra decoding sequence adds
    (its floor: reading that sequence's KV cache, `per_seq_floor_ms`). The modelled
    all-reduce time is taken out first, so TP runs compare on the same footing."""

    experiment: str
    cell: str
    workload: str
    points: int
    fixed_ms: float
    per_seq_ms: float
    weights_floor_ms: float
    per_seq_floor_ms: float
    r2: float

    @property
    def fixed_over_floor_ms(self) -> float:
        return self.fixed_ms - self.weights_floor_ms

    @property
    def per_seq_stretch(self) -> float:
        """Measured per-sequence cost over its KV-read floor (1.0 = at peak bandwidth)."""
        return self.per_seq_ms / self.per_seq_floor_ms

    def batch_at(self, itl_ms: float) -> float:
        """Batch whose typical step takes `itl_ms` on this fit."""
        return max(0.0, (itl_ms - self.fixed_ms) / self.per_seq_ms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "cell": self.cell,
            "workload": self.workload,
            "points": self.points,
            "fixed_ms": self.fixed_ms,
            "per_seq_ms": self.per_seq_ms,
            "weights_floor_ms": self.weights_floor_ms,
            "per_seq_floor_ms": self.per_seq_floor_ms,
            "fixed_over_floor_ms": self.fixed_over_floor_ms,
            "per_seq_stretch": self.per_seq_stretch,
            "r2": self.r2,
        }


def fit_steps(points: Sequence[PointSummary]) -> list[StepFit]:
    """One fit per (experiment, cell, workload) with 3+ load points; speculative
    decoding is left out (a step there verifies several tokens per sequence)."""
    groups: dict[tuple[str, str, str], list[PointSummary]] = {}
    for p in points:
        if not p.speculative:
            groups.setdefault((p.experiment, p.cell, p.workload), []).append(p)
    fits = []
    for (exp, cell, wl), ps in sorted(groups.items()):
        if len(ps) < 3:
            continue
        x = np.array([p.batch for p in ps])
        y = np.array([p.itl_p50_ms - p.floor_parts["comm_ms"] for p in ps])
        if np.ptp(x) <= 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
        per_seq_floor = float(np.mean([p.floor_parts["kv_ms"] / p.batch for p in ps]))
        fits.append(
            StepFit(
                experiment=exp,
                cell=cell,
                workload=wl,
                points=len(ps),
                fixed_ms=float(intercept),
                per_seq_ms=float(slope),
                weights_floor_ms=float(np.mean([p.floor_parts["weights_ms"] for p in ps])),
                per_seq_floor_ms=per_seq_floor,
                r2=1 - ss_res / ss_tot,
            )
        )
    return fits


def observe_runs(
    runs: Sequence[RunRow], *, roots: Sequence[Path] = (), grid_s: float = GRID_S
) -> tuple[list[Observation], list[str]]:
    """Observations for every run with readable Parquet rows, and why others were skipped."""
    obs, skipped = [], []
    for run in runs:
        path = resolve_requests(run.requests_uri, roots)
        if path is None:
            skipped.append(f"{run.run_id}: requests.parquet not found ({run.requests_uri})")
            continue
        try:
            obs.append(observe(run, Requests.read(path), setup_for(run), grid_s))
        except ValueError as e:
            skipped.append(f"{run.run_id}: {e}")
    return obs, skipped
