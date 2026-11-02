"""A synthetic results DB with stored runs whose batch, context and step time are known."""

import json
import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from loom_bench.records import RequestRecord, RequestStatus
from loom_bench.registry import load_registry
from loom_bench.store.db import upgrade
from loom_bench.store.parquet import write_requests

REGISTRY = load_registry()
BY_ID = {m.id: m for m in REGISTRY.models}


def steady_requests(
    n: int, *, prompt: int = 1000, out: int = 100, itl_s: float = 0.03, warmup: int = 0
) -> list[RequestRecord]:
    """`n` requests decoding together: sent at 0, first token at 0.1 s, one token every
    `itl_s` after that."""
    recs = []
    for i in range(n):
        first = 0.1
        recs.append(
            RequestRecord(
                request_id=f"r{i}",
                status=RequestStatus.OK,
                sent_at_s=0.0,
                first_token_at_s=first,
                finished_at_s=first + itl_s * (out - 1),
                itl_s=[itl_s] * (out - 1),
                prompt_tokens=prompt,
                completion_tokens=out,
                warmup=i < warmup,
            )
        )
    return recs


def provenance(model_id: str, **patch: Any) -> dict[str, Any]:
    spec = BY_ID[model_id].model_copy(update=patch)
    return {"config": {"model": spec.model_dump(mode="json"), "launch": {"env": {}}}}


def summary(window_s: float, *, tpot_p95: float = 31.0, **server: Any) -> dict[str, Any]:
    return {
        "window_s": window_s,
        "ttft_ms": {"p50": 100.0, "p95": 150.0},
        "tpot_ms": {"p50": 30.0, "p95": tpot_p95},
        "server": server,
        "slo": {"ttft_ms": {"p95": 1000}, "tpot_ms": {"p95": 50}},
    }


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "loom.db"
        self.url = f"sqlite:///{self.path}"
        upgrade(self.url)
        self.experiment_id = uuid.uuid4()
        con = sqlite3.connect(self.path)
        con.execute(
            "insert into bench_experiments (id, name, spec, spec_hash, status, created_at,"
            " spent_micros) values (?, 'synthetic', '{}', 'x', 'completed',"
            " '2026-11-01 00:00:00', 0)",
            (self.experiment_id.hex,),
        )
        con.commit()
        con.close()

    def add_run(
        self,
        *,
        cell: str,
        model_id: str,
        load: float,
        records: list[RequestRecord],
        window_s: float,
        workload: str = "fixed-1k-1k",
        rep: int = 0,
        relative_uri: bool = False,
        summ: dict[str, Any] | None = None,
        **patch: Any,
    ) -> str:
        run_id = uuid.uuid4()
        rel = Path("results") / str(self.experiment_id) / "runs" / str(run_id) / "requests.parquet"
        write_requests(records, self.root.parent / rel if relative_uri else self.root / rel)
        uri = str(rel) if relative_uri else str(self.root / rel)
        con = sqlite3.connect(self.path)
        con.execute(
            "insert into bench_runs (id, experiment_id, cell_key, config_hash, workload,"
            " load_mode, load_value, repetition, status, provenance, summary, requests_uri)"
            " values (?, ?, ?, 'h', ?, 'open_loop', ?, ?, 'completed', ?, ?, ?)",
            (
                run_id.hex,
                self.experiment_id.hex,
                cell,
                workload,
                load,
                rep,
                json.dumps(provenance(model_id, **patch)),
                json.dumps(summ or summary(window_s)),
                uri,
            ),
        )
        con.commit()
        con.close()
        return str(run_id)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    root = tmp_path / "repo" / "results"
    root.mkdir(parents=True)
    return Store(root)


@pytest.fixture
def make_store(tmp_path: Path) -> Callable[[str], Store]:
    def make(name: str) -> Store:
        root = tmp_path / name / "results"
        root.mkdir(parents=True)
        return Store(root)

    return make
