"""Flat exports of runs: one row per run, summary and provenance as dotted columns.

Column order is stable: the fixed run columns, then `summary.*` sorted, then
`provenance.*` sorted. Lists and empty objects are kept as canonical JSON text.
"""

from __future__ import annotations

import csv
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from loom_bench.provenance import canonical_json
from loom_bench.store.models import BenchRun

RUN_COLUMNS = (
    "id",
    "experiment_id",
    "cell_key",
    "config_hash",
    "workload",
    "load_mode",
    "load_value",
    "repetition",
    "status",
    "requests_uri",
    "started_at",
    "finished_at",
)


def flatten(doc: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in doc.items():
        name = f"{prefix}.{key}"
        if isinstance(value, Mapping) and value:
            nested = flatten(value, name)
        elif isinstance(value, Mapping | list):
            nested = {name: canonical_json(value)}
        else:
            nested = {name: value}
        for k, v in nested.items():
            if k in out:
                raise ValueError(f"column {k!r} produced twice; a key contains '.'")
            out[k] = v
    return out


def run_table(runs: Iterable[BenchRun]) -> tuple[list[str], list[dict[str, Any]]]:
    """Column names and one dict per run (missing values are None)."""
    rows: list[dict[str, Any]] = []
    summary_cols: set[str] = set()
    prov_cols: set[str] = set()
    for run in runs:
        row = {c: getattr(run, c) for c in RUN_COLUMNS}
        row["id"], row["experiment_id"] = str(run.id), str(run.experiment_id)
        summary = flatten(run.summary or {}, "summary")
        prov = flatten(run.provenance, "provenance")
        summary_cols.update(summary)
        prov_cols.update(prov)
        rows.append(row | summary | prov)
    columns = [*RUN_COLUMNS, *sorted(summary_cols), *sorted(prov_cols)]
    return columns, [{c: row.get(c) for c in columns} for row in rows]


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def export_runs_csv(runs: Iterable[BenchRun], path: str | Path) -> Path:
    columns, rows = run_table(runs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(row[c]) for c in columns])
    return out


def _arrow_column(values: list[Any]) -> pa.Array:
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError):  # mixed types across runs
        return pa.array(
            [v if v is None or isinstance(v, str) else canonical_json(v) for v in values]
        )


def export_runs_parquet(runs: Iterable[BenchRun], path: str | Path) -> Path:
    columns, rows = run_table(runs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({c: _arrow_column([row[c] for row in rows]) for c in columns})
    pq.write_table(table, out)
    return out
