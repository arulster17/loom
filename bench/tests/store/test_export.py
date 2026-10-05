import csv
import uuid
from datetime import UTC, datetime

import pyarrow.parquet as pq

from loom_bench.store.export import RUN_COLUMNS, export_runs_csv, export_runs_parquet, flatten
from loom_bench.store.models import BenchRun

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _run(rep: int, summary: dict, provenance: dict) -> BenchRun:
    return BenchRun(
        id=uuid.UUID(int=rep),
        experiment_id=uuid.UUID(int=99),
        cell_key="vllm/l40s/chat",
        config_hash="h",
        workload="chat",
        load_mode="open_loop",
        load_value=4.0,
        repetition=rep,
        status="completed",
        provenance=provenance,
        summary=summary,
        requests_uri=None,
        started_at=T0,
        finished_at=None,
    )


RUNS = [
    _run(
        0,
        {"ttft_s": {"p50": 0.1, "p99": 0.4}, "goodput_rps": 12.5},
        {"engine": {"name": "vllm", "args": {}}, "git": {"dirty": False}, "tags": ["a"]},
    ),
    _run(1, {"ttft_s": {"p50": 0.2}, "slo_ok": True}, {"engine": {"name": "sglang"}}),
]

EXPECTED_COLUMNS = [
    *RUN_COLUMNS,
    "summary.goodput_rps",
    "summary.slo_ok",
    "summary.ttft_s.p50",
    "summary.ttft_s.p99",
    "provenance.engine.args",
    "provenance.engine.name",
    "provenance.git.dirty",
    "provenance.tags",
]


def test_flatten_dotted_keys():
    assert flatten({"a": {"b": {"c": 1}}, "l": [1, {"z": 2, "y": 1}], "e": {}}, "p") == {
        "p.a.b.c": 1,
        "p.l": '[1,{"y":1,"z":2}]',
        "p.e": "{}",
    }


def test_csv_columns_and_values(tmp_path):
    path = export_runs_csv(RUNS, tmp_path / "out" / "runs.csv")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with path.open(newline="") as f:
        header = next(csv.reader(f))
    assert header == EXPECTED_COLUMNS
    assert rows[0]["id"] == str(uuid.UUID(int=0))
    assert rows[0]["summary.ttft_s.p99"] == "0.4"
    assert rows[1]["summary.ttft_s.p99"] == ""
    assert rows[1]["summary.slo_ok"] == "true"
    assert rows[0]["provenance.git.dirty"] == "false"
    assert rows[0]["provenance.tags"] == '["a"]'
    assert rows[0]["started_at"] == "2026-10-04T12:00:00+00:00"
    assert rows[0]["finished_at"] == ""


def test_column_order_independent_of_run_order(tmp_path):
    a = export_runs_csv(RUNS, tmp_path / "a.csv")
    b = export_runs_csv(list(reversed(RUNS)), tmp_path / "b.csv")
    assert a.read_text().splitlines()[0] == b.read_text().splitlines()[0]


def test_parquet_export(tmp_path):
    mixed = _run(2, {"goodput_rps": "n/a"}, {})
    table = pq.read_table(export_runs_parquet([*RUNS, mixed], tmp_path / "runs.parquet"))
    assert table.column_names == EXPECTED_COLUMNS
    assert table.column("summary.ttft_s.p50").to_pylist() == [0.1, 0.2, None]
    assert table.column("summary.goodput_rps").to_pylist() == ["12.5", None, "n/a"]
    assert table.column("started_at").to_pylist()[0] == T0
