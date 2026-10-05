"""An experiment replays a production trace: its token lengths and its arrival times."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from loom_bench.loadgen.arrivals import read_azure_trace, trace_offsets_at_rate
from loom_bench.runner import run_experiment
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.parquet import read_requests

from .conftest import mock_experiment

TRACE = Path(__file__).parent / "fixtures" / "azure-tiny.csv"
DURATION_S = 2.0
RATES = (4.0, 8.0)


def trace_doc(arrival: dict | None = None) -> dict:
    return {
        "profile": "trace-azure-code",
        "overrides": {"path": str(TRACE), "max_input_len": 512, "max_output_len": 16},
        "load": {
            "mode": "open_loop",
            "values": list(RATES),
            "duration_s": DURATION_S,
            "arrival": arrival or {"kind": "trace", "path": str(TRACE), "format": "azure"},
            "drain_timeout_s": 5,
            "scrape_interval_s": 0.1,
        },
    }


def test_trace_arrivals_take_their_rate_from_the_load_value():
    load = mock_experiment(workloads=[trace_doc()]).workloads[0].load
    assert load.arrival_for(4.0) == {
        "kind": "trace",
        "path": str(TRACE),
        "format": "azure",
        "time_scale": None,
        "rate": 4.0,
        "max_rows": None,
    }
    for extra in ({"rate": 2}, {"time_scale": 0.5}):
        arrival = {"kind": "trace", "path": str(TRACE), "format": "azure", **extra}
        with pytest.raises(ValidationError, match="load value"):
            mock_experiment(workloads=[trace_doc(arrival)])


@pytest.mark.timeout(120)
async def test_mock_experiment_replays_trace_arrivals_and_lengths(ctx):
    exp = mock_experiment(workloads=[trace_doc()], name="trace")
    outcome = await run_experiment(exp, ctx)
    assert outcome.status.value == "completed", outcome.reason
    with session_scope(ctx.db_url) as s:
        runs = repo.list_runs(s, experiment_id=outcome.experiment_id)
    assert {r.status for r in runs} == {"completed"} and len(runs) == len(RATES) * 2

    rows = read_azure_trace(TRACE)
    for run in runs:
        rate = run.load_value
        expected = trace_offsets_at_rate(rows, rate, DURATION_S)
        records = sorted(read_requests(run.requests_uri), key=lambda r: r.scheduled_at_s)
        assert len(records) == round(rate * DURATION_S) == len(expected)
        # the trace's bursts survive: arrivals land where the scaled trace puts them
        sent = np.array([r.scheduled_at_s for r in records])
        np.testing.assert_allclose(sent - sent[0], expected, atol=1e-6)
        # request i carries trace row i's prompt length (the mock counts simple tokens)
        assert [r.prompt_tokens for r in records] == [
            row.input_tokens for row in rows[: len(records)]
        ]
