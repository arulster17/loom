from datetime import UTC, datetime, timedelta

from loom_bench.jobs import LoadJob, LoadJobResult, TokenizerSpec
from loom_bench.providers.base import Host, HostRequest
from loom_bench.records import LoadMode, Market, RequestRecord, RequestStatus


def test_load_job_json_round_trip():
    job = LoadJob(
        run_id="r1",
        base_url="http://127.0.0.1:8000",
        engine="mock",
        served_model="mock-model",
        workload={"kind": "synthetic", "input_len": 128},
        tokenizer=TokenizerSpec(kind="simple"),
        mode=LoadMode.OPEN_LOOP,
        load_value=2.0,
        duration_s=10,
    )
    assert LoadJob.model_validate_json(job.model_dump_json()) == job


def test_load_job_result_rebuilds_records():
    rec = RequestRecord(request_id="a", status=RequestStatus.OK, sent_at_s=0.0, itl_s=[0.1])
    res = LoadJobResult(
        run_id="r1",
        mode=LoadMode.CLOSED_LOOP,
        load_value=4,
        t_measure_start_s=0,
        t_measure_end_s=1,
        records=[rec.to_row()],
        scrapes=[(0.5, "vllm:num_requests_running 1\n")],
        started_at="2026-10-04T00:00:00Z",
        finished_at="2026-10-04T00:00:01Z",
    )
    again = LoadJobResult.model_validate_json(res.model_dump_json())
    assert again.request_records() == [rec]
    assert again.scrapes == [(0.5, "vllm:num_requests_running 1\n")]


def test_host_round_trip():
    now = datetime(2026, 10, 4, tzinfo=UTC)
    host = Host(
        provider="mock",
        host_id="h1",
        request=HostRequest(ttl_s=60, market=Market.SPOT),
        hourly_micros=1_000_000,
        launched_at=now,
        ttl_at=now + timedelta(seconds=60),
    )
    assert Host.model_validate_json(host.model_dump_json()) == host
