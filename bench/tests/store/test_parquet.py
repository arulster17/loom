import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from loom_bench.records import RequestRecord, RequestStatus
from loom_bench.store.parquet import REQUEST_SCHEMA, read_requests, write_requests


def _records() -> list[RequestRecord]:
    return [
        RequestRecord(
            request_id="r1",
            status=RequestStatus.OK,
            sent_at_s=0.1 + 0.2,
            scheduled_at_s=0.25,
            first_token_at_s=0.5,
            finished_at_s=2.0,
            itl_s=[0.01, 0.02, 1e-9],
            prompt_tokens=128,
            completion_tokens=3,
            cached_prompt_tokens=64,
            expected_prompt_tokens=128,
            max_tokens=128,
            finish_reason="stop",
            http_status=200,
            output_text="héllo 👋",
            meta={"prefix_group": 3, "tags": ["a", "b"], "nested": {"x": 1.5, "n": None}},
        ),
        RequestRecord(
            request_id="r2",
            status=RequestStatus.TIMEOUT,
            sent_at_s=1.0,
            http_status=None,
            error="read timeout",
            warmup=True,
        ),
        RequestRecord(request_id="r3", status=RequestStatus.ABORTED, sent_at_s=2.0, itl_s=[]),
    ]


def test_round_trip_is_lossless(tmp_path):
    records = _records()
    uri = write_requests(records, tmp_path / "runs" / "r.parquet")
    assert read_requests(uri) == records


def test_file_uri_and_empty_run(tmp_path):
    uri = f"file://{tmp_path}/empty.parquet"
    write_requests([], uri)
    assert read_requests(uri) == []


def test_file_schema_is_explicit(tmp_path):
    path = tmp_path / "r.parquet"
    write_requests(_records(), path)
    schema = pq.read_schema(path)
    assert schema.equals(REQUEST_SCHEMA, check_metadata=False)
    assert schema.field("itl_s").type == pa.list_(pa.field("item", pa.float64(), nullable=False))
    assert schema.field("meta").type == pa.string()
    assert schema.metadata[b"loom.requests.schema_version"] == b"1"
    table = pq.read_table(path)
    assert table.column("status").to_pylist() == ["ok", "timeout", "aborted"]
    assert table.column("ttft_s").to_pylist()[0] == pytest.approx(0.2)


def test_unsupported_scheme_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported URI scheme"):
        write_requests(_records(), "gs://bucket/r.parquet")
