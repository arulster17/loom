"""Per-request rows as Parquet. One file per run; Postgres keeps only its URI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from loom_bench.records import RequestRecord

REQUESTS_SCHEMA_VERSION = "1"

_FIELDS: list[pa.Field[Any]] = [
    pa.field("request_id", pa.string(), nullable=False),
    pa.field("status", pa.string(), nullable=False),
    pa.field("sent_at_s", pa.float64(), nullable=False),
    pa.field("scheduled_at_s", pa.float64()),
    pa.field("first_token_at_s", pa.float64()),
    pa.field("finished_at_s", pa.float64()),
    pa.field("itl_s", pa.list_(pa.field("item", pa.float64(), nullable=False)), nullable=False),
    pa.field("prompt_tokens", pa.int64()),
    pa.field("completion_tokens", pa.int64()),
    pa.field("cached_prompt_tokens", pa.int64()),
    pa.field("expected_prompt_tokens", pa.int64()),
    pa.field("max_tokens", pa.int64()),
    pa.field("finish_reason", pa.string()),
    pa.field("http_status", pa.int64()),
    pa.field("error", pa.string()),
    pa.field("warmup", pa.bool_(), nullable=False),
    pa.field("output_text", pa.string()),
    pa.field("meta", pa.string(), nullable=False),  # JSON object
    # Derived on write for ad-hoc analysis; ignored on read.
    pa.field("ttft_s", pa.float64()),
    pa.field("e2e_s", pa.float64()),
    pa.field("tpot_s", pa.float64()),
    pa.field("queue_delay_s", pa.float64()),
]
REQUEST_SCHEMA = pa.schema(
    _FIELDS, metadata={"loom.requests.schema_version": REQUESTS_SCHEMA_VERSION}
)


def _resolve(uri: str | Path) -> tuple[pafs.FileSystem, str]:
    """Filesystem and path for `uri`. The one place remote schemes get added."""
    text = str(uri)
    parsed = urlparse(text)
    if parsed.scheme == "file":
        return pafs.LocalFileSystem(), parsed.path
    if parsed.scheme:
        raise ValueError(f"unsupported URI scheme {parsed.scheme!r}: {text}")
    return pafs.LocalFileSystem(), Path(text).resolve().as_posix()


def records_to_table(records: Sequence[RequestRecord]) -> pa.Table:
    rows = []
    for r in records:
        row = r.to_row()
        row["meta"] = json.dumps(row["meta"], ensure_ascii=False)
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=REQUEST_SCHEMA)


def table_to_records(table: pa.Table) -> list[RequestRecord]:
    records = []
    for row in table.to_pylist():
        row["meta"] = json.loads(row["meta"])
        records.append(RequestRecord.from_row(row))
    return records


def write_requests(records: Sequence[RequestRecord], uri: str | Path) -> str:
    """Write `records` to `uri` and return the URI to store on the run."""
    fs, path = _resolve(uri)
    fs.create_dir(path.rsplit("/", 1)[0], recursive=True)
    pq.write_table(records_to_table(records), path, filesystem=fs)
    return str(uri)


def read_requests(uri: str | Path) -> list[RequestRecord]:
    fs, path = _resolve(uri)
    return table_to_records(pq.read_table(path, filesystem=fs, schema=REQUEST_SCHEMA))
