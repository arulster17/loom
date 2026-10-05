"""Waitlist signups as a demand signal: count them and record the count in `docs/waitlist.md`."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from loom_bench.registry import REPO_ROOT
from loom_bench.store.models import WaitlistSignup

DEFAULT_DOCS_PATH = REPO_ROOT / "docs" / "waitlist.md"
TABLE_HEADER = "| Date | Count | Source |"


def waitlist_count(session: Session) -> int:
    """Number of signups in the `waitlist_signups` table."""
    return session.scalar(select(func.count()).select_from(WaitlistSignup)) or 0


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def record_count(
    docs_path: str | Path | None,
    count: int,
    source: str,
    *,
    on: dt.date | None = None,
) -> Path:
    """Add `(date, count, source)` to the table in `docs_path` (default `docs/waitlist.md`).

    Recording the same source again on the same date replaces that row's count.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    source = source.strip()
    if not source or "|" in source or "\n" in source:
        raise ValueError(f"source must be one line of text without '|': {source!r}")
    day = (on or dt.datetime.now(dt.UTC).date()).isoformat()
    path = Path(docs_path) if docs_path is not None else DEFAULT_DOCS_PATH

    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header = next(i for i, line in enumerate(lines) if line.strip() == TABLE_HEADER)
    except StopIteration:
        raise ValueError(f"{path} has no table starting with {TABLE_HEADER!r}") from None
    first = header + 2  # after the separator row
    end = first
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1

    row = f"| {day} | {count} | {source} |"
    for i in range(first, end):
        if _cells(lines[i])[0::2] == [day, source]:
            lines[i] = row
            break
    else:
        lines.insert(end, row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
