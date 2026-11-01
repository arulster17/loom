"""Structured JSON logs, one object per line on stderr.

Usage: `log.info("batch.posted", extra=fields(org_id=..., amount_micros=...))`. The
message is a stable event name; details go in fields. Billing never sees prompts or
completions (usage events carry token counts only), and webhook payloads are stored in
`stripe_events`, never logged.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

_FIELDS = "loom_fields"


def fields(**kv: Any) -> dict[str, Any]:
    return {_FIELDS: kv}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, _FIELDS, None)
        if isinstance(extra, dict):
            out.update(extra)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn's access log would print request lines; keep its errors only.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
