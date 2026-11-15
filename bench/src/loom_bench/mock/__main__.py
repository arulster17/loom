"""python -m loom_bench.mock [--host H] [--port P] [--config mock.yaml] [--model M]...
[--set FIELD=VALUE]...

Timing knobs (`--set`, or the YAML): inter-token latency is about `step_base_ms +
decode_ms_per_seq * running sequences`; time to first token about `step_base_ms +
prefill_ms_per_token * uncached prompt tokens` plus any queueing; `time_scale` scales
both (0.01 = 100x faster).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app


def _override(text: str) -> tuple[str, Any]:
    field, sep, value = text.partition("=")
    if not sep or not field:
        raise argparse.ArgumentTypeError(f"expected FIELD=VALUE, got {text!r}")
    return field.strip(), yaml.safe_load(value)


def build_config(
    path: Path | None, models: list[str] | None, overrides: list[tuple[str, Any]]
) -> MockConfig:
    data: dict[str, Any] = {}
    if path is not None:
        data = MockConfig.from_yaml(path).model_dump()
    data.update(overrides)
    if models:
        data["models"] = models
    return MockConfig.model_validate(data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m loom_bench.mock",
        description="OpenAI-compatible mock backend with a simulated batching GPU.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, help="YAML mapping of MockConfig fields")
    parser.add_argument(
        "--model", action="append", help="served model id (repeatable; replaces `models`)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        type=_override,
        default=[],
        metavar="FIELD=VALUE",
        help="override one MockConfig field (YAML value), e.g. --set step_base_ms=20",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    try:
        config = build_config(args.config, args.model, args.overrides)
    except ValueError as e:
        parser.error(str(e))
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
