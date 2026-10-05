"""python -m loom_bench.mock [--host H] [--port P] [--config mock.yaml]"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m loom_bench.mock",
        description="OpenAI-compatible mock backend with a simulated batching GPU.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, help="YAML mapping of MockConfig fields")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    config = MockConfig.from_yaml(args.config) if args.config else MockConfig()
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
