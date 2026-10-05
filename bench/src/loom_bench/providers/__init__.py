"""Benchmark host providers, constructed by experiment `provider.kind`.

Implementations import lazily so optional dependencies (uvicorn for mock,
boto3 for aws_ec2) load only when that provider is used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom_bench.experiment import ProviderSpec
    from loom_bench.prices import PriceBook
    from loom_bench.providers.base import Provider


def build_wheel(out_dir: Path) -> Path:
    """Build the loom-bench wheel that GPU hosts install to run `bench job run`."""
    from loom_bench.registry import REPO_ROOT

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--package", "loom-bench", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = sorted(out_dir.glob("loom_bench-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"uv build produced no wheel in {out_dir}")
    return wheels[-1]


def export_requirements(extra: str = "") -> str:
    """The dependencies GPU hosts install next to the wheel: `uv.lock` exported for
    loom-bench (plus `extra`), every package pinned to its locked version and hashes,
    loom-bench itself left out. Hosts install it with `pip --require-hashes --no-deps`."""
    from loom_bench.registry import REPO_ROOT

    cmd = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-workspace",
        "--package",
        "loom-bench",
        "--format",
        "requirements-txt",
    ]
    if extra:
        cmd += ["--extra", extra]
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout


def make_provider(spec: ProviderSpec, *, prices: PriceBook, work_dir: Path) -> Provider:
    if spec.kind == "mock":
        from loom_bench.providers.mock import MockProvider

        return MockProvider(hourly_micros=spec.hourly_price)
    if spec.kind == "local":
        from loom_bench.providers.local import LocalProvider

        return LocalProvider(spec)
    from loom_bench.providers.aws_ec2 import AwsEc2Provider, load_aws_settings

    settings = load_aws_settings()
    if settings.region != spec.region:
        raise ValueError(
            f"experiment region {spec.region} != AWS settings region {settings.region}"
        )
    wheel = settings.wheel_path or build_wheel(work_dir / "wheel")
    return AwsEc2Provider(settings, prices=prices, wheel_path=wheel)
