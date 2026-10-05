"""Benchmark host providers, constructed by experiment `provider.kind`.

Implementations import lazily so optional dependencies (uvicorn for mock,
boto3 for aws_ec2) load only when that provider is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom_bench.experiment import ProviderSpec
    from loom_bench.providers.base import Provider

PROVIDER_KINDS = ("mock", "local", "aws_ec2")


def make_provider(spec: ProviderSpec) -> Provider:
    if spec.kind == "mock":
        from loom_bench.providers.mock import MockProvider

        return MockProvider(hourly_micros=spec.hourly_price)
    if spec.kind == "local":
        from loom_bench.providers.local import LocalProvider

        return LocalProvider(spec)
    raise NotImplementedError(f"provider {spec.kind!r} is not available in this build")


def spot_interruption_errors() -> tuple[type[BaseException], ...]:
    """Exceptions meaning the host was reclaimed mid-work (retried once on a new host)."""
    return ()
