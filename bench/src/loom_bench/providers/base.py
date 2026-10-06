"""Provider contract: where benchmark pools come from.

Implementations: `mock` (in-process simulated GPU), `local` (an endpoint you
already run), `aws_ec2` (tagged spot/on-demand VM running the engine in Docker),
`runpod` (one RunPod pod per engine image, driven over SSH). A Phase 1 `k8s` provider
will reuse the Helm chart.

Lifecycle per host: provision -> start_engine (cold) -> run jobs and evals ->
[stop_engine -> start_engine (warm) -> run jobs and evals]* -> teardown.
Providers report spend inputs (accrual price, market, start time); the runner's
budget guard decides when to abort. Separately, each host carries its as-run cost
price and its basis, which the runner records in every run's provenance. Every
provisioned resource carries a TTL and is recorded so the reaper can remove it if
the runner dies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from loom_bench.jobs import EvalJob, EvalJobResult, LoadJob, LoadJobResult
from loom_bench.provenance import PriceBasis
from loom_bench.records import Market


class HostRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cloud: str | None = None  # aws | gcp | runpod | None for mock/local
    region: str | None = None
    instance_type: str | None = None
    market: Market = Market.LOCAL
    gpus: int = 1
    disk_gb: int = 0
    # Engine image the host is created for. runpod runs the engine as the pod's own
    # container, so a pod serves one image; aws_ec2, mock and local ignore it.
    image: str | None = None
    ttl_s: int  # hard lifetime; the host must not outlive this even if the runner dies
    tags: dict[str, str] = Field(default_factory=dict)  # loom:experiment, loom:owner, ...


class Host(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    host_id: str
    request: HostRequest
    hourly_micros: int  # budget accrual rate (spot x safety multiplier); 0 if unpriced
    launched_at: datetime  # UTC; billing starts here
    ttl_at: datetime
    # As-run cost price (storage included, no safety multiplier) and how it was obtained;
    # None when the host is unpriced or its price was not observable.
    as_run_micros: int | None = None
    price_basis: PriceBasis | None = None
    info: dict[str, Any] = Field(default_factory=dict)  # az, ami, private ip, ...


class EngineLaunch(BaseModel):
    """Fully rendered engine invocation (see loom_bench.engines)."""

    model_config = ConfigDict(extra="forbid")

    engine: str  # vllm | sglang | mock
    image: str  # repo@sha256:... (ignored by mock)
    model_repo: str
    model_revision: str
    served_model: str
    args: list[str]  # engine CLI args after the entrypoint
    env: dict[str, str] = Field(default_factory=dict)
    port: int = 8000
    gpus: int = 1
    ready_timeout_s: float = 1800.0


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str  # with the API prefix, reachable from where this provider runs jobs
    metrics_url: str | None
    engine: str
    served_model: str
    # Stage name -> seconds since the stage clock started (provisioning for cold,
    # engine start for warm), e.g. instance_running, image_pulled, weights_ready,
    # engine_healthy, first_token.
    start_stages: dict[str, float] = Field(default_factory=dict)
    warm: bool = False
    system: dict[str, Any] = Field(default_factory=dict)  # cuda, driver, gpu names, image digest


class HostLost(RuntimeError):
    """The instance is gone or going (TTL self-shutdown, manual termination, ...).

    Raised by any provider; the runner records it and may retry the cell elsewhere.
    """

    def __init__(
        self,
        host_id: str,
        *,
        state: str,
        reason_code: str | None,
        reason_message: str | None,
        detected_at: datetime,
        seconds_since_launch: float,
    ) -> None:
        super().__init__(
            f"{host_id} is {state} ({reason_code or 'no reason'}) after {seconds_since_launch:.0f}s"
        )
        self.host_id = host_id
        self.state = state
        self.reason_code = reason_code
        self.reason_message = reason_message
        self.detected_at = detected_at
        self.seconds_since_launch = seconds_since_launch


class SpotInterrupted(HostLost):
    """EC2 reclaimed the spot instance. The runner records it for the interruption rate."""


class Provider(Protocol):
    name: str

    async def provision(self, req: HostRequest) -> Host: ...

    async def start_engine(self, host: Host, launch: EngineLaunch, *, warm: bool) -> Endpoint: ...

    async def stop_engine(self, host: Host) -> None: ...

    async def run_job(self, host: Host, job: LoadJob) -> LoadJobResult:
        """Execute where latency is measured correctly for this provider."""
        ...

    async def run_eval(self, host: Host, job: EvalJob) -> EvalJobResult:
        """Execute where `Endpoint.base_url` is reachable (on the host for cloud providers)."""
        ...

    async def teardown(self, host: Host) -> None:
        """Idempotent. Must succeed even if the host already self-terminated."""
        ...

    async def reap(self, now: datetime) -> list[str]:
        """Delete this provider's resources whose TTL passed; return their ids."""
        ...
