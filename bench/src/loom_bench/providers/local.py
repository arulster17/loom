"""Local provider: an OpenAI-compatible endpoint that is already running.

Nothing is provisioned or billed (market `local`); the engine is never
restarted, so a local experiment has exactly one cell.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import httpx

from loom_bench.experiment import LocalProviderSpec
from loom_bench.jobexec import execute_load_job
from loom_bench.jobs import LoadJob, LoadJobResult
from loom_bench.providers.base import Endpoint, EngineLaunch, Host, HostRequest
from loom_bench.records import Market


class LocalProvider:
    name = "local"

    def __init__(self, settings: LocalProviderSpec) -> None:
        self.settings = settings

    async def provision(self, req: HostRequest) -> Host:
        now = datetime.now(UTC)
        return Host(
            provider=self.name,
            host_id=f"local-{uuid.uuid4().hex[:12]}",
            request=req.model_copy(update={"market": Market.LOCAL}),
            hourly_micros=0,
            launched_at=now,
            ttl_at=now + timedelta(seconds=req.ttl_s),
            info={"base_url": self.settings.base_url},
        )

    async def start_engine(self, host: Host, launch: EngineLaunch, *, warm: bool) -> Endpoint:
        s = self.settings
        t0 = time.monotonic()
        system: dict[str, object] = {"engine": s.engine}
        async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
            resp = await http.get(f"{s.base_url.rstrip('/')}/models")
            resp.raise_for_status()
            served = [m.get("id") for m in resp.json().get("data", [])]
            if s.served_model not in served:
                raise RuntimeError(f"{s.base_url} does not serve {s.served_model!r}: {served}")
            root = s.base_url.rstrip("/").removesuffix("/v1")
            try:
                version = await http.get(f"{root}/version")
                if version.status_code == 200:
                    system["engine_version"] = version.json().get("version")
            except (httpx.HTTPError, ValueError):
                pass
        return Endpoint(
            base_url=s.base_url,
            metrics_url=s.metrics_url,
            engine=s.engine,
            served_model=s.served_model,
            start_stages={"engine_healthy": time.monotonic() - t0},
            warm=warm,
            system=system,
        )

    async def stop_engine(self, host: Host) -> None:
        return None

    async def run_job(self, host: Host, job: LoadJob) -> LoadJobResult:
        return await execute_load_job(job)

    async def teardown(self, host: Host) -> None:
        return None

    async def reap(self, now: datetime) -> list[str]:
        return []
