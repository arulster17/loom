"""Gateway settings: the platform's LOOM_* variables plus the gateway's own
(docs/phase1-contracts.md §2)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from loom_platform.settings import PlatformSettings

# Measured RunPod cold starts (docs/phase1-runpod-plan.md §5), the Retry-After of a
# request to a cold model whose pool hash says nothing better. Gateway constants in M1;
# config/serving.yaml from M2.
COLD_START_ESTIMATE_S: dict[str, int] = {
    "qwen3-8b": 240,
    "qwen3-8b-fp8": 240,
    "llama-3.3-70b-instruct": 450,
    "llama-3.3-70b-instruct-fp8": 450,
}
DEFAULT_COLD_START_S = 300

# /v1/completions without max_tokens: OpenAI's and vLLM's default, sent explicitly.
COMPLETIONS_DEFAULT_MAX_TOKENS = 16


DEV_CORS_ORIGINS = "http://localhost:3000"


class GatewaySettings(PlatformSettings):
    # Recorded on usage events.
    cloud: str = "local"
    region: str | None = None

    # Comma-separated origins allowed by CORS (the web app).
    cors_origins: str = DEV_CORS_ORIGINS

    # mTLS to the pods' proxies (contracts §9). Unset: plain TLS with the system trust
    # store (or plain HTTP for http:// replica URLs, which only tests and local dev use).
    # Outside LOOM_ENV=dev all three are required and http:// replicas are never used.
    gateway_client_cert: str | None = None
    gateway_client_key: str | None = None
    pod_ca_cert: str | None = None

    gateway_host: str = "0.0.0.0"
    gateway_port: Annotated[int, Field(gt=0, lt=65536)] = 8080
    gateway_metrics_port: Annotated[int, Field(gt=0, lt=65536)] = 9100
    log_level: str = "INFO"

    # In-process copy of an auth record; never longer than its Redis entry's remaining
    # TTL, so revocation stays within LOOM_AUTH_CACHE_TTL_S (contracts §6).
    auth_local_ttl_s: Annotated[float, Field(ge=0, le=1)] = 1.0
    # LISTEN on loom_auth_changed (tests turn it off to check the TTL bound alone).
    auth_listen: bool = True
    # How long a pool state read from Redis is reused (contracts §10: at most 1 s).
    pool_cache_ttl_s: Annotated[float, Field(ge=0, le=1)] = 1.0

    upstream_connect_timeout_s: Annotated[float, Field(gt=0)] = 5.0
    # A replica that failed is tried last for this long.
    replica_penalty_s: Annotated[float, Field(ge=0)] = 10.0

    # Shutdown (SIGTERM, contracts §6 "Shutdown"): in-flight requests get this long to
    # finish on their own; then open streams are cut (an error chunk and [DONE], billed
    # for the tokens streamed, status aborted) and anything else still running is
    # cancelled (settled, releasing its reservation). The process exits within about
    # this + SHUTDOWN_SETTLE_S + 2 s; the orchestrator's stop timeout must be longer.
    shutdown_grace_s: Annotated[float, Field(ge=0)] = 10.0

    # Append-only file for settles Redis refused, replayed until it accepts them
    # (loom_gateway.spool). Unset: a per-process temp file, which survives a Redis
    # outage but not a restart; production puts it on a volume.
    settle_spool_path: str | None = None

    redis_max_connections: Annotated[int, Field(gt=0)] = 256
    pg_pool_max: Annotated[int, Field(gt=0)] = 10

    def production_problems(self) -> list[str]:
        problems = super().production_problems()
        if self.env == "dev":
            return problems
        if not self.pod_ca_cert:
            problems.append("LOOM_POD_CA_CERT is not set (pods would be trusted by the system CAs)")
        if not (self.gateway_client_cert and self.gateway_client_key):
            problems.append("LOOM_GATEWAY_CLIENT_CERT / _KEY are not set (no mTLS to pods)")
        if self.cors_origins == DEV_CORS_ORIGINS:
            problems.append("LOOM_CORS_ORIGINS is the docker-compose default (localhost)")
        if not self.settle_spool_path:
            problems.append(
                "LOOM_SETTLE_SPOOL_PATH is not set (settles spooled during a Redis outage "
                "would not survive a restart)"
            )
        return problems

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
