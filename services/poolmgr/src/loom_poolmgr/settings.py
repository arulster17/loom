"""Pool manager settings: the platform's `LOOM_*` variables plus `LOOM_POOLMGR_*`
(docs/phase1-m2-spec.md §11) and `RUNPOD_API_KEY`.

- `LOOM_POOLMGR_SERVING_YAML`: the serving config (else `LOOM_SERVING_YAML`, else
  `config/serving.yaml`);
- `LOOM_POOLMGR_CA_CERT`, `LOOM_POOLMGR_CA_KEY`: the Loom pod CA, to issue pod server
  certificates;
- `LOOM_POOLMGR_CLIENT_CERT`, `LOOM_POOLMGR_CLIENT_KEY`: the manager's client
  certificate, signed by that CA;
- `LOOM_POOLMGR_SSH_KEY`: the private key serving pods authorize (`<key>.pub` next to it);
- `LOOM_POOLMGR_LEASE_BUCKET`: the S3 bucket of the lease records;
- `LOOM_POOLMGR_PROXY_BUNDLE`: the `loom-podproxy` bundle tarball (its sha256 must match
  `config/serving-artifacts.lock`);
- `LOOM_POOLMGR_METRICS_PORT`: Prometheus (9102; 0 = none);
- `LOOM_POOLMGR_LOCAL_COLD_START_S`: `provider: local`, seconds before a pod runs (20);
- `LOOM_POOLMGR_LOCAL_HOST`: `provider: local`, the host the gateway reaches the local
  pods' proxies at (127.0.0.1; any other value makes them listen on all interfaces);
- `LOOM_POOLMGR_STATE_DIR`: where the manager keeps its per-pod SSH known_hosts files
  (and, in dev, a generated SSH key and CA); required outside dev (known_hosts must
  survive a restart), a temporary directory in dev when unset;
- `LOOM_POOLMGR_ALLOW_CIDRS`: dev only, `provider: local` only: comma-separated CIDRs
  added to the local pods' proxy allowlist (compose: its network's fixed subnet, so the
  gateway container reaches them while `gateway_egress_cidrs` stays empty);
- `LOOM_POOLMGR_IDLE_STOP_S`: dev only: replaces every model's `idle_stop_s` (unset: the
  serving file's; compose sets 600, so `loom-m2-acceptance` sees an idle stop in minutes
  while the M1 acceptance never sees a model go cold mid-run);
- `RUNPOD_API_KEY`: the serving account's key (`provider: runpod` only).

Outside `LOOM_ENV=dev` the manager refuses to start without the CA, the client
certificate, the SSH key, the lease bucket, and (with the serving config) a non-empty
`gateway_egress_cidrs` and a provider other than `local`/`fake`. With `provider: runpod`
the proxy bundle and the lease bucket are required in dev too: the placeholders exist
only for `local`/`fake`.
"""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import SettingsConfigDict

from loom_platform.serving import LOCAL_PROVIDERS, ServingConfig
from loom_platform.settings import PlatformSettings

DEFAULT_METRICS_PORT = 9102
DEFAULT_LOCAL_COLD_START_S = 20.0


class PoolmgrSettings(PlatformSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOOM_", extra="ignore", frozen=True, populate_by_name=True
    )

    poolmgr_serving_yaml: Path | None = None
    poolmgr_ca_cert: Path | None = None
    poolmgr_ca_key: Path | None = None
    poolmgr_client_cert: Path | None = None
    poolmgr_client_key: Path | None = None
    poolmgr_ssh_key: Path | None = None
    poolmgr_lease_bucket: str | None = None
    poolmgr_proxy_bundle: Path | None = None
    poolmgr_metrics_port: Annotated[int, Field(ge=0, lt=65536)] = DEFAULT_METRICS_PORT
    poolmgr_local_cold_start_s: Annotated[float, Field(ge=0)] = DEFAULT_LOCAL_COLD_START_S
    poolmgr_local_host: str = "127.0.0.1"
    poolmgr_state_dir: Path | None = None
    poolmgr_allow_cidrs: str = ""
    poolmgr_idle_stop_s: Annotated[int, Field(ge=0)] | None = None
    runpod_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RUNPOD_API_KEY", "runpod_api_key")
    )

    @field_validator("poolmgr_lease_bucket", "runpod_api_key", "poolmgr_idle_stop_s", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("poolmgr_allow_cidrs")
    @classmethod
    def _cidrs(cls, v: str) -> str:
        for part in _split(v):
            try:
                ipaddress.ip_network(part, strict=False)
            except ValueError:
                raise ValueError(f"LOOM_POOLMGR_ALLOW_CIDRS: {part!r} is not a CIDR") from None
        return v

    @property
    def allow_cidrs(self) -> tuple[str, ...]:
        """`LOOM_POOLMGR_ALLOW_CIDRS` as a list (normalised, e.g. `10.0.0.0/24`)."""
        return tuple(
            str(ipaddress.ip_network(p, strict=False)) for p in _split(self.poolmgr_allow_cidrs)
        )

    def production_problems(self, config: ServingConfig | None = None) -> list[str]:
        """The platform's checks plus the manager's own (see the module docstring)."""
        problems = super().production_problems()
        if config is not None and config.provider == "runpod":
            # Whatever LOOM_ENV is: no paid pod is ever bootstrapped with the dev
            # placeholder bundle or without lease records (spec §11).
            if not self.poolmgr_proxy_bundle:
                problems.append("LOOM_POOLMGR_PROXY_BUNDLE is not set (provider runpod)")
            if not self.poolmgr_lease_bucket:
                problems.append("LOOM_POOLMGR_LEASE_BUCKET is not set (provider runpod)")
        if self.env == "dev":
            return problems
        # Dev-only knobs (compose and the acceptance).
        if self.allow_cidrs:
            problems.append("LOOM_POOLMGR_ALLOW_CIDRS is dev only")
        if self.poolmgr_idle_stop_s is not None:
            problems.append("LOOM_POOLMGR_IDLE_STOP_S is dev only")
        required = {
            "LOOM_POOLMGR_CA_CERT": self.poolmgr_ca_cert,
            "LOOM_POOLMGR_CA_KEY": self.poolmgr_ca_key,
            "LOOM_POOLMGR_CLIENT_CERT": self.poolmgr_client_cert,
            "LOOM_POOLMGR_CLIENT_KEY": self.poolmgr_client_key,
            "LOOM_POOLMGR_SSH_KEY": self.poolmgr_ssh_key,
            "LOOM_POOLMGR_LEASE_BUCKET": self.poolmgr_lease_bucket,
            "LOOM_POOLMGR_PROXY_BUNDLE": self.poolmgr_proxy_bundle,
            # The pods' known_hosts files must survive a restart (spec §11).
            "LOOM_POOLMGR_STATE_DIR": self.poolmgr_state_dir,
        }
        problems += [f"{name} is not set" for name, value in required.items() if not value]
        if config is not None:
            if config.provider in LOCAL_PROVIDERS:
                problems.append(f"provider {config.provider} is dev only")
            if not config.network.gateway_egress_cidrs:
                problems.append("network.gateway_egress_cidrs is empty")
            if config.provider == "runpod" and self.runpod_api_key is None:
                problems.append("RUNPOD_API_KEY is not set")
        return problems


def _split(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


@lru_cache(maxsize=1)
def get_poolmgr_settings() -> PoolmgrSettings:
    return PoolmgrSettings()
