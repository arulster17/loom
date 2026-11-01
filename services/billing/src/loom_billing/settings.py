"""Billing settings: `LOOM_*` variables (docs/phase1-contracts.md §2) plus the Stripe keys.

Stripe mode is explicit, `LOOM_STRIPE_MODE`:

- `stripe` (the default): **real Stripe**, test mode only. Both `STRIPE_SECRET_KEY` and
  `STRIPE_WEBHOOK_SECRET` (the names Stripe's own docs use) are required by anything that
  talks to Stripe (the HTTP API refuses to start without them); a live key (`sk_live_`,
  `rk_live_`) is refused at startup.
- `stub`: **local stub** (`loom_billing.stripe_stub`), local dev only (compose sets it).
  Billing calls the stub's Checkout API at `LOOM_STRIPE_STUB_URL` and verifies webhooks
  with a secret it shares with the stub, `LOOM_STRIPE_STUB_WEBHOOK_SECRET` or the file
  named by `LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE` (compose: generated per stack by
  `loom-devcerts --secret`). There is no default: a secret written in code would let
  anyone forge a purchase. Stripe keys set together with `stub` are refused.

The stub is never chosen because keys are missing: a deployment that forgot its keys
fails to start instead of silently accepting forged webhooks.
"""

from __future__ import annotations

import socket
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict

from loom_platform.settings import PlatformSettings

PositiveInt = Annotated[int, Field(gt=0)]
NonNegInt = Annotated[int, Field(ge=0)]
StripeMode = Literal["stripe", "stub"]

# Micro-dollars per US cent: Stripe amounts are integer cents.
MICROS_PER_CENT = 10_000
# The API key billing sends to the stub (the stub does not check it). Not a secret.
STUB_SECRET_KEY = "sk_test_loom_local_stub"
# Shortest stub webhook secret accepted.
MIN_STUB_SECRET_CHARS = 24


class BillingSettings(PlatformSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOOM_", extra="ignore", frozen=True, populate_by_name=True
    )

    # Usage stream consumer (§7).
    usage_group: str = "billing"
    usage_dead_stream: str = "loom:usage:dead"
    # Consumer name inside the group; default billing-<hostname>.
    billing_consumer: str | None = None
    usage_read_count: PositiveInt = 500
    usage_block_ms: PositiveInt = 2000
    # Pending entries of another consumer idle this long are claimed (crashed process).
    usage_claim_idle_ms: PositiveInt = 60_000

    # Periodic jobs.
    batch_interval_s: PositiveInt = 60
    reconcile_interval_s: PositiveInt = 60
    partition_interval_s: PositiveInt = 86_400
    partition_months_ahead: PositiveInt = 3

    # HTTP: webhook + internal API, and Prometheus.
    billing_host: str = "0.0.0.0"
    billing_port: PositiveInt = 8090
    billing_metrics_port: NonNegInt = 9101  # 0 = no metrics server
    # Bearer token the web app sends to POST /internal/checkout-sessions. Unset = the
    # endpoint answers 503 (never a default secret).
    billing_internal_token: SecretStr | None = None

    # Notifications outbox (a Redis stream the web/email layer consumes).
    notify_stream: str = "loom:notify"
    low_balance_micros: NonNegInt = 1_000_000  # $1
    # Auto-recharge hook (stubbed: emits an outbox event instead of charging a card).
    auto_recharge_enabled: bool = False

    # Checkout amounts the web app may ask for.
    checkout_min_micros: PositiveInt = 5_000_000  # $5
    checkout_max_micros: PositiveInt = 500_000_000  # $500

    # stripe (default: real Stripe, test mode) or stub (local dev, explicit opt-in).
    stripe_mode: StripeMode = "stripe"
    # Real Stripe (test mode only).
    stripe_secret_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("STRIPE_SECRET_KEY", "stripe_secret_key")
    )
    stripe_webhook_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("STRIPE_WEBHOOK_SECRET", "stripe_webhook_secret"),
    )
    stripe_api_base: str = "https://api.stripe.com"
    stripe_webhook_tolerance_s: PositiveInt = 300

    # Local stub. Billing calls the stub at stripe_stub_url; the browser opens checkout
    # pages at stripe_stub_public_url; the stub posts webhooks to stripe_stub_webhook_url.
    stripe_stub_url: str = "http://localhost:12111"
    stripe_stub_public_url: str = "http://localhost:12111"
    stripe_stub_webhook_url: str = "http://localhost:8090/stripe/webhook"
    # Shared by stub and billing; one of the two is required in stub mode, no default.
    stripe_stub_webhook_secret: SecretStr | None = None
    stripe_stub_webhook_secret_file: Path | None = None
    stripe_stub_host: str = "0.0.0.0"
    stripe_stub_port: PositiveInt = 12111

    @model_validator(mode="after")
    def _stripe_keys(self) -> BillingSettings:
        sk = self.stripe_secret_key.get_secret_value() if self.stripe_secret_key else None
        wh = self.stripe_webhook_secret.get_secret_value() if self.stripe_webhook_secret else None
        if sk and ("_live_" in sk or not sk.startswith(("sk_test_", "rk_test_"))):
            raise ValueError("STRIPE_SECRET_KEY must be a test-mode key (sk_test_/rk_test_)")
        if bool(sk) != bool(wh):
            raise ValueError("set both STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET, or neither")
        if wh and not wh.startswith("whsec_"):
            raise ValueError("STRIPE_WEBHOOK_SECRET must start with whsec_")
        if self.stripe_mode == "stub":
            if sk:
                raise ValueError("LOOM_STRIPE_MODE=stub with Stripe keys set: unset one")
            if len(self._stub_secret() or "") < MIN_STUB_SECRET_CHARS:
                raise ValueError(
                    "LOOM_STRIPE_MODE=stub needs LOOM_STRIPE_STUB_WEBHOOK_SECRET or "
                    f"LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE (at least {MIN_STUB_SECRET_CHARS} "
                    "characters, e.g. from `loom-devcerts --secret stripe-stub-webhook`)"
                )
        if self.checkout_min_micros > self.checkout_max_micros:
            raise ValueError("checkout_min_micros > checkout_max_micros")
        return self

    @property
    def consumer_name(self) -> str:
        return self.billing_consumer or f"billing-{socket.gethostname()}"

    def _stub_secret(self) -> str | None:
        if self.stripe_stub_webhook_secret is not None and self.stripe_stub_webhook_secret_file:
            raise ValueError(
                "set LOOM_STRIPE_STUB_WEBHOOK_SECRET or LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE, "
                "not both"
            )
        if self.stripe_stub_webhook_secret is not None:
            return self.stripe_stub_webhook_secret.get_secret_value().strip()
        if self.stripe_stub_webhook_secret_file is not None:
            try:
                return self.stripe_stub_webhook_secret_file.read_text().strip()
            except OSError as e:
                raise ValueError(f"LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE: {e}") from None
        return None

    def _require_stripe_keys(self) -> tuple[str, str]:
        if self.stripe_secret_key is None or self.stripe_webhook_secret is None:
            raise ValueError(
                "real Stripe (LOOM_STRIPE_MODE=stripe, the default) needs STRIPE_SECRET_KEY "
                "and STRIPE_WEBHOOK_SECRET (test mode); for local dev set LOOM_STRIPE_MODE=stub"
            )
        return (
            self.stripe_secret_key.get_secret_value(),
            self.stripe_webhook_secret.get_secret_value(),
        )

    def stripe_api(self) -> tuple[str, str]:
        """(API base URL, secret key) for Checkout calls in the current mode. Raises
        ValueError when real Stripe is selected without its keys."""
        if self.stripe_mode == "stub":
            return self.stripe_stub_url, STUB_SECRET_KEY
        return self.stripe_api_base, self._require_stripe_keys()[0]

    def webhook_secret(self) -> str:
        """The secret webhooks are verified with (raises like `stripe_api`)."""
        if self.stripe_mode == "stub":
            secret = self._stub_secret()
            assert secret  # checked at validation
            return secret
        return self._require_stripe_keys()[1]


@lru_cache(maxsize=1)
def get_billing_settings() -> BillingSettings:
    return BillingSettings()
