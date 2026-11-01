"""Billing settings: `LOOM_*` variables (docs/phase1-contracts.md §2) plus the Stripe keys.

Stripe mode is decided by two variables without the `LOOM_` prefix, the names Stripe's
own docs use:

- both `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` set: **real Stripe**, test mode
  only. A live key (`sk_live_`, `rk_live_`) is refused at startup.
- neither set: **local stub** (`loom_billing.stripe_stub`). Billing calls the stub's
  Checkout API at `LOOM_STRIPE_STUB_URL` and verifies webhooks with
  `LOOM_STRIPE_STUB_WEBHOOK_SECRET`, a fixed dev value shared with the stub.
- only one set: a configuration error.
"""

from __future__ import annotations

import socket
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict

from loom_platform.settings import PlatformSettings

PositiveInt = Annotated[int, Field(gt=0)]
NonNegInt = Annotated[int, Field(ge=0)]
StripeMode = Literal["stripe", "stub"]

# Micro-dollars per US cent: Stripe amounts are integer cents.
MICROS_PER_CENT = 10_000
# Not a secret: the stub and billing share it in local dev only. Real Stripe never sees it.
STUB_WEBHOOK_SECRET = "whsec_loom_local_stub_not_a_secret"
STUB_SECRET_KEY = "sk_test_loom_local_stub"


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
    stripe_stub_webhook_secret: SecretStr = SecretStr(STUB_WEBHOOK_SECRET)
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
        if self.checkout_min_micros > self.checkout_max_micros:
            raise ValueError("checkout_min_micros > checkout_max_micros")
        return self

    @property
    def stripe_mode(self) -> StripeMode:
        return "stripe" if self.stripe_secret_key else "stub"

    @property
    def consumer_name(self) -> str:
        return self.billing_consumer or f"billing-{socket.gethostname()}"

    def stripe_api(self) -> tuple[str, str]:
        """(API base URL, secret key) for Checkout calls in the current mode."""
        if self.stripe_secret_key is not None:
            return self.stripe_api_base, self.stripe_secret_key.get_secret_value()
        return self.stripe_stub_url, STUB_SECRET_KEY

    def webhook_secret(self) -> str:
        if self.stripe_webhook_secret is not None:
            return self.stripe_webhook_secret.get_secret_value()
        return self.stripe_stub_webhook_secret.get_secret_value()


@lru_cache(maxsize=1)
def get_billing_settings() -> BillingSettings:
    return BillingSettings()
