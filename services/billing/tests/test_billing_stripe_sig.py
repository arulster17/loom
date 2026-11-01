"""Stripe-Signature verification and billing settings (no servers needed)."""

from __future__ import annotations

import hashlib
import hmac

import pytest
from pydantic import ValidationError

from loom_billing.api import create_app
from loom_billing.settings import BillingSettings
from loom_billing.stripe_api import StripeClient, form_encode
from loom_billing.stripe_sig import SignatureError, compute, sign, verify

SECRET = "whsec_test_abc123"
BODY = b'{"id":"evt_1","type":"checkout.session.completed"}'
NOW = 1_700_000_000


def test_signature_matches_stripes_scheme():
    # Independent computation of Stripe's documented scheme.
    expected = hmac.new(SECRET.encode(), f"{NOW}.".encode() + BODY, hashlib.sha256).hexdigest()
    assert compute(BODY, SECRET, NOW) == expected
    assert sign(BODY, SECRET, NOW) == f"t={NOW},v1={expected}"


def test_accepts_valid_signature():
    assert verify(BODY, sign(BODY, SECRET, NOW), SECRET, now=NOW + 10) == NOW


def test_accepts_any_v1_during_secret_roll_and_ignores_v0():
    good = compute(BODY, SECRET, NOW)
    header = f"t={NOW},v1={'0' * 64},v0=abc,v1={good}"
    assert verify(BODY, header, SECRET, now=NOW)


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        f"v1={compute(BODY, SECRET, NOW)}",  # no timestamp
        f"t={NOW}",  # no signature
        f"t={NOW},v0={compute(BODY, SECRET, NOW)}",  # only the legacy scheme
        f"t=abc,v1={compute(BODY, SECRET, NOW)}",
        f"t={NOW},v1={compute(BODY, 'whsec_other', NOW)}",  # wrong secret
        f"t={NOW + 1},v1={compute(BODY, SECRET, NOW)}",  # timestamp swapped
    ],
)
def test_rejects_bad_headers(header):
    with pytest.raises(SignatureError):
        verify(BODY, header, SECRET, now=NOW)


def test_rejects_tampered_body():
    with pytest.raises(SignatureError):
        verify(BODY + b" ", sign(BODY, SECRET, NOW), SECRET, now=NOW)


def test_tolerance_window():
    header = sign(BODY, SECRET, NOW)
    verify(BODY, header, SECRET, now=NOW + 300)
    with pytest.raises(SignatureError, match="tolerance"):
        verify(BODY, header, SECRET, now=NOW + 301)  # replayed too late
    with pytest.raises(SignatureError, match="tolerance"):
        verify(BODY, header, SECRET, now=NOW - 301)  # from the future
    verify(BODY, header, SECRET, now=NOW + 3000, tolerance_s=3600)


STUB_ENV = (
    "STRIPE_SECRET_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "LOOM_STRIPE_MODE",
    "LOOM_STRIPE_STUB_WEBHOOK_SECRET",
    "LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE",
)


@pytest.fixture
def clean_env(monkeypatch):
    for k in STUB_ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_missing_keys_never_fall_back_to_the_stub(clean_env):
    """Without keys and without LOOM_STRIPE_MODE=stub, nothing that talks to Stripe
    starts: before, it silently ran the stub with a public webhook secret, so anyone
    could sign a checkout event and credit any org."""
    s = BillingSettings()
    assert s.stripe_mode == "stripe"
    with pytest.raises(ValueError, match="STRIPE_SECRET_KEY"):
        s.webhook_secret()
    with pytest.raises(ValueError, match="STRIPE_SECRET_KEY"):
        s.stripe_api()
    with pytest.raises(ValueError, match="LOOM_STRIPE_MODE=stub"):
        create_app(s)


def test_stub_mode_is_explicit_and_needs_its_own_secret(clean_env, tmp_path):
    clean_env.setenv("LOOM_STRIPE_MODE", "stub")
    with pytest.raises(ValidationError, match="STUB_WEBHOOK_SECRET"):
        BillingSettings()  # no default secret
    clean_env.setenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET", "short")
    with pytest.raises(ValidationError, match="at least"):
        BillingSettings()
    secret = "whsec_" + "k" * 30
    clean_env.setenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET", secret)
    s = BillingSettings()
    assert s.stripe_mode == "stub" and s.webhook_secret() == secret
    assert s.stripe_api() == ("http://localhost:12111", "sk_test_loom_local_stub")
    assert secret not in repr(s)

    path = tmp_path / "stripe-stub-webhook.secret"
    path.write_text(secret + "\n")
    clean_env.setenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE", str(path))
    with pytest.raises(ValidationError, match="not both"):
        BillingSettings()
    clean_env.delenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET")
    assert BillingSettings().webhook_secret() == secret
    clean_env.setenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE", str(tmp_path / "missing"))
    with pytest.raises(ValidationError):
        BillingSettings()

    clean_env.setenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE", str(path))
    clean_env.setenv("STRIPE_SECRET_KEY", "sk_test_51abc")
    clean_env.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xyz")
    with pytest.raises(ValidationError, match="unset one"):
        BillingSettings()  # stub and real keys at once


def test_settings_empty_keys_are_unset(clean_env):
    # docker-compose sets STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY:-}, i.e. "" when unset.
    # Regression: "" used to select real Stripe's API base with an empty key. Empty is
    # unset: with the compose opt-in that is the stub; without it, nothing starts.
    clean_env.setenv("STRIPE_SECRET_KEY", "")
    clean_env.setenv("STRIPE_WEBHOOK_SECRET", "")
    clean_env.setenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET", "")
    s = BillingSettings()
    assert s.stripe_secret_key is None and s.stripe_webhook_secret is None
    with pytest.raises(ValueError, match="LOOM_STRIPE_MODE=stub"):
        s.stripe_api()
    secret = "whsec_" + "e" * 30
    clean_env.setenv("LOOM_STRIPE_MODE", "stub")
    clean_env.setenv("LOOM_STRIPE_STUB_WEBHOOK_SECRET", secret)
    s = BillingSettings()  # empty keys alongside stub are not "keys set"
    assert s.stripe_mode == "stub" and s.webhook_secret() == secret
    assert s.stripe_api() == ("http://localhost:12111", "sk_test_loom_local_stub")


def test_settings_real_stripe_from_env(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_51abc")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_xyz")
    s = BillingSettings()
    assert s.stripe_mode == "stripe"
    assert s.webhook_secret() == "whsec_xyz"
    assert s.stripe_api() == ("https://api.stripe.com", "sk_test_51abc")
    assert "sk_test_51abc" not in repr(s)  # secrets stay out of logs


@pytest.mark.parametrize(
    "env",
    [
        {"STRIPE_SECRET_KEY": "sk_live_51abc", "STRIPE_WEBHOOK_SECRET": "whsec_x"},
        {"STRIPE_SECRET_KEY": "rk_live_51abc", "STRIPE_WEBHOOK_SECRET": "whsec_x"},
        {"STRIPE_SECRET_KEY": "pk_test_51abc", "STRIPE_WEBHOOK_SECRET": "whsec_x"},
        {"STRIPE_SECRET_KEY": "sk_test_51abc"},  # only one of the pair
        {"STRIPE_WEBHOOK_SECRET": "whsec_x"},
        {"STRIPE_SECRET_KEY": "sk_test_51abc", "STRIPE_WEBHOOK_SECRET": "nope"},
    ],
)
def test_settings_refuse_live_or_partial_keys(monkeypatch, env):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError):
        BillingSettings()


def test_client_refuses_live_key():
    with pytest.raises(ValueError):
        StripeClient("https://api.stripe.com", "sk_live_123")


def test_form_encoding_uses_stripe_brackets():
    body = form_encode(
        {"mode": "payment", "metadata": {"org_id": "o"}, "line_items": [{"quantity": 1}],
         "skip": None, "flag": True}
    )  # fmt: skip
    assert body == (
        b"mode=payment&metadata%5Borg_id%5D=o&line_items%5B0%5D%5Bquantity%5D=1&flag=true"
    )


def test_billing_refuses_dev_defaults_outside_dev(clean_env, capsys):
    from loom_billing import cli

    prod = {
        "env": "prod",
        "database_url": "postgresql://loom:s3cret@db/loom_platform",
        "stripe_secret_key": "sk_test_51abc",
        "stripe_webhook_secret": "whsec_xyz",
        "billing_internal_token": "a-real-random-token",
    }
    assert BillingSettings(**prod).production_problems() == []
    stub = {**prod, "stripe_secret_key": None, "stripe_webhook_secret": None,
            "stripe_mode": "stub", "stripe_stub_webhook_secret": "whsec_" + "k" * 30}  # fmt: skip
    assert any("LOOM_STRIPE_MODE=stub" in p for p in BillingSettings(**stub).production_problems())
    for token in (None, "loom-dev-internal-token"):
        [p] = BillingSettings(**{**prod, "billing_internal_token": token}).production_problems()
        assert "LOOM_BILLING_INTERNAL_TOKEN" in p
        # The worker serves no API: it does not need the token.
        assert (
            BillingSettings(**{**prod, "billing_internal_token": token}).production_problems(
                serves_api=False
            )
            == []
        )
    assert BillingSettings(**{**stub, "env": "dev"}).production_problems() == []

    # The entrypoint exits before starting anything, whatever the command.
    for command in ("all", "worker", "stripe-stub"):
        with pytest.raises(SystemExit) as e:
            cli.main([command])
        assert e.value.code == 2
    assert "refusing to start" in capsys.readouterr().err
