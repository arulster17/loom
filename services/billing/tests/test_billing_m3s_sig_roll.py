"""Stripe-Signature during a webhook secret roll (docs/phase1-m3-spec.md §8.1: "any `v1`
matches"; done-when B23, no delivery fails while the secret is rolled).

While an endpoint secret is being rolled Stripe sends two `v1` values in one header, one
per secret, in no promised order. The box knows only one of the two secrets, so a header
whose matching `v1` is not the last one must still verify: the position of the matching
`v1` among the others never decides.
"""

from __future__ import annotations

import time

import pytest
from billing_fixtures import STUB_WEBHOOK_SECRET
from billing_m3s_kit import (
    billing_client,
    body_of,
    checkout_event,
    deliver,
    error_code,
    m3s_settings,
    make_org,
    purchases,
)

from loom_billing.stripe_sig import SignatureMismatch, compute, verify

SECRET = "whsec_test_box_secret_" + "b" * 18
OLD_SECRET = "whsec_test_rolled_out_" + "o" * 18
NEW_SECRET = "whsec_test_rolled_in__" + "n" * 18
BODY = b'{"id":"evt_roll","type":"checkout.session.completed"}'
NOW = 1_800_000_000
M = 10_000


def _good(t: int = NOW) -> str:
    return compute(BODY, SECRET, t)


def _others(t: int = NOW) -> list[str]:
    """`v1` values that are well-formed (64 lowercase hex chars, like the good one) but
    are not the box's: the HMAC under the other secret of the roll, and the box secret's
    HMAC over another timestamp."""
    vals = [
        compute(BODY, OLD_SECRET, t),
        compute(BODY, NEW_SECRET, t),
        compute(BODY, SECRET, t - 1),
    ]
    good = _good(t)
    for v in vals:
        assert len(v) == len(good) == 64 and v.isascii() and v != good
    return vals


def _header(t: int, *v1s: str) -> str:
    return ",".join([f"t={t}", *(f"v1={v}" for v in v1s)])


# ------------------------------------------------------------------ verify (unit)


@pytest.mark.parametrize("other_ix", [0, 1, 2])
def test_any_v1_matches_when_the_valid_one_comes_first(other_ix):
    """§8.1 any `v1` matches: the box's `v1` first, another secret's after it."""
    header = _header(NOW, _good(), _others()[other_ix])
    assert verify(BODY, header, SECRET, now=NOW) == NOW
    assert verify(BODY, header, SECRET, now=NOW + 300) == NOW  # still within the tolerance


@pytest.mark.parametrize("other_ix", [0, 1, 2])
def test_any_v1_matches_when_the_valid_one_comes_last(other_ix):
    """§8.1 any `v1` matches: another secret's `v1` first, the box's after it."""
    header = _header(NOW, _others()[other_ix], _good())
    assert verify(BODY, header, SECRET, now=NOW) == NOW


def test_any_v1_matches_when_the_valid_one_is_in_the_middle_of_three():
    """§8.1 any `v1` matches: the box's `v1` between two that are not."""
    old, new, _ = _others()
    assert verify(BODY, _header(NOW, old, _good(), new), SECRET, now=NOW) == NOW
    assert verify(BODY, _header(NOW, new, _good(), old), SECRET, now=NOW - 300) == NOW


def test_any_v1_matches_for_either_secret_of_the_roll_on_either_box():
    """§8.1 / B23: one header carrying both secrets' `v1` verifies on a box holding the
    old secret and on one already holding the new one, whatever the order."""
    old = compute(BODY, OLD_SECRET, NOW)
    new = compute(BODY, NEW_SECRET, NOW)
    for header in (_header(NOW, old, new), _header(NOW, new, old)):
        assert verify(BODY, header, OLD_SECRET, now=NOW) == NOW, header
        assert verify(BODY, header, NEW_SECRET, now=NOW) == NOW, header


@pytest.mark.parametrize("count", [1, 2, 3])
def test_no_v1_matching_with_a_current_t_is_a_signature_mismatch(count):
    """§8.1: a well-formed header with a current `t` and no matching `v1` is refused."""
    header = _header(NOW, *_others()[:count])
    with pytest.raises(SignatureMismatch):
        verify(BODY, header, SECRET, now=NOW)


# ------------------------------------------------------------------ webhook route


@pytest.mark.postgres
@pytest.mark.redis
@pytest.mark.parametrize("valid_first", [True, False])
async def test_webhook_any_v1_matches_during_a_secret_roll(pool, redis_client, valid_first):
    """§8.1 through `POST /stripe/webhook` (B23): a delivery signed with both secrets of a
    roll is credited whether the box's `v1` comes first or last."""
    org = await make_org(pool)
    ev = checkout_event(org, 2000)
    raw = body_of(ev)
    t = int(time.time())
    good = compute(raw, STUB_WEBHOOK_SECRET, t)
    other = compute(raw, "whsec_test_the_other_secret_of_the_roll", t)
    header = _header(t, good, other) if valid_first else _header(t, other, good)
    async with billing_client(m3s_settings(), pool, redis_client) as c:
        r = await deliver(c, ev, header=header)
    assert r.status_code == 200, (r.status_code, error_code(r))
    assert r.json()["result"] == "credited"
    assert await purchases(pool, org) == [2000 * M]
