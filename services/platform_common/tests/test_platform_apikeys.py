import hashlib

import pytest

from loom_platform.apikeys import (
    KEY_LEN,
    PREFIX_LEN,
    InvalidApiKey,
    bearer_token,
    generate_api_key,
    hash_api_key,
    is_well_formed,
    key_prefix,
)


def test_generated_key_shape_and_hash():
    k = generate_api_key()
    assert len(k.plaintext) == KEY_LEN == 61
    assert k.plaintext.startswith("loom_sk_")
    assert k.prefix == k.plaintext[:PREFIX_LEN]
    assert len(k.prefix) == 20
    assert k.key_hash == hashlib.sha256(k.plaintext.encode()).digest()
    assert len(k.key_hash) == 32
    assert is_well_formed(k.plaintext)
    assert key_prefix(k.plaintext) == k.prefix


def test_keys_are_unique_and_secret_not_in_repr():
    keys = [generate_api_key() for _ in range(200)]
    assert len({k.plaintext for k in keys}) == 200
    assert len({k.prefix for k in keys}) == 200
    k = keys[0]
    assert k.plaintext[PREFIX_LEN + 1 :] not in repr(k)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "sk-abc",
        "loom_sk_short_x",
        "loom_sk_" + "a" * 12 + "_" + "b" * 39,  # secret too short
        "loom_sk_" + "a" * 12 + "_" + "b" * 41,
        "loom_sk_" + "a" * 12 + "-" + "b" * 40,
        "loom_sk_" + "a" * 11 + "!_" + "b" * 40,
        " loom_sk_" + "a" * 12 + "_" + "b" * 40,
    ],
)
def test_malformed_keys_are_rejected_before_hashing(bad):
    assert not is_well_formed(bad)
    with pytest.raises(InvalidApiKey):
        hash_api_key(bad)
    with pytest.raises(InvalidApiKey):
        key_prefix(bad)


def test_hash_is_deterministic():
    k = generate_api_key()
    assert hash_api_key(k.plaintext) == hash_api_key(k.plaintext) == k.key_hash


def test_bearer_token():
    assert bearer_token("Bearer abc") == "abc"
    assert bearer_token("bearer   abc ") == "abc"
    for bad in (None, "", "Basic abc", "Bearer ", "abc"):
        with pytest.raises(InvalidApiKey):
            bearer_token(bad)
