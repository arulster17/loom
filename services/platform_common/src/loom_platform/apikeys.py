"""API keys: `loom_sk_<12 base62 lookup chars>_<40 base62 secret chars>` (61 characters).

Only the SHA-256 of the full key (`api_keys.key_hash`) and its first 20 characters
(`api_keys.prefix`, e.g. `loom_sk_4f9QzT0aLm2X`, shown in the dashboard) are stored.
The plaintext is returned once, at creation.

The 40-character secret carries 238 bits of entropy, so a single unsalted SHA-256 is
the right hash: there is nothing to brute-force, and the gateway can look a key up by
hash in O(1). The format check runs first, so malformed tokens are rejected without a
cache or database lookup.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import string
from dataclasses import dataclass

KEY_TAG = "loom_sk_"
LOOKUP_LEN = 12
SECRET_LEN = 40
PREFIX_LEN = len(KEY_TAG) + LOOKUP_LEN  # 20
KEY_LEN = PREFIX_LEN + 1 + SECRET_LEN  # 61

_BASE62 = string.digits + string.ascii_uppercase + string.ascii_lowercase
KEY_RE = re.compile(rf"^{KEY_TAG}[0-9A-Za-z]{{{LOOKUP_LEN}}}_[0-9A-Za-z]{{{SECRET_LEN}}}$")
PREFIX_RE = re.compile(rf"^{KEY_TAG}[0-9A-Za-z]{{{LOOKUP_LEN}}}$")


class InvalidApiKey(ValueError):
    """Not a Loom API key (wrong shape). Maps to 401 invalid_api_key."""


@dataclass(frozen=True)
class NewApiKey:
    plaintext: str  # show once, never store or log
    prefix: str
    key_hash: bytes  # 32 bytes, api_keys.key_hash

    def __repr__(self) -> str:  # keep the secret out of logs and tracebacks
        return f"NewApiKey(prefix={self.prefix!r})"


def _random_base62(n: int) -> str:
    return "".join(secrets.choice(_BASE62) for _ in range(n))


def generate_api_key() -> NewApiKey:
    prefix = KEY_TAG + _random_base62(LOOKUP_LEN)
    plaintext = f"{prefix}_{_random_base62(SECRET_LEN)}"
    return NewApiKey(plaintext=plaintext, prefix=prefix, key_hash=hash_api_key(plaintext))


def is_well_formed(key: str) -> bool:
    return KEY_RE.match(key) is not None


def hash_api_key(key: str) -> bytes:
    """SHA-256 of the full key. Raises InvalidApiKey for anything not shaped like one."""
    if not is_well_formed(key):
        raise InvalidApiKey("malformed API key")
    return hashlib.sha256(key.encode("ascii")).digest()


def key_prefix(key: str) -> str:
    if not is_well_formed(key):
        raise InvalidApiKey("malformed API key")
    return key[:PREFIX_LEN]


def bearer_token(authorization: str | None) -> str:
    """The key from an `Authorization: Bearer <key>` header value."""
    if not authorization:
        raise InvalidApiKey("missing Authorization header")
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise InvalidApiKey("Authorization must be 'Bearer <key>'")
    return token.strip()
