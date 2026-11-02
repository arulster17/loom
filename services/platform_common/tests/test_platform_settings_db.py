import pytest

from loom_platform.db import libpq_url
from loom_platform.migrate import load_migrations
from loom_platform.settings import PlatformSettings, enforce_environment


def test_settings_defaults_and_env(monkeypatch):
    for name in ("LOOM_DATABASE_URL", "LOOM_REDIS_URL", "LOOM_DEFAULT_KEY_RPM"):
        monkeypatch.delenv(name, raising=False)
    s = PlatformSettings()
    assert s.database_url.endswith("/loom_platform")
    assert s.reservation_ttl_ms == (600 + 60) * 1000
    monkeypatch.setenv("LOOM_REDIS_URL", "redis://r:1/2")
    monkeypatch.setenv("LOOM_DEFAULT_KEY_RPM", "7")
    s = PlatformSettings()
    assert s.redis_url == "redis://r:1/2"
    assert s.default_key_rpm == 7
    monkeypatch.setenv("LOOM_DEFAULT_KEY_RPM", "0")
    with pytest.raises(ValueError):
        PlatformSettings()


def test_loom_env_is_strict_unless_dev(monkeypatch, capsys):
    monkeypatch.delenv("LOOM_ENV", raising=False)
    monkeypatch.delenv("LOOM_DATABASE_URL", raising=False)
    s = PlatformSettings()
    assert s.env == "prod"  # unset is prod: forgetting it never relaxes anything
    assert s.production_problems() == [
        "LOOM_DATABASE_URL has no password or the docker-compose one",
        "LOOM_REDIS_URL has no password (Redis AUTH)",
        "LOOM_REDIS_URL is not TLS (rediss://)",
    ]
    with pytest.raises(SystemExit) as e:
        enforce_environment(s, "svc")
    assert e.value.code == 2 and "refusing to start" in capsys.readouterr().err
    real = PlatformSettings(
        database_url="postgresql://loom:s3cret@db:5432/loom_platform",
        redis_url="rediss://:r3dis@cache:6380/0",
    )
    assert real.production_problems() == []
    assert PlatformSettings(database_url="postgresql://loom@db/x").production_problems()
    plain = real.model_copy(update={"redis_url": "redis://:r3dis@cache:6379/0"})
    assert plain.production_problems() == ["LOOM_REDIS_URL is not TLS (rediss://)"]
    monkeypatch.setenv("LOOM_ENV", "dev")
    assert PlatformSettings().production_problems() == []
    enforce_environment(PlatformSettings(), "svc")  # dev: starts
    monkeypatch.setenv("LOOM_ENV", "staging")
    with pytest.raises(ValueError):
        PlatformSettings()


def test_libpq_url():
    assert libpq_url("postgresql+psycopg://a:b@h:1/d") == "postgresql://a:b@h:1/d"
    assert libpq_url("postgres://a@h/d") == "postgresql://a@h/d"
    with pytest.raises(ValueError):
        libpq_url("mysql://x")
    with pytest.raises(ValueError):
        libpq_url("nothing")


def test_migrations_are_numbered_without_gaps():
    migs = load_migrations()
    assert [m.version for m in migs] == list(range(1, len(migs) + 1))
    assert all(len(m.checksum) == 64 for m in migs)


def test_redis_must_not_evict_outside_dev(monkeypatch, capsys):
    import os

    import redis

    from loom_platform.settings import redis_policy_problems

    url = os.environ.get("LOOM_TEST_REDIS_URL") or "redis://localhost:6379/15"
    monkeypatch.setattr(redis.Redis, "config_get", lambda self, name: {name: "allkeys-lru"})
    assert redis_policy_problems(url) == [
        "Redis maxmemory-policy is 'allkeys-lru', not 'noeviction'"
    ]
    monkeypatch.setattr(redis.Redis, "config_get", lambda self, name: {name: "noeviction"})
    assert redis_policy_problems(url) == []

    def refused(self, name):  # managed Redis (ElastiCache) disables CONFIG
        raise redis.exceptions.ResponseError("unknown command 'CONFIG'")

    monkeypatch.setattr(redis.Redis, "config_get", refused)
    assert redis_policy_problems(url) == []
    assert "make sure it is noeviction" in capsys.readouterr().err
    # Unreachable: a warning, not a refusal (readiness covers it).
    monkeypatch.undo()
    assert redis_policy_problems("redis://127.0.0.1:1/0", timeout_s=0.2) == []
    assert "unreachable" in capsys.readouterr().err
