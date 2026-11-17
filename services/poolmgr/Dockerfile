# syntax=docker/dockerfile:1
# Loom pool manager (services/poolmgr/README.md). Build from the repository root (the uv
# workspace); docker-compose.yml builds it as loom-poolmgr:dev and runs `loom-poolmgr run`
# on the local backend: each "pod" is the bench mock engine behind loom-podproxy, as
# subprocesses of the manager (hence the `local` extra, the bench's `mock` extra).
# Prometheus on 9102 (docs/phase1-contracts.md §2), served once the manager holds the
# single-writer lock and has adopted its pods: the HEALTHCHECK reads it.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOOM_MODELS_YAML=/app/config/models.yaml \
    LOOM_SERVING_YAML=/app/config/serving.yaml \
    LOOM_POOLMGR_METRICS_PORT=9102

WORKDIR /app

# Third-party dependencies first: this layer is reused until uv.lock or a pyproject changes.
# uv needs every workspace member's pyproject.toml to read the lock.
COPY pyproject.toml uv.lock README.md ./
COPY bench/pyproject.toml bench/
COPY services/platform_common/pyproject.toml services/platform_common/
COPY services/gateway/pyproject.toml services/gateway/
COPY services/billing/pyproject.toml services/billing/
COPY services/podproxy/pyproject.toml services/podproxy/
COPY services/devstack/pyproject.toml services/devstack/
COPY services/poolmgr/pyproject.toml services/poolmgr/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package loom-poolmgr --extra local --no-install-workspace

# The bench's sources and the data it reads at run time (prices for the price check).
COPY config config
COPY bench/src bench/src
COPY bench/prices.yaml bench/
COPY services/platform_common/src services/platform_common/src
COPY services/podproxy/src services/podproxy/src
COPY services/poolmgr/src services/poolmgr/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package loom-poolmgr --extra local

# LOOM_POOLMGR_STATE_DIR: per-pod directories, known_hosts, a dev SSH key. The local
# backend's pods die with the container, so compose keeps no volume for it.
RUN mkdir -p /var/lib/loom-poolmgr
EXPOSE 9102
HEALTHCHECK --interval=5s --timeout=3s --retries=12 --start-period=10s \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/metrics' % os.environ.get('LOOM_POOLMGR_METRICS_PORT', '9102'), timeout=2)"]
CMD ["loom-poolmgr", "run"]
