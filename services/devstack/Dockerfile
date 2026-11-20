# syntax=docker/dockerfile:1
# Loom local-stack tools (docs/local-dev.md): the mock pods (bench mock engine behind the
# pod proxy), the pool simulator, the dev certificates and the one-shot schema migration.
# Build context: the repository root (docker-compose.yml builds it as loom-devstack:dev).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv
# The docker CLI (a static binary) for `loom-m3-acceptance`, which runs on the production
# box in the `acceptance` compose service with the docker socket: it reads the containers'
# logs and stops and starts services (M3 spec §6.12). Pinned by digest.
COPY --from=docker:27.3.1-cli@sha256:328eb399a065780c2cebe9224de003aa14084cf69efae882ac27430f921819b7 /usr/local/bin/docker /usr/local/bin/docker

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOOM_MODELS_YAML=/app/config/models.yaml

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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package loom-devstack --extra acceptance --no-install-workspace

COPY config config
COPY bench/src bench/src
COPY bench/workloads bench/workloads
COPY services/platform_common/src services/platform_common/src
COPY services/podproxy/src services/podproxy/src
COPY services/devstack/src services/devstack/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package loom-devstack --extra acceptance

# The pod proxy's port; the engine stays on 127.0.0.1:8000 inside the container.
EXPOSE 8443
CMD ["loom-mockpod"]
