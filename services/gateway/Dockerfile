# syntax=docker/dockerfile:1.7
# Loom API gateway. Build from the repository root (the uv workspace):
#   docker build -f services/gateway/Dockerfile -t loom-gateway .
# Public API on 8080, Prometheus metrics on 9100 (docs/phase1-contracts.md §2).

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

# Dependencies first (cached until uv.lock or a member's pyproject changes).
COPY pyproject.toml uv.lock ./
COPY bench/pyproject.toml bench/
COPY services/platform_common/pyproject.toml services/platform_common/
COPY services/gateway/pyproject.toml services/gateway/
COPY services/billing/pyproject.toml services/billing/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package loom-gateway --no-install-workspace

COPY bench/src bench/src
COPY services/platform_common/src services/platform_common/src
COPY services/gateway/src services/gateway/src
COPY services/billing/src services/billing/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package loom-gateway --no-editable

FROM python:3.12-slim
RUN useradd --system --uid 10001 --no-create-home loom
COPY --from=build /opt/venv /opt/venv
COPY config/models.yaml /app/config/models.yaml
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LOOM_MODELS_YAML=/app/config/models.yaml
WORKDIR /app
USER loom
EXPOSE 8080 9100
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]
CMD ["python", "-m", "loom_gateway"]
