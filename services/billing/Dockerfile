# Loom billing: worker + Stripe webhook API, and the local Stripe stub.
#
# Build from the repo root (the uv workspace):
#   docker build -f services/billing/Dockerfile -t loom-billing .
# Run:
#   docker run loom-billing                # worker + API (8090), metrics 9101
#   docker run loom-billing worker         # consumer, batches, reconcile only
#   docker run loom-billing api            # webhook / internal API only
#   docker run -p 12111:12111 loom-billing stripe-stub
# Settings are LOOM_* variables (docs/phase1-contracts.md, services/billing/README.md).

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    LOOM_MODELS_YAML=/app/config/models.yaml

WORKDIR /app

# Dependencies first (cached until uv.lock or a pyproject changes). Every workspace
# member's pyproject must be present for uv to read the workspace.
COPY pyproject.toml uv.lock README.md ./
COPY bench/pyproject.toml bench/pyproject.toml
COPY services/platform_common/pyproject.toml services/platform_common/pyproject.toml
COPY services/gateway/pyproject.toml services/gateway/pyproject.toml
COPY services/billing/pyproject.toml services/billing/pyproject.toml
RUN uv sync --frozen --no-dev --package loom-billing --no-install-workspace

COPY config config
COPY bench/src bench/src
COPY services/platform_common/src services/platform_common/src
COPY services/billing/src services/billing/src
RUN uv sync --frozen --no-dev --package loom-billing

RUN useradd --system --uid 10001 --no-create-home loom
USER loom

EXPOSE 8090 9101 12111
ENTRYPOINT ["/app/.venv/bin/loom-billing"]
CMD ["all"]
