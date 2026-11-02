# Loom web app (Next.js standalone server). Build from the repo root, so the model
# registry can be copied in:
#
#   docker build -f apps/web/Dockerfile -t loom-web .
#
# Runtime env: see apps/web/README.md (DATABASE_URL, AUTH_SECRET, AUTH_URL,
# EMAIL_SERVER, LOOM_GATEWAY_URL, LOOM_BILLING_URL, LOOM_REDIS_URL, ...).

# node:22-alpine, pinned by digest (multi-arch index).
ARG NODE_IMAGE=node:22-alpine@sha256:fe4f6d74e20370acb04ad889077d14d0f64a9c8ea6a42bc60cddf1e844699914

FROM ${NODE_IMAGE} AS deps
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM ${NODE_IMAGE} AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web/ ./
RUN npm run build

FROM ${NODE_IMAGE} AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    LOOM_MODELS_YAML=/app/config/models.yaml
RUN addgroup -S -g 1001 loom && adduser -S -u 1001 -G loom loom
COPY --from=build --chown=loom:loom /app/.next/standalone ./
COPY --from=build --chown=loom:loom /app/.next/static ./.next/static
COPY --from=build --chown=loom:loom /app/public ./public
COPY --chown=loom:loom config/models.yaml /app/config/models.yaml
USER loom
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD wget -qO- http://127.0.0.1:3000/signin >/dev/null || exit 1
CMD ["node", "server.js"]
