# syntax=docker/dockerfile:1
# Caddy with the per-client rate-limit module (docs/phase1-m3-spec.md §6.7, §15 D2).
# Build context: deploy/caddy/. go.mod and go.sum pin Caddy and caddy-ratelimit; the
# Caddyfile is not in the image (compose mounts the release bundle's, §6.12).
# Both base images are pinned by digest (multi-arch: the instance is arm64).
FROM golang:1.23-alpine@sha256:383395b794dffa5b53012a212365d40c8e37109a626ca30d6151c8348d380b5f AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/caddy .

FROM caddy:2.8.4-alpine@sha256:af32e97399febea808609119bb21544d0265c58a02836576e32a2d082c262c17
COPY --from=build /out/caddy /usr/bin/caddy
RUN caddy list-modules | grep -q '^http.handlers.rate_limit$'
