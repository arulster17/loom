# gateway

`loom_gateway`: the OpenAI-compatible public API (FastAPI on uvloop + httptools). Built on
`loom_platform`; the contracts it implements are in
[docs/phase1-contracts.md](../../docs/phase1-contracts.md) (§6 reserve/settle, §8 request
order, §9 pod contract, §10 pools, §11 errors, §12 cost headers).

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | Streaming (SSE) and non-streaming; `tools`, `response_format` and sampling parameters pass through |
| `POST /v1/completions` | Same; without `max_tokens` it sends OpenAI's default of 16 explicitly |
| `GET /v1/models`, `GET /v1/models/{id}` | Servable registry rows: context length, prices (micros and USD per 1M), registry status, pool availability. No key needed |
| `GET /health`, `GET /ready` | Liveness; readiness pings Redis |
| `:9100/metrics` | Prometheus (`loom_gateway_*`), never on the public port |

Run: `python -m loom_gateway` (settings from `LOOM_*`, see `settings.py`), or the image:
`docker build -f services/gateway/Dockerfile -t loom-gateway .` from the repository root.

## Per request

1. 413 body cap, 401 key format, auth record: in-process copy (at most 1 s and never past
   its Redis entry), Redis (`LOOM_AUTH_CACHE_TTL_S`), Postgres. `LISTEN loom_auth_changed`
   drops cached records on commit, so revocation and suspension take milliseconds; the TTL
   alone bounds them at 4 s.
2. 404 model, 400 validation (`max_tokens` over `LOOM_MAX_TOKENS_CAP`, `n`, `best_of`, ...).
3. One `reserve.lua` call (402/429), loading the org from Postgres and retrying once on
   `noinit`.
4. 503 `model_unavailable` with `Retry-After` unless the model's pool is `ready` (a cold
   model also gets a wake request). No fallback to another model or GPU type. The 503
   releases the reservation (`release.lua`: tpm refunded, the rpm token stays spent, no
   usage event), so only a request with the balance and rate limit to run can wake a GPU.
5. Dispatch over mTLS (client certificate, pod CA, server name = pod id, TLS 1.3,
   `X-Loom-Pod-Token`) to the replica with the fewest open requests; a replica that just
   failed goes last. Retry once on another replica, only before the first byte reaches the
   client.
6. One `settle.lua` call when the response ends, which appends the usage event. Streams
   ask the engine for usage on every chunk and bill what was forwarded (the prompt plus
   each choice's last running count, as vLLM counts per choice with `n > 1`), so a client
   abort or a lost pod bills exactly the tokens streamed.

Cost: `X-Loom-Request-Id`, `X-Loom-Price-Version`, `X-Loom-Max-Cost-Micros` on every
admitted response; `X-Loom-Cost-Micros` / `X-Loom-Cost-USD` on non-streaming ones;
`usage.cost_micros` in the response body and in a stream's final usage chunk.

Logs are JSON lines with ids, model, status, tokens, cost and timings. Prompts,
completions, keys and headers are never logged.

## Tests

```
docker compose up -d postgres redis
uv run pytest services/gateway/tests
uv run pytest services/gateway/tests/test_gateway_overhead.py -s   # prints the overhead
```

The fake engine (`tests/gateway_testkit.py`) is vLLM-shaped, serves mTLS with test
certificates generated per session, honours `continuous_usage_stats`, and injects
failures (status codes, dropped connections, delays).
