# Loom

Verified-quality, transparent-cost inference for open-weight models.

Phase 0 is the **Benchmark Lab** (`/bench`): reproducible benchmarks of models, serving
engines (vLLM, SGLang), GPU types and clouds on correctness, latency, throughput and
cost per token, with full provenance for every number.

- Plan, stack and schemas: [docs/PLAN.md](docs/PLAN.md)
- Build spec: [inference-platform-build-prompt.md](inference-platform-build-prompt.md)

## Quick start (no GPU needed)

```bash
uv sync
uv run pytest
uv run bench --help
```

Licensed under Apache-2.0.
