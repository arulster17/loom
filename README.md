# Loom

Verified-quality, transparent-cost inference for open-weight models.

Phase 0 is the **Benchmark Lab** (`bench/`): reproducible benchmarks of models, serving
engines (vLLM, SGLang), GPU types and clouds on latency, throughput, quality and
$/1M tokens at an SLO, with confidence intervals and full provenance for every number.
Phase 1 (gateway, billing, dashboard) is planned in [docs/PLAN.md](docs/PLAN.md).

**Status:** Phase 0 is built and runs end to end on the mock backend. The first GPU runs
(Qwen3-8B vLLM vs SGLang on 1x L40S, Llama 3.3 70B TP=4 on 4x L40S) run on RunPod
Secure Cloud, because the AWS GPU spot quota is 0; the RunPod provider is built and its
smoke test is next (the EC2 provider stays as the secondary path). No measured results
are published yet.

## Quick start (no GPU, no cloud)

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync
mkdir -p results && export LOOM_DATABASE_URL=sqlite:///results/loom.db
uv run bench run bench/experiments/mock-smoke.yaml   # ~15 s on the simulated GPU
uv run bench report                                  # leaderboard in reports/
uv run bench site build                              # results site in site/_build/
```

`uv run pytest -q` runs the test suite. With Docker, `docker compose up -d postgres`
gives the default results database instead of SQLite (unset `LOOM_DATABASE_URL`).

## Docs

[docs/README.md](docs/README.md) indexes everything: architecture, local development,
running experiments, the quality gate, the cost model, the RunPod runbook, AWS setup and
runbook, security, and how to add a model, engine, GPU type, workload profile or eval
task.

Licensed under Apache-2.0.
