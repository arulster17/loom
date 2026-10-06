# Loom docs

## Start here

- [PLAN.md](PLAN.md): decisions, stack, phase plan, Postgres schemas, registry schema, open questions.
- [architecture.md](architecture.md): Phase 0 components, data flow, module contracts, where each measurement lives.
- [local-dev.md](local-dev.md): setup, tests, the mock backend, mock experiments end to end, lint, CI.

## Benchmark Lab

- [benchmark-lab.md](benchmark-lab.md): running experiments: commands, experiment YAML, budget rails, provenance, results layout.
- [load-generators.md](load-generators.md): the native load generator and the `vllm bench` / `sglang` wrappers.
- [quality-gate.md](quality-gate.md): eval suites, the gate's decision rule, sample sizes, logprob divergence, sanity checks.
- [cost-model.md](cost-model.md): $/1M tokens at SLO: goodput, allocation, CIs, what the hourly price includes, margins.
- [site.md](site.md): the public results site: snapshot export, build, deployment, waitlist form.

## Operations

- [aws-setup.md](aws-setup.md): one-time AWS account setup: quotas, HF token secret, Terraform, runner policy, settings.
- [runbook-runpod.md](runbook-runpod.md): GPU runs on RunPod (the primary path): preflight, monitoring spend, incidents, teardown checks, smoke test.
- [runbook.md](runbook.md): GPU runs on AWS: preflight, monitoring spend, incidents, reproducing, publishing, teardown.
- [security.md](security.md): Phase 0 security measures and their limits; Phase 1 requirements.

## Reference (how to add ...)

- [how-to/add-model.md](how-to/add-model.md): a model in `config/models.yaml` (config only).
- [how-to/add-engine.md](how-to/add-engine.md): a serving engine (code).
- [how-to/add-gpu-type.md](how-to/add-gpu-type.md): an instance type in `bench/prices.yaml` (config only).
- [how-to/add-workload-profile.md](how-to/add-workload-profile.md): a workload profile in `bench/workloads/` (config only).
- [how-to/add-eval-task.md](how-to/add-eval-task.md): an eval task for the quality gate (code).

## Policy

- [legal/competitor-benchmarking.md](legal/competitor-benchmarking.md): no tooling for third-party APIs; public list prices only.
- [waitlist.md](waitlist.md): waitlist signup counts, recorded as a demand signal.
