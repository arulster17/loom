# Loom docs

## Start here

- [PLAN.md](PLAN.md): decisions, stack, phase plan, Postgres schemas, registry schema, open questions.
- [architecture.md](architecture.md): Phase 0 components, data flow, module contracts, where each measurement lives.
- [phase1-runpod-plan.md](phase1-runpod-plan.md): Phase 1 serving on RunPod (draft for approval): where things run, gateway-to-pod security, replicas, costs, milestones.
- [local-dev.md](local-dev.md): setup, tests, the mock backend, mock experiments end to end, lint, CI.

## Benchmark Lab

- [benchmark-lab.md](benchmark-lab.md): running experiments: commands, experiment YAML, budget rails, provenance, results layout.
- [load-generators.md](load-generators.md): the native load generator and the `vllm bench` / `sglang` wrappers.
- [quality-gate.md](quality-gate.md): eval suites, the gate's decision rule, sample sizes, logprob divergence, sanity checks.
- [perf/roofline-and-profiling.md](perf/roofline-and-profiling.md): why the numbers come out as they do: decode-step floors (`bench roofline`), achieved share per config, the fp8-kv8 2.4x explained, engine profile windows (`profile:`, `bench profile analyze`).
- [cost-model.md](cost-model.md): $/1M tokens at SLO: goodput, allocation, CIs, what the hourly price includes, margins.
- [perf/tensorrt-llm.md](perf/tensorrt-llm.md): TensorRT-LLM as a third engine: what it does differently from vLLM, the pinned version and its API gaps, the measured comparison (46938d1f, 9647a23b, 08204560) and the profiled "why" (71e2c32c, ad5b1408).
- [perf/speculative-decoding.md](perf/speculative-decoding.md): speculative decoding in plain words, the Qwen3-8B n-gram vs EAGLE3 study (measured, 53e4c107) and why EAGLE3 did not help the 70B on 4x L40S.
- [site.md](site.md): the public results site: snapshot export, build, deployment, waitlist form.

## Write-ups

- [writeups/70b-fp8-on-h100.md](writeups/70b-fp8-on-h100.md): serving Llama 3.3 70B under a 50 ms TPOT SLO: the cross-socket 4x L40S dead end, FP8 on 2x H100 SXM, and making the quality gate decide.
- README charts: drawn by [`scripts/readme_charts.py`](../scripts/readme_charts.py) from the committed report CSVs into [img/](img/).

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
- [how-to/price-a-model.md](how-to/price-a-model.md): a swept winner and its price in `config/models.yaml` (cost-plus, market, or both).

## Policy

- [legal/competitor-benchmarking.md](legal/competitor-benchmarking.md): no tooling for third-party APIs; public list prices only.
- [waitlist.md](waitlist.md): waitlist signup counts, recorded as a demand signal.
