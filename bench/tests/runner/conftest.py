from pathlib import Path
from typing import Any

import pytest

from loom_bench.budget import load_budget
from loom_bench.experiment import EXPERIMENTS_DIR, Experiment
from loom_bench.prices import load_prices
from loom_bench.registry import load_registry
from loom_bench.runner import RunnerContext
from loom_bench.store.db import upgrade

SMOKE = EXPERIMENTS_DIR / "mock-smoke.yaml"
ABORT = EXPERIMENTS_DIR / "mock-budget-abort.yaml"
QWEN = EXPERIMENTS_DIR / "qwen3-8b-vllm-vs-sglang.yaml"
LLAMA = EXPERIMENTS_DIR / "llama-3.3-70b-tp4.yaml"
QWEN_RUNPOD = EXPERIMENTS_DIR / "qwen3-8b-vllm-vs-sglang-runpod.yaml"
LLAMA_RUNPOD = EXPERIMENTS_DIR / "llama-3.3-70b-tp4-runpod.yaml"
RUNPOD_SMOKE = EXPERIMENTS_DIR / "runpod-smoke.yaml"
LLAMA_FP8 = EXPERIMENTS_DIR / "llama-3.3-70b-fp8-tp4.yaml"
LLAMA_FP8_RUNPOD = EXPERIMENTS_DIR / "llama-3.3-70b-fp8-tp4-runpod.yaml"
RUNPOD_SMOKE_FP8 = EXPERIMENTS_DIR / "runpod-smoke-fp8.yaml"
# Approved 2026-10-08: the 2x H100 SXM 70B run and its smoke.
LLAMA_H100_RUNPOD = EXPERIMENTS_DIR / "llama-3.3-70b-h100-tp2-runpod.yaml"
RUNPOD_SMOKE_H100 = EXPERIMENTS_DIR / "runpod-smoke-h100.yaml"
QWEN_QUALITY_RUNPOD = EXPERIMENTS_DIR / "qwen3-8b-quality-runpod.yaml"
# Approved 2026-10-09: the 70B H100 run's evals and gate again, on tool-calling data v2.
LLAMA_H100_QUALITY_RUNPOD = EXPERIMENTS_DIR / "llama-3.3-70b-h100-tp2-quality-runpod.yaml"
# Approved 2026-10-09: the Qwen3-8B config sweep, its smoke, and the follow-up that runs
# the sweep's winner on L40 and RTX 6000 Ada (a draft until the sweep has run).
QWEN_SWEEP_RUNPOD = EXPERIMENTS_DIR / "qwen3-8b-config-sweep-runpod.yaml"
RUNPOD_SMOKE_8B_SWEEP = EXPERIMENTS_DIR / "runpod-smoke-8b-sweep.yaml"
QWEN_WINNER_ADA_RUNPOD = EXPERIMENTS_DIR / "qwen3-8b-winner-ada-runpod.yaml"


@pytest.fixture(autouse=True)
def _no_aws_settings(monkeypatch):
    """Plans use the aws_ec2 provider's default accrual terms, whatever the shell has."""
    monkeypatch.delenv("LOOM_AWS_CONFIG", raising=False)


@pytest.fixture
def db(tmp_path) -> str:
    url = f"sqlite:///{tmp_path / 'loom.db'}"
    upgrade(url)
    return url


@pytest.fixture
def ctx(db, tmp_path) -> RunnerContext:
    return RunnerContext(
        db_url=db,
        out_dir=tmp_path / "results",
        registry=load_registry(),
        prices=load_prices(),
        budget=load_budget(),
    )


def mock_doc(**over: Any) -> dict[str, Any]:
    """A small, fast mock experiment as a dict; keyword args replace top-level keys."""
    doc: dict[str, Any] = {
        "name": "unit",
        "description": "unit test experiment",
        "model": "qwen3-8b",
        "provider": {"kind": "mock", "hourly_price": "$1", "time_scale": 0.01},
        "variants": [{"name": "a"}],
        "workloads": [
            {
                "profile": "fixed-128-128",
                "overrides": {"input_len": 32, "output_len": 8},
                "load": {
                    "mode": "closed_loop",
                    "values": [2],
                    "num_requests": 6,
                    "warmup_requests": 1,
                    "scrape_interval_s": 0.05,
                },
            }
        ],
        "repetitions": 2,
        "slo": {"ttft_ms": {"p95": 1000}, "max_error_rate": 0.01},
        "budget": {"max_spend": "$1", "ttl_minutes": 10, "accrual_interval_s": 0.2},
    }
    doc.update(over)
    return doc


def mock_experiment(**over: Any) -> Experiment:
    return Experiment.model_validate(mock_doc(**over))


def write_yaml(path: Path, doc: dict[str, Any]) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(doc))
    return path
