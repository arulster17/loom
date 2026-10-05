from datetime import UTC, datetime

import pytest
from site_helpers import SPEC, TTFT_SGLANG, Store, make_runs, new_db, store_runs

from loom_bench.provenance import GitInfo
from loom_bench.records import Market
from loom_bench.store.db import session_scope
from loom_bench.store.repo import (
    create_experiment,
    record_cold_start,
    record_eval_run,
    record_gate_decision,
    update_experiment_status,
)


@pytest.fixture
def empty_db(tmp_path) -> str:
    return new_db(tmp_path / "empty.db")


@pytest.fixture(scope="module")
def populated(tmp_path_factory) -> Store:
    """One completed experiment: three Qwen3-8B configs on chat, evals, gates, a cold start."""
    url = new_db(tmp_path_factory.mktemp("db") / "loom.db")
    runs = [
        *make_runs("vllm-bf16"),
        *make_runs("sglang-bf16", engine="sglang", ttft=TTFT_SGLANG),
        *make_runs("vllm-awq", quantization="awq", ttft=TTFT_SGLANG, market=Market.SPOT),
    ]
    with session_scope(url) as s:
        exp = create_experiment(
            s,
            name="qwen3-8b-chat",
            spec=SPEC,
            git=GitInfo(sha="0123456789abcdef0123456789abcdef01234567", dirty=False),
            budget_micros=50_000_000,
            created_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
        hashes = store_runs(s, exp.id, runs)
        record_cold_start(
            s,
            experiment_id=exp.id,
            kind="cold",
            stages={"image_pull": 61.0, "weights": 39.0},
            total_s=100.0,
            config_hash=hashes["vllm-bf16"],
        )
        for cell, score in (("vllm-bf16", 0.71), ("sglang-bf16", 0.70), ("vllm-awq", 0.60)):
            record_eval_run(
                s,
                experiment_id=exp.id,
                config_hash=hashes[cell],
                task="gsm8k",
                task_version="1",
                n=500,
                score=score,
                ci_low=score - 0.02,
                ci_high=score + 0.02,
                provenance={},
            )
        for cell, decision in (("sglang-bf16", "pass"), ("vllm-awq", "fail")):
            record_gate_decision(
                s,
                experiment_id=exp.id,
                baseline_config_hash=hashes["vllm-bf16"],
                candidate_config_hash=hashes[cell],
                decision=decision,
                details={},
            )
        update_experiment_status(s, exp.id, "completed")
        exp_id = exp.id
    return Store(url=url, experiment_id=exp_id, hashes=hashes)
