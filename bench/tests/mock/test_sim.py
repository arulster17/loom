import random
import statistics
from dataclasses import dataclass

import pytest

from loom_bench.mock.config import MockConfig
from loom_bench.mock.sim import (
    SimRequest,
    SimSequence,
    Simulator,
    VirtualClock,
    prompt_block_hashes,
)
from loom_bench.tokenize import SimpleTokenizer

TOK = SimpleTokenizer()


@dataclass
class RunResult:
    seqs: list[SimSequence]
    elapsed_s: float
    sim: Simulator

    @property
    def mean_ttft(self) -> float:
        return statistics.mean(s.first_token_s - s.arrival_s for s in self.seqs)

    @property
    def mean_tpot(self) -> float:
        return statistics.mean(
            (s.finished_s - s.first_token_s) / (s.num_output - 1) for s in self.seqs
        )

    @property
    def output_tps(self) -> float:
        return sum(s.num_output for s in self.seqs) / self.elapsed_s


def closed_loop(
    config: MockConfig, concurrency: int, n_requests: int, prompt: int, output: int
) -> RunResult:
    clock = VirtualClock()
    sim = Simulator(config, clock)
    done: list[SimSequence] = []
    submitted = 0

    def submit() -> None:
        nonlocal submitted
        sim.add(SimRequest(prompt_tokens=prompt, output_tokens=output))
        submitted += 1

    for _ in range(min(concurrency, n_requests)):
        submit()
    while sim.has_work:
        plan = sim.schedule()
        assert plan is not None
        clock.advance(plan.duration_s)
        for seq in sim.finish_step(plan):
            if seq.finished:
                done.append(seq)
                if submitted < n_requests:
                    submit()
    return RunResult(done, clock.now, sim)


def run_one(sim: Simulator, clock: VirtualClock, request: SimRequest) -> SimSequence:
    seq = sim.add(request)
    while not seq.finished:
        plan = sim.schedule()
        assert plan is not None
        clock.advance(plan.duration_s)
        sim.finish_step(plan)
    return seq


CFG = MockConfig(max_num_seqs=32, max_batched_tokens=2048, kv_capacity_tokens=1 << 20)


def test_latency_rises_with_concurrency():
    low = closed_loop(CFG, concurrency=1, n_requests=8, prompt=256, output=64)
    mid = closed_loop(CFG, concurrency=16, n_requests=64, prompt=256, output=64)
    high = closed_loop(CFG, concurrency=128, n_requests=256, prompt=256, output=64)
    assert low.mean_tpot < mid.mean_tpot < high.mean_tpot
    assert low.mean_ttft < mid.mean_ttft < high.mean_ttft
    # Beyond max_num_seqs requests queue, so TTFT grows much faster than TPOT.
    assert high.mean_ttft > 5 * mid.mean_ttft


def test_throughput_saturates():
    tps = {
        c: closed_loop(CFG, concurrency=c, n_requests=4 * c, prompt=128, output=64).output_tps
        for c in (1, 8, 32, 64, 128)
    }
    assert tps[8] > 4 * tps[1]
    assert tps[32] > tps[8]
    # Capped at max_num_seqs=32: more clients add queueing, not throughput.
    assert tps[128] == pytest.approx(tps[32], rel=0.1)
    assert tps[64] == pytest.approx(tps[32], rel=0.1)


def test_step_cost_model():
    cfg = MockConfig(step_base_ms=10, decode_ms_per_seq=1, prefill_ms_per_token=0.5, time_scale=2)
    sim = Simulator(cfg, VirtualClock())
    sim.add(SimRequest(prompt_tokens=100, output_tokens=3))
    plan = sim.schedule()
    assert (plan.num_prefill_tokens, plan.num_decode_seqs) == (100, 0)
    assert plan.duration_s == pytest.approx((10 + 50) / 1000 * 2)
    sim.finish_step(plan)
    plan = sim.schedule()
    assert (plan.num_prefill_tokens, plan.num_decode_seqs) == (0, 1)
    assert plan.duration_s == pytest.approx((10 + 1) / 1000 * 2)


def test_chunked_prefill_respects_token_budget():
    cfg = MockConfig(max_batched_tokens=512)
    clock = VirtualClock()
    sim = Simulator(cfg, clock)
    seq = sim.add(SimRequest(prompt_tokens=1300, output_tokens=2))
    chunks = []
    while not seq.finished:
        plan = sim.schedule()
        chunks.append(plan.num_prefill_tokens)
        clock.advance(plan.duration_s)
        sim.finish_step(plan)
    assert chunks == [512, 512, 276, 0]
    assert seq.num_output == 2


def test_max_num_seqs_limits_running():
    cfg = MockConfig(max_num_seqs=4)
    sim = Simulator(cfg, VirtualClock())
    for _ in range(10):
        sim.add(SimRequest(prompt_tokens=10, output_tokens=5))
    sim.schedule()
    assert (sim.num_running, sim.num_waiting) == (4, 6)


def text(n: int, seed: int) -> str:
    return TOK.random_text(n, random.Random(seed))


def request_for(prompt: str) -> SimRequest:
    pieces = TOK.pieces(prompt)
    return SimRequest(
        prompt_tokens=len(pieces), output_tokens=4, block_hashes=prompt_block_hashes(pieces, 16)
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_prefix_cache_skips_prefill(enabled):
    clock = VirtualClock()
    sim = Simulator(MockConfig(prefix_caching=enabled), clock)
    prefix = text(1000, seed=1)
    first = request_for(prefix + " " + text(100, seed=2))
    second = request_for(prefix + " " + text(100, seed=3))
    a = run_one(sim, clock, first)
    b = run_one(sim, clock, second)
    assert a.num_cached_tokens == 0
    ttft_a = a.first_token_s - a.arrival_s
    ttft_b = b.first_token_s - b.arrival_s
    if enabled:
        assert b.num_cached_tokens == 1000 // 16 * 16
        assert ttft_b < 0.3 * ttft_a
        assert sim.stats.prefix_cache_hits == b.num_cached_tokens
        assert sim.stats.prefix_cache_queries == first.prompt_tokens + second.prompt_tokens
    else:
        assert b.num_cached_tokens == 0
        assert ttft_b == pytest.approx(ttft_a)
        assert sim.stats.prefix_cache_queries == 0


def test_prefix_cache_identical_prompt_recomputes_last_block():
    clock = VirtualClock()
    sim = Simulator(MockConfig(), clock)
    req = request_for(text(64, seed=1))
    run_one(sim, clock, req)
    again = run_one(sim, clock, req)
    assert again.num_cached_tokens == 48  # at least one token is computed to produce logits


def test_prefix_cache_lru_eviction():
    clock = VirtualClock()
    sim = Simulator(MockConfig(kv_capacity_tokens=2048, max_model_len=2048), clock)
    first = request_for(text(1000, seed=1))
    run_one(sim, clock, first)
    # 128 blocks total. Blocks are freed tail-first, so the first prompt's leading
    # blocks are the last of it to go; three unrelated prompts evict all of it.
    for seed in (2, 3, 4):
        run_one(sim, clock, request_for(text(1000, seed=seed)))
    assert run_one(sim, clock, first).num_cached_tokens == 0


def test_preemption_under_kv_pressure():
    cfg = MockConfig(kv_capacity_tokens=1024, max_model_len=512, max_num_seqs=16)
    result = closed_loop(cfg, concurrency=8, n_requests=8, prompt=100, output=300)
    assert result.sim.stats.num_preemptions > 0
    assert sum(s.preemptions for s in result.seqs) == result.sim.stats.num_preemptions
    assert len(result.seqs) == 8
    assert all(s.num_output == 300 for s in result.seqs)
    assert result.sim.kv.used_blocks == 0


def test_kv_usage_tracks_running_blocks():
    cfg = MockConfig(kv_capacity_tokens=1600, max_model_len=800)
    clock = VirtualClock()
    sim = Simulator(cfg, clock)
    sim.add(SimRequest(prompt_tokens=400, output_tokens=10))
    plan = sim.schedule()
    assert sim.kv.usage == pytest.approx(25 / 100)
    clock.advance(plan.duration_s)
    sim.finish_step(plan)
    assert 0 < sim.kv.usage <= 1


def test_abort_frees_kv():
    sim = Simulator(MockConfig(), VirtualClock())
    seq = sim.add(SimRequest(prompt_tokens=100, output_tokens=10))
    plan = sim.schedule()
    sim.abort(seq)
    assert sim.finish_step(plan) == []
    assert sim.kv.used_blocks == 0
    assert not sim.has_work


def test_queue_time_histogram_and_counters():
    cfg = MockConfig(max_num_seqs=1)
    result = closed_loop(cfg, concurrency=4, n_requests=4, prompt=50, output=5)
    stats = result.sim.stats
    assert stats.queue_time.count == 4
    assert stats.queue_time.total > 0
    assert stats.prompt_tokens == 200
    assert stats.generation_tokens == 20
    assert stats.request_success == {"length": 4}


def test_deterministic():
    cfg = MockConfig(kv_capacity_tokens=1024, max_model_len=512, max_num_seqs=16)
    runs = [closed_loop(cfg, concurrency=8, n_requests=16, prompt=100, output=200) for _ in "ab"]
    timings = [
        [(s.seq_id, s.first_token_s, s.finished_s, s.preemptions) for s in r.seqs] for r in runs
    ]
    assert timings[0] == timings[1]
    assert runs[0].sim.stats == runs[1].sim.stats


def test_rejects_request_over_max_model_len():
    sim = Simulator(MockConfig(max_model_len=100), VirtualClock())
    with pytest.raises(ValueError):
        sim.add(SimRequest(prompt_tokens=90, output_tokens=11))


def test_config_requires_kv_for_one_sequence():
    with pytest.raises(ValueError):
        MockConfig(kv_capacity_tokens=1000, max_model_len=2000)
