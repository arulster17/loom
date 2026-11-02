"""Speculative decoding in the mock backend: the simulator drafts, verifies and emits
several tokens per step, counts it under vLLM v0.30.0's names, and the server streams one
chunk per step, so the whole spec-decode measurement path runs offline."""

import json
import statistics

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from loom_bench.metrics.prometheus import summarize_scrapes
from loom_bench.metrics.summary import summarize_run
from loom_bench.mock.config import MockConfig
from loom_bench.mock.metrics import render_metrics
from loom_bench.mock.sim import SimRequest, SimSequence, Simulator, VirtualClock
from loom_bench.records import RequestRecord, RequestStatus
from loom_bench.tokenize import SimpleTokenizer

TOK = SimpleTokenizer()
MODEL = "mock-model"


def run_one(sim: Simulator, clock: VirtualClock, request: SimRequest) -> SimSequence:
    seq = sim.add(request)
    while not seq.finished:
        plan = sim.schedule()
        assert plan is not None
        clock.advance(plan.duration_s)
        sim.finish_step(plan)
    return seq


def closed_loop_tpot(cfg: MockConfig, concurrency: int, n: int, prompt: int, output: int) -> float:
    """Mean per-request TPOT (simulated s) with `concurrency` requests always in flight."""
    clock = VirtualClock()
    sim = Simulator(cfg, clock)
    done: list[SimSequence] = []
    submitted = 0
    for _ in range(concurrency):
        sim.add(SimRequest(prompt_tokens=prompt, output_tokens=output))
        submitted += 1
    while sim.has_work:
        plan = sim.schedule()
        assert plan is not None
        clock.advance(plan.duration_s)
        for seq in sim.finish_step(plan):
            if seq.finished:
                done.append(seq)
                if submitted < n:
                    sim.add(SimRequest(prompt_tokens=prompt, output_tokens=output))
                    submitted += 1
    assert done
    return statistics.mean(
        (s.finished_s - s.first_token_s) / (s.num_output - 1)
        for s in done
        if s.finished_s is not None and s.first_token_s is not None
    )


def stream_events(client: httpx.Client, path: str, body: dict) -> list[dict | str]:
    events: list[dict | str] = []
    with client.stream("POST", path, json={**body, "stream": True}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line:
                data = line.removeprefix("data: ")
                events.append(data if data == "[DONE]" else json.loads(data))
    return events


def metrics(client: httpx.Client) -> dict[str, float]:
    samples = {}
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for s in family.samples:
            labels = ",".join(f"{k}={v}" for k, v in sorted(s.labels.items()) if k != "model_name")
            samples[f"{s.name}{{{labels}}}" if labels else s.name] = s.value
    return samples


BIG = {"max_num_seqs": 256, "max_batched_tokens": 8192, "kv_capacity_tokens": 1 << 21}


def test_no_speculation_by_default():
    clock = VirtualClock()
    sim = Simulator(MockConfig(), clock)
    run_one(sim, clock, SimRequest(prompt_tokens=10, output_tokens=20))
    assert sim.stats.spec_drafts == 0 and sim.stats.spec_accepted_per_pos == []
    assert "spec_decode" not in render_metrics(sim, "m").decode()


def test_a_request_emits_exactly_its_tokens_with_speculation():
    clock = VirtualClock()
    cfg = MockConfig(spec_tokens=3, spec_acceptance=0.8, seed=3)
    sim = Simulator(cfg, clock)
    seq = run_one(sim, clock, SimRequest(prompt_tokens=10, output_tokens=101))
    s = sim.stats
    assert seq.num_output == 101 == s.generation_tokens
    # Every step emits 1 + accepted tokens; the first token comes from prefill.
    assert s.spec_accepted_tokens == 101 - 1 - (s.spec_drafts + _undrafted_decodes(s, 101))
    assert s.spec_draft_tokens <= 3 * s.spec_drafts
    # Position i is accepted only when 0..i-1 were.
    pos = s.spec_accepted_per_pos
    assert len(pos) == 3 and pos[0] >= pos[1] >= pos[2] and sum(pos) == s.spec_accepted_tokens
    assert seq.num_computed == seq.num_tokens - 1  # the KV holds all but the last sample


def _undrafted_decodes(stats, output_tokens: int) -> int:
    """Decode steps with no draft: tokens = 1 (prefill) + drafts + accepted + undrafted."""
    return output_tokens - 1 - stats.spec_drafts - stats.spec_accepted_tokens


def test_acceptance_matches_the_configured_probability():
    clock = VirtualClock()
    cfg = MockConfig(spec_tokens=4, spec_acceptance=0.6, seed=7, **BIG)
    sim = Simulator(cfg, clock)
    for _ in range(20):
        run_one(sim, clock, SimRequest(prompt_tokens=8, output_tokens=400))
    s = sim.stats
    by_pos = [n / s.spec_drafts for n in s.spec_accepted_per_pos]
    for i, rate in enumerate(by_pos):
        assert rate == pytest.approx(0.6 ** (i + 1), abs=0.03)
    expected_len = 1 + sum(0.6 ** (i + 1) for i in range(4))
    assert 1 + s.spec_accepted_tokens / s.spec_drafts == pytest.approx(expected_len, abs=0.05)


def test_draft_rate_below_one_leaves_steps_undrafted():
    clock = VirtualClock()
    sim = Simulator(MockConfig(spec_tokens=3, spec_draft_rate=0.3, seed=1, **BIG), clock)
    run_one(sim, clock, SimRequest(prompt_tokens=8, output_tokens=1000))
    s = sim.stats
    undrafted = _undrafted_decodes(s, 1000)
    assert s.spec_drafts / (s.spec_drafts + undrafted) == pytest.approx(0.3, abs=0.06)


def test_speculation_helps_at_low_batch_and_fades_at_high_batch():
    """The mock's cost model reproduces the crossover the study looks for: verify work
    grows with every running sequence, drafting is a fixed cost per step."""
    base = {"step_base_ms": 10.0, "decode_ms_per_seq": 0.3, "seed": 5, **BIG}
    plain = MockConfig(**base)
    spec = MockConfig(**base, spec_tokens=3, spec_acceptance=0.6, spec_overhead_ms=2.0)

    def tpot(cfg: MockConfig, concurrency: int) -> float:
        return closed_loop_tpot(cfg, concurrency, 2 * concurrency, prompt=32, output=200)

    low = tpot(plain, 1) / tpot(spec, 1)
    high = tpot(plain, 128) / tpot(spec, 128)
    assert low > 1.5  # ~2.2 tokens a step for ~1.2x the step time
    assert high < 1.0  # 4x the verify work on 128 sequences outweighs it


def test_scraped_counters_summarize_to_the_simulators_truth():
    clock = VirtualClock()
    sim = Simulator(MockConfig(spec_tokens=3, spec_acceptance=0.7, seed=2, **BIG), clock)
    before = render_metrics(sim, MODEL).decode()
    for _ in range(5):
        run_one(sim, clock, SimRequest(prompt_tokens=8, output_tokens=300))
    after = render_metrics(sim, MODEL).decode()
    m = summarize_scrapes("mock", [(0.0, before), (1.0, after)], speculative=True)
    s = sim.stats
    assert m.spec_drafts == s.spec_drafts and m.spec_accepted_tokens == s.spec_accepted_tokens
    assert m.spec_mean_accepted_length == pytest.approx(1 + s.spec_accepted_tokens / s.spec_drafts)
    assert m.spec_acceptance_rate == pytest.approx(s.spec_accepted_tokens / s.spec_draft_tokens)
    assert m.spec_acceptance_by_position == pytest.approx(
        {str(i): n / s.spec_drafts for i, n in enumerate(s.spec_accepted_per_pos)}
    )
    assert "spec_decode" not in m.missing


def test_server_streams_one_chunk_per_step(make_client):
    c = make_client(spec_tokens=3, spec_acceptance=0.9, seed=4)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a story."}],
        "max_tokens": 60,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    events = stream_events(c, "/v1/chat/completions", body)
    chunks = events[1:-2]  # role chunk first; usage and [DONE] last
    pieces = [ch["choices"][0]["delta"]["content"] for ch in chunks]
    counts = [TOK.count(p) for p in pieces]
    usage = events[-2]["usage"]
    assert sum(counts) == usage["completion_tokens"] == TOK.count("".join(pieces))
    assert max(counts) > 1 and max(counts) <= 4  # at most k drafts + 1
    assert statistics.mean(counts) > 2
    # The same text as without streaming.
    plain = c.post("/v1/chat/completions", json={**body, "stream": False}).json()
    assert plain["choices"][0]["message"]["content"] == "".join(pieces)
    m = metrics(c)
    assert m["vllm:spec_decode_num_drafts_total"] > 0
    assert m["vllm:spec_decode_num_accepted_tokens_per_pos_total{position=0}"] > 0


def test_completions_stream_with_logprobs_and_echo(make_client):
    c = make_client(spec_tokens=3, spec_acceptance=0.9, seed=4)
    body = {"model": MODEL, "prompt": "one two three", "max_tokens": 30, "logprobs": 1}
    events = stream_events(c, "/v1/completions", {**body, "echo": True})
    texts = [e["choices"][0]["text"] for e in events if isinstance(e, dict) and e["choices"]]
    lps = [e["choices"][0]["logprobs"] for e in events if isinstance(e, dict) and e["choices"]]
    assert texts[0].startswith("one two three")
    for text, lp in zip(texts, lps, strict=True):
        assert "".join(lp["tokens"]) == text  # one logprob entry per token in the chunk
    plain = c.post("/v1/completions", json={**body, "echo": True}).json()
    assert plain["choices"][0]["text"] == "".join(texts)


def test_tokens_per_chunk_reads_acceptance_from_the_client_side():
    """RunSummary.tokens_per_chunk: completion tokens over content chunks per request."""
    recs = [
        RequestRecord(
            request_id=str(i),
            status=RequestStatus.OK,
            sent_at_s=0.0,
            first_token_at_s=0.1,
            finished_at_s=1.0,
            itl_s=[0.1] * gaps,
            prompt_tokens=10,
            completion_tokens=tokens,
        )
        for i, (gaps, tokens) in enumerate([(9, 30), (29, 30), (14, 30)])
    ]
    s = summarize_run(recs, window_s=1.0, gpus=1)
    assert s.tokens_per_chunk is not None
    assert s.tokens_per_chunk.min == pytest.approx(1.0)  # a request no draft helped
    assert s.tokens_per_chunk.max == pytest.approx(3.0)
    assert s.tokens_per_chunk.p50 == pytest.approx(2.0)


CONTINUOUS = {"include_usage": True, "continuous_usage_stats": True}


@pytest.mark.parametrize("n", [1, 3])
def test_multi_token_chunks_keep_continuous_usage_per_choice(make_client, n):
    """Under speculation each chunk's running usage is the tokens its choice has streamed."""
    c = make_client(spec_tokens=3, spec_acceptance=0.9, seed=4)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a story."}],
        "max_tokens": 40,
        "ignore_eos": True,
        "n": n,
        "stream_options": CONTINUOUS,
    }
    events = stream_events(c, "/v1/chat/completions", body)
    chunks = [e for e in events if isinstance(e, dict) and e["choices"]]
    seen: dict[int, int] = {}
    multi = False
    for ch in chunks:
        (choice,) = ch["choices"]
        got = TOK.count(choice["delta"].get("content") or "")
        multi |= got > 1
        seen[choice["index"]] = seen.get(choice["index"], 0) + got
        assert ch["usage"]["completion_tokens"] == seen[choice["index"]]
    assert multi and set(seen) == set(range(n)) and all(v == 40 for v in seen.values())


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_an_abort_inside_a_multi_token_step_streams_exactly_the_tokens_before_it(make_client, path):
    """The injected abort cuts the step's chunk at `abort_at`: the last usage equals the
    tokens received, as the gateway bills."""
    c = make_client(spec_tokens=3, spec_acceptance=0.9, seed=4, abort_rate=1.0)
    prompt = {"messages": [{"role": "user", "content": "hi"}]}
    if path == "/v1/completions":
        prompt = {"prompt": "hi"}
    body = {
        "model": MODEL,
        **prompt,
        "max_tokens": 40,
        "ignore_eos": True,
        "stream": True,
        "stream_options": CONTINUOUS,
    }
    for _ in range(8):
        received: list[dict] = []
        with (
            pytest.raises(httpx.RemoteProtocolError),
            c.stream("POST", path, json=body) as response,
        ):
            for line in response.iter_lines():
                if line.startswith("data: {"):
                    received.append(json.loads(line[6:]))
        texts = [
            (e["choices"][0].get("delta") or {}).get("content") or e["choices"][0].get("text")
            for e in received
            if e["choices"]
        ]
        tokens = sum(TOK.count(t or "") for t in texts)
        if received:
            assert received[-1]["usage"]["completion_tokens"] == tokens < 40
