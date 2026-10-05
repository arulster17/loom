"""Simulated continuous-batching engine, modelled on the vLLM V1 scheduler.

Pure and synchronous: a driver calls `schedule()`, waits `plan.duration_s`
(real sleep or a `VirtualClock` advance), then calls `finish_step(plan)`.

Each sequence schedules `num_tokens - num_computed` tokens per step, capped by
the shared token budget, so prefill, chunked prefill, decode (one token) and
recompute after preemption are the same code path. A token is sampled when a
step brings `num_computed` up to `num_tokens`.
"""

from __future__ import annotations

import bisect
import hashlib
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from loom_bench.mock.config import MockConfig

Clock = Callable[[], float]

# vLLM V1 request latency buckets (seconds).
QUEUE_TIME_BUCKETS = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
)  # fmt: skip


class VirtualClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def prompt_block_hashes(pieces: Sequence[str], block_size: int) -> tuple[bytes, ...]:
    """Chained hashes of the full blocks of a tokenized prompt (vLLM-style).

    Block i's hash covers all tokens up to the end of block i, so equal hashes
    mean equal prefixes.
    """
    hashes: list[bytes] = []
    prev = b""
    for start in range(0, len(pieces) - block_size + 1, block_size):
        h = hashlib.blake2b(prev, digest_size=16)
        for piece in pieces[start : start + block_size]:
            data = piece.encode()
            h.update(len(data).to_bytes(4, "big"))
            h.update(data)
        prev = h.digest()
        hashes.append(prev)
    return tuple(hashes)


@dataclass(frozen=True, slots=True)
class SimRequest:
    prompt_tokens: int
    output_tokens: int
    block_hashes: tuple[bytes, ...] = ()
    finish_reason: str = "length"

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1 or self.output_tokens < 1:
            raise ValueError("prompt_tokens and output_tokens must be >= 1")


class SeqStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(eq=False, slots=True)
class SimSequence:
    seq_id: int
    request: SimRequest
    arrival_s: float
    status: SeqStatus = SeqStatus.WAITING
    num_computed: int = 0
    num_output: int = 0
    # Prefix-cache hit at first admission; what the API reports as cached tokens.
    num_cached_tokens: int | None = None
    first_scheduled_s: float | None = None
    first_token_s: float | None = None
    finished_s: float | None = None
    preemptions: int = 0
    # KV blocks held: shared content-addressed prompt blocks, then private ones.
    hashed_blocks: list[bytes] = field(default_factory=list)
    private_blocks: int = 0

    @property
    def num_tokens(self) -> int:
        return self.request.prompt_tokens + self.num_output

    @property
    def finished(self) -> bool:
        return self.status is SeqStatus.FINISHED

    @property
    def num_blocks(self) -> int:
        return len(self.hashed_blocks) + self.private_blocks


@dataclass(slots=True)
class Histogram:
    bounds: tuple[float, ...]
    counts: list[int] = field(init=False)  # per bucket, non-cumulative; last is +Inf
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self.counts = [0] * (len(self.bounds) + 1)

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.bounds, value)] += 1
        self.total += value
        self.count += 1


@dataclass(slots=True)
class SimStats:
    num_preemptions: int = 0
    prefix_cache_queries: int = 0  # tokens looked up
    prefix_cache_hits: int = 0  # tokens found
    prompt_tokens: int = 0
    generation_tokens: int = 0
    request_success: dict[str, int] = field(default_factory=dict)  # by finish reason
    queue_time: Histogram = field(default_factory=lambda: Histogram(QUEUE_TIME_BUCKETS))


@dataclass(frozen=True, slots=True)
class StepPlan:
    scheduled: tuple[tuple[SimSequence, int], ...]  # (sequence, tokens computed this step)
    num_decode_seqs: int
    num_prefill_tokens: int
    duration_s: float


class KVCache:
    """Block accounting with an LRU prefix cache.

    Hashed blocks with no running reference stay resident and reusable but
    count as free; allocation evicts them least-recently-freed first.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._private = 0
        self._refs: dict[bytes, int] = {}
        self._evictable: OrderedDict[bytes, None] = OrderedDict()

    @property
    def used_blocks(self) -> int:
        return self._private + len(self._refs)

    @property
    def free_blocks(self) -> int:
        return self.num_blocks - self.used_blocks

    @property
    def usage(self) -> float:
        return self.used_blocks / self.num_blocks

    def num_hits(self, hashes: Sequence[bytes]) -> int:
        """Length of the leading run of `hashes` resident in the cache."""
        n = 0
        for h in hashes:
            if h not in self._refs and h not in self._evictable:
                break
            n += 1
        return n

    def can_allocate(self, num_private: int, acquire: Sequence[bytes] = ()) -> bool:
        revived = sum(1 for h in acquire if h not in self._refs)
        return num_private + revived <= self.free_blocks

    def allocate(self, seq: SimSequence, num_private: int, acquire: Sequence[bytes] = ()) -> None:
        for h in acquire:
            if h in self._refs:
                self._refs[h] += 1
            else:
                del self._evictable[h]
                self._refs[h] = 1
            seq.hashed_blocks.append(h)
        self._private += num_private
        seq.private_blocks += num_private
        self._evict_overflow()

    def promote(self, seq: SimSequence, h: bytes) -> None:
        """Turn one of `seq`'s private blocks into the cached block `h`."""
        self._private -= 1
        seq.private_blocks -= 1
        if h in self._refs:
            self._refs[h] += 1  # computed concurrently by another sequence; share it
        else:
            self._evictable.pop(h, None)
            self._refs[h] = 1
        seq.hashed_blocks.append(h)
        self._evict_overflow()

    def release(self, seq: SimSequence) -> None:
        self._private -= seq.private_blocks
        # Tail blocks first, so a request's deepest blocks are evicted before its prefix.
        for h in reversed(seq.hashed_blocks):
            self._refs[h] -= 1
            if self._refs[h] == 0:
                del self._refs[h]
                self._evictable[h] = None
        seq.hashed_blocks.clear()
        seq.private_blocks = 0

    def _evict_overflow(self) -> None:
        while self.used_blocks + len(self._evictable) > self.num_blocks:
            self._evictable.popitem(last=False)


class Simulator:
    def __init__(self, config: MockConfig, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self.kv = KVCache(config.num_kv_blocks)
        self.stats = SimStats()
        self._waiting: deque[SimSequence] = deque()
        self._running: list[SimSequence] = []
        self._next_id = 0

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def has_work(self) -> bool:
        return bool(self._running or self._waiting)

    def add(self, request: SimRequest) -> SimSequence:
        if request.prompt_tokens + request.output_tokens > self.config.max_model_len:
            raise ValueError("request exceeds max_model_len")
        seq = SimSequence(seq_id=self._next_id, request=request, arrival_s=self.clock())
        self._next_id += 1
        self._waiting.append(seq)
        return seq

    def abort(self, seq: SimSequence) -> None:
        if seq.status is SeqStatus.WAITING:
            self._waiting.remove(seq)
        elif seq.status is SeqStatus.RUNNING:
            self.kv.release(seq)
            self._running.remove(seq)
        else:
            return
        seq.status = SeqStatus.ABORTED

    def schedule(self) -> StepPlan | None:
        """Pick the work for the next step; None when idle."""
        budget = self.config.max_batched_tokens
        scheduled: list[tuple[SimSequence, int]] = []
        preempted = False

        idx = 0
        while idx < len(self._running) and budget > 0:
            seq = self._running[idx]
            n = min(seq.num_tokens - seq.num_computed, budget)
            need = self._blocks_needed(seq, n)
            while not self.kv.can_allocate(need):
                victim = self._running.pop()
                self._preempt(victim)
                preempted = True
                if victim is seq:
                    break
            else:
                self.kv.allocate(seq, need)
                scheduled.append((seq, n))
                budget -= n
                idx += 1

        # Like vLLM, admit nothing new in a step that had to preempt.
        if not preempted:
            budget = self._admit(budget, scheduled)

        if not scheduled:
            return None
        num_decode = 0
        num_prefill = 0
        for seq, n in scheduled:
            if n == 1 and seq.num_computed >= seq.request.prompt_tokens:
                num_decode += 1
            else:
                num_prefill += n
        cfg = self.config
        ms = (
            cfg.step_base_ms
            + cfg.decode_ms_per_seq * num_decode
            + cfg.prefill_ms_per_token * num_prefill
        )
        return StepPlan(tuple(scheduled), num_decode, num_prefill, ms / 1000 * cfg.time_scale)

    def finish_step(self, plan: StepPlan) -> list[SimSequence]:
        """Apply a step's results; returns sequences that sampled a token."""
        now = self.clock()
        emitted: list[SimSequence] = []
        for seq, n in plan.scheduled:
            if seq.status is not SeqStatus.RUNNING:
                continue  # aborted while the step ran
            seq.num_computed += n
            self._cache_full_blocks(seq)
            if seq.num_computed < seq.num_tokens:
                continue  # chunked prefill not done yet
            seq.num_output += 1
            self.stats.generation_tokens += 1
            if seq.first_token_s is None:
                seq.first_token_s = now
                self.stats.prompt_tokens += seq.request.prompt_tokens
            if seq.num_output >= seq.request.output_tokens:
                self._finish(seq, now)
            emitted.append(seq)
        return emitted

    def _admit(self, budget: int, scheduled: list[tuple[SimSequence, int]]) -> int:
        bs = self.config.block_size
        while self._waiting and budget > 0 and len(self._running) < self.config.max_num_seqs:
            seq = self._waiting[0]
            hashes = seq.request.block_hashes if self.config.prefix_caching else ()
            hit_blocks = self.kv.num_hits(hashes)
            # At least one token must be computed to produce logits.
            hit_blocks = min(hit_blocks, (seq.num_tokens - 1) // bs)
            hit_tokens = hit_blocks * bs
            n = min(seq.num_tokens - hit_tokens, budget)
            need = -(-(hit_tokens + n) // bs) - hit_blocks
            if not self.kv.can_allocate(need, hashes[:hit_blocks]):
                break
            self._waiting.popleft()
            self.kv.allocate(seq, need, hashes[:hit_blocks])
            seq.num_computed = hit_tokens
            seq.status = SeqStatus.RUNNING
            self._running.append(seq)
            if self.config.prefix_caching:
                self.stats.prefix_cache_queries += seq.num_tokens
                self.stats.prefix_cache_hits += hit_tokens
            if seq.first_scheduled_s is None:
                seq.first_scheduled_s = self.clock()
                seq.num_cached_tokens = hit_tokens
                self.stats.queue_time.observe(seq.first_scheduled_s - seq.arrival_s)
            scheduled.append((seq, n))
            budget -= n
        return budget

    def _blocks_needed(self, seq: SimSequence, n: int) -> int:
        bs = self.config.block_size
        return max(0, -(-(seq.num_computed + n) // bs) - seq.num_blocks)

    def _cache_full_blocks(self, seq: SimSequence) -> None:
        if not self.config.prefix_caching:
            return
        hashes = seq.request.block_hashes
        full = min(seq.num_computed // self.config.block_size, len(hashes))
        for i in range(len(seq.hashed_blocks), full):
            self.kv.promote(seq, hashes[i])

    def _preempt(self, seq: SimSequence) -> None:
        # Recompute-style: drop the KV, keep generated tokens, rejoin at the queue head.
        self.kv.release(seq)
        seq.num_computed = 0
        seq.status = SeqStatus.WAITING
        seq.preemptions += 1
        self.stats.num_preemptions += 1
        self._waiting.appendleft(seq)

    def _finish(self, seq: SimSequence, now: float) -> None:
        seq.status = SeqStatus.FINISHED
        seq.finished_s = now
        self.kv.release(seq)
        self._running.remove(seq)
        reason = seq.request.finish_reason
        self.stats.request_success[reason] = self.stats.request_success.get(reason, 0) + 1
