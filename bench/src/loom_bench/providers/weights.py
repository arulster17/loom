"""Which pinned checkpoints a host holds, and what each engine start must download.

Engines run offline (`HF_HUB_OFFLINE=1`) against the host's Hugging Face cache, so every
checkpoint a launch loads (the model and any speculative draft, `engines.launch_weights`)
must be on the host before the engine starts. A host's first (cold) start downloads its
launch's checkpoints. A warm restart onto a launch with a checkpoint the host does not
hold yet (a variant serving another registry entry or an `hf` override, e.g. the FP8 row
after its BF16 baseline on one pod) downloads that one too, at its pinned revision, as the
cold start does, and checks the ones it already holds offline. 2026-10-08's
runpod-smoke-h100 (4e50b5a5) failed because warm restarts downloaded nothing: the FP8
cell's engine start found RedHatAI/Qwen3-8B-FP8-dynamic "not cached".

The aws_ec2, runpod and mock providers each keep one `HostWeights`; the planner prices the
same downloads (`plan.Estimator.download_bytes`).
"""

from __future__ import annotations

from dataclasses import dataclass

from loom_bench.engines import launch_weights
from loom_bench.providers.base import EngineLaunch

Checkpoint = tuple[str, str]  # (HF repo, pinned revision)


def flat(checkpoints: tuple[Checkpoint, ...]) -> list[str]:
    """`repo revision repo revision ...`, as the start scripts take them."""
    return [x for pair in checkpoints for x in pair]


@dataclass(frozen=True)
class WeightsPlan:
    """One engine start's checkpoints: those to download (with the HF token, online) and
    those an earlier start on the host already downloaded (checked offline)."""

    fetch: tuple[Checkpoint, ...]
    cached: tuple[Checkpoint, ...]


class HostWeights:
    """The checkpoints each host's cache holds, as this provider downloaded them."""

    def __init__(self) -> None:
        self._held: dict[str, set[Checkpoint]] = {}

    def plan(self, host_id: str, launch: EngineLaunch, *, warm: bool) -> WeightsPlan:
        # A cold start is a new host: nothing is cached yet.
        held = self._held.get(host_id, set()) if warm else set()
        need = launch_weights(launch)
        return WeightsPlan(
            fetch=tuple(c for c in need if c not in held),
            cached=tuple(c for c in need if c in held),
        )

    def downloaded(self, host_id: str, plan: WeightsPlan) -> None:
        """Record that the host now holds `plan`'s checkpoints (its start passed
        weights_ready)."""
        self._held.setdefault(host_id, set()).update(plan.fetch)

    def held(self, host_id: str) -> frozenset[Checkpoint]:
        return frozenset(self._held.get(host_id, set()))

    def forget(self, host_id: str) -> None:
        self._held.pop(host_id, None)
