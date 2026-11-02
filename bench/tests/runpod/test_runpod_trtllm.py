"""The TensorRT-LLM vs vLLM run and its smoke on the runpod provider, against FakeRunpod
and the pod simulator, at test scale: the shipped specs' model, provider and both
variants (one pod per engine image), with their replicated passes and the in-run gate.

What it pins down before a pod is paid for: the TRT-LLM pod's start script runs the
launcher with the checkpoint's snapshot directory moved into the pod's weight cache
(PodSim starts the engine only if that checkpoint is cached), its load jobs scrape
/prometheus/metrics, and an engine that rejects completions logprobs, as TRT-LLM 1.2.1
does, loses only the divergence half of its eval: task scores stand, `divergence_failed`
names the reason, and the gate is INCONCLUSIVE (blocked), not missing."""

import json
from typing import Any

import pytest

from loom_bench.engines import ENTRYPOINTS
from loom_bench.experiment import EXPERIMENTS_DIR
from loom_bench.jobs import LoadJob

from . import test_runpod_two_checkpoints as two
from .fakes import script_array

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("runpod-e2e")]

REAL = EXPERIMENTS_DIR / "qwen3-8b-trtllm-vs-vllm-runpod.yaml"
SMOKE = EXPERIMENTS_DIR / "runpod-smoke-trtllm.yaml"
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"


class TrtllmPodSim(two.RecordingPodSim):
    """The TRT-LLM pod's mock engine rejects completions logprobs, as 1.2.1 does."""

    async def start(self, target: Any, script: str) -> Any:
        self.mock = {"completions_logprobs": "trtllm-launch" not in script}
        return await super().start(target, script)


@pytest.fixture(scope="module", params=[REAL, SMOKE], ids=lambda p: p.stem)
def trt_run(request, tmp_path_factory):
    path = request.param
    return two.run_on_pod_sim(
        path, tmp_path_factory.mktemp(path.stem), replicates=True, pod_sim=TrtllmPodSim
    )


def _events(outcome, ctx) -> list[dict]:
    path = ctx.out_dir / str(outcome.experiment_id) / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_pod_per_engine_and_the_launcher_on_the_trtllm_pod(trt_run):
    outcome, sim, _, cells, _ = trt_run
    assert outcome.status.value == "completed", outcome.reason
    starts = sim.scripts("start_engine")
    assert len(starts) == 2 and len({c.host_key for c in cells}) == 2
    vllm_cmd, trt_cmd = (script_array(s, "ENGINE_CMD") for s in starts)
    assert vllm_cmd[:2] == ["vllm", "serve"]
    assert trt_cmd[:4] == list(ENTRYPOINTS["trtllm"])
    assert trt_cmd[5] == f"/opt/loom/hf/hub/models--Qwen--Qwen3-8B/snapshots/{QWEN_REV}"
    assert trt_cmd[trt_cmd.index("--host") + 1] == "127.0.0.1"
    assert json.loads(trt_cmd[4]) == {
        "guided_decoding_backend": "xgrammar",
        "return_perf_metrics": True,
    }


def test_trtllm_load_jobs_scrape_its_prometheus_path(trt_run):
    _, sim, _, _, _ = trt_run
    urls = {j.engine: j.metrics_url for j in sim.jobs if isinstance(j, LoadJob)}
    assert urls["trtllm"].endswith("/prometheus/metrics")
    assert urls["vllm"].endswith(":8000/metrics")


def test_no_completions_logprobs_blocks_only_the_divergence_half(trt_run):
    outcome, _, ctx, _, _ = trt_run
    events = _events(outcome, ctx)
    quality = {e["cell"]: e for e in events if e["kind"] == "quality"}
    assert set(quality) == {"vllm-bf16", "trtllm-bf16"}  # task scores stand on both
    assert not [e for e in events if e["kind"] == "quality_failed"]
    failed = [e for e in events if e["kind"] == "divergence_failed"]
    assert [e["cell"] for e in failed] == ["trtllm-bf16"]
    assert "logprobs is not supported" in json.dumps(failed[0])
    (gate,) = outcome.gates
    assert (gate.cell, gate.baseline) == ("trtllm-bf16", "vllm-bf16")
    assert str(gate.decision) == "inconclusive" and gate.blocked
