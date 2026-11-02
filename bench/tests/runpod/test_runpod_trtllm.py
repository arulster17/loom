"""The TensorRT-LLM vs vLLM run and its smoke on the runpod provider, against FakeRunpod
and the pod simulator, at test scale: the shipped specs' model, provider and three
variants (vLLM BF16, TRT-LLM BF16, TRT-LLM FP8 on its own pod), with their replicated
passes and the in-run gates.

What it pins down before a pod is paid for: each TRT-LLM pod's start script runs the
launcher with Loom's prompt-logprobs wrapper and the checkpoint's snapshot directory moved
into the pod's weight cache (PodSim starts the engine only if that checkpoint is cached),
the FP8 pod downloads the ModelOpt checkpoint, its load jobs scrape /prometheus/metrics,
and both TRT-LLM cells get a gate with a measured divergence. The simulated TRT-LLM
engine refuses echo+logprobs scoring unless the engine command carries the wrapper, as
trtllm-serve does, so dropping the wrapper fails here as it would on a pod.
"""

import json
from typing import Any

import pytest

from loom_bench.engines import ENTRYPOINTS, TRTLLM_SERVE_SOURCE
from loom_bench.experiment import EXPERIMENTS_DIR
from loom_bench.jobs import LoadJob

from . import test_runpod_two_checkpoints as two
from .fakes import script_array, script_pairs

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("runpod-e2e")]

REAL = EXPERIMENTS_DIR / "qwen3-8b-trtllm-vs-vllm-runpod.yaml"
SMOKE = EXPERIMENTS_DIR / "runpod-smoke-trtllm.yaml"
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"
FP8 = ("nvidia/Qwen3-8B-FP8", "2cebc4c89e25abc17668c81b01dceaf3d8b914d5")


class TrtllmPodSim(two.RecordingPodSim):
    """A TRT-LLM pod's engine answers echo+logprobs scoring only through the wrapper."""

    async def start(self, target: Any, script: str) -> Any:
        cmd = script_array(script, "ENGINE_CMD")
        stock_trtllm = "trtllm-launch" in cmd and TRTLLM_SERVE_SOURCE not in cmd
        self.mock = {"completions_logprobs": not stock_trtllm}
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


def test_three_pods_and_the_wrapper_on_each_trtllm_pod(trt_run):
    outcome, sim, _, cells, _ = trt_run
    assert outcome.status.value == "completed", outcome.reason
    starts = sim.scripts("start_engine")
    assert len(starts) == 3 and len({c.host_key for c in cells}) == 3
    vllm_cmd, bf16_cmd, fp8_cmd = (script_array(s, "ENGINE_CMD") for s in starts)
    assert vllm_cmd[:2] == ["vllm", "serve"]
    for cmd in (bf16_cmd, fp8_cmd):
        assert cmd[:5] == list(ENTRYPOINTS["trtllm"])  # bash launcher + wrapper source
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert bf16_cmd[6] == f"/opt/loom/hf/hub/models--Qwen--Qwen3-8B/snapshots/{QWEN_REV}"
    assert fp8_cmd[6] == f"/opt/loom/hf/hub/models--nvidia--Qwen3-8B-FP8/snapshots/{FP8[1]}"
    assert json.loads(fp8_cmd[5])["kv_cache_config"] == {"dtype": "fp8"}
    assert script_pairs(starts[2], "FETCH_WEIGHTS") == [FP8]
    assert fp8_cmd[fp8_cmd.index("--served_model_name") + 1] == "qwen3-8b-fp8-modelopt"


def test_trtllm_load_jobs_scrape_its_prometheus_path(trt_run):
    _, sim, _, _, _ = trt_run
    urls = {j.engine: j.metrics_url for j in sim.jobs if isinstance(j, LoadJob)}
    assert urls["trtllm"].endswith("/prometheus/metrics")
    assert urls["vllm"].endswith(":8000/metrics")


def test_both_trtllm_cells_get_the_full_gate_with_a_measured_divergence(trt_run):
    outcome, _, ctx, _, _ = trt_run
    events = _events(outcome, ctx)
    quality = {e["cell"]: e for e in events if e["kind"] == "quality"}
    assert set(quality) == {"vllm-bf16", "trtllm-bf16", "trtllm-fp8"}
    assert not [e for e in events if e["kind"] in ("quality_failed", "divergence_failed")]
    gates = {g.cell: g for g in outcome.gates}
    assert set(gates) == {"trtllm-bf16", "trtllm-fp8"}
    assert {g.baseline for g in gates.values()} == {"vllm-bf16"}
    gate_events = [e for e in events if e["kind"] == "gate"]
    assert len(gate_events) == 2
    for e in gate_events:
        assert "not measured" not in json.dumps(e)
        assert "KL" in json.dumps(e)


def test_without_the_wrapper_the_divergence_fails_visibly(tmp_path):
    # The same smoke with stock trtllm-serve (no wrapper): task scores stand, the
    # divergence half fails with trtllm-serve's own refusal, and the gates are blocked.
    import loom_bench.engines as engines

    stock = ("bash", "-c", engines.TRTLLM_LAUNCHER, "trtllm-launch", "stock")
    mp = pytest.MonkeyPatch()
    mp.setitem(engines.ENTRYPOINTS, "trtllm", stock)
    try:
        outcome, _, ctx, _, _ = two.run_on_pod_sim(
            SMOKE, tmp_path, replicates=False, pod_sim=TrtllmPodSim
        )
    finally:
        mp.undo()
    events = _events(outcome, ctx)
    failed = [e for e in events if e["kind"] == "divergence_failed"]
    assert sorted(e["cell"] for e in failed) == ["trtllm-bf16", "trtllm-fp8"]
    assert all("logprobs is not supported" in json.dumps(e) for e in failed)
    assert all(str(g.decision) == "inconclusive" and g.blocked for g in outcome.gates)
