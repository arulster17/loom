"""The roofline model: shapes from pinned configs, bytes, floors."""

import json
from pathlib import Path

import pytest

from loom_bench.profile.arch import ARCHITECTURES, BASE_REPO, architecture, model_bytes
from loom_bench.profile.cli import default_setups
from loom_bench.profile.hardware import LINKS, gpu
from loom_bench.profile.roofline import ServingSetup, decode_step, max_batch_at, prefill
from loom_bench.registry import load_registry

REGISTRY = load_registry()
BY_ID = {m.id: m for m in REGISTRY.models}
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@pytest.mark.parametrize("repo", sorted(ARCHITECTURES))
def test_architectures_match_the_pinned_config_json(repo: str) -> None:
    arch = ARCHITECTURES[repo]
    pinned = {m.hf.repo: m.hf.revision for m in REGISTRY.models}
    assert pinned[repo] == arch.revision, "copied from the revision config/models.yaml pins"
    cfg_path = (
        HF_CACHE
        / f"models--{repo.replace('/', '--')}"
        / "snapshots"
        / arch.revision
        / "config.json"
    )
    if not cfg_path.exists():
        pytest.skip(f"{repo}@{arch.revision} config.json not in the local HF cache")
    cfg = json.loads(cfg_path.read_text())
    assert cfg["num_hidden_layers"] == arch.layers
    assert cfg["hidden_size"] == arch.hidden
    assert cfg["num_attention_heads"] == arch.heads
    assert cfg["num_key_value_heads"] == arch.kv_heads
    assert cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]) == arch.head_dim
    assert cfg["intermediate_size"] == arch.intermediate
    assert cfg["vocab_size"] == arch.vocab
    assert cfg["tie_word_embeddings"] == arch.tie_word_embeddings


@pytest.mark.parametrize("model_id", [m.id for m in REGISTRY.models])
def test_byte_accounting_reproduces_every_checkpoint_size(model_id: str) -> None:
    """Parameter counts x precision land within 0.5% of each pinned checkpoint's size
    (FP8 ones carry per-channel scales on top), so the bytes per step are right."""
    spec = BY_ID[model_id]
    mb = model_bytes(spec)
    assert mb.checkpoint_bytes == pytest.approx(spec.hf.size_bytes, rel=0.005)


def test_every_registry_repo_has_an_architecture() -> None:
    for m in REGISTRY.models:
        assert architecture(m.hf.repo).repo == BASE_REPO.get(m.hf.repo, m.hf.repo)


def test_qwen3_8b_numbers() -> None:
    bf16 = model_bytes(BY_ID["qwen3-8b"])
    assert bf16.step_weight_bytes / 1e9 == pytest.approx(15.136, abs=0.01)  # embedding not read
    assert bf16.kv_bytes_per_token == 2 * 36 * 8 * 128 * 2  # 144 KiB
    fp8 = model_bytes(BY_ID["qwen3-8b-fp8"].model_copy(update={"kv_cache_dtype": "fp8"}))
    assert fp8.weights == "fp8" and fp8.kv == "fp8"
    assert fp8.kv_bytes_per_token == bf16.kv_bytes_per_token / 2
    # LM head stays BF16 in the FP8 checkpoint, so weights are a bit over half.
    assert fp8.step_weight_bytes / bf16.step_weight_bytes == pytest.approx(0.541, abs=0.002)


def test_decode_floor_is_weights_plus_kv_over_bandwidth() -> None:
    s = ServingSetup.from_spec(BY_ID["qwen3-8b"])
    one = decode_step(s, 1, 1024)
    assert one.weights_ms == pytest.approx(15.136e9 / 864e9 * 1000, rel=1e-3)  # ~17.5 ms
    assert one.bound == "memory" and one.comm_ms == 0
    big = decode_step(s, 64, 64 * 1536)
    assert big.kv_ms == pytest.approx(64 * 1536 * 147456 / 864e9 * 1000, rel=1e-6)
    assert big.total_ms == pytest.approx(big.weights_ms + big.kv_ms)
    # floors grow with batch and with context
    assert decode_step(s, 8, 8 * 1024).total_ms < decode_step(s, 16, 16 * 1024).total_ms
    assert decode_step(s, 8, 8 * 1024).total_ms < decode_step(s, 8, 8 * 4096).total_ms


def test_compute_bound_at_huge_batch() -> None:
    s = ServingSetup.from_spec(BY_ID["qwen3-8b"])
    assert decode_step(s, 4096, 4096 * 16).bound == "compute"


def test_tensor_parallel_splits_bytes_and_adds_the_link() -> None:
    setups = {s.label: s for s in default_setups(REGISTRY)}
    tp4 = setups["70B bf16 TP4 L40S"]
    assert tp4.tp == 4 and tp4.link is LINKS["pcie-host-p2p-off"]
    f = decode_step(tp4, 1, 1024)
    assert f.weights_ms == pytest.approx(139.0e9 / 4 / 864e9 * 1000, rel=0.01)  # ~40.2 ms
    # 160 all-reduces of one 16 KB token at the measured 27.6 us each
    assert f.comm_ms == pytest.approx(160 * 0.0276, rel=0.01)
    h100 = setups["70B bf16 TP2 H100"]
    assert h100.link is LINKS["nvlink4"] and h100.gpu is gpu("H100")


def test_measured_link_passes_through_its_two_points() -> None:
    link = LINKS["pcie-host-p2p-off"]
    assert link.allreduce_ms(16_384) == pytest.approx(0.0276, rel=1e-6)
    assert link.allreduce_ms(2048 * 8192 * 2) == pytest.approx(2.76, rel=1e-6)


def test_prefill_is_flop_bound_and_fp8_halves_it() -> None:
    bf16 = ServingSetup.from_spec(BY_ID["qwen3-8b"])
    fp8 = ServingSetup.from_spec(BY_ID["qwen3-8b-fp8"])
    p = prefill(bf16, 1024)
    assert p.compute_ms > p.weights_ms
    # 2 FLOPs per weight per token over the linear layers and LM head (7.57B weights; the
    # embedding table is a lookup), plus a little causal attention
    linear = 2 * 7.567e9 * 1024 / 362.05e12 * 1000
    assert p.compute_ms == pytest.approx(linear, rel=0.03)
    assert prefill(fp8, 1024).compute_ms < 0.6 * p.compute_ms


def test_max_batch_at_inverts_the_floor() -> None:
    s = ServingSetup.from_spec(BY_ID["qwen3-8b"])
    b = max_batch_at(s, 50.0, 1536)
    assert decode_step(s, b, b * 1536).total_ms == pytest.approx(50.0, abs=0.01)
    assert max_batch_at(s, 1.0, 1536) == 0.0


def test_unknown_gpu_is_refused() -> None:
    with pytest.raises(ValueError, match="no roofline figures"):
        gpu("A10G")
