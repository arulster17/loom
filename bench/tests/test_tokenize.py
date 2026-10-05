import random
from pathlib import Path

import huggingface_hub
import pytest
from tokenizers import Tokenizer as _Tok
from tokenizers import models, pre_tokenizers

from loom_bench.jobs import LoadJob, TokenizerSpec
from loom_bench.loadgen.base import prepare_load
from loom_bench.records import LoadMode
from loom_bench.tokenize import (
    HFTokenizer,
    SimpleTokenizer,
    SnapshotMismatch,
    Tokenizer,
    hf_cache_folder,
    verify_snapshot,
)


def test_simple_tokenizer_round_trips_and_counts():
    tok = SimpleTokenizer()
    text = "What is 37 + 48?\n  Answer:  85."
    assert "".join(tok.pieces(text)) == text
    assert tok.count("hello world") == 2
    assert tok.count("123") == 3  # digits split individually


def test_random_text_exact_length_and_deterministic():
    tok = SimpleTokenizer()
    for n in (0, 1, 7, 1000):
        assert tok.count(tok.random_text(n, random.Random(1))) == n
    assert tok.random_text(50, random.Random(3)) == tok.random_text(50, random.Random(3))


def test_truncate():
    tok = SimpleTokenizer()
    assert tok.truncate("a b c d", 2) == "a b"
    assert tok.truncate("a b", 10) == "a b"


def test_protocol():
    assert isinstance(SimpleTokenizer(), Tokenizer)


REPO = "meta-llama/Llama-3.3-70B-Instruct"
REV = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
WORDS = ("hello", "world", "model", "cache", "token")


def snapshot(root: Path, repo: str = REPO, revision: str = REV) -> Path:
    """A Hugging Face cache snapshot holding a tiny word-level tokenizer.json."""
    path = root / hf_cache_folder(repo) / "snapshots" / revision
    path.mkdir(parents=True)
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(WORDS)}}
    tok = _Tok(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path / "tokenizer.json"))
    return path


@pytest.fixture
def no_hub(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("tried the Hugging Face Hub")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)


def test_hf_tokenizer_loads_a_local_snapshot_without_the_hub(tmp_path, no_hub):
    tok = HFTokenizer(REPO, REV, local_dir=str(snapshot(tmp_path)))
    assert tok.name == f"{REPO}@{REV}"
    assert tok.count("hello world model") == 3
    assert tok.truncate("hello world model", 2) == "hello world"
    assert tok.count(tok.random_text(7, random.Random(1))) == 7


@pytest.mark.parametrize(("repo", "revision"), [(REPO, "0" * 40), ("Qwen/Qwen3-8B", REV)])
def test_local_snapshot_must_be_the_pinned_revision(tmp_path, no_hub, repo, revision):
    local = snapshot(tmp_path)
    with pytest.raises(SnapshotMismatch, match="is not the snapshot of"):
        HFTokenizer(repo, revision, local_dir=str(local))


def test_missing_local_snapshot_fails(tmp_path):
    absent = tmp_path / hf_cache_folder(REPO) / "snapshots" / REV
    with pytest.raises(SnapshotMismatch, match="not found"):
        verify_snapshot(absent, REPO, REV)


def test_prepare_load_uses_the_local_snapshot(tmp_path, no_hub):
    spec = TokenizerSpec(kind="hf", repo=REPO, revision=REV, local_dir=str(snapshot(tmp_path)))
    job = LoadJob(
        run_id="r1",
        base_url="http://127.0.0.1:8000/v1",
        engine="vllm",
        served_model="m",
        workload={
            "name": "w",
            "description": "d",
            "content": "synthetic",
            "kind": "synthetic",
            "endpoint": "completions",
            "input_len": 6,
            "output_len": 2,
        },
        tokenizer=spec,
        mode=LoadMode.CLOSED_LOOP,
        load_value=1,
        num_requests=2,
    )
    ctx = prepare_load(job)
    assert ctx.tokenizer.name == f"{REPO}@{REV}"
    assert len(ctx.requests) == 2


def test_local_dir_needs_a_pinned_hf_repo(tmp_path):
    with pytest.raises(ValueError, match="local_dir needs"):
        TokenizerSpec(kind="hf", repo=REPO, local_dir=str(tmp_path))
    with pytest.raises(ValueError, match="local_dir needs"):
        TokenizerSpec(kind="simple", local_dir=str(tmp_path))
