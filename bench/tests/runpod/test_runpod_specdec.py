"""The speculative decoding smoke (and so the run), end to end on the runpod provider at test scale
(one tiny workload, a small suite, one repetition) against FakeRunpod and the pod
simulator. The simulator's engine speculates when the launch carries
`--speculative-config`, as vLLM would, so the paths no real run has taken are covered for
free first: warm restarts onto n-gram (no draft weights) and EAGLE3 (its head downloaded
on the restart), vLLM's spec-decode counters scraped into each run's summary (and their
absence on the BF16 row), both speculative rows scored on BF16's divergence reference and
gated against it, and the leaderboard's speculative-decoding table."""

import json
from typing import Any

import pytest
from sqlalchemy import select

from loom_bench.cost import CostAllocation
from loom_bench.experiment import EXPERIMENTS_DIR
from loom_bench.jobs import EvalJob, LoadJob
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.draft_misses import analyze_outputs
from loom_bench.report.leaderboard import build_leaderboard, render_markdown
from loom_bench.slo import Slo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchRun
from loom_bench.store.parquet import read_requests
from loom_bench.tokenize import SimpleTokenizer

from . import test_runpod_two_checkpoints as two
from .fakes import script_array, script_pairs, script_var

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("runpod-e2e")]

SMOKE = EXPERIMENTS_DIR / "runpod-smoke-specdec.yaml"
EAGLE3 = ("RedHatAI/Qwen3-8B-speculator.eagle3", "08610ffa01dd9f16731fe8f627b85905b6aa51c4")


class SpeculatingPodSim(two.RecordingPodSim):
    """The pod's mock engine drafts `num_speculative_tokens` per step when the rendered
    launch has a speculative config, accepting fewer drafts for n-gram than EAGLE3."""

    async def start(self, target: Any, script: str) -> Any:
        cmd = script_array(script, "ENGINE_CMD")
        self.mock = {}
        if "--speculative-config" in cmd:
            cfg = json.loads(cmd[cmd.index("--speculative-config") + 1])
            ngram = cfg["method"] == "ngram"
            self.mock = {
                "spec_tokens": cfg["num_speculative_tokens"],
                "spec_acceptance": 0.4 if ngram else 0.7,
                "spec_draft_rate": 0.5 if ngram else 1.0,
            }
        return await super().start(target, script)


@pytest.fixture(scope="module")
def spec_run(tmp_path_factory):
    # At test scale the smoke and the run are the same experiment (their provider,
    # variants and quality section are equal: bench/tests/runner/test_specdec_8b.py), so
    # one of them runs here.
    path = SMOKE
    return two.run_on_pod_sim(
        path,
        tmp_path_factory.mktemp(path.stem),
        replicates=True,
        pod_sim=SpeculatingPodSim,
    )


def _runs(ctx) -> list[BenchRun]:
    with session_scope(ctx.db_url) as s:
        return list(s.scalars(select(BenchRun)))


def test_three_rows_on_one_pod_with_the_eagle3_head_fetched_on_its_restart(spec_run):
    outcome, sim, _, cells, _ = spec_run
    assert outcome.status.value == "completed", outcome.reason
    assert [c.variant for c in cells] == ["bf16", "bf16-ngram", "bf16-eagle3"]
    assert len(sim.downloads) == 1  # one pod
    cold, ngram, eagle = sim.scripts("start_engine")
    bf16 = (cells[0].spec.hf.repo, cells[0].spec.hf.revision)
    assert script_pairs(cold, "FETCH_WEIGHTS") == [bf16]
    assert script_var(ngram, "WARM") == "1" and script_pairs(ngram, "FETCH_WEIGHTS") == []
    assert script_var(eagle, "WARM") == "1" and script_pairs(eagle, "FETCH_WEIGHTS") == [EAGLE3]
    (downloads,) = sim.downloads.values()
    assert downloads == [bf16, EAGLE3]


def test_spec_counters_land_in_every_speculating_run_and_nowhere_else(spec_run):
    _, _, ctx, cells, _ = spec_run
    by_hash = {c.config_hash: c.variant for c in cells}
    runs = _runs(ctx)
    assert runs and {r.status for r in runs} == {"completed"}
    for run in runs:
        server = run.summary["server"]
        variant = by_hash[run.config_hash]
        assert "spec_decode" not in server["missing"], variant
        if variant == "bf16":
            assert server["spec_drafts"] is None
            assert server["spec_mean_accepted_length"] is None
        else:
            assert server["spec_drafts"] > 0, variant
            assert 1 <= server["spec_mean_accepted_length"] <= 4
            assert set(server["spec_acceptance_by_position"]) == {"0", "1", "2"}
        assert run.summary["tokens_per_chunk"]["n"] > 0


def test_both_speculative_rows_are_scored_on_bf16s_reference_and_gated(spec_run):
    outcome, sim, _, (bf16, ngram, eagle), _ = spec_run
    evals = [j for j in sim.jobs if isinstance(j, EvalJob)]
    assert len(evals) == 3 * 3  # three replicated passes per row
    # Divergence runs on each row's first pass only.
    first = [j for j in evals if j.divergence is not None]
    assert [j.divergence for j in first] == ["capture_and_floor", "score", "score"]
    for job in first[1:]:
        assert job.reference is not None and job.reference.config_hash == bf16.config_hash
    assert {(g.cell, g.baseline) for g in outcome.gates} == {
        (ngram.key, bf16.key),
        (eagle.key, bf16.key),
    }


def test_the_leaderboard_shows_acceptance_and_speed_up_by_load(spec_run):
    _, _, ctx, _, _ = spec_run
    results = analyze_runs(
        _runs(ctx),
        slo=Slo(max_error_rate=0.01),
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(ctx.prices),
    )
    report = build_leaderboard(results)
    (board,) = report.boards
    t = board.specdec
    assert t is not None and t.reference == "bf16"
    assert set(t.crossover) == {"bf16-ngram", "bf16-eagle3"}
    rows = {r.config: r for r in t.rows}
    assert rows["bf16"].accepted_length is None
    assert rows["bf16-eagle3"].accepted_length is not None
    # EAGLE3 accepts more than n-gram in this simulation, by construction.
    assert rows["bf16-eagle3"].accepted_length.mean > rows["bf16-ngram"].accepted_length.mean
    assert "#### Speculative decoding by load" in render_markdown(report)


def test_only_the_eagle3_rows_keep_output_text_and_draft_misses_reads_it(spec_run):
    _, sim, ctx, cells, _ = spec_run
    by_hash = {c.config_hash: c.variant for c in cells}
    # The job asks for text only on the EAGLE3 row's load runs.
    load_jobs = [j for j in sim.jobs if isinstance(j, LoadJob)]
    assert {j.keep_output for j in load_jobs} == {True, False}
    stored: dict[str, list] = {}
    for run in _runs(ctx):
        recs = [r for r in read_requests(str(run.requests_uri)) if r.ok and not r.warmup]
        stored.setdefault(by_hash[run.config_hash], []).extend((str(run.id), r) for r in recs)
    assert all(r.output_text for _, r in stored["bf16-eagle3"])
    assert not any(r.output_text for v in ("bf16", "bf16-ngram") for _, r in stored[v])
    # The analysis on the mock's text: a toy vocabulary of the pieces the simple
    # tokenizer splits; the "draft" knows every other one.
    tok = SimpleTokenizer()
    vocab: dict[str, int] = {}

    def encode(text: str) -> list[int]:
        return [vocab.setdefault(p, len(vocab)) for p in tok.pieces(text)]

    for _, r in stored["bf16-eagle3"]:
        encode(r.output_text or "")
    draft = frozenset(i for i in vocab.values() if i % 2 == 0)
    names = {i: p for p, i in vocab.items()}
    report = analyze_outputs(
        stored["bf16-eagle3"], encode=encode, decode_token=names.__getitem__, draft_vocab=draft
    )
    assert report.low.n + report.rest.n == len(stored["bf16-eagle3"])
    assert report.skipped_no_text == 0
    assert all(0 <= r.outside_share <= 1 and r.tokens > 0 for r in report.requests)
    assert report.finding()
    # Without stored text there is nothing to analyse.
    empty = analyze_outputs(
        stored["bf16"], encode=encode, decode_token=names.__getitem__, draft_vocab=draft
    )
    assert empty.requests == [] and empty.skipped_no_text == len(stored["bf16"])


def test_bench_draft_misses_reads_the_runs_head_and_writes_its_report(
    spec_run, tmp_path, monkeypatch
):
    import huggingface_hub
    from tokenizers import Tokenizer, models, pre_tokenizers
    from typer.testing import CliRunner

    import loom_bench.report.draft_misses as dm
    from loom_bench.cli import app

    outcome, _, ctx, _, _ = spec_run
    words = sorted(
        {
            w
            for run in _runs(ctx)
            for r in read_requests(str(run.requests_uri))
            for w in (r.output_text or "").split()
        }
    )
    assert words  # the EAGLE3 row stored text
    tok = Tokenizer(
        models.WordLevel({"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}, "[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok_file = tmp_path / "tokenizer.json"
    tok.save(str(tok_file))
    asked: list[tuple] = []

    def fake_download(repo, filename, revision=None, **_):
        asked.append(("tokenizer", repo, revision))
        return str(tok_file)

    def fake_vocab(repo, revision, **_):
        asked.append(("draft", repo, revision))
        return frozenset(range(0, len(words) + 1, 2))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(dm, "draft_vocab_from_hub", fake_vocab)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "draft-misses",
            str(outcome.experiment_id),
            "--cell",
            "bf16-eagle3",
            "--db",
            ctx.db_url,
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert ("draft", *EAGLE3) in asked
    assert ("tokenizer", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218") in asked
    report = json.loads((out / "draft_misses.json").read_text())
    assert report["requests"] and report["skipped_no_text"] == 0
    assert (
        (out / "draft_misses.md")
        .read_text()
        .startswith(("No request had", f"{report['low']['n']} of"))
    )
    # A cell without a draft model is refused.
    plain = CliRunner().invoke(
        app,
        ["draft-misses", str(outcome.experiment_id), "--cell", "bf16-ngram", "--db", ctx.db_url],
    )
    assert plain.exit_code == 2 and "no draft model" in plain.output
