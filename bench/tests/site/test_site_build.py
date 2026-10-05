import json
import re
import shutil
from pathlib import Path

import pytest
import yaml
from site_helpers import REPO_URL, export, is_external, parse, resolve

from loom_bench.registry import REPO_ROOT, load_registry
from loom_bench.site import SiteConfig, WaitlistConfig, build_site, load_snapshot
from loom_bench.site.build import stylesheet

MODEL_PAGES = [f"models/{m.id}.html" for m in load_registry().models]  # one per model
PAGES = {"index.html", "methodology.html", "pricing.html", "harness.html", *MODEL_PAGES}
PENDING = "No published results yet — first runs pending."
WAITLIST_ON = SiteConfig(
    waitlist=WaitlistConfig(
        action_url="https://forms.example.org/f/abc",
        extra_fields={"source": "results-page"},
    )
)


@pytest.fixture
def empty_site(empty_db, tmp_path) -> Path:
    export(empty_db, tmp_path / "data")
    build_site(tmp_path / "data", tmp_path / "out", SiteConfig())
    return tmp_path / "out"


@pytest.fixture(scope="module")
def full_site(populated, tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("full")
    export(populated.url, root / "data")
    build_site(root / "data", root / "out", SiteConfig())
    return root / "out"


def _html(site: Path) -> dict[str, Path]:
    return {p.relative_to(site).as_posix(): p for p in site.rglob("*.html")}


def _text(site: Path, page: str) -> str:
    return (site / page).read_text(encoding="utf-8")


def _allowed_external(data: Path) -> set[str]:
    """Links the site may make: the repo, model weights, and sources named in the data."""
    found = set()
    for path in data.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        found.update(re.findall(r'"(https?://[^"]+)"', text))
    return found


def _check_site(site: Path) -> None:
    pages = _html(site)
    assert set(pages) >= PAGES
    assert (site / "style.css").read_text(encoding="utf-8") == stylesheet()
    allowed = _allowed_external(site / "data")
    parsed = {name: parse(path) for name, path in pages.items()}
    for name, page in parsed.items():
        assert page.errors == [], (name, page.errors)
        assert page.assets == [("link", "data:,"), ("link", "../" * name.count("/") + "style.css")]
        for href in page.links:
            if is_external(href):
                assert href.startswith((REPO_URL, "https://huggingface.co/")) or href in allowed, (
                    name,
                    href,
                )
                continue
            target, fragment = resolve(pages[name], href)
            assert target.is_file(), (name, href)
            if fragment and target.suffix == ".html":
                rel = target.relative_to(site.resolve()).as_posix()
                assert fragment in parsed[rel].ids, (name, href)
    css = (site / "style.css").read_text(encoding="utf-8")
    assert "url(" not in css and "@import" not in css


def test_empty_snapshot_renders_honest_empty_state(empty_site):
    _check_site(empty_site)
    index = _text(empty_site, "index.html")
    assert PENDING in index
    assert "not measured yet" in index
    assert "Qwen3 8B" in index and "Llama 3.3 70B Instruct" in index
    model = _text(empty_site, "models/qwen3-8b.html")
    assert PENDING in model
    assert "Configuration to be benchmarked" in model
    assert "b968826d9c46dd6066d109eabc6255188de91218" in model
    methodology = _text(empty_site, "methodology.html")
    assert "none yet: first runs pending" in methodology
    assert "Illustrative" in methodology and "not a measurement" in methodology
    pricing = _text(empty_site, "pricing.html")
    assert "Public list prices only." in pricing
    assert "Not benchmarked yet" in pricing
    assert "https://www.together.ai/pricing" in pricing
    for page in ("index.html", *MODEL_PAGES):
        assert re.search(r"\$\d", _text(empty_site, page)) is None, page  # no money figures


def test_populated_snapshot_renders_results(full_site, populated):
    _check_site(full_site)
    snap = load_snapshot(full_site / "data")
    qwen = next(m for m in snap.models if m.model_id == "qwen3-8b")
    best = next(r for r in qwen.results if r.name == "sglang-bf16")
    awq = next(r for r in qwen.results if r.name == "vllm-awq")

    index = _text(full_site, "index.html")
    assert PENDING not in index
    assert "sglang-bf16" in index
    assert "$0.8717" in index  # cheapest ranked config's $/1M out at SLO
    assert "No published results for this model yet" in index  # Llama

    model = _text(full_site, "models/qwen3-8b.html")
    assert f"bench reproduce {best.reproduce_run_id()}" in model
    assert "cell: sglang-bf16" in model
    for run_id in best.run_ids + awq.run_ids:
        assert f"../data/provenance/{run_id}.json" in model
    assert "quality gate failed" in model
    assert "100 s (median of 1)" in model
    assert "$0.8503 – $0.8936" in model  # on-demand cost CI, storage included
    assert "$/1M out at SLO, spot" in model and "$/1M out at SLO, as run" in model
    assert "committed 1y</th>" not in model  # no committed price in the price book
    assert "Price as run" in model and "observed at launch" in model
    assert "$/1M in at SLO" not in model  # all_output: input has no separate price
    assert "$0.0000" not in model

    methodology = _text(full_site, "methodology.html")
    assert "From the published results" in methodology
    assert "On-demand (ranked)" in methodology and "As run" in methodology
    assert "so on-demand is the default" not in methodology
    assert "Illustrative" not in methodology
    assert "TTFT p95 ≤ 600 ms" in methodology
    assert "data/experiments/" in methodology


def test_provenance_files_are_published(full_site):
    files = list((full_site / "data" / "provenance").glob("*.json"))
    assert len(files) == 36
    assert json.loads(files[0].read_text())["provenance"]["config_hash"]


def test_manifest_only_snapshot_builds(empty_db, tmp_path):
    export(empty_db, tmp_path / "full")
    data = tmp_path / "bare"
    data.mkdir()
    manifest = json.loads((tmp_path / "full" / "manifest.json").read_text())
    manifest["models"] = []
    (data / "manifest.json").write_text(json.dumps(manifest))
    pages = build_site(data, tmp_path / "out", SiteConfig())
    assert {p.relative_to(tmp_path / "out").as_posix() for p in pages} == {
        "index.html",
        "methodology.html",
        "pricing.html",
        "harness.html",
    }
    assert PENDING in _text(tmp_path / "out", "index.html")
    assert "no pricing data" in _text(tmp_path / "out", "pricing.html")


def test_committed_snapshot_builds(tmp_path):
    data = REPO_ROOT / "site" / "data"
    snap = load_snapshot(data)
    assert snap.manifest.experiment_ids == []
    pages = build_site(data, tmp_path / "out", REPO_ROOT / "site" / "config.yaml")
    assert len(pages) == len(PAGES)
    assert PENDING in _text(tmp_path / "out", "index.html")


def test_waitlist_without_endpoint_renders_no_form(empty_site):
    index = _text(empty_site, "index.html")
    assert "Waitlist opens soon" in index
    assert "<form" not in index and "<script" not in index
    assert f'<a href="{REPO_URL}">' in index


def test_waitlist_form_when_endpoint_set(empty_db, tmp_path):
    export(empty_db, tmp_path / "data")
    build_site(tmp_path / "data", tmp_path / "out", WAITLIST_ON)
    index = _text(tmp_path / "out", "index.html")
    assert '<form id="waitlist-form" action="https://forms.example.org/f/abc" method="post">' in (
        index
    )
    assert re.search(r'<input id="waitlist-email" name="email" type="email" required', index)
    assert 'name="_gotcha" type="text" tabindex="-1"' in index
    assert '<div class="hp" aria-hidden="true">' in index
    assert '<input type="hidden" name="source" value="results-page">' in index
    assert "We only use this to tell you when Loom opens; no sharing." in index
    assert "Waitlist opens soon" not in index
    script = re.search(r"<script>(.*?)</script>", index, re.S)
    assert script is not None
    assert len(script.group(1).strip().splitlines()) <= 25
    assert parse(tmp_path / "out" / "index.html").errors == []


def test_site_config_from_yaml(tmp_path):
    repo_cfg = yaml.safe_load((REPO_ROOT / "site" / "config.yaml").read_text())
    assert repo_cfg["waitlist"]["action_url"] is None
    path = tmp_path / "config.yaml"
    path.write_text(
        "waitlist: {action_url: 'https://buttondown.example/api', method: get, field: addr}\n"
    )
    build_site(REPO_ROOT / "site" / "data", tmp_path / "out", path)
    index = _text(tmp_path / "out", "index.html")
    assert 'method="get"' in index and 'name="addr" type="email"' in index


def test_build_refuses_to_clobber_unrelated_directory(empty_db, tmp_path):
    export(empty_db, tmp_path / "data")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="not a previous site build"):
        build_site(tmp_path / "data", tmp_path / "out", SiteConfig())
    with pytest.raises(ValueError, match="overwrite the snapshot"):
        build_site(tmp_path / "data", tmp_path, SiteConfig())


def test_rebuild_replaces_previous_build(empty_db, tmp_path):
    export(empty_db, tmp_path / "data")
    build_site(tmp_path / "data", tmp_path / "out", SiteConfig())
    (tmp_path / "out" / "models" / "stale.html").write_text("old")
    build_site(tmp_path / "data", tmp_path / "out", SiteConfig())
    assert not (tmp_path / "out" / "models" / "stale.html").exists()
    shutil.rmtree(tmp_path / "out")
