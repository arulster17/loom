"""Shared by the site tests: store fixture helpers and HTML checks."""

import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from loom_bench.provenance import GitInfo
from loom_bench.site import export_snapshot
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.repo import record_run

TESTS_DIR = str(Path(__file__).resolve().parents[1])
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from report.factories import SLO, TTFT_SGLANG, make_runs  # noqa: E402

__all__ = ["SLO", "TTFT_SGLANG", "make_runs"]

EXPORT_GIT = GitInfo(sha="fedcba9876543210fedcba9876543210fedcba98", dirty=False, branch="main")
GENERATED_AT = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
SPEC = {"name": "qwen3-8b-chat", "slo": SLO.model_dump(mode="json")}
REPO_URL = "https://github.com/arulster17/loom"


@dataclass(frozen=True)
class Store:
    url: str
    experiment_id: uuid.UUID
    hashes: dict[str, str]  # cell key -> config hash


def store_runs(session, experiment_id, runs) -> dict[str, str]:
    hashes = {}
    for run in runs:
        hashes[run.cell_key] = run.config_hash
        record_run(
            session,
            experiment_id=experiment_id,
            config_hash=run.config_hash,
            provenance=run.provenance,
            status=run.status,
            summary=run.summary,
            cell_key=run.cell_key,
            workload=run.workload,
            load_mode=run.load_mode,
            load_value=run.load_value,
            repetition=run.repetition,
        )
    return hashes


def new_db(path: Path) -> str:
    url = f"sqlite:///{path}"
    upgrade(url)
    return url


def export(url: str, out: Path, experiment_ids="latest", **kwargs):
    with session_scope(url) as s:
        return export_snapshot(
            s, out, experiment_ids, git=EXPORT_GIT, generated_at=GENERATED_AT, **kwargs
        )


VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())


@dataclass
class Page(HTMLParser):
    """Parses one page; records links, ids, assets and tag-nesting errors."""

    links: list[str] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    assets: list[tuple[str, str]] = field(default_factory=list)  # (tag, url)
    errors: list[str] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        HTMLParser.__init__(self, convert_charrefs=True)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            if a["id"] in self.ids:
                self.errors.append(f"duplicate id {a['id']}")
            self.ids.add(a["id"])
        if tag == "a" and "href" in a:
            self.links.append(a["href"])
        if tag in ("link", "script", "img", "iframe", "source", "video", "audio"):
            url = a.get("href") or a.get("src")
            if url:
                self.assets.append((tag, url))
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            self.errors.append(f"end tag for void element {tag}")
        elif not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1] if self.stack else 'nothing'}")
        else:
            self.stack.pop()


def parse(path: Path) -> Page:
    page = Page()
    page.feed(path.read_text(encoding="utf-8"))
    page.close()
    if page.stack:
        page.errors.append(f"unclosed: {page.stack}")
    return page


def is_external(url: str) -> bool:
    return urlsplit(url).scheme != ""


def resolve(page_path: Path, href: str) -> tuple[Path, str]:
    parts = urlsplit(href)
    target = (page_path.parent / parts.path).resolve() if parts.path else page_path.resolve()
    return target, parts.fragment
