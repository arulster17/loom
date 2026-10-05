import datetime as dt
import shutil

import pytest
from pydantic import ValidationError

from loom_bench.registry import REPO_ROOT
from loom_bench.site import WaitlistConfig, load_site_config, record_count, waitlist_count
from loom_bench.store.db import session_scope
from loom_bench.store.repo import add_waitlist_signup

DAY = dt.date(2026, 10, 4)


@pytest.fixture
def docs(tmp_path):
    path = tmp_path / "waitlist.md"
    shutil.copy(REPO_ROOT / "docs" / "waitlist.md", path)
    return path


def _rows(path):
    lines = path.read_text().splitlines()
    start = lines.index("| Date | Count | Source |") + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def test_waitlist_count(empty_db):
    with session_scope(empty_db) as s:
        assert waitlist_count(s) == 0
        add_waitlist_signup(s, "a@example.com", source="results-page")
        add_waitlist_signup(s, "b@example.com")
        add_waitlist_signup(s, "A@Example.com")  # already signed up
        assert waitlist_count(s) == 2


def test_record_count_appends_rows(docs):
    before = docs.read_text()
    record_count(docs, 3, "waitlist_signups table", on=DAY)
    record_count(docs, 5, "Formspree dashboard", on=DAY + dt.timedelta(days=1))
    assert _rows(docs) == [
        "| 2026-10-04 | 3 | waitlist_signups table |",
        "| 2026-10-05 | 5 | Formspree dashboard |",
    ]
    assert docs.read_text().startswith(before.rstrip("\n"))


def test_record_count_replaces_same_day_and_source(docs):
    record_count(docs, 3, "waitlist_signups table", on=DAY)
    record_count(docs, 4, "Formspree dashboard", on=DAY)
    record_count(docs, 7, "waitlist_signups table", on=DAY)
    assert _rows(docs) == [
        "| 2026-10-04 | 7 | waitlist_signups table |",
        "| 2026-10-04 | 4 | Formspree dashboard |",
    ]


def test_record_count_keeps_text_after_the_table(docs):
    docs.write_text(docs.read_text() + "\nNotes below the table.\n")
    record_count(docs, 1, "manual", on=DAY)
    text = docs.read_text()
    assert "| 2026-10-04 | 1 | manual |\n\nNotes below the table.\n" in text


@pytest.mark.parametrize(
    ("count", "source"), [(-1, "x"), (True, "x"), (1.0, "x"), (1, ""), (1, "a|b"), (1, "a\nb")]
)
def test_record_count_rejects_bad_input(docs, count, source):
    with pytest.raises(ValueError):
        record_count(docs, count, source, on=DAY)


def test_record_count_needs_the_table(tmp_path):
    path = tmp_path / "other.md"
    path.write_text("# No table here\n")
    with pytest.raises(ValueError, match="no table"):
        record_count(path, 1, "manual", on=DAY)


def test_repo_waitlist_doc_has_an_empty_table():
    path = REPO_ROOT / "docs" / "waitlist.md"
    assert "No waitlist backend is live yet" in path.read_text()
    assert _rows(path) == []


def test_repo_site_config_has_no_endpoint():
    cfg = load_site_config()
    assert cfg.waitlist.action_url is None and not cfg.waitlist.enabled
    assert cfg.waitlist.method == "POST" and cfg.waitlist.field == "email"
    assert cfg.repo == "https://github.com/arulster17/loom"


def test_waitlist_config_validation():
    assert WaitlistConfig(method="post").method == "POST"
    for bad in ({"method": "PUT"}, {"field": "e mail"}, {"action_url": "not a url"}, {"x": 1}):
        with pytest.raises(ValidationError):
            WaitlistConfig.model_validate(bad)
