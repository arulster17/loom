"""CommitPackFT-format code edits: one JSON object per line with `old_file`,
`old_contents`, `new_contents`, `message` (or `subject`) and `license`
(bigcode/commitpackft, `data/<language>/data.jsonl`).

No dataset ships with Loom; an `edit_dataset` profile pins the file it reads.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from loom_bench.loadgen.arrivals import data_path

# The instruction around each edit. Part of what a run measures: changing it changes the
# requests, so do it only with a new profile.
EDIT_PROMPT = (
    "Apply this change to the file `{file}`: {message}\n\n"
    "Reply with the complete updated file and nothing else.\n\n"
    "```\n{old}\n```"
)


@dataclass(frozen=True, slots=True)
class Edit:
    file: str
    message: str
    old: str
    new: str
    license: str

    @property
    def prompt(self) -> str:
        return EDIT_PROMPT.format(file=self.file, message=self.message.strip(), old=self.old)


def read_edits(path: str | Path, licenses: Collection[str] | None = None) -> list[Edit]:
    """Edits that change the file, in file order; with `licenses`, only those whose
    repository license is listed."""
    p = data_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"edit dataset not found at {p} (set the profile's `path`)")
    allowed = set(licenses) if licenses is not None else None
    out = []
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            old, new = row.get("old_contents"), row.get("new_contents")
            message = row.get("message") or row.get("subject")
            lic = str(row.get("license") or "unknown")
            if not (isinstance(old, str) and isinstance(new, str) and isinstance(message, str)):
                continue
            if old == new or not old.strip() or not message.strip():
                continue
            if allowed is not None and lic not in allowed:
                continue
            out.append(Edit(str(row.get("old_file") or "file"), message, old, new, lic))
    return out
