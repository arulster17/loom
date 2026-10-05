"""ShareGPT-format conversations: `[{"conversations": [{"from": ..., "value": ...}]}]`.

No dataset ships with Loom; users download one, check its terms, and point a
`chat_dataset` profile at it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loom_bench.loadgen.arrivals import data_path

Message = dict[str, str]

_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
}


def read_conversations(path: str | Path) -> list[list[dict[str, Any]]]:
    """Raw `conversations` lists from a JSON array or JSONL file."""
    p = data_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ShareGPT dataset not found at {p} (set the profile's `path`)")
    with p.open() as f:
        if p.suffix == ".jsonl":
            items = [json.loads(line) for line in f if line.strip()]
        else:
            items = json.load(f)
    return [item["conversations"] for item in items if isinstance(item.get("conversations"), list)]


def to_chat(turns: list[dict[str, Any]]) -> tuple[list[Message], str] | None:
    """History up to the last assistant reply, and that reply as the reference.

    Returns None unless roles are an optional system turn followed by strictly
    alternating user/assistant turns (chat templates reject anything else).
    """
    msgs: list[Message] = []
    for turn in turns:
        role = _ROLES.get(str(turn.get("from", "")).lower())
        text = turn.get("value")
        if role is None or not isinstance(text, str) or not text.strip():
            return None
        msgs.append({"role": role, "content": text})
    body = msgs[1:] if msgs and msgs[0]["role"] == "system" else msgs
    for i, m in enumerate(body):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None
    if len(body) % 2:
        msgs.pop()  # trailing user turn has no reference reply
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return None
    return msgs[:-1], msgs[-1]["content"]
