"""Write items.json: the 45 "multiple" items of Loom's tool_calling set, self-contained.

Loom-only helper (upstream gets items.json). Run from the repo root:

    uv run python repro/trtllm-qwen3-strict-tool-prefix/make_items.py

Each item carries its functions inline as plain (non-strict) OpenAI tool definitions and
the function it should call; repro.py adds strict mode itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from loom_bench.quality.tasks.tool_calling import load_data

HERE = Path(__file__).parent


def main() -> None:
    data = load_data()
    items = [
        {
            "id": item.item_id,
            "query": item.query,
            "expected_function": item.expected_name,
            "tools": data.tools_for(item, strict=False),
        }
        for item in data.items
        if item.category == "multiple"
    ]
    out = {
        "source": f"Loom tool_calling data version {data.version} (sha256 {data.sha256}), "
        "category 'multiple'",
        "items": items,
    }
    (HERE / "items.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items")


if __name__ == "__main__":
    main()
