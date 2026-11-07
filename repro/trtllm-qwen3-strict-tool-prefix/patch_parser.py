#!/usr/bin/env python3
"""Apply the one-character fix to the installed Qwen3 tool parser (run inside the image).

Changes the strict-tools structural tag's forced prefix from `{"name":"` to `{"name": "`,
which is what Qwen3's chat template writes. Fails loudly if the line is not found once.
"""

import importlib.util
import sys
from pathlib import Path

OLD = """begin='<tool_call>\\n{"name":"' + name"""
NEW = """begin='<tool_call>\\n{"name": "' + name"""

spec = importlib.util.find_spec("tensorrt_llm.serve.tool_parser.qwen3_tool_parser")
if spec is None or spec.origin is None:
    sys.exit("qwen3_tool_parser not found")
path = Path(spec.origin)
src = path.read_text()
if src.count(OLD) != 1:
    sys.exit(f"{path}: expected exactly one {OLD!r}, found {src.count(OLD)}")
path.write_text(src.replace(OLD, NEW))
print(f"patched {path}: {OLD!r} -> {NEW!r}")
