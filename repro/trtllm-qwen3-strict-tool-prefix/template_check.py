#!/usr/bin/env python3
"""Show the mismatch without a GPU: the chat template's tool call vs the parser's prefix.

Run inside the TensorRT-LLM image (it has transformers and tensorrt_llm):

    python3 template_check.py Qwen/Qwen3-8B
"""

import json
import sys

from transformers import AutoTokenizer

from tensorrt_llm.serve.tool_parser.qwen3_tool_parser import Qwen3ToolParser

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(model)
messages = [
    {"role": "user", "content": "Turn the house temperature down to 19 celsius."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "set_thermostat",
                         "arguments": {"temperature": 19, "unit": "celsius"}},
        }],
    },
]
rendered = tok.apply_chat_template(messages, tokenize=False)
start = rendered.index("<tool_call>")
template_call = rendered[start:rendered.index("</tool_call>", start) + len("</tool_call>")]
forced = Qwen3ToolParser().structure_info()("set_thermostat").begin

print("chat template writes:   ", json.dumps(template_call))
print("strict tag forces begin:", json.dumps(forced))
prefix_t = template_call[: len('<tool_call>\n{"name": "')]
prefix_f = forced[: len('<tool_call>\n{"name":"')]
print("template prefix:", json.dumps(prefix_t), tok.tokenize(prefix_t))
print("forced prefix:  ", json.dumps(prefix_f), tok.tokenize(prefix_f))
