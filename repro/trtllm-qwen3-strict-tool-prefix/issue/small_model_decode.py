"""Optional: greedy-decode a small Qwen3 on CPU with no grammar, with TRT-LLM's grammar, and
with the spaced grammar, on multi-tool strict items. This uses xgrammar's HF logits
processor and no TensorRT-LLM.

    python small_model_decode.py multi_items.json [Qwen/Qwen3-0.6B]

Not the benchmarked model (Qwen3-8B): a smaller model's tool choice can differ.
"""

import collections
import json
import sys

import torch
import xgrammar as xgr
from transformers import AutoModelForCausalLM, AutoTokenizer

from trtllm_qwen3_strict_prefix_repro import (
    build_strict_structural_tag,
    spaced_structure_info,
    trtllm_qwen3_structure_info,
)

REVISIONS = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca"}


def called_name(text: str) -> str | None:
    if "<tool_call>" not in text:
        return None
    body = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
    try:
        return json.loads(body)["name"]
    except (ValueError, KeyError, TypeError):
        return "<unparsable>"


def main() -> None:
    items = json.load(open(sys.argv[1]))
    model_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-0.6B"
    rev = REVISIONS.get(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.float32)
    model.eval()
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    info = xgr.TokenizerInfo.from_huggingface(
        tok, vocab_size=model.config.vocab_size, stop_token_ids=[stop]
    )
    compiler = xgr.GrammarCompiler(info)
    modes = {
        "none": None,
        "trtllm": trtllm_qwen3_structure_info,
        "spaced": spaced_structure_info,
    }
    tally = {m: collections.Counter() for m in modes}
    for item in items:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": item["query"]}],
            tools=item["tools"],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        row = []
        for mode, get_info in modes.items():
            procs = []
            if get_info is not None:
                grammar = compiler.compile_structural_tag(
                    build_strict_structural_tag(item["tools"], get_info)
                )
                procs = [xgr.contrib.hf.LogitsProcessor(grammar)]
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=96,
                    do_sample=False,
                    logits_processor=procs,
                    eos_token_id=stop,
                    pad_token_id=stop,
                )
            text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=False)
            name = called_name(text)
            verdict = "right" if name == item["expected"] else f"wrong:{name}"
            tally[mode][verdict == "right"] += 1
            row.append(f"{mode}={verdict}")
            print(f"  {item['item_id']} {mode}: {text[:160]!r}", file=sys.stderr, flush=True)
        print(item["item_id"], " ".join(row), flush=True)
    for mode, c in tally.items():
        print(f"{model_id} {mode:7} right function {c[True]}/{c[True] + c[False]}")


if __name__ == "__main__":
    main()
