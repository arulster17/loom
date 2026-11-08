#!/usr/bin/env python3
"""Count how often Qwen3 calls the right function, with plain vs strict tools.

Standard library only. Each item offers 3 or 4 functions and a request that one of them
answers; the model must call that one. Requests are greedy (temperature 0), non-thinking,
tool_choice "auto", one at a time.

    python3 repro.py --mode plain
    python3 repro.py --mode strict
    python3 repro.py --mode strict --show m-thermo-1

"strict" sends each tool with "strict": true and additionalProperties: false on every
object schema, which makes trtllm-serve constrain the call with an xgrammar structural
tag. Exit code 0 always; read the summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def strict_schema(schema: dict) -> dict:
    out = dict(schema)
    if out.get("type") == "object":
        out["additionalProperties"] = False
    if "properties" in out:
        out["properties"] = {k: strict_schema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = strict_schema(out["items"])
    return out


def tools_for(item: dict, strict: bool) -> list[dict]:
    tools = copy.deepcopy(item["tools"])
    if strict:
        for t in tools:
            t["function"]["parameters"] = strict_schema(t["function"]["parameters"])
            t["function"]["strict"] = True
    return tools


def chat(base_url: str, body: dict) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="served model name")
    ap.add_argument("--mode", choices=["plain", "strict"], required=True)
    ap.add_argument("--items", default=str(Path(__file__).with_name("items.json")))
    ap.add_argument("--show", action="append", default=[], help="print this item's full reply")
    ap.add_argument("--out", help="write one JSON line per item here")
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text(encoding="utf-8"))["items"]
    strict = args.mode == "strict"
    rows, ok = [], 0
    for item in items:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": item["query"]}],
            "tools": tools_for(item, strict),
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = chat(args.base_url, body)
        if "error" in resp:
            got, verdict, msg = None, "request_error", {"error": resp["error"]}
        else:
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            got = [c["function"]["name"] for c in calls]
            if len(calls) == 1 and got[0] == item["expected_function"]:
                verdict = "ok"
            elif not calls:
                verdict = "no_tool_call"
            elif len(calls) > 1:
                verdict = "several_calls"
            else:
                verdict = "wrong_function"
        ok += verdict == "ok"
        row = {"id": item["id"], "expected": item["expected_function"], "called": got,
               "verdict": verdict}
        rows.append(row)
        if verdict != "ok":
            print(f"  {item['id']:<14} expected {item['expected_function']:<22} "
                  f"got {verdict}: {got}", flush=True)
        if item["id"] in args.show:
            print(f"--- {item['id']} ({args.mode}) query: {item['query']!r}")
            print(json.dumps(msg, indent=1, ensure_ascii=False))
            print("---")
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps({**row, "mode": args.mode, "message": msg}) + "\n")

    print(f"{args.mode}: {ok}/{len(items)} called the expected function")
    return 0


if __name__ == "__main__":
    sys.exit(main())
