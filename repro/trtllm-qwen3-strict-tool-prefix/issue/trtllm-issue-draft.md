<!--
How to file: open https://github.com/NVIDIA/TensorRT-LLM/issues/new?template=06-bug-report.yml
and paste each "## ..." section below into the form field of the same name:

  Title                        -> Title
  ## System Info               -> System Info
  ## Who can help?             -> Who can help? (optional; left blank here)
  ## Information / ## Tasks    -> tick the same boxes in the form
  ## Reproduction              -> Reproduction
  ## Expected behavior         -> Expected behavior
  ## actual behavior           -> actual behavior
  ## additional notes          -> additional notes
  ## Before submitting...      -> tick the box after searching existing issues

Do not paste this comment or the "## ..." heading lines themselves.
-->

**Title:** [Bug]: Strict tool calls with `--tool_parser qwen3` pick the wrong function: the structural-tag prefix `{"name":"` rejects the space Qwen3 writes

## System Info

- CPU architecture: x86_64 (AWS g6e.xlarge: 4 vCPU, 32 GiB host RAM)
- GPU: 1x NVIDIA L40S, 48 GB
- NVIDIA driver 595.91.07 on the host (CUDA 13.2). The container's banner: "CUDA Forward Compatibility mode ENABLED. Using CUDA 13.4 driver version 615.71.09 with kernel driver version 595.91.07."
- OS: Ubuntu 24.04 (AWS Deep Learning Base OSS Nvidia Driver GPU AMI, 20261006), Docker 29.8.2 with the NVIDIA container runtime
- Container: `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2` (`@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a`; NVIDIA PyTorch 26.09 base)
- Libraries in the container: `tensorrt_llm 1.4.0rc2`, `xgrammar 0.1.32` (as pinned), `transformers 5.5.4`
- PyTorch backend, `guided_decoding_backend: xgrammar`
- Models: Qwen/Qwen3-8B @ b968826 (BF16) and nvidia/Qwen3-8B-FP8 @ 2cebc4c (ModelOpt FP8); both affected
- Server: `trtllm-serve <model> --backend pytorch --tool_parser qwen3`; requests use `chat_template_kwargs: {"enable_thinking": false}`
- The same parser code is in v1.4.0rc2.post1 and on `main` as of fe6938adb1cd0a17f16eb7da68d4b824e2909f51 (2026-11-08). `qwen3_tool_parser.py` last changed in 03d4be4 (#16866), which is already in v1.4.0rc2. Main now pins `xgrammar==0.2.8` (`requirements.txt` line 76), and the CPU repro below gives the same result with it.

## Who can help?

<!-- optional: tag maintainers -->

## Information

- [ ] The official example scripts
- [x] My own modified scripts

## Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

## Reproduction

**The cause.** For tools sent with `"strict": true`, `_build_tool_strict_guided_decoding_params` (`tensorrt_llm/serve/openai_server.py`, lines 315-392 at v1.4.0rc2) builds an xgrammar `triggered_tags` structural tag from the tool parser's `structure_info()`. For `qwen3`, `Qwen3ToolParser.structure_info` (`tensorrt_llm/serve/tool_parser/qwen3_tool_parser.py`, lines 262-267) returns:

```python
begin='<tool_call>\n{"name":"' + name + '", "arguments":',
end="}\n</tool_call>",
trigger="<tool_call>",
```

Qwen3's chat template writes every tool call as `<tool_call>\n{"name": "<name>", "arguments": <args>}\n</tool_call>`, with a space after `"name":` and another after `"arguments":`. The tag forbids both spaces:

- Right after `<tool_call>\n{"name":`, the grammar rejects the token Qwen3's template has next, ` "` (id 330).
- For this tool set (`get_weather`, `set_thermostat`, `set_timer`) it allows only `"`, `"g` or `"s`, so the function name has to start in a context the model never produces.
- With several tools offered, the model often starts the wrong name and then fills that tool's schema.

**1. Deterministic check on CPU (no GPU, no weights).** The script below renders the correct call through Qwen3-8B's chat template (tokenizer only), builds the structural tag the way v1.4.0rc2 does (begin/end/trigger copied verbatim, same `triggered_tags` and `ResponseFormat` dump), compiles it with `GrammarCompiler.compile_structural_tag` as `grammar_matcher.py` does, and feeds the template's tokens through a `GrammarMatcher`. It then repeats this with one, then both, spaces restored.

```bash
pip install "xgrammar==0.1.32" "transformers==5.5.4"   # 1.4.0rc2's pins
python trtllm_qwen3_strict_prefix_repro.py
```

Output (Python 3.12, CPU, also with tokenizers 0.22.2 and torch 2.15.0):

```
Qwen3-8B chat template, assistant reply for the correct call:
  "<tool_call>\n{\"name\": \"set_thermostat\", \"arguments\": {\"temperature\": 19, \"unit\": \"celsius\"}}\n</tool_call>"
  tokens: ['<tool_call>', '\n', '{"', 'name', '":', ' "', 'set', '_', 'ther', 'mostat', '",', ' "', 'arguments', '":', ' {"', 'temperature', '":', ' ', '1', '9', ',', ' "', 'unit', '":', ' "', 'c', 'elsius', '"}}\n', '</tool_call>']

== TensorRT-LLM 1.4.0rc2 qwen3 structure_info (as shipped)
tag begin for set_thermostat: "<tool_call>\n{\"name\":\"set_thermostat\", \"arguments\":"
REJECTED token 5: id 330 ' "'
  accepted so far: '<tool_call>\n{"name":'
  the grammar allows 3 tokens here, e.g. ['"', '"g', '"s']

== only the space after "name": restored
tag begin for set_thermostat: "<tool_call>\n{\"name\": \"set_thermostat\", \"arguments\":"
REJECTED token 14: id 5212 ' {"'
  accepted so far: '<tool_call>\n{"name": "set_thermostat", "arguments":'
  the grammar allows 5 tokens here, e.g. ['{', '{\n', '{\n\n', '{\n\n\n', '{"']

== spaced as the chat template (suggested fix)
tag begin for set_thermostat: "<tool_call>\n{\"name\": \"set_thermostat\", \"arguments\": "
ACCEPTED all 29 tokens of the template's reply; <|im_end|> ends: True
```

Both spaces matter: restoring only the one after `"name":` still rejects ` {"` after `"arguments":`.

With `xgrammar==0.2.8` (main's pin) and the same transformers, the result is the same: id 330 ` "` rejected first, id 5212 ` {"` rejected with only the first space restored, and all 29 tokens accepted with both. The only difference is the allowed-token list after `"arguments":`, which has 7 entries instead of 5 (it adds `{\r\n` and `{\r\n\r\n`).

<details><summary>trtllm_qwen3_strict_prefix_repro.py</summary>

```python
"""CPU repro: TensorRT-LLM 1.4.0rc2's strict-tool grammar for `--tool_parser qwen3` rejects
the tool call Qwen3's own chat template writes.

No GPU and no model weights: only Qwen3-8B's tokenizer and chat template, plus xgrammar.

    pip install "xgrammar==0.1.32" "transformers==5.5.4"   # the versions TRT-LLM 1.4.0rc2 pins
    python trtllm_qwen3_strict_prefix_repro.py

What it does:
1. Renders a correct call, set_thermostat(temperature=19, unit="celsius"), as an assistant
   turn through Qwen3-8B's chat template, then tokenizes the assistant's reply text.
2. Builds the structural tag the way TensorRT-LLM v1.4.0rc2 does for tools sent with
   "strict": true:
   - tensorrt_llm/serve/openai_server.py, _build_tool_strict_guided_decoding_params
     (lines 315-392): a "triggered_tags" format, with one tag per tool;
   - begin/end/trigger copied verbatim from
     tensorrt_llm/serve/tool_parser/qwen3_tool_parser.py, Qwen3ToolParser.structure_info
     (lines 262-267);
   - wrapped as ResponseFormat(type="structural_tag", format=...) and dumped with
     model_dump_json(by_alias=True, exclude_none=True) (tensorrt_llm/serve/openai_protocol.py,
     ResponseFormat, lines 288-295);
   - compiled with GrammarCompiler.compile_structural_tag
     (tensorrt_llm/_torch/pyexecutor/grammar_matcher.py, lines 110-111).
   One difference: the tokenizer info here comes from xgrammar.TokenizerInfo.from_huggingface,
   while TRT-LLM builds it from the encoded vocab and the same HF metadata.
3. Feeds the template's tokens through an xgrammar GrammarMatcher and reports the first
   token it rejects, with what the grammar allows there instead.
4. Repeats it with only the space after "name": restored, then with both spaces as the
   template writes them (the suggested fix).
"""

import json
from typing import Any, Literal, Optional

import xgrammar as xgr
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# The tools of one multi-tool item, exactly as a client sends them in strict mode
# ("strict": true, additionalProperties: false on every object schema).
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                     "description": "Temperature unit"}},
            "required": ["city"], "additionalProperties": False},
        "strict": True}},
    {"type": "function", "function": {
        "name": "set_thermostat", "description": "Set the home thermostat.",
        "parameters": {"type": "object", "properties": {
            "temperature": {"type": "number"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["temperature", "unit"], "additionalProperties": False},
        "strict": True}},
    {"type": "function", "function": {
        "name": "set_timer", "description": "Start a countdown timer.",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "integer"}, "label": {"type": "string"}},
            "required": ["minutes"], "additionalProperties": False},
        "strict": True}},
]  # fmt: skip
USER = "Turn the house temperature down to 19 celsius."
CALL = {"name": "set_thermostat", "arguments": {"temperature": 19, "unit": "celsius"}}


# --- TensorRT-LLM v1.4.0rc2, copied ----------------------------------------------------------


def trtllm_qwen3_structure_info(name: str) -> dict[str, str]:
    """Qwen3ToolParser.structure_info, qwen3_tool_parser.py lines 262-267 (verbatim strings)."""
    return dict(
        begin='<tool_call>\n{"name":"' + name + '", "arguments":',
        end="}\n</tool_call>",
        trigger="<tool_call>",
    )


def spaced_structure_info(name: str) -> dict[str, str]:
    """The suggested fix: the same strings, spaced as Qwen3's chat template writes them."""
    return dict(
        begin='<tool_call>\n{"name": "' + name + '", "arguments": ',
        end="}\n</tool_call>",
        trigger="<tool_call>",
    )


def name_space_only_structure_info(name: str) -> dict[str, str]:
    """Only the space after "name": restored, to see whether "arguments": needs its space too."""
    return dict(
        begin='<tool_call>\n{"name": "' + name + '", "arguments":',
        end="}\n</tool_call>",
        trigger="<tool_call>",
    )


class ResponseFormat(BaseModel):
    """openai_protocol.py ResponseFormat (lines 288-295), only the fields this path sets."""

    type: Literal["text", "json", "json_schema", "json_object", "regex", "ebnf", "structural_tag"]
    schema_: Optional[dict] = None
    json_schema: Optional[dict] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    format: Optional[xgr.structural_tag.Format] = None


def build_strict_structural_tag(tools: list[dict[str, Any]], get_info) -> str:
    """_build_tool_strict_guided_decoding_params (openai_server.py lines 359-392), minus the
    parser lookup: one tag per tool, json_schema content for strict tools, triggered_tags."""
    tags, triggers = [], set()
    for tool in tools:
        fn = tool["function"]
        info = get_info(fn["name"])
        triggers.add(info["trigger"])
        if fn.get("strict") and fn.get("parameters"):
            content = {"type": "json_schema", "json_schema": fn["parameters"]}
        else:
            content = {"type": "any_text"}
        tags.append({"begin": info["begin"], "content": content, "end": info["end"]})
    stag_format = {"type": "triggered_tags", "triggers": sorted(triggers), "tags": tags}
    resp_format = ResponseFormat(type="structural_tag", format=stag_format)
    return resp_format.model_dump_json(by_alias=True, exclude_none=True)


# --- the check ------------------------------------------------------------------------------


def template_reply(tok) -> str:
    """The assistant reply text Qwen3's template writes for CALL (generation prompt removed)."""
    messages = [{"role": "user", "content": USER}]
    kwargs = dict(tools=TOOLS, tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    call = {"type": "function", "function": {"name": CALL["name"], "arguments": CALL["arguments"]}}
    full = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": "", "tool_calls": [call]}], **kwargs
    )
    assert full.startswith(prompt), "the template's assistant turn does not extend the prompt"
    reply = full[len(prompt) :]
    end = reply.index("<|im_end|>")  # what the model writes before its stop token
    return reply[:end]


def run(label: str, get_info, tok, info: xgr.TokenizerInfo, reply_ids: list[int]) -> None:
    compiler = xgr.GrammarCompiler(info)
    stag = build_strict_structural_tag(TOOLS, get_info)
    matcher = xgr.GrammarMatcher(compiler.compile_structural_tag(stag))
    print(f"\n== {label}")
    print("tag begin for set_thermostat:", json.dumps(get_info("set_thermostat")["begin"]))
    accepted = ""
    for pos, tid in enumerate(reply_ids):
        piece = tok.decode([tid])
        if not matcher.accept_token(tid):
            bitmask = xgr.allocate_token_bitmask(1, info.vocab_size)
            matcher.fill_next_token_bitmask(bitmask)
            allowed = [
                i for i in range(info.vocab_size)
                if (int(bitmask[0, i // 32]) >> (i % 32)) & 1
            ]  # fmt: skip
            shown = sorted(tok.decode([i]) for i in allowed)[:8]
            print(f"REJECTED token {pos}: id {tid} {piece!r}")
            print(f"  accepted so far: {accepted!r}")
            print(f"  the grammar allows {len(allowed)} tokens here, e.g. {shown}")
            return
        accepted += piece
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    ended = matcher.accept_token(stop) and matcher.is_terminated()
    print(f"ACCEPTED all {len(reply_ids)} tokens of the template's reply; <|im_end|> ends: {ended}")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    reply = template_reply(tok)
    reply_ids = tok.encode(reply, add_special_tokens=False)
    print("Qwen3-8B chat template, assistant reply for the correct call:")
    print("  " + json.dumps(reply))
    print("  tokens:", [tok.decode([t]) for t in reply_ids])
    info = xgr.TokenizerInfo.from_huggingface(tok, stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>")])
    run("TensorRT-LLM 1.4.0rc2 qwen3 structure_info (as shipped)", trtllm_qwen3_structure_info,
        tok, info, reply_ids)  # fmt: skip
    run("only the space after \"name\": restored", name_space_only_structure_info,
        tok, info, reply_ids)  # fmt: skip
    run("spaced as the chat template (suggested fix)", spaced_structure_info, tok, info, reply_ids)


if __name__ == "__main__":
    main()
```

</details>

**2. End-to-end A/B on GPU.** Standalone script: https://github.com/arulster17/loom/tree/b3c4f15/repro/trtllm-qwen3-strict-tool-prefix (`run.sh`, a standard-library client, 45 multi-tool items, raw replies in `result-bf16/`). It serves Qwen3-8B BF16 with `trtllm-serve --tool_parser qwen3` on the pinned 1.4.0rc2 image and runs the 45 items three ways, one request at a time (greedy, non-thinking, `tool_choice: "auto"`). Between runs 2 and 3 the only change is the two spaces in `structure_info`, patched inside the image.

| Run (1x L40S, 2026-11-07, one greedy pass) | Right function |
|---|---|
| 1. plain tools, stock parser | 45/45 |
| 2. strict tools, stock parser | 27/45 (all 18 misses are the wrong function) |
| 3. strict tools, both spaces restored | 45/45 |

40 of the 45 items offer 3 functions, 5 offer 4.

## Expected behavior

With `"strict": true` tools, the grammar should accept the tool call exactly as Qwen3's chat template writes it, `<tool_call>\n{"name": "<name>", "arguments": {...}}\n</tool_call>`. The model should then pick the same function it picks with plain tools: `set_thermostat(temperature=19, unit="celsius")` for the request below, as TensorRT-LLM does without `"strict"` and as vLLM does with it.

## actual behavior

The grammar rejects the template's call at the space after `"name":` (CPU check above), and on multi-tool requests the model often calls the wrong function.

Example: one item from our function-calling benchmark, run through our quality gate on the same 1.4.0rc2 image. The request in the details block below is exactly what the gate's client sent for this item: one user turn ("Turn the house temperature down to 19 celsius."), three tools in strict mode (`"strict": true`, `additionalProperties: false`), `tool_choice: "auto"`, greedy decoding with a seed.

- **TensorRT-LLM 1.4.0rc2, BF16, strict:** wrong in all 3 passes. The stored reply, as `trtllm-serve` returned it after its tool parser:
  `"tool_calls": [{"name": "get_weather", "arguments": "{\"city\": \"house\", \"unit\": \"celsius\"}"}]`, with `content` empty.
- **Same request without `"strict": true` on TensorRT-LLM:** right in all 3 passes, `set_thermostat` with `temperature` 19 and `unit` `"celsius"`.
- **vLLM 0.30.0 (hermes parser), same strict request:** right in all 3 passes. We keep only the verdict for passing items, not the raw reply. The verdict is an exact match on name `set_thermostat` with `temperature` = 19 and `unit` = `"celsius"`.

<details><summary>The gate's request for this item (JSON)</summary>

```json
{
  "model": "qwen3-8b",
  "temperature": 0.0,
  "seed": 1234,
  "chat_template_kwargs": {
    "enable_thinking": false
  },
  "messages": [
    {
      "role": "user",
      "content": "Turn the house temperature down to 19 celsius."
    }
  ],
  "max_tokens": 512,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "City name"
            },
            "unit": {
              "type": "string",
              "enum": [
                "celsius",
                "fahrenheit"
              ],
              "description": "Temperature unit"
            }
          },
          "required": [
            "city"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    },
    {
      "type": "function",
      "function": {
        "name": "set_thermostat",
        "description": "Set the home thermostat.",
        "parameters": {
          "type": "object",
          "properties": {
            "temperature": {
              "type": "number"
            },
            "unit": {
              "type": "string",
              "enum": [
                "celsius",
                "fahrenheit"
              ]
            }
          },
          "required": [
            "temperature",
            "unit"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    },
    {
      "type": "function",
      "function": {
        "name": "set_timer",
        "description": "Start a countdown timer.",
        "parameters": {
          "type": "object",
          "properties": {
            "minutes": {
              "type": "integer"
            },
            "label": {
              "type": "string"
            }
          },
          "required": [
            "minutes"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "tool_choice": "auto",
  "stream": false
}
```

</details>

The GPU repro (step 2) is not this request: it sends no seed, names the model `Qwen/Qwen3-8B`, has no `stream` field, sends requests one at a time instead of up to 16 at once, and starts the server with `--max_seq_len 8192` instead of 32768 and without the gate's perf-metric options (`return_perf_metrics`, `enable_iter_perf_stats`). In the repro, strict mode returned `get_weather` again for this item, with `city` "New York", and the patched parser returned `set_thermostat(temperature=19, unit="celsius")`. The made-up city differs between setups ("house" in both of our BF16 benchmark runs, "New York" in both FP8 benchmark runs and in this BF16 repro), but the function does not: all five strict replies we stored for this item call `get_weather`. The repro's 18 strict misses are the same 18 items as the gate's 18 `wrong_function` failures on BF16.

## additional notes

**Measured impact (our quality gate).** 135 single-call items: 90 offer one tool, 45 offer 3 or 4 tools. Scoring needs the exact function name and typed arguments. Greedy decoding; each score is the mean of 3 passes. The tools are identical except for `"strict": true` and `additionalProperties: false`.

| Engine / model | plain tools | strict tools | strict − plain | strict, multi-tool items |
|---|---|---|---|---|
| vLLM 0.30.0, Qwen3-8B BF16 | 97.8% | 97.8% | 0.0 pts | 100% |
| TensorRT-LLM 1.4.0rc2, Qwen3-8B BF16 | 97.8% | 84.0% | −13.8 pts | 58.5% |
| TensorRT-LLM 1.4.0rc2, Qwen3-8B-FP8 | 98.3% | 86.4% | −11.9 pts | 64.4% |

- Against vLLM BF16 with strict tools (97.8%), TensorRT-LLM strict is 13.8 points lower on BF16 and 11.4 points lower on FP8.
- One-tool items score the same with and without strict. The whole loss is on multi-tool items, almost all of it `wrong_function` (18 of the 19 BF16 multi-tool failures).
- Otherwise Qwen3-8B BF16's logits on TensorRT-LLM match vLLM's closely (teacher-forced top-5 KL 0.0006 nats, top-1 agreement 99.2%).

**Suggested fix.** Match the chat template in `Qwen3ToolParser.structure_info`:

```python
begin='<tool_call>\n{"name": "' + name + '", "arguments": ',
```

`end="}\n</tool_call>"` already matches. A regression test could render one call through the Qwen3 chat template and check that the grammar accepts its tokens, as the CPU script does. Three things to check with it:

- **Named `tool_choice` uses the same string.** `_build_forced_tool_call_decoding` (`openai_server.py`, lines 409-474 at v1.4.0rc2) returns `info.begin` from the same `structure_info` as `begin_prefix`, and `render/chat.py` appends it to the rendered prompt text; what follows is constrained by a plain JSON-schema grammar. With the spaced string, that prompt would end in `"arguments": ` with a trailing space, while the template tokenizes that space together with the brace (` {"`, id 5212). This path needs its own test and possibly its own handling: for example, strip the trailing space from the forced prefix, or keep the space out of `begin` and let the grammar that follows accept it.
- **Scope.** With `--tool_parser auto`, `MODEL_TYPE_TO_TOOL_PARSER` (`tool_parser_factory.py`, lines 25-31) picks this parser for model types `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5`, `qwen3_5_moe` and `qwen3_next`. We have only tested Qwen3-8B, and have not checked which of those families' templates write the spaced JSON form.
- **Docstring.** The `Qwen3ToolParser` class docstring (`qwen3_tool_parser.py`, line 20) shows the same unspaced format, `<tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>`.

**What other implementations use.** vLLM and xgrammar's own Qwen3 tag keep the spaces; SGLang has the same unspaced string.

- vLLM 0.30.0, `vllm/tool_parsers/structural_tag_registry.py`, `_hermes_tool_tags` (lines 335-356): begin `'<tool_call>\n{"name": "' + name + '", "arguments": '`, end `"}\n</tool_call>"`. We used vLLM's hermes parser for Qwen3.
- xgrammar's own built-in `qwen_3` structural tag (`python/xgrammar/builtin_structural_tag.py` at v0.2.8, lines 1582-1583, `get_qwen_3_structural_tag`): `TOOL_CALL_BEGIN_PREFIX = '<tool_call>\n{"name": "'` and `ARGUMENTS_FIELD_PREFIX = '", "arguments": '`.
- SGLang: `Qwen25Detector.structure_info` (`python/sglang/srt/function_call/qwen25_detector.py`) uses the same `{"name":"` string in SGLang 0.5.21, and it is unchanged on SGLang main. In our runs SGLang 0.5.21 (`qwen25` parser) shows a similar strict-mode drop on Qwen3-8B: 86.7% vs 97.8% plain, mostly `wrong_function` on multi-tool items.

**Supplementary, weak evidence (not in the linked repro): Qwen3-0.6B on CPU.** No TensorRT-LLM: xgrammar 0.1.32's HF logits processor, the same 45 multi-tool items with strict tools, greedy.
- Right function: 41/45 unconstrained, 40/45 with the 1.4.0rc2 tag, 42/45 with the spaced tag.
- Two items were wrong only under the 1.4.0rc2 tag, and the spaced tag got both right: "define sonder" became `search_books` instead of `define_word`, and a BMI item became `log_workout` instead of `calculate_bmi`.
- Under the 1.4.0rc2 tag the call is unspaced at the forced prefix by construction, and the arguments after it were mostly unspaced too (33 of 43 calls, e.g. `{"temperature":19,"unit":"celsius"}`; the other 10 partly spaced). Unconstrained and under the spaced tag, all 43 calls were spaced as the template writes them.

## Before submitting a new issue...

- [ ] Make sure you already searched for relevant issues, and checked the documentation and examples for answers to frequently asked questions.
