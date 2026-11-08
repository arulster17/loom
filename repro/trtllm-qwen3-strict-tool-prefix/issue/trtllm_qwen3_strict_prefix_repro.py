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
