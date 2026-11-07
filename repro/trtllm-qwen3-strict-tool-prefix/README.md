# Repro: Qwen3 strict tools force `{"name":"` and pick the wrong function

**Status: written, not yet run on a GPU.** The numbers under "What Loom saw" come from
Loom's quality gate (docs/perf/tensorrt-llm.md), not from this script. Fill in "Result"
once `run.sh` has run.

## The bug

With strict tools, TensorRT-LLM's `qwen3` tool parser constrains the call with an
xgrammar structural tag. The tag fires on `<tool_call>` and forces this text
(`tensorrt_llm/serve/tool_parser/qwen3_tool_parser.py`, `structure_info`; the same on
v1.4.0rc2 and on main at 03d4be4):

```python
begin='<tool_call>\n{"name":"' + name + '", "arguments":',
```

Qwen3's chat template, and so its training data, writes `{"name": "` and
`"arguments": {` with a space after each colon. vLLM's hermes structural tag writes the
spaces too. Run through xgrammar with Qwen3's tokenizer, the stock tag rejects the
template's own call at the space after `"name":`; with only that space restored it still
rejects at the space after `"arguments":`; with both restored it accepts the whole call. Forcing the
model off its usual tokens just before it names the function changes which function it
picks when it has several to choose from.

## Files

| File | What it does |
|---|---|
| `run.sh` | The whole repro: versions, weights, template check, then three runs |
| `repro.py` | Standard-library client: sends the 45 items, counts right-function calls |
| `items.json` | 45 items, each with 3 functions to choose from and the right one |
| `template_check.py` | No GPU work: prints the template's tool call next to the forced prefix |
| `patch_parser.py` | The fix (both spaces restored), applied inside the image for the third run |
| `options.yaml` | `guided_decoding_backend: xgrammar` (strict tools need a grammar backend) |
| `make_items.py` | Loom only: regenerates `items.json` from Loom's tool_calling data |

## Run

Needs Docker with the NVIDIA runtime and a GPU that holds the model (an L40S 48 GB for
Qwen3-8B BF16, about 30 min including downloads):

```bash
./run.sh
# FP8 (fits ~24 GB):
MODEL=nvidia/Qwen3-8B-FP8 REVISION=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 ./run.sh
```

It serves the model with `trtllm-serve --tool_parser qwen3` and runs the same 45 requests
three times (greedy, non-thinking, `tool_choice: auto`):

1. plain tools, stock parser;
2. strict tools (`"strict": true`, `additionalProperties: false`), stock parser;
3. strict tools, parser patched to `{"name": "` ... `"arguments": ` (both spaces).
   Nothing else changes from run 2.

Output goes to `out/`: `summary.txt`, one `.jsonl` per run with every reply, the server
logs and `versions.txt`.

## What Loom saw (TensorRT-LLM 1.4.0rc2, L40S, Qwen3-8B)

- tool_calling_strict, all 135 items: BF16 −13.8 points vs vLLM BF16, FP8 −11.4.
- BF16 strict has 22 failures. 18 are the wrong function, all on the 45 items with several
  functions (this set). Plain tools: no failures on these 45.
- Example, `m-thermo-1`: "Turn the house temperature down to 19 celsius." became
  `get_weather(city="house")` instead of `set_thermostat`.
- Seen in two separate runs (0.837 and 0.844 strict).

## Result

Not run yet.
