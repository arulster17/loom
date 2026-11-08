# Repro: Qwen3 strict tools force `{"name":"` and pick the wrong function

**Status: run once on an L40S (BF16, 2026-11-07); reproduces.** Strict tools with the stock
parser: 27/45 right function. The same with both spaces restored: 45/45, as with plain
tools. See "Result". The numbers under "What Loom saw" come from Loom's quality gate
(docs/perf/tensorrt-llm.md), not from this script.

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
| `result-bf16/` | Output of the one L40S BF16 run (see "Result") |

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

One run, 2026-11-07, AWS g6e.xlarge (1x L40S 48 GB), BF16 `Qwen/Qwen3-8B` @ b968826,
greedy, one pass over the 45 items. Versions (`result-bf16/versions.txt`):

```
tensorrt_llm 1.4.0rc2 | xgrammar 0.1.32 | transformers 5.5.4
nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
NVIDIA L40S, 595.91.07
```

| Run | Right function | Failures |
|---|---|---|
| 1. plain tools, stock parser | **45/45** | none |
| 2. strict tools, stock parser | **27/45** | 18, all wrong function |
| 3. strict tools, both spaces restored | **45/45** | none |

`summary.txt` prints these in that order (runs 2 and 3 are both labelled `strict:`).

`m-thermo-1`, "Turn the house temperature down to 19 celsius.":

- plain: `set_thermostat({"temperature": 19, "unit": "celsius"})`
- strict, stock: `get_weather({"city": "New York", "unit": "celsius"})` (Loom's gate run
  got `get_weather(city="house")`; same wrong function, different made-up city)
- strict, patched: `set_thermostat({"temperature": 19, "unit": "celsius"})`

The 18 strict-stock misses (`result-bf16/strict-stock.txt`): m-thermo-1, m-timer-1,
m-currency-1, m-route-1, m-table-1, m-translate-1, m-cart-1, m-currency-2, m-alarm-1,
m-timer-2, m-currency-3, m-nearby-1, m-email-1, m-music-1, m-track-1, m-image-1,
m-bmi-1, m-translate-2. The count matches the 18 wrong-function failures Loom's gate saw.

`template_check.txt` shows the mismatch at the token level: the template's prefix
tokenizes as `... '":', 'Ġ"'`, the forced one as `... '":"'`, a token the chat template
never puts there.

So, on these 45 items: restoring the two spaces recovers everything strict tools lost
(27/45 back to 45/45), and nothing else changed between runs 2 and 3. This is one greedy
run on 45 items with one model and precision (FP8 not run); it shows the prefix is the
cause here, not how large the effect is in general.

Files: `result-bf16/` holds the summary, versions, template check, the three per-run
`.txt` and `.jsonl` (every reply). Server logs stayed in `out/` (git-ignored).

Two script fixes were needed on the first real run (the script had only met a fake
server): `xgrammar` has no `__version__`, so `versions.txt` now reads versions from
package metadata; and `docker logs | grep -m1 patched` could die of SIGPIPE under
`pipefail`, so the log is saved first and grepped from the file.
