# Repro: Qwen3 strict tools force `{"name":"` and pick the wrong function

**Status: run once on an L40S (BF16, 2026-11-07); reproduces.** Strict tools with the stock
parser: 27/45 right function. The same with both spaces restored: 45/45, as with plain
tools. See "Result". The numbers under "What Loom saw" come from Loom's quality gate
(docs/perf/tensorrt-llm.md), not from this script.

## The bug

With strict tools, TensorRT-LLM's `qwen3` tool parser constrains the call with an
xgrammar structural tag. The tag fires on `<tool_call>` and forces this text
(`tensorrt_llm/serve/tool_parser/qwen3_tool_parser.py`, `structure_info`; the same on
v1.4.0rc2 and on main as of fe6938a, 2026-11-08; the file last changed in 03d4be4, #16866):

```python
begin='<tool_call>\n{"name":"' + name + '", "arguments":',
```

Qwen3's chat template, and so its training data, writes `{"name": "` and
`"arguments": {` with a space after each colon. vLLM's hermes structural tag writes the
spaces too. Run through xgrammar with Qwen3's tokenizer, the stock tag rejects the
template's own call at the space after `"name":`; with only that space restored it still
rejects at the space after `"arguments":`; with both restored it accepts the whole call
(xgrammar 0.1.32, as 1.4.0rc2 pins it, and the same with 0.2.8, which main now pins).
Forcing the model off its usual tokens just before it names the function changes which function it
picks when it has several to choose from.

## Files

| File | What it does |
|---|---|
| `run.sh` | The whole repro: versions, weights, template check, then three runs |
| `repro.py` | Standard-library client: sends the 45 items, counts right-function calls |
| `items.json` | 45 items (40 offer 3 functions, 5 offer 4) and the function each should call |
| `template_check.py` | No GPU work: prints the template's tool call next to the forced prefix |
| `patch_parser.py` | The fix (both spaces restored), applied inside the image for the third run |
| `options.yaml` | `guided_decoding_backend: xgrammar` (strict tools need a grammar backend) |
| `make_items.py` | Loom only: regenerates `items.json` from Loom's tool_calling data |
| `result-bf16/` | Output of the one L40S BF16 run (see "Result") |

## Run

Needs, on the host: Docker with the NVIDIA container runtime, `nvidia-smi`, `curl` and
`python3` (the client is standard library only), internet access for the image and
weights, and a GPU that holds the model (an L40S 48 GB for Qwen3-8B BF16, about 30 min
including downloads). The container runs as root and writes the weights into
`$HOME/.cache/huggingface` (override with `HF_CACHE=...`), so the files it adds there are
owned by root.

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

- tool_calling_strict, all 135 items, mean of 3 greedy passes: BF16 84.0% and FP8 86.4%,
  against 97.8% for vLLM 0.30.0 BF16 with the same strict tools (−13.8 and −11.4 points).
  With plain tools TensorRT-LLM scored 97.8% (BF16) and 98.3% (FP8), so FP8 lost 11.9
  points to strict tools on its own numbers.
- BF16 strict has 22 failures. 18 are the wrong function, all on the 45 items with several
  functions (this set). Plain tools: no failures on these 45.
- Example, `m-thermo-1`: "Turn the house temperature down to 19 celsius." became
  `get_weather(city="house")` instead of `set_thermostat`.
- Seen in two separate BF16 runs: 0.837 in a one-pass run, and 0.840 in the three-pass
  gate run (passes 0.837, 0.837, 0.844).

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
- strict, stock: `get_weather({"city": "New York", "unit": "celsius"})` (Loom's BF16 gate
  run stored `get_weather(city="house")` and its FP8 gate run `get_weather(city="New York")`:
  the made-up city varies, the wrong function does not)
- strict, patched: `set_thermostat({"temperature": 19, "unit": "celsius"})`

The 18 strict-stock misses (`result-bf16/strict-stock.txt`): m-thermo-1, m-timer-1,
m-currency-1, m-route-1, m-table-1, m-translate-1, m-cart-1, m-currency-2, m-alarm-1,
m-timer-2, m-currency-3, m-nearby-1, m-email-1, m-music-1, m-track-1, m-image-1,
m-bmi-1, m-translate-2. These are the same 18 items as the 18 wrong-function failures in
Loom's BF16 gate run (and in its earlier one-pass BF16 run). The wrong function picked
differs on a few of them (e.g. `m-translate-1`: `search_books` here, `define_word` in the
gate run). The gate sent its requests differently (a seed, up to 16 at a time,
`--max_seq_len 32768`, perf-metric options on the server), so it is not this script.

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
