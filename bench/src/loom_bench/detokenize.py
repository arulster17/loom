"""How engines render one token's text in completions logprobs.

Byte-level BPE tokenizers (Qwen, Llama 3) can split one character across tokens:
" √" can be the two tokens b" \\xe2\\x88" and b"\\x9a". Engines render such tokens
differently in a completions response (`tokens` and the `top_logprobs` keys):

- vLLM (v0.30.0, `LogprobsProcessor._correct_decoded_token`) decodes each token alone
  and, when that ends in U+FFFD, re-decodes it after up to 4 preceding sequence tokens:
  a token that cannot finish a character yet renders as "", and the token that finishes
  it carries the character's whole text (" √"). `incremental_text` is that rule.
- SGLang (v0.5.21, `_lossless_token_text`) decodes each token alone and renders a
  genuine fragment (bytes that are not valid UTF-8) as its raw bytes in latin-1, one
  character per byte: " â\\x88", then "\\x9a". `per_token_text` is that rule.

Both see the same token ids; only the strings differ. The mock backend renders with
these functions, and the divergence scorer uses them to bring a per-token response
to vLLM's form before comparing it with a reference.
"""

from __future__ import annotations

from collections.abc import Sequence

REPLACEMENT = "�"
# vLLM looks back at most this many sequence tokens: enough for any UTF-8 sequence.
MAX_CONTEXT = 4


def _decode(data: bytes) -> str:
    # What a tokenizer's decode() does with a byte-level token's bytes.
    return data.decode("utf-8", "replace")


def incremental_text(token: bytes, context: Sequence[bytes]) -> str:
    """vLLM's text for a token with raw bytes `token` after the sequence tokens
    `context` (oldest first): its own decode when that is complete, else the text it
    completes together with the preceding fragment tokens, else ""."""
    alone = _decode(token)
    if not alone.endswith(REPLACEMENT):
        return alone
    for n in range(1, min(len(context), MAX_CONTEXT) + 1):
        ctx = list(context[-n:])
        full = _decode(b"".join(ctx) + token)
        if full.endswith(REPLACEMENT):
            continue
        # Trailing context tokens that were fragments rendered "": their text is this
        # token's. Only the clean prefix before them is already accounted for.
        clean_end = len(ctx)
        for j in range(len(ctx) - 1, -1, -1):
            if _decode(ctx[j]).endswith(REPLACEMENT):
                clean_end = j
            else:
                break
        prefix = _decode(b"".join(ctx[:clean_end]))
        if full.startswith(prefix):
            return full[len(prefix) :]
        common = 0
        for a, b in zip(prefix, full, strict=False):
            if a != b:
                break
            common += 1
        return full[common:]
    return ""


def per_token_text(token: bytes) -> str:
    """SGLang's text for a token with raw bytes `token`: its decode, or, for bytes that
    are not valid UTF-8 (a fragment), the bytes themselves as latin-1."""
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return token.decode("latin-1")
