"""OpenAI-compatible HTTP front end (vLLM flavoured) over the simulated engine."""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from loom_bench import __version__
from loom_bench.mock.config import MockConfig
from loom_bench.mock.content import (
    TOKENIZER,
    Output,
    arithmetic_answer,
    fit_output,
    free_text,
    json_text,
    message_text,
    render_chat_prompt,
)
from loom_bench.mock.engine import AsyncEngine
from loom_bench.mock.logprobs import MAX_LOGPROBS, TokenLogprob, token_logprobs
from loom_bench.mock.metrics import CONTENT_TYPE, render_metrics
from loom_bench.mock.sim import SimRequest, SimSequence, prompt_block_hashes

# ---------------------------------------------------------------- request bodies


class _Body(BaseModel):
    model_config = ConfigDict(extra="allow")


class StreamOptions(_Body):
    include_usage: bool = False


class JsonSchemaFormat(_Body):
    name: str | None = None
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")


class ResponseFormat(_Body):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaFormat | None = None


class FunctionDef(_Body):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(_Body):
    type: Literal["function"] = "function"
    function: FunctionDef


class ChatMessage(_Body):
    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class _GenerateRequest(_Body):
    model: str
    n: int = Field(1, ge=1, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    min_tokens: int = Field(0, ge=0)
    ignore_eos: bool = False
    response_format: ResponseFormat | None = None

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage


class ChatCompletionRequest(_GenerateRequest):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=MAX_LOGPROBS)
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None


class CompletionRequest(_GenerateRequest):
    prompt: str | list[str]
    max_tokens: int = Field(16, ge=1)  # OpenAI / vLLM default for completions
    logprobs: int | None = Field(None, ge=0, le=MAX_LOGPROBS)
    echo: bool = False


# ---------------------------------------------------------------- errors


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = {
            "error": {"message": message, "type": error_type, "param": param, "code": code}
        }


class StreamAborted(Exception):
    """Injected fault: the connection drops mid-stream."""


def _error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(error.body, status_code=error.status)


# ---------------------------------------------------------------- generation


@dataclass(frozen=True, slots=True)
class Generation:
    prompt_pieces: tuple[str, ...]
    output: Output
    finish_reason: str

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_pieces)

    @property
    def completion_tokens(self) -> int:
        return len(self.output.pieces)


class MockServer:
    def __init__(self, config: MockConfig) -> None:
        self.config = config
        self.engine = AsyncEngine(config)
        self.ready_at = math.inf
        self._faults = random.Random(config.seed)

    def check_ready(self) -> None:
        if time.monotonic() < self.ready_at:
            raise ApiError(
                503, "Server is starting up.", error_type="server_error", code="service_unavailable"
            )

    def check_model(self, model: str) -> None:
        if model not in self.config.models:
            raise ApiError(
                404,
                f"The model `{model}` does not exist.",
                param="model",
                code="model_not_found",
            )

    def inject_error(self) -> None:
        if self._faults.random() < self.config.error_rate:
            if self._faults.random() < 0.5:
                raise ApiError(500, "Injected internal error.", error_type="server_error")
            raise ApiError(
                503, "Injected overload.", error_type="server_error", code="service_unavailable"
            )

    def abort_index(self, num_tokens: int) -> int | None:
        """Index of the token before which an injected abort drops the stream."""
        if self._faults.random() < self.config.abort_rate:
            return self._faults.randrange(num_tokens)
        return None

    def generate(
        self,
        prompt: str,
        *,
        question: str,
        max_tokens: int | None,
        request: _GenerateRequest,
        tool: FunctionDef | None = None,
        continuation: bool = False,
    ) -> Generation:
        cfg = self.config
        pieces = tuple(TOKENIZER.pieces(prompt))
        if max_tokens is None:
            max_tokens = cfg.max_model_len - len(pieces)
        if max_tokens < 1 or len(pieces) + max_tokens > cfg.max_model_len:
            raise ApiError(
                400,
                f"This model's maximum context length is {cfg.max_model_len} tokens. "
                f"However, you requested {len(pieces) + max_tokens} tokens "
                f"({len(pieces)} in the prompt, {max_tokens} in the completion).",
                param="max_tokens",
                code="context_length_exceeded",
            )
        if request.min_tokens > max_tokens:
            raise ApiError(400, "min_tokens must not exceed max_tokens.", param="min_tokens")

        text = self._natural_text(prompt, question, request.response_format, tool)
        if continuation:
            # Completions continue the prompt; the space joins the first token, adding none.
            text = " " + text
        output = fit_output(
            text,
            seed=cfg.seed,
            prompt=prompt,
            max_tokens=max_tokens,
            min_tokens=request.min_tokens,
            ignore_eos=request.ignore_eos,
        )
        if output.hit_length:
            finish_reason = "length"
        else:
            finish_reason = "tool_calls" if tool is not None else "stop"
        return Generation(pieces, output, finish_reason)

    def _natural_text(
        self,
        prompt: str,
        question: str,
        response_format: ResponseFormat | None,
        tool: FunctionDef | None,
    ) -> str:
        seed, degrade = self.config.seed, self.config.degrade
        if tool is not None:
            return json_text(
                tool.parameters, seed=seed, prompt=prompt, degrade=degrade, purpose="tool"
            )
        if response_format is not None and response_format.type != "text":
            spec = response_format.json_schema
            schema = spec.schema_ if response_format.type == "json_schema" and spec else None
            return json_text(schema, seed=seed, prompt=prompt, degrade=degrade, purpose="json")
        answer = arithmetic_answer(question, seed=seed, prompt=prompt, degrade=degrade)
        if answer is not None:
            return answer
        return free_text(seed=seed, prompt=prompt, mean_tokens=self.config.mean_output_tokens)

    def submit(self, gen: Generation) -> SimSequence:
        return self.engine.add(
            SimRequest(
                prompt_tokens=gen.prompt_tokens,
                output_tokens=gen.completion_tokens,
                block_hashes=prompt_block_hashes(gen.prompt_pieces, self.config.block_size),
                finish_reason=gen.finish_reason,
            )
        )

    async def run_to_completion(self, seq: SimSequence) -> None:
        try:
            async for _ in self.engine.tokens(seq):
                pass
        finally:
            self.engine.abort(seq)

    def logprobs(
        self, gen: Generation, *, k: int, include_prompt: bool = False
    ) -> list[TokenLogprob | None]:
        """Per-token logprobs for the output, preceded by the prompt's when asked
        (the first prompt token has none, as in vLLM)."""
        start = 1 if include_prompt else gen.prompt_tokens
        entries: list[TokenLogprob | None] = [None] if include_prompt else []
        entries += token_logprobs(
            [*gen.prompt_pieces, *gen.output.pieces],
            seed=self.config.seed,
            start=start,
            k=k,
            noise=self.config.logprob_noise,
        )
        return entries


def _usage(requests: Iterable[tuple[Generation, SimSequence]]) -> dict[str, Any]:
    prompt = completion = cached = 0
    for gen, seq in requests:
        prompt += gen.prompt_tokens
        completion += gen.completion_tokens
        cached += seq.num_cached_tokens or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def _chat_logprob(entry: TokenLogprob) -> dict[str, Any]:
    return {
        "token": entry.token,
        "logprob": entry.logprob,
        "bytes": list(entry.token.encode()),
        "top_logprobs": [
            {"token": t, "logprob": lp, "bytes": list(t.encode())} for t, lp in entry.top
        ],
    }


def _completion_logprobs(
    tokens: list[str], entries: list[TokenLogprob | None], offset: int
) -> dict[str, Any]:
    text_offset = []
    for token in tokens:
        text_offset.append(offset)
        offset += len(token)
    return {
        "tokens": tokens,
        "token_logprobs": [e.logprob if e else None for e in entries],
        # vLLM returns the top-k plus the sampled token.
        "top_logprobs": [{**dict(e.top), e.token: e.logprob} if e else None for e in entries],
        "text_offset": text_offset,
    }


def _select_tool(req: ChatCompletionRequest) -> FunctionDef | None:
    """The mock always calls a tool when tools are offered, unless told not to."""
    if not req.tools or req.tool_choice == "none":
        return None
    if isinstance(req.tool_choice, dict):
        name = req.tool_choice.get("function", {}).get("name")
        for tool in req.tools:
            if tool.function.name == name:
                return tool.function
        raise ApiError(400, f"tool_choice names unknown tool {name!r}.", param="tool_choice")
    return req.tools[0].function


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message_text(message.model_dump())
    return ""


async def _parse[T: BaseModel](request: Request, model: type[T]) -> T:
    try:
        body = await request.json()
    except ValueError as e:
        raise ApiError(400, "Request body is not valid JSON.") from e
    try:
        return model.model_validate(body)
    except ValidationError as e:
        err = e.errors()[0]
        param = ".".join(str(p) for p in err["loc"]) or None
        raise ApiError(400, f"{param}: {err['msg']}", param=param) from e


# ---------------------------------------------------------------- handlers


@dataclass(frozen=True, slots=True)
class _Envelope:
    """Fields shared by every body or chunk of one response."""

    id: str
    object: str
    model: str
    created: int = field(default_factory=lambda: int(time.time()))

    def body(self, choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": choices,
            **extra,
        }

    def sse(self, choices: list[dict[str, Any]], **extra: Any) -> str:
        return f"data: {json.dumps(self.body(choices, **extra), separators=(',', ':'))}\n\n"


_DONE = "data: [DONE]\n\n"


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def _chat_completions(server: MockServer, request: Request) -> Response:
    server.check_ready()
    req = await _parse(request, ChatCompletionRequest)
    server.check_model(req.model)
    tool = _select_tool(req)
    tools = [t.model_dump(by_alias=True, exclude_none=True) for t in req.tools or []]
    prompt = render_chat_prompt([m.model_dump(exclude_none=True) for m in req.messages], tools)
    gen = server.generate(
        prompt,
        question=_last_user_text(req.messages),
        max_tokens=req.max_completion_tokens or req.max_tokens,
        request=req,
        tool=tool,
    )
    logprobs = server.logprobs(gen, k=req.top_logprobs or 0) if req.logprobs else None
    server.inject_error()
    seq = server.submit(gen)
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    call_id = f"call_{uuid.uuid4().hex[:24]}"

    if not req.stream:
        await server.run_to_completion(seq)
        message: dict[str, Any] = {"role": "assistant", "content": gen.output.text}
        if tool is not None:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call(call_id, tool.name, gen.output.text)],
            }
        choice = {
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": gen.finish_reason,
        }
        if logprobs is not None:
            choice["logprobs"] = {"content": [_chat_logprob(e) for e in logprobs if e]}
        envelope = _Envelope(response_id, "chat.completion", req.model)
        return JSONResponse(envelope.body([choice], usage=_usage([(gen, seq)])))

    envelope = _Envelope(response_id, "chat.completion.chunk", req.model)
    pieces = gen.output.pieces
    abort_at = server.abort_index(len(pieces))

    def chunk(delta: dict[str, Any], **extra: Any) -> str:
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": None, **extra}
        return envelope.sse([choice])

    async def events() -> AsyncIterator[str]:
        try:
            async for n in server.engine.tokens(seq):
                i = n - 1
                if i == abort_at:
                    raise StreamAborted
                if i == 0:
                    # vLLM sends the role with the first token, not before prefill.
                    if tool is None:
                        yield chunk({"role": "assistant", "content": ""})
                    else:
                        header = {"index": 0, **_tool_call(call_id, tool.name, "")}
                        yield chunk({"role": "assistant", "content": None, "tool_calls": [header]})
                if tool is None:
                    delta: dict[str, Any] = {"content": pieces[i]}
                else:
                    delta = {"tool_calls": [{"index": 0, "function": {"arguments": pieces[i]}}]}
                entry = logprobs[i] if logprobs is not None else None
                yield chunk(
                    delta,
                    logprobs={"content": [_chat_logprob(entry)]} if entry else None,
                    finish_reason=gen.finish_reason if n == len(pieces) else None,
                )
            if req.include_usage:
                yield envelope.sse([], usage=_usage([(gen, seq)]))
            yield _DONE
        finally:
            server.engine.abort(seq)

    return StreamingResponse(events(), media_type="text/event-stream")


async def _completions(server: MockServer, request: Request) -> Response:
    server.check_ready()
    req = await _parse(request, CompletionRequest)
    server.check_model(req.model)
    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    if not prompts:
        raise ApiError(400, "prompt must not be empty.", param="prompt")
    if req.stream and len(prompts) > 1:
        raise ApiError(400, "Streaming supports a single prompt.", param="prompt")
    gens = [
        server.generate(p, question=p, max_tokens=req.max_tokens, request=req, continuation=True)
        for p in prompts
    ]
    k = req.logprobs
    # With echo, logprobs cover the prompt tokens first, then the output tokens.
    logprobs = [
        server.logprobs(g, k=k, include_prompt=req.echo) if k is not None else None for g in gens
    ]
    server.inject_error()
    seqs = [server.submit(g) for g in gens]
    envelope = _Envelope(f"cmpl-{uuid.uuid4().hex}", "text_completion", req.model)

    if not req.stream:
        try:
            await asyncio.gather(*(server.run_to_completion(s) for s in seqs))
        finally:
            for seq in seqs:
                server.engine.abort(seq)
        choices = []
        for i, (prompt, gen, lps) in enumerate(zip(prompts, gens, logprobs, strict=True)):
            tokens = [*gen.prompt_pieces, *gen.output.pieces] if req.echo else [*gen.output.pieces]
            choices.append(
                {
                    "index": i,
                    "text": (prompt if req.echo else "") + gen.output.text,
                    "logprobs": _completion_logprobs(tokens, lps, 0) if lps is not None else None,
                    "finish_reason": gen.finish_reason,
                }
            )
        return JSONResponse(envelope.body(choices, usage=_usage(zip(gens, seqs, strict=True))))

    gen, seq, lps = gens[0], seqs[0], logprobs[0]
    pieces = gen.output.pieces
    abort_at = server.abort_index(len(pieces))
    base = gen.prompt_tokens if req.echo else 0

    async def events() -> AsyncIterator[str]:
        offset = 0
        try:
            async for n in server.engine.tokens(seq):
                i = n - 1
                if i == abort_at:
                    raise StreamAborted
                if i == 0 and req.echo:
                    tokens, first = [*gen.prompt_pieces, pieces[0]], 0
                else:
                    tokens, first = [pieces[i]], base + i
                text = "".join(tokens)
                choice = {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": gen.finish_reason if n == len(pieces) else None,
                }
                if lps is not None:
                    entries = lps[first : first + len(tokens)]
                    choice["logprobs"] = _completion_logprobs(tokens, entries, offset)
                yield envelope.sse([choice])
                offset += len(text)
            if req.include_usage:
                yield envelope.sse([], usage=_usage([(gen, seq)]))
            yield _DONE
        finally:
            server.engine.abort(seq)

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------------------------------------------------------------- app


def create_app(config: MockConfig | None = None) -> FastAPI:
    config = config or MockConfig()
    server = MockServer(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        server.engine.start()
        server.ready_at = time.monotonic() + config.startup_delay_s * config.time_scale
        yield
        await server.engine.stop()

    app = FastAPI(title="Loom mock backend", version=__version__, lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(ApiError(exc.status_code, str(exc.detail)))

    @app.get("/health")
    async def health() -> Response:
        server.check_ready()
        return Response(status_code=200)

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"version": __version__}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "loom-mock",
                    "max_model_len": config.max_model_len,
                }
                for model in config.models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _chat_completions(server, request)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _completions(server, request)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            render_metrics(server.engine.sim, config.models[0]), media_type=CONTENT_TYPE
        )

    return app
