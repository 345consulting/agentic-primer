# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The seam between a mock model and a real one.

Scaffolding, never a lesson. Every chapter takes its model as an argument,
so the only difference between a mock run and a live run is this function.

The mock model replies from a fixed list. That makes a chapter's trace
identical on every run, which is what lets the mechanics be read rather than
guessed at. A live run proves the same mechanics against a real model
provider, and shows you which parts of the trace were the mock's cooperation.
"""

from support.trace import WIRE_FRAME, WIRE_REQUEST, WIRE_RESPONSE, ModelKind, Span

import json
import os
from collections.abc import Callable, Iterator, Sequence
from itertools import count
from time import monotonic
from typing import Any, Self, override

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_deepseek import ChatDeepSeek

# Cheap enough that running a chapter live is routine rather than an event.
# `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` were both retired
# and consolidated into this name, backed by DeepSeek-V4.1-Flash.
LIVE_MODEL = "deepseek-flash"

LIVE_KEY = "DEEPSEEK_API_KEY"

# Parameters the client will send only if we set them. Anything absent from the
# request body was chosen by the model provider, not by us -- which means the run is
# not reproducible from the record alone.
CONFIGURABLE = ("temperature", "max_tokens", "top_p", "stop", "seed", "tools", "tool_choice")

# Headers whose value is a credential wherever it appears. Never recorded, in
# either direction, whatever else the rules below allow.
SECRET_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization", "set-cookie"})

# Request header values are recorded only for names named here, and the check
# is an allowlist rather than a denylist on purpose. A denylist has to be right
# about every auth header that will ever exist -- `authorization` here,
# `x-api-key` at Anthropic, `api-key` at Azure, whatever a proxy adds -- and
# when it is wrong a live key lands in a file on disk, silently and for good.
# An allowlist that is wrong records nothing. That is the direction to be wrong
# in, and it is why the response side, which guards the model provider's secrets
# rather than ours, is allowed the weaker rule.
RECORDED_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "connection",
        "content-length",
        "content-type",
        "host",
        "user-agent",
    }
)

# The OpenAI SDK's generated-client telemetry: os, arch, runtime, package
# version, retry count. Worth recording because it is the answer to how plain
# a "plain" request really is -- eleven of chapter 1's eighteen headers are a
# fingerprint of the machine that sent it. No credential has this prefix, and
# the family grows, so it is matched by prefix rather than enumerated.
RECORDED_REQUEST_PREFIX = "x-stainless-"


# How many pieces a mocked tool call's arguments are split into. Three is
# enough for a fragment to be visible mid-stream and for the last one to
# complete it -- `ch15_stream_tools` is what that middle state is for.
MOCK_ARGUMENT_PIECES = 3


def _chunked_tool_calls(reply: AIMessage) -> Iterator[ChatGenerationChunk]:
    """A reply's tool calls, with each call's arguments split mid-JSON.

    A real provider streams a call's arguments as fragments of the JSON
    string it is building, and picks its own boundaries -- mid-key, mid-value,
    anywhere. This splits into equal thirds, which is enough to put a
    half-finished argument on the wire; `ch15_stream_tools` is about what a
    harness may and may not do with one.
    """
    for call in reply.tool_calls:
        serialized = json.dumps(call["args"])
        width = max(1, len(serialized) // MOCK_ARGUMENT_PIECES)
        pieces = [serialized[at : at + width] for at in range(0, len(serialized), width)]
        for index, piece in enumerate(pieces):
            first = index == 0
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        tool_call_chunk(
                            # Name and id arrive once, with the first fragment.
                            name=call["name"] if first else None,
                            args=piece,
                            id=call["id"] if first else None,
                            index=0,
                        )
                    ],
                )
            )


class MockModel(BaseChatModel):
    """Replies from a fixed list, one per call, in order.

    It ignores what it is sent. That is deliberate: a chapter about the loop
    should not also be a chapter about prompting.
    """

    replies: Sequence[AIMessage]
    calls: int = 0
    watching: Span | None = None

    @property
    @override
    def _llm_type(self) -> str:
        return "mock"

    @override
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Self:
        """Accept a declaration and ignore it, exactly as it ignores context.

        A real model is told what it may ask for; a mock model already knows what
        it will ask for. Recording the declaration keeps it honest --
        the trace shows what was offered even though nothing consulted it.
        """
        if self.watching is not None:
            self.watching.add_note(
                "tools declared to the mock model",
                names=[getattr(t, "name", str(t)) for t in tools],
                consulted=False,
            )
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.calls >= len(self.replies):
            raise RuntimeError(
                f"mock model exhausted at call ['{self.calls + 1}'] of "
                f"['{len(self.replies)}'] -- the loop ran longer than the mock model expected"
            )
        reply = self.replies[self.calls]
        self.calls += 1
        if self.watching is not None:
            # What the mock model was handed and threw away. Without this the mock
            # column silently implies its reply was responsive to the context;
            # recording it keeps the mock honest about its own limits.
            self.watching.add_note(
                "context ignored by the mock model",
                messages=len(messages),
                roles=[m.type for m in messages],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """The scripted reply, split into words and yielded one at a time.

        `ch14_stream` is the first chapter to call this instead of
        `_generate`. A reply carrying tool calls is out of scope here --
        `ch15_stream_tools` chunks a call's own arguments, a different shape
        from splitting plain text, and gets its own mock support there.
        """
        if self.calls >= len(self.replies):
            raise RuntimeError(
                f"mock model exhausted at call ['{self.calls + 1}'] of "
                f"['{len(self.replies)}'] -- the loop ran longer than the mock model expected"
            )
        reply = self.replies[self.calls]
        self.calls += 1
        if self.watching is not None:
            self.watching.add_note(
                "context ignored by the mock model",
                messages=len(messages),
                roles=[m.type for m in messages],
            )
        if reply.tool_calls:
            yield from _chunked_tool_calls(reply)
            return
        words = str(reply.content).split(" ")
        for index, word in enumerate(words):
            piece = word if index == len(words) - 1 else f"{word} "
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


class ThinkingModel(ChatDeepSeek):
    """`ChatDeepSeek`, with the reasoning it was sent put back on the way out.

    `langchain-deepseek` carries `reasoning_content` inbound, into
    `additional_kwargs` -- verified against the wire, and the subject of the
    finding dated 2026-09-01. It does not carry it back out: the assistant
    message it re-serializes for the next turn has `content`, `role` and
    `tool_calls` and nothing else.

    DeepSeek's thinking mode requires it back, and answers 400 when it is
    missing -- but only sometimes, because whether turn one produced any
    reasoning is the model's choice. A multi-turn tool-calling run therefore
    fails intermittently, which is the worst way for a defect to present.

    This is a workaround for a library defect, not a lesson. It lives here
    because a chapter that carried it would be teaching the shape of someone
    else's bug.
    """

    @override
    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        sent = self._convert_input(input_).to_messages()
        # The payload's messages are positional with the input's, so pair them
        # and restore what the conversion dropped. Only assistant messages that
        # actually carried reasoning are touched.
        for message, serialized in zip(sent, payload.get("messages", []), strict=True):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if isinstance(message, AIMessage) and reasoning:
                serialized["reasoning_content"] = reasoning
        return payload


def build_live_model(span: Span, max_tokens: int | None = None) -> BaseChatModel:
    """A real model provider, with the wire recorded either side of the library.

    Fails loudly when the key is absent -- never a default. The client reads
    the key from the environment itself; we check that it is there and never
    touch the value, because a secret this code never holds is a secret it
    cannot leak into a trace.

    The span is required. What is worth seeing is not the wire format alone
    but what the library changed on the way through, so there is no reason to
    build a live model that records neither.
    """
    if LIVE_KEY not in os.environ:
        raise RuntimeError(
            f"environment variable ['{LIVE_KEY}'] is not set; a live run has no fallback"
        )
    # `max_tokens` is the first of CONFIGURABLE this primer ever sets. Left
    # None it stays absent from the request body and the model provider
    # chooses, which is what `defaulted_by_model_provider` has been reporting
    # since ch01_single_call.
    return ThinkingModel(
        model=LIVE_MODEL,
        timeout=60,
        max_tokens=max_tokens,
        http_client=httpx.Client(
            event_hooks=build_wire_hooks(span), transport=RecordingTransport(span)
        ),
    )


def build_model(
    model_kind: ModelKind,
    mock_model_replies: Sequence[AIMessage],
    span: Span,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """The one switch.

    `mock_model_replies` is what the mock model will say and is ignored when
    live. `span` is where either kind records itself -- the raw HTTP on a live
    run, the thrown-away context on a mock one. Both are required:
    every chapter has both, and a default here would only make it possible to
    build a model that quietly records less than the one beside it.
    """
    if model_kind == "live":
        return build_live_model(span, max_tokens)
    return MockModel(replies=mock_model_replies, watching=span)


# `None` is "this header was present and its value was not recorded" -- which
# is a different fact from the header being absent, and the page says so.
def _record(headers: httpx.Headers, allowed: Callable[[str], bool]) -> dict[str, str | None]:
    """Every header name, with the value only where `allowed` says so.

    `None` means the header was present and its value was withheld, which is a
    different fact from the header being absent -- and the page says so.
    `SECRET_HEADERS` is refused here, before any rule is consulted, so no
    caller can widen its way past a credential.
    """
    return {
        name: headers[name]
        if name.lower() not in SECRET_HEADERS and allowed(name.lower())
        else None
        for name in sorted(headers.keys())
    }


def _request_headers(headers: httpx.Headers) -> dict[str, str | None]:
    """Request values by allowlist: only names this code has heard of.

    A denylist has to be right about every auth header that will ever exist --
    `authorization` here, `x-api-key` at Anthropic, `api-key` at Azure,
    whatever a proxy adds -- and when it is wrong a live key lands in a file on
    disk, silently and for good. An allowlist that is wrong records nothing.
    """
    return _record(
        headers,
        lambda name: name in RECORDED_REQUEST_HEADERS or name.startswith(RECORDED_REQUEST_PREFIX),
    )


def _response_headers(headers: httpx.Headers) -> dict[str, str | None]:
    """Response values by deny-set: everything that is not a credential.

    The weaker rule is deliberate. This side guards the model provider's secrets,
    not ours -- a miss here records DeepSeek's cookie, where a miss on the
    request side records our key. Different blast radius, different default.
    """
    # Every name is allowed; SECRET_HEADERS is refused inside `_record` regardless.
    return _record(headers, lambda _name: True)


def _body(text: str) -> Any:
    """A response body as JSON where it is JSON, and as text where it is not.

    Written against one model provider, this assumed every response was JSON --
    and the first status code with an empty body crashed the run. A recorder
    that only works on the happy path is not a recorder.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class _RecordedStream(httpx.SyncByteStream):
    """A response body that records each frame as it passes, and passes it on.

    An event hook cannot do this: hooks fire once, when the response line
    arrives, and the body has not been sent yet. Reading it there is what
    `on_response` used to do, and it forced the whole stream to buffer before
    anything downstream could iterate it. A stream that records on the way
    through keeps the timing honest and the wire visible at the same time.

    The first version recorded after the `for` loop inside `__iter__`, and
    every live run silently recorded nothing: the OpenAI SDK reads an SSE
    stream until it sees `[DONE]`, then calls `.close()` directly, and never
    drives the generator to a natural `StopIteration`. Code placed after a
    `yield` only runs on that path or on `GeneratorExit`, neither of which
    happened. `close()` is what every caller reliably invokes regardless of
    how the read ends, which is why the accumulator is an instance attribute
    and the note is added there instead.
    """

    def __init__(self, inner: httpx.SyncByteStream, span: Span) -> None:
        self._inner: httpx.SyncByteStream = inner
        self._span: Span = span
        self._parts: list[bytes] = []

    @override
    def __iter__(self) -> Iterator[bytes]:
        for raw in self._inner:
            self._parts.append(raw)
            yield raw

    @override
    def close(self) -> None:
        # Not after the `for` loop in `__iter__`: an SSE client reads until it
        # has seen `[DONE]` and then closes the stream directly, without ever
        # asking this generator for one more value -- code placed after that
        # loop depends on `StopIteration`, which never arrives. `close()` is
        # what every caller reliably invokes, whichever way the read ends, so
        # this is the one place the accumulated frames are guaranteed to land.
        self._inner.close()
        self._span.add_note(
            WIRE_FRAME,
            frames=len(self._parts),
            raw=b"".join(self._parts).decode(errors="replace"),
        )


class RecordingTransport(httpx.BaseTransport):
    """`httpx`'s default transport, with a streamed body teed into the span."""

    def __init__(self, span: Span, inner: httpx.BaseTransport | None = None) -> None:
        self._span: Span = span
        self._inner: httpx.BaseTransport = inner or httpx.HTTPTransport()

    @override
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            assert isinstance(response.stream, httpx.SyncByteStream)
            response.stream = _RecordedStream(response.stream, self._span)
        return response


def build_wire_hooks(span: Span) -> dict[str, list[Any]]:
    """httpx hooks that record the actual bytes, either side of the library.

    Request header values are recorded only for names on an allowlist. The
    Authorization header carries the key and a trace is a file on disk, so the
    rule is not "redact the known secret", it is "record only the known safe"
    -- an auth header this code has never heard of is withheld by default.

    Response header values are recorded, less `SECRET_HEADERS`. They are the
    model provider's own account of the call -- its trace id, its rate limits, what
    cache the answer came out of -- and none of them is a secret of ours.
    """
    attempts = count(1)

    def on_request(request: httpx.Request) -> None:
        body = json.loads(request.content) if request.content else None
        # One `invoke` may be several HTTP calls: the client retries on some
        # failures by itself. Numbering the attempts turns a confusing repeat
        # into a fact -- and you are billed for every one of them.
        span.add_note(
            WIRE_REQUEST,
            attempt=next(attempts),
            at_ms=monotonic() * 1000,
            method=request.method,
            url=str(request.url),
            headers=_request_headers(request.headers),
            defaulted_by_model_provider=[k for k in CONFIGURABLE if not body or k not in body],
            body=body,
        )

    def on_response(response: httpx.Response) -> None:
        # `.read()` blocks until the whole body has arrived -- correct for an
        # ordinary response, wrong for an event stream: it would force the
        # entire reply to buffer before `ch14_stream`'s loop ever gets to
        # iterate it, collapsing every chunk's arrival into one instant.
        # Status and headers are available the moment the response line is,
        # streamed or not; only the body recording waits on `.read()`.
        streaming = response.headers.get("content-type", "").startswith("text/event-stream")
        if not streaming:
            response.read()
        span.add_note(
            WIRE_RESPONSE,
            at_ms=monotonic() * 1000,
            status=response.status_code,
            headers=_response_headers(response.headers),
            body=(
                "(streamed -- see this span's `wire frame` notes for the raw bytes)"
                if streaming
                else _body(response.text)
            ),
        )

    return {"request": [on_request], "response": [on_response]}
