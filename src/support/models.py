# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The seam between a mock model and a real one.

Scaffolding, never a lesson. Every chapter takes its model as an argument,
so the only difference between a mock run and a live run is this function.

The mock model replies from a fixed list. That makes a chapter's trace
identical on every run, which is what lets the mechanics be read rather than
guessed at. A live run proves the same mechanics against a real model
provider, and shows you which parts of the trace were the mock's cooperation.
"""

from support.trace import WIRE_REQUEST, WIRE_RESPONSE, ModelKind, Span

import json
import os
from collections.abc import Callable, Sequence
from itertools import count
from time import monotonic
from typing import Any, Self, override

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek

# Cheap enough that running a chapter live is routine rather than an event,
# and the 0731 retrain is tuned for tool calling -- which is the subject from
# chapter 2 onward.
LIVE_MODEL = "deepseek-v4-flash"

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


def build_live_model(span: Span) -> BaseChatModel:
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
    return ChatDeepSeek(
        model=LIVE_MODEL, timeout=60, http_client=httpx.Client(event_hooks=_build_wire_hooks(span))
    )


def build_model(
    model_kind: ModelKind, mock_model_replies: Sequence[AIMessage], span: Span
) -> BaseChatModel:
    """The one switch.

    `mock_model_replies` is what the mock model will say and is ignored when
    live. `span` is where either kind records itself -- the raw HTTP on a live
    run, the thrown-away context on a mock one. Both are required:
    every chapter has both, and a default here would only make it possible to
    build a model that quietly records less than the one beside it.
    """
    if model_kind == "live":
        return build_live_model(span)
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


def _build_wire_hooks(span: Span) -> dict[str, list[Any]]:
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
        response.read()
        span.add_note(
            WIRE_RESPONSE,
            at_ms=monotonic() * 1000,
            status=response.status_code,
            headers=_response_headers(response.headers),
            body=json.loads(response.text),
        )

    return {"request": [on_request], "response": [on_response]}
