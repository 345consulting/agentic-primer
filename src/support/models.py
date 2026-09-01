"""The seam between a scripted model and a real one.

Scaffolding, never a lesson. Every chapter takes its model as an argument,
so the only difference between a mock run and a live run is this function.

The scripted model replies from a fixed list. That makes a chapter's trace
identical on every run, which is what lets the mechanics be read rather than
guessed at. A live run proves the same mechanics against a real provider --
and shows you which parts of the trace were the mock's cooperation.
"""

from support.trace import Span

import json
import os
from collections.abc import Sequence
from itertools import count
from time import monotonic
from typing import Any, Literal, Self

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek

type Kind = Literal["mock", "live"]

# Cheap enough that running a chapter live is routine rather than an event,
# and the 0731 retrain is tuned for tool calling -- which is the subject from
# chapter 2 onward.
LIVE_MODEL = "deepseek-v4-flash"
LIVE_KEY = "DEEPSEEK_API_KEY"


class Scripted(BaseChatModel):
    """Replies from a fixed list, one per call, in order.

    It ignores what it is sent. That is deliberate: a chapter about the loop
    should not also be a chapter about prompting.
    """

    replies: Sequence[AIMessage]
    calls: int = 0
    watching: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Self:
        """Accept a declaration and ignore it, exactly as it ignores context.

        A real model is told what it may ask for; a script already knows what
        it will ask for. Recording the declaration keeps the mock honest --
        the trace shows what was offered even though nothing consulted it.
        """
        if self.watching is not None:
            self.watching.note(
                "tools declared to the script",
                names=[getattr(t, "name", str(t)) for t in tools],
                consulted=False,
            )
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.calls >= len(self.replies):
            raise RuntimeError(
                f"scripted model exhausted: call {self.calls + 1} of {len(self.replies)} "
                "-- the loop ran longer than the script expected"
            )
        reply = self.replies[self.calls]
        self.calls += 1
        if self.watching is not None:
            # What the script was handed and threw away. Without this the mock
            # column silently implies its reply was responsive to the context;
            # recording it keeps the mock honest about its own limits.
            self.watching.note(
                "context ignored by the script",
                messages=len(messages),
                roles=[m.type for m in messages],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


# Parameters the client will send only if we set them. Anything absent from the
# request body was chosen by the provider, not by us -- which means the run is
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
# in, and it is why the response side, which guards the provider's secrets
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


def _recorded(name: str, values: bool) -> bool:
    """Whether this header's value goes in the record.

    A response records everything but a credential; a request records only
    what is named. Both refuse `SECRET_HEADERS` first, so no later rule can
    widen its way past them.
    """
    if name in SECRET_HEADERS:
        return False
    if values:
        return True
    return name in RECORDED_REQUEST_HEADERS or name.startswith(RECORDED_REQUEST_PREFIX)


# `None` is "this header was present and its value was not recorded" -- which
# is a different fact from the header being absent, and the page says so.
type Headers = dict[str, str | None]


def _headers(headers: httpx.Headers, values: bool) -> Headers:
    return {
        name: headers[name] if _recorded(name.lower(), values) else None
        for name in sorted(headers.keys())
    }


def _wire(span: Span) -> dict[str, list[Any]]:
    """httpx hooks that record the actual bytes, either side of the library.

    Request header values are recorded only for names on an allowlist. The
    Authorization header carries the key and a trace is a file on disk, so the
    rule is not "redact the known secret", it is "record only the known safe"
    -- an auth header this code has never heard of is withheld by default.

    Response header values are recorded, less `SECRET_HEADERS`. They are the
    provider's own account of the call -- its trace id, its rate limits, what
    cache the answer came out of -- and none of them is a secret of ours.
    """
    attempts = count(1)

    def sent(request: httpx.Request) -> None:
        body = json.loads(request.content) if request.content else None
        # One `invoke` may be several HTTP calls: the client retries on some
        # failures by itself. Numbering the attempts turns a confusing repeat
        # into a fact -- and you are billed for every one of them.
        span.note(
            "wire request",
            attempt=next(attempts),
            at_ms=monotonic() * 1000,
            method=request.method,
            url=str(request.url),
            headers=_headers(request.headers, values=False),
            defaulted_by_provider=[k for k in CONFIGURABLE if not body or k not in body],
            body=body,
        )

    def received(response: httpx.Response) -> None:
        response.read()
        span.note(
            "wire response",
            at_ms=monotonic() * 1000,
            status=response.status_code,
            headers=_headers(response.headers, values=True),
            body=json.loads(response.text),
        )

    return {"request": [sent], "response": [received]}


def live(wire: Span | None = None) -> BaseChatModel:
    """A real provider. Fails loudly when the key is absent -- never a default."""
    if LIVE_KEY not in os.environ:
        raise RuntimeError(f"{LIVE_KEY} is not set; a live run has no fallback")
    # The client reads the key from the environment itself. We check that it is
    # there and never touch the value -- a secret this code never holds is a
    # secret it cannot leak into a trace.
    if wire is None:
        return ChatDeepSeek(model=LIVE_MODEL, timeout=60)
    # The same call, with the wire recorded either side of the library. What is
    # worth seeing is not the wire format alone but what the library changed on
    # the way through -- so the trace carries both, in one span.
    return ChatDeepSeek(
        model=LIVE_MODEL, timeout=60, http_client=httpx.Client(event_hooks=_wire(wire))
    )


def model(kind: Kind, replies: Sequence[AIMessage] = (), wire: Span | None = None) -> BaseChatModel:
    """The one switch.

    `replies` is the script and is ignored when live; `wire` records the raw
    HTTP and is ignored when mocked, because a scripted model makes no request.
    """
    if kind == "live":
        return live(wire)
    return Scripted(replies=replies, watching=wire)
