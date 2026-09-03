# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

""" "stream": true -- a reply arrives in pieces.

Every request until now carried `"stream": false`, implicitly -- nothing
ever set it, and `model.invoke(messages)` waits for the whole reply before
returning anything. This chapter is the first to call `model.stream(...)`
instead, and the one thing every chapter before it assumed quietly stops
being true: a reply is a sequence now, not an object.

**`stream` groups the pieces, nested inside `model`.** Not a span per chunk
-- a chunk has no duration of its own, nothing nested inside it, so it fits
`Span.add_note` the way `WIRE_REQUEST`/`WIRE_RESPONSE` already do, each one
timestamped the same way. One `stream` span holds all of them, so they read
as a group on the page instead of interleaving with `context` and `reply`.

**`model_provider_ms` was one number; a stream needs two.** `first_token_ms`
is how long before anything arrived at all -- the number a user actually
waits on before seeing a response start. `last_token_ms` is how long the
whole thing took, the number `model_provider_ms` already reported for a
non-streaming call. Both are computed the same way `model_provider_ms`
always has been: the difference between two recorded timestamps, here the
first and last `chunk` notes instead of one request/response pair.

**Accumulation, not parsing.** `AIMessageChunk` supports `+`, and summing
every chunk produces the same `AIMessage` `.invoke()` would have returned in
one piece -- `reply.content` is checkable against the same scripted answer
either way. That equality is this chapter's whole claim: streaming changes
how the answer arrives, not what it says.

**`MockModel.stream()` chunks a scripted reply by word,** because there is
no real generation happening to chunk for real -- the mock's job, as
everywhere else, is a trace shaped like a live one without a network call.
The live column is where the interesting case actually lives: a real server
picks its own chunk boundaries, mid-word or mid-token, in a way no script
would ever produce, which the primer's own gotchas already name -- anything
provable only under mock was the mock's cooperation, not the framework's
guarantee.

Tool calls are out of scope here on purpose. `ch15_stream_tools` is where a
call's own arguments arrive as fragments of a JSON string, unusable until
the stream ends -- a different shape of problem from splitting plain text,
and this chapter keeps the two apart the way `ch04` kept failure classes
apart before combining them.

**The first live run's timing was a lie, and the cause was already named in
this primer's own gotchas.** `build_wire_hooks`' response hook called
`.read()` unconditionally -- correct for every chapter before this one, and
wrong here: `.read()` blocks until the whole body has arrived, so it forced
an entire SSE reply to buffer before this chapter's loop ever got to iterate
it. The first run recorded fifty-odd chunks nine milliseconds apart --
not real arrival times, just how fast Python can loop over memory that was
already there. `first_token_ms` and `last_token_ms` were both reporting
roughly the same number, which was the tell: two genuinely different
latencies do not coincidentally collapse into one unless something upstream
already did the waiting for you.

The fix checks `content-type` and skips `.read()` for `text/event-stream`
specifically -- status and headers are available the moment the response
line is, streamed or not, so only the body recording needed to change.
Ordinary chapters are untouched; a regression test in
`tests/support/test_models.py` asserts `.read()` is still called for them
and is not called for a stream. Rerun after the fix: `first_token_ms=505`,
`last_token_ms=1474` -- a real split, because nothing had already consumed
the answer before the loop got to it.
"""

from support.models import build_model
from support.trace import CHUNK, ENDED, ModelKind, Trace

from time import monotonic

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

SYSTEM_PROMPT = "You answer briefly and plainly."

QUESTION = "How much is milk?"

MOCK_REPLY = AIMessage("Milk is 1.20.")


def ask_streaming(trace: Trace, model_kind: ModelKind) -> AIMessage:
    """One call, its reply accumulated from pieces instead of returned whole."""
    with trace.span("model", model_kind=model_kind) as span:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(QUESTION)]
        span.add_context(messages)
        model = build_model(model_kind, [MOCK_REPLY], span)

        with trace.span("stream") as stream_span:
            requested_at = monotonic() * 1000
            chunks: list[AIMessageChunk] = []
            timestamps: list[float] = []
            for chunk in model.stream(messages):
                at_ms = monotonic() * 1000
                stream_span.add_note(CHUNK, text=chunk.content, at_ms=at_ms)
                chunks.append(chunk)
                timestamps.append(at_ms)
            accumulated = chunks[0]
            for piece in chunks[1:]:
                accumulated = accumulated + piece
            # The split `model_provider_ms` cannot make: how long before
            # anything arrived, versus how long the whole reply took.
            stream_span.add_note(
                "streamed",
                chunks=len(chunks),
                first_token_ms=timestamps[0] - requested_at,
                last_token_ms=timestamps[-1] - requested_at,
            )

        # `AIMessageChunk.type` is `"AIMessageChunk"`, not `"ai"` -- accurate
        # for a chunk, wrong for the page, which reads `message.type` as the
        # role. The accumulated result is the answer now, recorded the same
        # shape every other chapter's reply has been.
        reply = AIMessage(
            content=accumulated.content,
            id=accumulated.id,
            tool_calls=accumulated.tool_calls,
            response_metadata=accumulated.response_metadata,
            usage_metadata=accumulated.usage_metadata,
        )
        span.add_reply(reply)
    return reply


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch14_stream", model_kind=model_kind)

    with trace.span("scenario", name="the_reply_arrives_in_pieces") as span:
        reply = ask_streaming(trace, model_kind)
        span.add_note(ENDED, reason=str(reply.content))

    trace.close(turns=1, messages=2, ended="no_tool_calls")
    return trace
