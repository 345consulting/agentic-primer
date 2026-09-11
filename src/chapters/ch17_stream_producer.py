# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""the harness becomes the server.

Every chapter until now called a model and, eventually, printed or rendered
what came back. This one is the harness on the other side of a request: an
ASGI app that runs an agent's turn loop and streams what happens as it
happens, to whatever is on the other end of the connection.

**The mechanism is generic; the payload is not.** An ASGI app and
Server-Sent Events are plain web serving, the same shape a thousand
non-agentic services already use. What is specific to this ladder is what
goes out over that connection: not bytes of a reply arriving (`ch14`'s
subject), but *lifecycle events* -- `turn_started`, `model_replied`,
`tool_dispatched`, `tool_completed`, `turn_ended`, `run_ended` -- the same
named points a hook would attach to, reused here as a report to a client
instead of a control point in the loop.

**A hand-written `emit`, not `ch18`'s hook mechanism.** `support/hooks.py`'s
`BeforeHook`/`AfterHook` are shaped for exactly one thing: a tool call, in
and a result, out, with a verdict that can modify or veto it. `turn`,
`model` and `run` have no hook point built anywhere in this ladder --
`ch18`'s own docstring says they "get the same treatment only when a later
chapter needs one." This is that chapter. Importing half of `ch18`'s
mechanism for the two events that fit (`tool_dispatched`/`tool_completed`)
and hand-rolling the other four would teach a seam that isn't really there;
writing `emit` once, plainly, for all six keeps the one thing this chapter
adds -- observe, and only observe, at named points -- in one place.

**A frame only reports what already finished.** `model_replied` does not
fire as the model's stream produces chunks (`ch14`/`ch15`'s subject) -- it
fires once that stream reports done, carrying the whole reply as one block.
The reason is the consumer, not the mechanism: a client watching a
"thinking" indicator wants a complete thought, not a word at a time
re-assembled on their end -- this chapter converts a stream meant for
progressive rendering into a stream meant for status reporting, and the two
have different units. `tool_dispatched`/`tool_completed` need no such
gating -- a tool call is not a stream, dispatch and completion are already
two distinct moments in `execute_tool`, so the frames just name a boundary
that already exists.

**The event stream is a curated subset of the trace, not another view of
it.** `Trace` keeps recording everything, unaffected -- context payloads,
wire frames, every note -- because deleting `emit` from this chapter would
change nothing about what the trace already does. That is `ch16`'s own
finding about instrumentation, extended to a live consumer: a client
watching a run's progress is not auditing it, and the frames it receives
are deliberately smaller than what `Trace` records for exactly that reason.

**The agent's loop is not the ASGI connection.** A disconnected client is a
fact reported by `receive()`, not a reason to stop running -- the loop
finishes on its own terms (the model asking for nothing further, same as
every chapter since `ch03`), and a gone client just means nobody was
listening for the frames that kept being offered. `emit` checks a
connection flag and skips the `send()` call once it's false; nothing about
the loop itself changes.

Five scenarios:

    the_harness_streams_lifecycle_events_as_an_asgi_app
        an in-process ASGI call, scope/receive/send driven directly -- the
        client receives five distinct frames as the run actually
        progresses: started, turn started, model replied, turn ended, run
        ended
    the_event_stream_is_smaller_than_the_trace
        the same run, Trace and the SSE frames both produced -- the trace
        holds far more than the frame count, proof this is a curation and
        not the same audit wearing SSE framing
    a_streamed_reply_arrives_as_one_complete_frame_not_pieces
        the model streams its reply chunk by chunk underneath -- exactly
        one model_replied frame reaches the client, gated on the
        provider's own completion signal, carrying the whole content
    tool_dispatched_and_tool_completed_are_two_frames_not_one
        a real tool call -- dispatch and completion were already two
        distinct moments in execute_tool; this makes them two frames
    a_client_that_disconnects_mid_run_does_not_crash_the_loop
        receive() reports the client gone partway through -- the loop
        keeps running to its own completion; it only stops trying to
        send frames nobody is reading
"""

from support.agent import execute_tool
from support.models import build_model
from support.trace import ModelKind, Span, Trace

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool, tool

type Json = dict[str, Any]
type Send = Callable[[Json], Awaitable[None]]
type Receive = Callable[[], Awaitable[Json]]


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


@dataclass
class Connection:
    """Whether anyone is still listening. A field on an object rather than
    a bare variable because `emit` is called many times across one run and
    each call has to see the *current* state -- a plain bool reassigned
    inside `_watch_for_disconnect` would rebind a local, not the one `emit`
    already captured.
    """

    connected: bool = True


def _frame(event: str, **fields: Any) -> bytes:
    """One SSE frame: `data: <json>`, blank-line terminated -- the format
    any SSE client already knows how to split on, same framing `ch14`'s
    own `WIRE_FRAME` notes described from the other side of a connection.
    """
    return f"data: {json.dumps({'event': event, **fields})}\n\n".encode()


async def _emit(
    send: Send, connection: Connection, record_on: Span, event: str, **fields: Any
) -> None:
    """The one place a lifecycle fact becomes a frame -- observe only,
    same as `ch18`'s safest power: this never changes what the loop does
    next, whether or not anyone receives it.

    `record_on` is the span this send actually belongs to -- `scenario`
    for `run_started`/`run_ended`, `turn` for everything in between -- so
    the trace shows exactly where each frame went out, not just a final
    list of event names with no position in the run at all. Recorded even
    when the client is gone (`connection.connected` is false): the harness
    still reached this point and still tried, which is the fact this
    chapter's fifth scenario needs to be able to show.
    """
    record_on.add_note("sse_frame_sent", event=event, **fields)
    if not connection.connected:
        return
    await send({"type": "http.response.body", "body": _frame(event, **fields), "more_body": True})


async def _watch_for_disconnect(receive: Receive, connection: Connection) -> None:
    """A disconnect is a message from `receive()`, not an exception from
    `send()` -- ASGI reports it the same way it reports anything else, so
    it is only noticed by asking. Called once between named points rather
    than awaited concurrently with the loop -- this chapter has no reason
    to run a receive loop alongside the agent's own.
    """
    message = await receive()
    if message["type"] == "http.disconnect":
        connection.connected = False


async def _run_agent_over_sse(
    scope: Json,
    receive: Receive,
    send: Send,
    *,
    trace: Trace,
    scenario_span: Span,
    model_kind: ModelKind,
    mock_replies: list[AIMessage],
    question: str,
    turn_cap: int,
    tools: Sequence[BaseTool] = (),
) -> tuple[int, int, str]:
    """One ASGI request, one run -- `ch03_the_loop`'s loop, hand-written
    again for the same reason `ch14` rewrote a model call for streaming:
    what changes here (a frame goes out as each point is reached, not
    after the whole run finishes) is this chapter's actual subject, and
    composing `run_turns` unchanged would have nothing to show.

    Returns turns, messages and ending, the same shape `run_turns` returns
    -- this chapter's addition is what happens *alongside* that return,
    not a different result.
    """
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    connection = Connection()
    await _watch_for_disconnect(receive, connection)
    await _emit(send, connection, scenario_span, "run_started")

    messages: list[BaseMessage] = [HumanMessage(question)]
    turns = 0
    ended = "turns_exhausted"

    while turns < turn_cap:
        turns += 1
        with trace.span("turn", number=turns) as turn_span:
            await _emit(send, connection, turn_span, "turn_started", number=turns)

            with trace.span("model", model_kind=model_kind) as span:
                span.add_context(messages, tools)
                # Bound unconditionally, even with an empty tool list --
                # `ch15_stream_tools` always binds too, and a ternary between
                # a bound and an unbound model here would type `.stream()` as
                # `BaseMessage`, not `AIMessageChunk`, the same way `ask_model`
                # avoids it by never streaming at all.
                model = build_model(model_kind, mock_replies[turns - 1 :], span).bind_tools(
                    list(tools)
                )
                chunks: list[AIMessageChunk] = []
                for chunk in model.stream(messages):
                    assert isinstance(chunk, AIMessageChunk)
                    chunks.append(chunk)
                accumulated = chunks[0]
                for piece in chunks[1:]:
                    accumulated = accumulated + piece
                reply = AIMessage(
                    content=accumulated.content,
                    id=accumulated.id,
                    tool_calls=accumulated.tool_calls,
                    response_metadata=accumulated.response_metadata,
                    usage_metadata=accumulated.usage_metadata,
                )
                span.add_reply(reply)
            messages.append(reply)

            await _watch_for_disconnect(receive, connection)
            await _emit(
                send,
                connection,
                turn_span,
                "model_replied",
                turn=turns,
                content=str(reply.content),
                tool_calls=bool(reply.tool_calls),
            )

            if not reply.tool_calls:
                ended = "no_tool_calls"
                await _emit(send, connection, turn_span, "turn_ended", number=turns)
                break

            for call in reply.tool_calls:
                await _watch_for_disconnect(receive, connection)
                await _emit(
                    send,
                    connection,
                    turn_span,
                    "tool_dispatched",
                    name=call["name"],
                    args=call["args"],
                )
                messages.append(
                    execute_tool(call, trace, lambda _n, args, _s: check_order.invoke(args))
                )
                await _watch_for_disconnect(receive, connection)
                await _emit(send, connection, turn_span, "tool_completed", name=call["name"])

            await _emit(send, connection, turn_span, "turn_ended", number=turns)

    await _emit(send, connection, scenario_span, "run_ended", turns=turns, ended=ended)
    await send({"type": "http.response.body", "body": b"", "more_body": False})
    return turns, len(messages), ended


async def _fake_receive() -> Json:
    """A client that sent its request and is only listening from here on."""
    return {"type": "http.request", "body": b"", "more_body": False}


def _disconnecting_receive(after: int) -> Receive:
    """A client gone after `after` checks -- `nonlocal` over a closure
    variable rather than a class, because this exists only to be handed
    straight to `_run_agent_over_sse` and read nowhere else.
    """
    calls = 0

    async def receive() -> Json:
        nonlocal calls
        calls += 1
        if calls > after:
            return {"type": "http.disconnect"}
        return await _fake_receive()

    return receive


def _collector() -> tuple[Send, list[Json]]:
    """A `send` that decodes every frame it's given into the list a
    scenario asserts against, instead of writing to a real socket.
    """
    frames: list[Json] = []

    async def send(message: Json) -> None:
        if message["type"] == "http.response.body" and message["body"]:
            for line in message["body"].decode().strip().split("\n\n"):
                frames.append(json.loads(line.removeprefix("data: ")))

    return send, frames


def the_harness_streams_lifecycle_events_as_an_asgi_app(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Five frames, in order, as an ordinary single-turn run progresses."""
    send, frames = _collector()
    with trace.span("scenario", name="the_harness_streams_lifecycle_events_as_an_asgi_app") as span:
        turns, messages, ended = asyncio.run(
            _run_agent_over_sse(
                {"type": "http"},
                _fake_receive,
                send,
                trace=trace,
                scenario_span=span,
                model_kind=model_kind,
                mock_replies=[AIMessage("Milk is 1.20.")],
                question="How much is milk?",
                turn_cap=1,
            )
        )
        span.add_note(
            "five_frames_arrived_in_order",
            events=[f["event"] for f in frames],
            turns=turns,
            messages=messages,
            ended=ended,
        )


def the_event_stream_is_smaller_than_the_trace(trace: Trace, model_kind: ModelKind) -> None:
    """A run with a real tool call -- the trace records more than the
    client ever receives, because the frames are a curation, not a mirror.

    Found by actually counting, not assumed: a single plain turn does not
    make this point. `Trace`'s own bookkeeping is efficient -- one span
    object covers both a start and an end -- while the frame vocabulary
    pays for an explicit start *and* end frame at both the run and turn
    level, for symmetry a client needs and a span object gets for free. At
    that trivial scale the frame count comes out even with the trace, not
    under it. It is the payloads -- context, reply, tool args, tool result,
    each one a note the client is never sent -- that make the trace grow
    faster than the frame count as a run does more, which is why this
    scenario uses the same two-turn tool call as
    `tool_dispatched_and_tool_completed_are_two_frames_not_one` rather
    than the simplest possible run.
    """

    def entries_recorded() -> int:
        # Spans and their notes together -- the actual size of what Trace
        # holds, not just how many spans were opened. A span count alone
        # would undercount by exactly the payloads (context, reply, args,
        # result) that make Trace an audit and not a shape.
        return sum(1 + len(s.notes) for _, s in trace.walk())

    call = tool_call(name="check_order", args={"order_id": "A100"}, id="c1")
    send, frames = _collector()
    with trace.span("scenario", name="the_event_stream_is_smaller_than_the_trace") as span:
        before = entries_recorded()
        asyncio.run(
            _run_agent_over_sse(
                {"type": "http"},
                _fake_receive,
                send,
                trace=trace,
                scenario_span=span,
                model_kind=model_kind,
                mock_replies=[
                    AIMessage("", tool_calls=[call]),
                    AIMessage("Order A100: shipped."),
                ],
                question="Check order A100.",
                turn_cap=2,
                tools=[check_order],
            )
        )
        trace_entries_added = entries_recorded() - before
        span.add_note(
            "the_stream_is_curated_not_mirrored",
            frame_count=len(frames),
            trace_entries_added=trace_entries_added,
            frames_are_fewer=len(frames) < trace_entries_added,
        )


def a_streamed_reply_arrives_as_one_complete_frame_not_pieces(
    trace: Trace, model_kind: ModelKind
) -> None:
    """The mock model's own `.stream()` yields this reply word by word
    (`ch14`'s own mechanism) -- exactly one `model_replied` frame reaches
    the client regardless, carrying the whole sentence.
    """
    send, frames = _collector()
    name = "a_streamed_reply_arrives_as_one_complete_frame_not_pieces"
    with trace.span("scenario", name=name) as span:
        asyncio.run(
            _run_agent_over_sse(
                {"type": "http"},
                _fake_receive,
                send,
                trace=trace,
                scenario_span=span,
                model_kind=model_kind,
                mock_replies=[AIMessage("Milk is one pound twenty, and butter is two pounds.")],
                question="How much is milk and butter?",
                turn_cap=1,
            )
        )
        replied = [f for f in frames if f["event"] == "model_replied"]
        span.add_note(
            "exactly_one_complete_frame",
            model_replied_frame_count=len(replied),
            content=replied[0]["content"] if replied else None,
        )


def tool_dispatched_and_tool_completed_are_two_frames_not_one(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A real tool call -- dispatch and completion were already two
    separate moments in `execute_tool`; this makes them two frames.
    """
    call = tool_call(name="check_order", args={"order_id": "A100"}, id="c1")
    name = "tool_dispatched_and_tool_completed_are_two_frames_not_one"
    send, frames = _collector()
    with trace.span("scenario", name=name) as span:
        asyncio.run(
            _run_agent_over_sse(
                {"type": "http"},
                _fake_receive,
                send,
                trace=trace,
                scenario_span=span,
                model_kind=model_kind,
                mock_replies=[
                    AIMessage("", tool_calls=[call]),
                    AIMessage("Order A100: shipped."),
                ],
                question="Check order A100.",
                turn_cap=2,
                tools=[check_order],
            )
        )
        events = [f["event"] for f in frames]
        span.add_note(
            "dispatch_and_completion_are_distinct",
            tool_dispatched_count=events.count("tool_dispatched"),
            tool_completed_count=events.count("tool_completed"),
            order=events,
        )


def a_client_that_disconnects_mid_run_does_not_crash_the_loop(
    trace: Trace, model_kind: ModelKind
) -> None:
    """The client vanishes after the second `receive()` check -- partway
    through the tool call -- and the loop finishes anyway, on its own
    terms, exactly as if someone were still listening.
    """
    call = tool_call(name="check_order", args={"order_id": "A100"}, id="c1")
    name = "a_client_that_disconnects_mid_run_does_not_crash_the_loop"
    send, frames = _collector()
    with trace.span("scenario", name=name) as span:
        turns, _messages, ended = asyncio.run(
            _run_agent_over_sse(
                {"type": "http"},
                _disconnecting_receive(after=2),
                send,
                trace=trace,
                scenario_span=span,
                model_kind=model_kind,
                mock_replies=[
                    AIMessage("", tool_calls=[call]),
                    AIMessage("Order A100: shipped."),
                ],
                question="Check order A100.",
                turn_cap=2,
                tools=[check_order],
            )
        )
        span.add_note(
            "the_loop_finished_though_nobody_was_listening",
            turns_completed=turns,
            ended=ended,
            frames_actually_delivered=len(frames),
            loop_ran_to_completion=ended == "no_tool_calls" and turns == 2,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch17_stream_producer", model_kind=model_kind)

    the_harness_streams_lifecycle_events_as_an_asgi_app(trace, model_kind)
    the_event_stream_is_smaller_than_the_trace(trace, model_kind)
    a_streamed_reply_arrives_as_one_complete_frame_not_pieces(trace, model_kind)
    tool_dispatched_and_tool_completed_are_two_frames_not_one(trace, model_kind)
    a_client_that_disconnects_mid_run_does_not_crash_the_loop(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
