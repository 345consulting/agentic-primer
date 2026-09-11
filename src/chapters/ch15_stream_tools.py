# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Tool arguments arrive as fragments of a JSON string.

`ch14_stream` streamed text, where every piece is useful on its own -- three
tokens of an answer are three tokens you can show someone. A tool call has no
useful partial form. `price_of(item="mi")` is not a partial answer; it is a
call to the wrong thing.

**The danger is not that a fragment fails to parse. It is that it does not.**
LangChain repairs partial JSON as it accumulates, so a half-arrived call still
presents as a well-formed `tool_calls` entry, and `invalid_tool_calls` stays
empty. The mocked call below arrives in fragments and reads, in order:

    raw '{"ite'             tool_calls  price_of()
    raw '{"item": "'        tool_calls  price_of(item="")
    raw '{"item": "milk"'   tool_calls  price_of(item="milk")   json still open
    raw '{"item": "milk"}'  tool_calls  price_of(item="milk")   complete

Four well-formed calls, and only the last one is trustworthy. The third is
right by luck -- the value happened to finish before the JSON did, and
nothing distinguishes it from the second, which was wrong. None flagged. A guard
reading `reply.tool_calls` on each chunk -- the obvious thing to write -- sees
a legitimate-looking call every time and has no signal that it is looking at
half a sentence. That is `ch02_tool_call`'s finding at a new layer: a
plausible value, indistinguishable from a true one, and no error anywhere.

**The signal is in the raw fragment, not the parsed call.** `tool_call_chunks`
keeps the JSON string as it actually arrived, and `json.loads` on it raises
until the last fragment lands. `arguments_are_complete` below is that check,
and it is the only thing in this chapter that can tell the four states apart.

**The live column is worse than the mock, and in a way a script could not
have produced.** DeepSeek streams the same call in twenty-four fragments, and
ten of them are judged before the arguments finish:

    raw ''                  tool_calls  (none)          eleven of these first
    raw ''                  tool_calls  price_of()      a call, before any JSON
    raw '{'                 tool_calls  price_of()
    raw '{"item'            tool_calls  price_of()
    raw '{"item": "'        tool_calls  price_of(item="")
    raw '{"item": "m'       tool_calls  price_of(item="m")
    raw '{"item": "milk'    tool_calls  price_of(item="milk")
    raw '{"item": "milk"}'  tool_calls  price_of(item="milk")   complete

`price_of(item="m")` is the one to sit with. It is a well-formed call, to a
real tool, with a plausible-looking string argument, and it is a call the
model never made. Here the `Literal` enum from `ch02_tool_call` would refuse
it -- `"m"` is not a grocery item -- which is the third time that typing
decision has turned out to be the only thing standing between the model's
output and something running. A tool whose argument is a path, a query or a
customer's name has no enum, and `"m"` would simply be dispatched.

**Which is the conflict `judge` inherits.** Streaming exists to show partial
work as it arrives; a check before dispatch needs the whole call. Both cannot
be satisfied for the same call at the same moment -- a judge that runs per
chunk judges fiction, and a judge that waits has nothing to say until the
stream is over. `ch20_judge` gets to choose; this chapter only shows that the
choice is forced.

The two scenarios are the same stream read two ways -- `judged_on_every_chunk`
checks each fragment as it lands and records what it would have allowed;
`judged_when_complete` waits, and dispatches once. Only the second one calls
the tool, and the trace shows the first one judging arguments that were not
finished arriving, without anything raising.
"""

from support.models import build_model
from support.trace import CHUNK, ENDED, ModelKind, Trace

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

QUESTION = "How much is milk?"

type GroceryItem = Literal["bread", "butter", "chips", "milk"]


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item, in pounds."""
    return f"1.20 for {item}"


TOOLS: dict[str, Any] = {price_of.name: price_of}

MOCK_REPLY = AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}])


def arguments_are_complete(accumulated: AIMessageChunk) -> bool:
    """Whether the raw argument string is finished JSON yet.

    The only check that can tell a fragment from a finished call. Reading
    `accumulated.tool_calls` cannot: the repair makes every state look
    well-formed, including the ones that are not.
    """
    if not accumulated.tool_call_chunks:
        return False
    raw = accumulated.tool_call_chunks[0]["args"] or ""
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def stream_the_call(trace: Trace, model_kind: ModelKind, judge_every_chunk: bool) -> AIMessageChunk:
    """One streamed reply carrying one tool call, checked either per chunk or
    once at the end -- the same stream, read two ways.
    """
    with trace.span("model", model_kind=model_kind) as span:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(QUESTION)]
        span.add_context(messages)
        model = build_model(model_kind, [MOCK_REPLY], span).bind_tools([price_of])

        # An empty chunk to add onto, so there is no `None` state to guard --
        # `+` on two chunks is how a streamed reply is assembled, and starting
        # from empty means the first chunk is not a special case.
        accumulated = AIMessageChunk(content="")
        with trace.span("stream") as stream_span:
            for chunk in model.stream(messages):
                merged = accumulated + chunk
                assert isinstance(merged, AIMessageChunk)
                accumulated = merged
                fragments = accumulated.tool_call_chunks
                raw = fragments[0]["args"] if fragments else ""
                complete = arguments_are_complete(accumulated)
                stream_span.add_note(
                    CHUNK,
                    raw_arguments=raw,
                    # What a check reading `tool_calls` would have seen -- and
                    # it is well-formed at every one of these, not just the last.
                    parsed_as=[call["args"] for call in accumulated.tool_calls],
                    arguments_complete=complete,
                )
                if judge_every_chunk and accumulated.tool_calls:
                    stream_span.add_note(
                        "judged",
                        on=[call["args"] for call in accumulated.tool_calls],
                        # The judge has no idea. Nothing it can read says so.
                        knew_it_was_a_fragment=complete,
                    )

        span.add_reply(AIMessage(content="", tool_calls=accumulated.tool_calls, id=accumulated.id))
    return accumulated


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch15_stream_tools", model_kind=model_kind)
    turns = 0

    with trace.span("scenario", name="judged_on_every_chunk") as span:
        turns += 1
        accumulated = stream_the_call(trace, model_kind, judge_every_chunk=True)
        # Nothing is dispatched: judging a fragment is not grounds to run it.
        # The count is read back from the trace rather than asserted, because
        # how many fragments a stream arrives in is never ours to decide.
        (stream_span,) = trace.find_spans("stream")
        verdicts = [note for note in stream_span.notes if note.label == "judged"]
        judged = len(verdicts)
        premature = len([n for n in verdicts if not n.payload["knew_it_was_a_fragment"]])
        span.add_note(
            ENDED, reason=f"judged {judged} times, {premature} of them early, dispatched nothing"
        )

    with trace.span("scenario", name="judged_when_complete") as span:
        turns += 1
        accumulated = stream_the_call(trace, model_kind, judge_every_chunk=False)
        (call,) = accumulated.tool_calls
        with trace.span("tool", name=call["name"], id=call["id"]) as tool_span:
            tool_span.add_note("args", **call["args"])
            result = TOOLS[call["name"]].invoke(call["args"])
            tool_span.add_note("result", value=result)
        span.add_note(ENDED, reason=f"dispatched once: {result}")

    endings = [str(span.require(ENDED).payload["reason"]) for span in trace.find_spans("scenario")]
    trace.close(turns=turns, messages=2, ended="; ".join(endings))
    return trace
