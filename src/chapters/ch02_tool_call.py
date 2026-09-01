"""Chapter 2 -- the model asks, we execute, turn two knows.

Chapter 1 was one invocation and a stop. Here the model replies with no answer
at all: it replies with a *request*, and the program has to do something about
it. That request-and-result round trip is the whole of tool calling, and it is
the first time the program does work between two model calls.

Still no framework. The tools are a dict of plain callables, dispatch is a
subscript, and the loop is written out twice rather than looped -- turn one and
turn two are separate blocks below so the difference between them is visible
rather than hidden inside a `while`. Chapter 4 collapses them.

Three things to read the trace for, none of which happened in chapter 1:

1. The tool declaration goes on the wire. `defaulted (7)` in chapter 1 becomes
   `defaulted (5)` here -- `tools` and `tool_choice` are ours now. What the
   model can ask for is something we said, in the request, every time.
2. The result re-enters the context as a message, not as a return value. There
   is nowhere else for it to go: the provider is stateless, so the only way it
   learns what the tool said is that we send it back in the next request.
3. `tool_call_id` is what ties a result to the request that asked for it.
   Order is not the tie -- chapter 4 runs two calls at once and the ids are all
   that survive.
"""

from support.models import Kind, model
from support.trace import Trace

import sys
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

SYSTEM = "You answer briefly and plainly. Use the tools you are given."

QUESTION = "Do we have enough flanges on hand to fill an order for 40?"


# A tool is a function plus a description of it. The decorator only builds the
# schema -- name, arguments, docstring -- that gets sent to the provider. The
# function underneath stays an ordinary callable we can call ourselves, which
# is exactly what happens below.
# A fact the model cannot know and cannot guess. That is the point: if the
# answer in turn two is right, it is right because the tool ran.
STOCK = {"flange": 17, "grommet": 240}

# The argument is an enum, not a string, and the enum is in the request body.
# This is the only thing constraining what the model may ask for -- the first
# live run of this chapter declared `part: str`, the model asked for "flanges",
# and a lookup with a default answered 0. The model then reported that as fact.
# See FINDINGS.md, 2026-09-01.
type Part = Literal["flange", "grommet"]


@tool
def stock_on_hand(part: Part) -> int:
    """How many of a part are currently in stock."""
    # No default. An argument outside the schema is a broken premise, not a
    # zero -- and a tool that answers anyway is worse than one that fails.
    return STOCK[part]


# Ours to dispatch, keyed by the name the model will use. This dict and the
# list handed to `bind_tools` are two different things that happen to agree
# here. Chapter 3 is what happens when they do not.
TOOLS: dict[str, Any] = {stock_on_hand.name: stock_on_hand}

SCRIPT = [
    # Turn one: no content at all, just a request. This is what a tool call
    # looks like -- the model stops mid-thought and waits for the program.
    AIMessage(
        "",
        tool_calls=[{"name": "stock_on_hand", "args": {"part": "flange"}, "id": "call_1"}],
    ),
    # Turn two: the answer, which exists only because the result came back.
    AIMessage("No -- there are 17 flanges on hand, which is 23 short of 40."),
]


def run(kind: Kind = "mock") -> Trace:
    trace = Trace(chapter="ch02_tool_call", kind=kind)
    messages: list[BaseMessage] = [SystemMessage(SYSTEM), HumanMessage(QUESTION)]

    # `bind_tools` does not teach the model anything. It puts a schema in the
    # request body, and it does so on every call -- there is no registration
    # step and nothing is remembered between requests.
    with trace.span("turn", number=1):
        with trace.span("model", kind=kind) as span:
            span.context(messages)
            declared = list(TOOLS.values())
            reply = model(kind, SCRIPT[:1], wire=span).bind_tools(declared).invoke(messages)
            span.reply(reply)
        assert isinstance(reply, AIMessage)
        messages.append(reply)

        # In chapter 1 this list was empty and the program stopped. It is not
        # empty, so the turn is not over: a turn is one invocation plus
        # whatever tools that invocation asked for.
        for call in reply.tool_calls:
            with trace.span("tool", name=call["name"], id=call["id"]) as span:
                span.note("args", **call["args"])
                # We dispatch. No framework is between the model's request and
                # this call -- which means every decision here is ours, and
                # chapter 3 is about the one we have not had to make yet.
                result = TOOLS[call["name"]].invoke(call["args"])
                span.note("result", value=result)
                # The result becomes a message. `tool_call_id` is the only
                # thing tying it to the request; the provider matches on that,
                # not on position.
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # A second invocation, with the same stateless provider. Everything it
    # knows about the tool run is in the list we are about to send.
    with trace.span("turn", number=2):
        with trace.span("model", kind=kind) as span:
            span.context(messages)
            reply = model(kind, SCRIPT[1:], wire=span).bind_tools(declared).invoke(messages)
            span.reply(reply)
        assert isinstance(reply, AIMessage)
        messages.append(reply)

    # Why this stops, read off the reply rather than asserted. There are two
    # turns here because two turns are written out, not because the model was
    # finished -- and if it asks for a tool in turn two, this chapter has
    # nowhere to put it. That gap is the whole of chapter 5.
    ended = "no tool_calls" if not reply.tool_calls else "out of written turns"
    trace.close(turns=2, messages=len(messages), ended=ended)
    return trace


if __name__ == "__main__":
    kind: Kind = "live" if "live" in sys.argv[1:] else "mock"
    run(kind).report()
