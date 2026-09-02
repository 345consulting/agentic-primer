# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The whole loop, and the two ways it is allowed to end.

Chapters 1 to 4 wrote their turns out by hand, two at a time, and said so:
`ended = "out of written turns"` was an admission that the number of turns was
a guess someone typed. Here it is a condition, so the number of turns becomes
a property of the run.

Nothing else is new. `ask_model` and `execute_tool` are chapter 4's, unchanged.
The loop is six lines and there is no framework in it:

    while True:
        reply = ask_model(...)          one invocation
        messages.append(reply)          the caller owns the history
        if not reply.tool_calls:        the model asked for nothing
            break
        for call in reply.tool_calls:   its tools, however many
            messages.append(execute_tool(call, trace))

That is a complete agent. Every chapter after this one answers a single
question -- what did it buy over these six lines? -- and some of the answers
will be "nothing you were missing".

The question is chosen so the loop has to run more than twice. The model
cannot know the answer until it has checked stock, and it cannot cost the
shortfall until it knows the price, so turn three exists *because* of turn
one's result. Two hand-written turns cannot show that; this is the first
chapter where the model's next move depends on what the last one returned.

And the loop needs a bound, because "it will stop on its own" is an assumption
rather than a property. `ch03_missing_tool` recorded a live model asking for
the same unavailable tool twice, so an unbounded `while` is a real spend. The
cap here is a number and nothing more -- `ch11_retry_exhausted` is where a
count becomes a policy.

Two endings, and the summary must tell them apart:

    ended = "no tool_calls"   the model was done
    ended = "turn cap"        we stopped it, and it was not done

A finished run and a capped run are indistinguishable otherwise, which is
`ch03_missing_tool`'s lesson applied to termination: a record that looks clean
is not the same as a run that went well. There are five further ways to end --
`ch06_loop_finish_reason` through `ch11_loop_veto` -- and each one is a stop
that this chapter would report as a completion.
"""

from support.models import build_model
from support.trace import ModelKind, Trace

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

# A genuine chain: the second call's *argument* is the first call's *result*.
# The model cannot ask what something costs until it has been told which thing,
# so the turns are forced to be sequential. An earlier draft asked "are we
# short of milk, and what would more cost?" -- and the live model called both
# tools in one reply, because "milk" was in the question. That is fan-out, and
# it belongs to ch12_two_tools.
USER_PROMPT = "Whatever we are lowest on -- what does one of them cost?"

STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

PRICE = {"bread": "2.40", "butter": "3.10", "chips": "1.75", "milk": "1.20"}

type GroceryItem = Literal["bread", "butter", "chips", "milk"]
# A test asserts these three agree; a Literal cannot be built from a dict.


@tool
def lowest_stock_item() -> str:
    """Which item there is least of in stock."""
    # No arguments, so nothing the model already knows can substitute for it.
    return min(STOCK_ON_HAND, key=lambda item: STOCK_ON_HAND[item])


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of one of an item, in pounds."""
    return PRICE[item]


TOOLS: dict[str, Any] = {lowest_stock_item.name: lowest_stock_item, price_of.name: price_of}

DECLARED_TOOLS = list(TOOLS.values())

# One list now, because there is one loop rather than turns written out. Each
# turn is still handed only what remains: a mock model is built per turn, since
# the wire hooks bind to that turn's span, so it cannot carry a position of its
# own between turns. Run past the end and it raises rather than inventing a
# reply -- the mock refusing to cover for a loop that ran longer than expected.
MOCK_MODEL_REPLIES = [
    AIMessage("", tool_calls=[{"name": "lowest_stock_item", "args": {}, "id": "call_1"}]),
    AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "butter"}, "id": "call_2"}]),
    AIMessage("You are lowest on butter, and one costs 3.10."),
]

# Six turns is nothing like a policy. It is a number that stops a runaway loop
# from being a bill, chosen to be comfortably more than this question needs.
# It is a parameter as well as a default so that the capped ending can be
# reached on purpose -- an ending no test could otherwise see.
TURN_CAP = 6


def ask_model(
    model_kind: ModelKind,
    mock_model_replies: Sequence[AIMessage],
    messages: list[BaseMessage],
    trace: Trace,
) -> AIMessage:
    """One invocation, recorded. Chapter 2's, unchanged."""
    with trace.span("model", model_kind=model_kind) as span:
        span.add_context(messages)
        reply = (
            build_model(model_kind, mock_model_replies, span)
            .bind_tools(DECLARED_TOOLS)
            .invoke(messages)
        )
        span.add_reply(reply)
    assert isinstance(reply, AIMessage)
    return reply


def execute_tool(call: ToolCall, trace: Trace) -> ToolMessage:
    """One tool call, dispatched by us. Chapter 4's, without the failure."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        result = TOOLS[call["name"]].invoke(call["args"])
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    trace = Trace(chapter="ch05_the_loop", model_kind=model_kind)
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]

    turns = 0
    ended = "turn cap"

    while turns < turn_cap:
        turns += 1
        with trace.span("turn", number=turns):
            remaining = MOCK_MODEL_REPLIES[turns - 1 :]
            reply = ask_model(model_kind, remaining, messages, trace)
            messages.append(reply)

            # The whole termination condition. In ch01 this list was empty on
            # the first reply and the program stopped; here it is empty when
            # the model has everything it needs.
            if not reply.tool_calls:
                ended = "no tool_calls"
                break

            for call in reply.tool_calls:
                messages.append(execute_tool(call, trace))

    # Which of the two endings happened. A run that finished and a run we
    # stopped are the same length, the same shape, and the same colour on the
    # page -- this line is the only thing that distinguishes them.
    trace.close(turns=turns, messages=len(messages), ended=ended)
    return trace
