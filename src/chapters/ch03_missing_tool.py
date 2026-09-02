# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""We ask for something no tool can answer, and nothing fails.

Chapter 2 asked a question the toolbox covered. This one does not: it asks
when the next delivery is, and there is no tool that knows. The toolbox has
one entry, `stock_on_hand`, and it answers a different question.

Watch what the model does, because it is not what a program would do. It does
not refuse and it does not invent a tool -- the declaration is enforced by the
model provider before a call exists, so a name we never sent cannot come back.
It reaches for the nearest thing it has, calls `stock_on_hand`, and then
explains in prose that it cannot answer the actual question.

So the run costs a turn, a tool call and two round trips, and produces an
answer to a question nobody asked plus an apology for the one they did.

Now read the trace as a harness would. Every span opened and closed. The tool
call succeeded and returned a real number. The reply came back with
`finish_reason: stop` and no tool calls, so the loop terminated normally. There
is no error anywhere, no exception, no retry, no status field set to anything
but success. A dashboard counting tool calls and completions scores this run
as clean.

The gap between what was asked and what the tools cover is invisible to every
mechanism in this file. That is the lesson, and it is the first one in this
primer that is not about a mechanism at all: some failures do not fail.

Nothing here is new machinery. The loop, the dispatch and the recording are
chapter 2's, unchanged, which is the point -- the code cannot tell.
"""

from support.models import build_model
from support.trace import ModelKind, Trace

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

# A question about the future. Nothing in the toolbox knows about deliveries,
# and nothing in the toolbox says so either.
USER_PROMPT = "When is the next delivery of milk expected?"

STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

type GroceryItem = Literal["bread", "butter", "chips", "milk"]
# A test asserts these two agree; a Literal cannot be built from a dict.


@tool
def stock_on_hand(item: GroceryItem) -> int:
    """How many of an item are currently in stock."""
    return STOCK_ON_HAND[item]


TOOLS: dict[str, Any] = {stock_on_hand.name: stock_on_hand}

DECLARED_TOOLS = list(TOOLS.values())

# What the mock model replies -- and here the script is a transcript. These
# are the two replies the live model actually gave, so the mock column shows
# the same behaviour rather than a tidier one someone imagined.
MOCK_MODEL_REACHES_FOR_THE_NEAREST_TOOL = [
    AIMessage("", tool_calls=[{"name": "stock_on_hand", "args": {"item": "milk"}, "id": "call_1"}])
]

MOCK_MODEL_ANSWERS = [
    AIMessage(
        "I don't have access to delivery schedules. There are currently 2 units of milk in stock."
    )
]


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
    """One tool call, dispatched by us. Chapter 2's, unchanged.

    There is no branch here for a tool that cannot be found, because that is
    not what goes wrong in this chapter. The name always resolves. The call
    always succeeds. What is missing is a tool nobody wrote.
    """
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        result = TOOLS[call["name"]].invoke(call["args"])
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch03_missing_tool", model_kind=model_kind)
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]

    with trace.span("turn", number=1):
        reply = ask_model(model_kind, MOCK_MODEL_REACHES_FOR_THE_NEAREST_TOOL, messages, trace)
        messages.append(reply)
        for call in reply.tool_calls:
            messages.append(execute_tool(call, trace))

    with trace.span("turn", number=2):
        reply = ask_model(model_kind, MOCK_MODEL_ANSWERS, messages, trace)
        messages.append(reply)

    # Every field here reports success, and the run did not answer the
    # question. `ended` is the honest stop reason and it is still the wrong
    # story -- there is no field in this summary, or anywhere in the trace,
    # for "answered a question nobody asked".
    ended = "no tool_calls" if not reply.tool_calls else "out of written turns"
    trace.close(turns=2, messages=len(messages), ended=ended)
    return trace
