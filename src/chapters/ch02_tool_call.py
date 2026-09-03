# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The model asks, we execute, and turn two knows.

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
   is nowhere else for it to go: the model provider is stateless, so the only way it
   learns what the tool said is that we send it back in the next request.
3. `tool_call_id` is what ties a result to the request that asked for it.
   Order is not the tie -- chapter 4 runs two calls at once and the ids are all
   that survive.
"""

from support.models import build_model
from support.scenario import Scenario, run_scenarios
from support.trace import ModelKind, Trace

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

USER_PROMPT = "Do we have enough milk for the week? We need four."


# A tool is a function plus a description of it. The decorator only builds the
# schema -- name, arguments, docstring -- that gets sent to the model provider. The
# function underneath stays an ordinary callable we can call ourselves, which
# is exactly what happens below.
# A fact the model cannot know and cannot guess. That is the point: if the
# answer in turn two is right, it is right because the tool ran.
STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

# The argument is an enum, not a string, and the enum is in the request body.
# This is the only thing constraining what the model may ask for -- the first
# live run of this chapter declared the argument as `str`, the model asked
# and a lookup with a default answered 0. The model then reported that as fact.
# See LEARNINGS.md, 2026-09-01.
type GroceryItem = Literal["bread", "butter", "chips", "milk"]
# A test asserts these two agree; a Literal cannot be built from a dict.


@tool
def stock_on_hand(item: GroceryItem) -> int:
    """How many of an item are currently in stock."""
    # No default. An argument outside the schema is a broken premise, not a
    # zero -- and a tool that answers anyway is worse than one that fails.
    return STOCK_ON_HAND[item]


# Ours to dispatch, keyed by the name the model will use. This dict and the
# list handed to `bind_tools` are two different things that happen to agree
# here. Chapter 3 is what happens when they do not.
TOOLS: dict[str, Any] = {stock_on_hand.name: stock_on_hand}

# What the mock model replies. A live run ignores both and answers for
# itself. Two, because this chapter writes out two turns.
MOCK_MODEL_REPLIES = [
    # No content at all, just a request. This is what a tool call looks like:
    # the model stops mid-thought and waits for the program.
    AIMessage(
        "",
        tool_calls=[{"name": "stock_on_hand", "args": {"item": "milk"}, "id": "call_1"}],
        response_metadata={"finish_reason": "tool_calls"},
    ),
    # The answer, which exists only because the result came back.
    AIMessage(
        "No -- there are two in the fridge, so you are two short.",
        response_metadata={"finish_reason": "stop"},
    ),
]

# One situation: a tool exists and it answers the question.
# ch04_tool_failures is the same axis from the other side.
SCENARIOS = [Scenario(name="agent_with_one_tool", mock_model_replies=MOCK_MODEL_REPLIES)]


# What we tell the model provider exists, built once. It is the same list on every
# call -- and it is sent on every call, because nothing is remembered between
# them. A stable order also keeps the request prefix stable, which is what the
# model provider's cache matches on.
DECLARED_TOOLS = list(TOOLS.values())


def ask_model(
    model_kind: ModelKind,
    mock_model_replies: Sequence[AIMessage],
    messages: list[BaseMessage],
    trace: Trace,
) -> AIMessage:
    """One invocation, recorded: the exact context in, the reply out.

    `ch01_single_call` wrote this out by hand without tools. It is written out
    again here, rather than imported from `support/agent.py`, because what is
    new in this chapter is inside it: `bind_tools`, and what that does and does
    not do. Chapters after `ch03_the_loop` import it instead.
    """
    with trace.span("model", model_kind=model_kind) as span:
        span.add_context(messages)
        # `bind_tools` does not teach the model anything. It puts a schema in
        # the request body, and it does so on every call -- there is no
        # registration step and nothing is remembered between requests.
        reply = (
            build_model(model_kind, mock_model_replies, span)
            .bind_tools(DECLARED_TOOLS)
            .invoke(messages)
        )
        span.add_reply(reply)
    # invoke() is typed as returning BaseMessage. Narrowing is for the type
    # checker, not for correctness -- worth seeing rather than hiding in a cast.
    assert isinstance(reply, AIMessage)
    return reply


def execute_tool(call: ToolCall, trace: Trace) -> ToolMessage:
    """One tool call, dispatched by us, as a message to send back.

    Named for the span it opens, as `ask_model` is -- the code and the trace
    should not need two vocabularies for the same two things.

    Nothing is between the model's request and this dispatch. Every decision
    here is ours, which is what makes chapter 3 possible.
    """
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        result = TOOLS[call["name"]].invoke(call["args"])
        span.add_note("result", value=result)
    # `tool_call_id` is the only thing tying this result to the request that
    # asked for it. The model provider matches on that, not on position.
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run_scenario(
    model_kind: ModelKind, scenario: Scenario, trace: Trace, _turn_cap: int
) -> tuple[int, int, str]:
    """Two turns, written out. No loop yet, so no turn cap to obey."""
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]

    # A turn is one invocation plus whatever tools that invocation asked for.
    with trace.span("turn", number=1):
        reply = ask_model(model_kind, scenario.mock_model_replies[:1], messages, trace)
        messages.append(reply)
        # In chapter 1 this list was empty and the program stopped. It is not
        # empty, so the turn is not over.
        for call in reply.tool_calls:
            messages.append(execute_tool(call, trace))

    # The same stateless model provider, told what happened only by the list.
    with trace.span("turn", number=2):
        reply = ask_model(model_kind, scenario.mock_model_replies[1:], messages, trace)
        messages.append(reply)

    # Why this stops, read off the reply rather than asserted. There are two
    # turns here because two turns are written out, not because the model was
    # finished -- and if it asks for a tool in turn two, this chapter has
    # nowhere to put it. That gap is the whole of ch03_the_loop.
    ended = "no_tool_calls" if not reply.tool_calls else "out of written turns"
    return 2, len(messages), ended


def run(model_kind: ModelKind = "mock") -> Trace:
    return run_scenarios("ch02_tool_call", model_kind, SCENARIOS, run_scenario, turn_cap=2)
