# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""It stopped for another reason, and the loop called it done.

`ch03_the_loop` ends when the reply carries no tool calls, and every chapter
since has agreed that this means the model was finished. It does not. It means
the model asked for nothing further, which is a weaker claim, and there are
several reasons a reply can arrive that way.

This is the first chapter that provokes a failure on purpose. `max_tokens` is
set low enough that the reply is cut off mid-sentence -- the first of the
seven CONFIGURABLE parameters this primer has ever set, after five chapters of
`defaulted_by_model_provider` reporting all of them.

Two scenarios, the same loop, one parameter different:

    answers    max_tokens unset      the model finishes, and the loop is right
    truncated  max_tokens = 24       the model is cut off, and the loop is wrong

Read the truncated scenario before reading the rest of this. The budget is
spent before the model emits anything: the reply has empty content, no tool
calls, and `finish_reason: length`. A loop that checks only `tool_calls` would
report a clean run that produced nothing whatsoever -- and would be within its
rights, because an empty list is exactly what "the model asked for nothing"
looks like.

The provider said so, in the same response, in a field the loop never read:

    finish_reason: "stop"     the model chose to stop
    finish_reason: "length"   we cut it off
    finish_reason: "content_filter"   it was refused

Only the first means finished. The other two produce a reply with no tool
calls and are therefore indistinguishable to a loop that infers termination
from an empty list -- which is what `ch01_single_call`'s docstring warned
about in point 4, as a footnote. In a loop it stops being a footnote.

So the loop learns to read the provider's own account before trusting its own
inference, and gains a third ending. `content_filter` has the same shape and
is not provoked here: a chapter that has to write a prompt designed to be
refused is teaching something else, and the failure would not be reproducible
anyway.
"""

from support.models import build_model
from support.trace import ModelKind, Trace

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

USER_PROMPT = "Whatever we are lowest on -- what does one of them cost, and why is it low?"

STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

PRICE = {"bread": "2.40", "butter": "3.10", "chips": "1.75", "milk": "1.20"}

type GroceryItem = Literal["bread", "butter", "chips", "milk"]
# A test asserts these three agree; a Literal cannot be built from a dict.

# The provider's own account of why it stopped. Only the first means finished.
FINISHED = "stop"


@tool
def lowest_stock_item() -> str:
    """Which item there is least of in stock."""
    return min(STOCK_ON_HAND, key=lambda item: STOCK_ON_HAND[item])


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of one of an item, in pounds."""
    return PRICE[item]


TOOLS: dict[str, Any] = {lowest_stock_item.name: lowest_stock_item, price_of.name: price_of}

DECLARED_TOOLS = list(TOOLS.values())

TURN_CAP = 6

# The mock cannot be cut off by a parameter it ignores, so it is scripted with
# what a cut-off reply looks like: content that stops mid-word, no tool calls,
# and `finish_reason: length` in the metadata. The shape has to match what
# DeepSeek actually returns, or the mock column would teach the wrong tell.
ASKS = AIMessage("", tool_calls=[{"name": "lowest_stock_item", "args": {}, "id": "call_1"}])
PRICES = AIMessage(
    "", tool_calls=[{"name": "price_of", "args": {"item": "butter"}, "id": "call_2"}]
)


@dataclass(frozen=True)
class Scenario:
    """One run of the same loop, under one condition."""

    name: str
    max_tokens: int | None
    mock_model_replies: Sequence[AIMessage]


SCENARIOS = [
    Scenario(
        name="answers",
        max_tokens=None,
        mock_model_replies=[
            ASKS,
            PRICES,
            AIMessage(
                "Butter, at 3.10 -- there is only one left, so it is the lowest we hold.",
                response_metadata={"finish_reason": "stop"},
            ),
        ],
    ),
    Scenario(
        name="truncated",
        max_tokens=24,
        # One reply, and it is empty. A budget of 24 is spent before the model
        # can emit anything at all -- the reasoning takes it, and what comes
        # back has no content, no tool calls, and `finish_reason: length`.
        # This is the live transcript, not a tidier failure someone imagined:
        # the first draft scripted a sentence cut off mid-word, and the model
        # never got as far as a sentence.
        mock_model_replies=[
            AIMessage("", response_metadata={"finish_reason": "length"}),
        ],
    ),
]


def ask_model(
    model_kind: ModelKind,
    scenario: Scenario,
    turn: int,
    messages: list[BaseMessage],
    trace: Trace,
) -> AIMessage:
    """One invocation, recorded, with this scenario's `max_tokens`."""
    with trace.span("model", model_kind=model_kind) as span:
        span.add_context(messages)
        reply = (
            build_model(
                model_kind,
                scenario.mock_model_replies[turn - 1 :],
                span,
                max_tokens=scenario.max_tokens,
            )
            .bind_tools(DECLARED_TOOLS)
            .invoke(messages)
        )
        span.add_reply(reply)
    assert isinstance(reply, AIMessage)
    return reply


def execute_tool(call: ToolCall, trace: Trace) -> ToolMessage:
    """One tool call, dispatched by us. ch02_tool_call's, unchanged."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        result = TOOLS[call["name"]].invoke(call["args"])
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run_scenario(
    model_kind: ModelKind, scenario: Scenario, trace: Trace, turn_cap: int
) -> tuple[int, int, str]:
    """ch03_the_loop's loop, with one line added.

    Returns this scenario's turns, its message count, and how it ended.
    """
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]
    turns = 0

    while turns < turn_cap:
        turns += 1
        with trace.span("turn", number=turns):
            reply = ask_model(model_kind, scenario, turns, messages, trace)
            messages.append(reply)
            if not reply.tool_calls:
                # The added line. Before trusting our own inference, read the
                # provider's -- it answered the question in the same response.
                said = reply.response_metadata.get("finish_reason", FINISHED)
                ended = "no tool_calls" if said == FINISHED else f"cut off: {said}"
                return turns, len(messages), ended
            for call in reply.tool_calls:
                messages.append(execute_tool(call, trace))
    return turns, len(messages), "turn cap"


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    trace = Trace(chapter="ch06_loop_finish_reason", model_kind=model_kind)
    turns = 0
    messages = 0
    endings = []

    # Two runs of one loop. The comparison unit on the page is the scenario,
    # not the turn -- the same mock and live columns, one row per condition.
    for scenario in SCENARIOS:
        with trace.span("scenario", name=scenario.name, max_tokens=scenario.max_tokens):
            ran, said, ended = run_scenario(model_kind, scenario, trace, turn_cap)
        turns += ran
        messages += said
        endings.append(f"{scenario.name}: {ended}")

    # Both scenarios produced a reply with no tool calls. Only one of them
    # finished, and the summary can say so only because the loop read the
    # field rather than inferring from the empty list.
    trace.close(turns=turns, messages=messages, ended="; ".join(endings))
    return trace
