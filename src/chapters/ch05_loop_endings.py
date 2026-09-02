# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Every way this loop can stop, and only one of them means finished.

A run ends for one of several reasons and the summary has to say which, or a
run that gave up reads exactly like a run that succeeded. Each ending is a
scenario, so they sit on one page and can be read against each other.

Two are here today:

    within_token_limit  max_tokens unset   the model finishes
    tokens_exhausted    max_tokens = 24    it never starts

and the rest arrive as scenarios rather than as chapters -- a budget spent, a
deadline passed, a context window full, a loop making no progress. `turns_
exhausted` already exists in the code and needs only a scenario to show it.
The one that cannot live here is a veto, which needs a guard to do the vetoing
and so waits for ch23_guards.

The first ending below is the one the earlier chapters could not tell apart.

`ch03_the_loop` ends when the reply carries no tool calls, and every chapter
since has agreed that this means the model was finished. It does not. It means
the model asked for nothing further, which is a weaker claim, and there are
several reasons a reply can arrive that way.

This is the first chapter that provokes a failure on purpose. `max_tokens` is
set low enough that the reply is cut off mid-sentence -- the first of the
seven CONFIGURABLE parameters this primer has ever set, after five chapters of
`defaulted_by_model_provider` reporting all of them.

Two scenarios, the same loop, one parameter different:

    within_token_limit  max_tokens unset   the model finishes, and the loop is right
    tokens_exhausted    max_tokens = 24    it never starts, and the loop is wrong

Two runs, the same empty `tool_calls`, and the only thing that tells them
apart is a field the loop was not reading.

Read the `tokens_exhausted` scenario before reading the rest of this. The budget is
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
from support.scenario import Scenario, run_scenarios
from support.trace import ModelKind, Trace

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

# The two tool calls both scenarios start with. `max_tokens` means nothing to
# a mock model, so what a cut-off reply looks like has to be scripted -- and
# the shape has to match what DeepSeek actually returns, or the mock column
# teaches the wrong tell. See `tokens_exhausted` below for what that
# turned out to be.
# Both carry `finish_reason: tool_calls`, because that is what a real reply
# containing tool calls comes back with. Leaving it off let the default fill
# in `stop`, and a capped run then reported that the model had finished when
# it had just asked for a tool -- the mock inventing the one fact this
# chapter is about.
ASKS = AIMessage(
    "",
    tool_calls=[{"name": "lowest_stock_item", "args": {}, "id": "call_1"}],
    response_metadata={"finish_reason": "tool_calls"},
)
PRICES = AIMessage(
    "",
    tool_calls=[{"name": "price_of", "args": {"item": "butter"}, "id": "call_2"}],
    response_metadata={"finish_reason": "tool_calls"},
)

SCENARIOS = [
    Scenario(
        name="within_token_limit",
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
        name="tokens_exhausted",
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


def _finish_reason(reply: AIMessage) -> str:
    """The provider's own account of why it stopped, or `stop` if it said nothing."""
    said: str = reply.response_metadata.get("finish_reason", FINISHED)
    return said


def _with_reason(ended: str, said: str) -> str:
    """Our conclusion, with the provider's word beside it rather than folded in.

    Two different facts: what the loop decided, and what it decided from. An
    earlier version translated one into the other -- `f"{said}_exhausted"` --
    which would have reported a refusal as `content_filter_exhausted`, and a
    refusal is not an exhausted resource. Quoting is safer than paraphrasing.
    """
    return f"{ended} (finish_reason = {said})"


def run_scenario(
    model_kind: ModelKind, scenario: Scenario, trace: Trace, turn_cap: int
) -> tuple[int, int, str]:
    """ch03_the_loop's loop, with one line added.

    Returns this scenario's turns, its message count, and how it ended.
    """
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]
    turns = 0
    # The last reply, so the capped ending can quote where the model had got
    # to. None means the loop never ran at all, which a cap of zero allows.
    reply: AIMessage | None = None

    while turns < turn_cap:
        turns += 1
        with trace.span("turn", number=turns):
            reply = ask_model(model_kind, scenario, turns, messages, trace)
            messages.append(reply)
            if not reply.tool_calls:
                # The added line. Before trusting its own inference the loop
                # reads the provider's, which arrived in the same response.
                # `answer_received` rather than `no_tool_calls`, because this
                # loop checked: the chapters before it only knew the list was
                # empty and said so.
                said = _finish_reason(reply)
                ended = "answer_received" if said == FINISHED else "tokens_exhausted"
                return turns, len(messages), _with_reason(ended, said)
            for call in reply.tool_calls:
                messages.append(execute_tool(call, trace))
    # Our own ending, and the provider's word for where the model had got
    # to: `tool_calls` means it was still going when we stopped it.
    said = _finish_reason(reply) if reply else "never asked"
    return turns, len(messages), _with_reason("turns_exhausted", said)


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    # The scenarios are the chapter; running them and recording the run
    # around them is bookkeeping, and lives in support/scenario.py.
    return run_scenarios("ch05_loop_endings", model_kind, SCENARIOS, run_scenario, turn_cap)
