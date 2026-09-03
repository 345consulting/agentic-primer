# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""What travels beside `messages`, and who is allowed to write it.

`turns`, `ended`, and `ch10_routing`'s `collected` have all been locals in a
`while` loop since `ch03_the_loop`. Every chapter has had state; none of them
named it. This one does: a value that travels alongside `messages` through
the whole run, is never sent to the model provider, and outlives any single
turn -- which a local variable inside one turn's tool dispatch cannot do.

**The value is a running total, and it closes a gap `ch09_who_retries` found
live.** `MAX_PER_ORDER` bounded one call to `place_order`. Asked for twenty
bottles against a limit of twelve, the live model did not correct its
argument -- it called the tool twice, twelve and eight, and satisfied the
user's intent by routing around a bound that only ever looked at one call at
a time. The finding, written down then: a tool cannot enforce a limit,
because it never sees the sequence it is part of.

`state.total_ordered` is that sequence, held somewhere for the first time.
Not in the tool -- `submit_order` below never receives `state` as an
argument, so there is nothing in it capable of reading or writing the total.
Not in the model -- it only ever sees `messages`, and `total_ordered` is
never written into one. The only thing holding it is the loop, in exactly one
function, `increment_order_total`, called from exactly one place.

**Two rules, and both are `increment_order_total`'s.** A read leaves the total alone --
`price_of` never changes it, whatever it returns. A write that failed leaves
it alone too: `increment_order_total` checks `result.status`, so a `place_order` call that
errored is not counted as if it had shipped. Only a dispatched, successful
write moves the number, which is the only thing that should be able to.

This chapter does not stop anything -- `state.total_ordered` can climb past
any number you like, and nothing here reads it back. Making it matter to what
happens next is `guards`, which needs a place to check before it can refuse
anything. This chapter only makes sure that place exists, and that what it
holds can be trusted.

`MessagesState` is LangGraph's name for the same idea, structured differently:
one shared object, `messages` one field of it among others, every node
handed the whole thing and choosing for itself what to forward to the model.

**The live run took a shortcut the mock never offered it.** The first
scenario is scripted as two turns, one `place_order` in each, because that is
what proves state has to outlive a single turn -- a local scoped to one
turn's dispatch loop would not survive to the second. The live model was
smarter than the script: it put both calls in one reply, the way
`ch06_two_tools` fans out, and answered in a single turn. `total_ordered`
still comes out right -- 5 + 3 = 8, `increment_order_total` runs once per
call regardless of how many turns they are spread across -- but this
particular run never needed a value that outlives a turn, because the model
never spread the calls across more than one.

The claim this chapter makes still holds; this run just is not the one that
demonstrates it. The mock proves the sharp version because it is scripted to
-- two turns, deliberately. The live column proves the weaker one, that the
total is right regardless of how the calls are grouped, and that is worth
having too. Nothing here can make a live model space its calls out to fit a
lesson; it only has to be true whichever way the model chooses to call.
"""

from support.agent import ask_model
from support.scenario import Outcome, RunOne, Scenario, run_scenarios
from support.trace import ModelKind, Trace

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

TURN_CAP = 4


@dataclass
class State:
    """What travels beside `messages`. Never serialized, never sent."""

    total_ordered: int = 0


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item, in pounds. A read: the total ignores it."""
    return f"1.20 for {item}"


@tool
def place_order(item: GroceryItem, quantity: int) -> str:
    """Order a quantity of an item. A write: the total counts it, once dispatched."""
    return f"ordered {quantity} x {item}"


TOOLS: dict[str, Any] = {price_of.name: price_of, place_order.name: place_order}

DECLARED_TOOLS = [place_order, price_of]

SCENARIOS = [
    Scenario(
        name="a_running_total_survives_between_calls",
        question="Order 5 bottles of milk for the cafe, and 3 for the office, separately.",
        tools=DECLARED_TOOLS,
        mock_model_replies=[
            AIMessage(
                "",
                tool_calls=[
                    {"name": "place_order", "args": {"item": "milk", "quantity": 5}, "id": "c1"}
                ],
            ),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "place_order", "args": {"item": "milk", "quantity": 3}, "id": "c2"}
                ],
            ),
            AIMessage("Ordered 5 for the cafe and 3 for the office -- 8 in total."),
        ],
    ),
    Scenario(
        name="reads_do_not_touch_the_total",
        question="How much is milk, and please order 4 bottles.",
        tools=DECLARED_TOOLS,
        mock_model_replies=[
            AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}]),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "place_order", "args": {"item": "milk", "quantity": 4}, "id": "c2"}
                ],
            ),
            AIMessage("Milk is 1.20 each; ordered 4 bottles."),
        ],
    ),
]


def execute_tool(call: ToolCall, trace: Trace) -> ToolMessage:
    """One tool call. `state` is not a parameter here -- the tool cannot
    reach what it must not write.
    """
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        result = TOOLS[call["name"]].invoke(call["args"])
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def increment_order_total(state: State, call: ToolCall, result: ToolMessage) -> None:
    """The one place `state` is written, and the two rules that govern it.

    A read is any call that is not `place_order`, and it falls through
    untouched. A write that did not succeed falls through too -- only a
    dispatched, successful `place_order` moves the total, because only that
    is a fact about what actually happened.
    """
    if call["name"] == "place_order" and result.status == "success":
        state.total_ordered += call["args"]["quantity"]


def run_one(model_kind: ModelKind, scenario: Scenario, trace: Trace, turn_cap: int) -> Outcome:
    """`run_turns`, with a second value carried beside `messages` and read by
    nothing outside this function -- not the tool, not the model.
    """
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(scenario.question)]
    state = State()
    turns = 0
    ended = "turns_exhausted"

    while turns < turn_cap:
        turns += 1
        with trace.span("turn", number=turns) as span:
            reply = ask_model(
                trace,
                model_kind,
                scenario.mock_model_replies[turns - 1 :],
                messages,
                scenario.tools,
                scenario.max_tokens,
            )
            messages.append(reply)
            if not reply.tool_calls:
                ended = "no_tool_calls"
                span.add_note("state", total_ordered=state.total_ordered)
                break
            for call in reply.tool_calls:
                result = execute_tool(call, trace)
                messages.append(result)
                increment_order_total(state, call, result)
            span.add_note("state", total_ordered=state.total_ordered)

    return turns, len(messages), ended


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    typed_run_one: RunOne = run_one
    return run_scenarios("ch11_state", model_kind, SCENARIOS, typed_run_one, turn_cap)
