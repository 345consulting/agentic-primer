# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The same two calls, together or in sequence, and the question decides.

`ch04_tool_failures` showed a fan-out under duress: two calls, one of them
raising, and the constraint that every `tool_call_id` needs a reply anyway.
This is the clean case, and it is here to be compared against its opposite.

Neither situation is a single turn. Two tool calls in one reply belong to one
turn -- the turn that asked for them -- and the run still needs another
invocation to turn their results into an answer. What differs between the two
situations is how many turns that takes, and why.

Two situations, the same three tools, and only the question changes:

    arguments_in_the_question   both arguments are already known   2 turns
    argument_from_a_result      one call must finish first         3 turns

The model issues everything it can at once and waits only where it must, so
the shape of a run is decided by the data dependencies in the question. Ask
"how much is milk and how many do we have" and both arguments are in the
sentence, so both calls go together. Ask "what does one of whatever we are
lowest on cost" and the second call's argument does not exist until the first
returns.

That is the whole of it: **turns are the depth of the question, not a property
of the agent.** The number of tools does not enter into it, and neither does
the harness.

Read the two token counts against each other. Both scenarios make exactly two
tool calls, so the work is identical; only the shape differs:

    arguments_in_the_question   2 turns   1141 billed input tokens
    argument_from_a_result      3 turns   1813 billed input tokens

Fifty-nine per cent more input for the same two calls, because every level of
depth is another full context resend and cost is the sum of prefixes. Depth
costs; breadth is nearly free.

No framework can fix that. A framework can run a turn's calls concurrently,
which is breadth; nothing parallelises a dependency, because that is what
"depends on" means. The two levers are the question and the tools -- naming
`milk` supplied an argument the model would otherwise have fetched, and a
single `price_of_lowest_stock_item` would collapse the second scenario to one
turn.

Both tool calls here run in order, on one thread. `seq` and `thread` have been
recorded since `ch01_single_call` for the chapter where that stops being true.
"""

from support.agent import execute_tool as dispatch_and_record
from support.scenario import Scenario, run_scenario, run_scenarios
from support.trace import ModelKind, Trace

from functools import partial
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

PRICE = {"bread": "2.40", "butter": "3.10", "chips": "1.75", "milk": "1.20"}

type GroceryItem = Literal["bread", "butter", "chips", "milk"]
# A test asserts these three agree; a Literal cannot be built from a dict.

TURN_CAP = 6


@tool
def lowest_stock_item() -> str:
    """Which item there is least of in stock."""
    return min(STOCK_ON_HAND, key=lambda item: STOCK_ON_HAND[item])


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of one of an item, in pounds."""
    return PRICE[item]


@tool
def stock_on_hand(item: GroceryItem) -> int:
    """How many of an item are currently in stock."""
    return STOCK_ON_HAND[item]


TOOLS: dict[str, Any] = {
    lowest_stock_item.name: lowest_stock_item,
    price_of.name: price_of,
    stock_on_hand.name: stock_on_hand,
}

DECLARED_TOOLS = list(TOOLS.values())

# Both arguments are in the question, so both calls go in one reply.
ASKS_FOR_BOTH = AIMessage(
    "",
    tool_calls=[
        {"name": "price_of", "args": {"item": "milk"}, "id": "c1"},
        {"name": "stock_on_hand", "args": {"item": "milk"}, "id": "c2"},
    ],
    response_metadata={"finish_reason": "tool_calls"},
)

# Neither of these could have been sent first: the item is not known until
# `lowest_stock_item` returns it.
ASKS_WHICH = AIMessage(
    "",
    tool_calls=[{"name": "lowest_stock_item", "args": {}, "id": "c1"}],
    response_metadata={"finish_reason": "tool_calls"},
)

ASKS_THE_PRICE = AIMessage(
    "",
    tool_calls=[{"name": "price_of", "args": {"item": "butter"}, "id": "c2"}],
    response_metadata={"finish_reason": "tool_calls"},
)

SCENARIOS = [
    Scenario(
        name="arguments_in_the_question",
        question="How much is milk, and how many do we have?",
        mock_model_replies=[
            ASKS_FOR_BOTH,
            AIMessage(
                "Milk is 1.20 and there are 2 in stock.",
                response_metadata={"finish_reason": "stop"},
            ),
        ],
    ),
    Scenario(
        name="argument_from_a_result",
        question="Whatever we are lowest on -- what does one of them cost?",
        mock_model_replies=[
            ASKS_WHICH,
            ASKS_THE_PRICE,
            AIMessage(
                "You are lowest on butter, and one costs 3.10.",
                response_metadata={"finish_reason": "stop"},
            ),
        ],
    ),
]


def execute_tool(
    call: ToolCall, _scenario: Scenario, _model_kind: ModelKind, trace: Trace
) -> ToolMessage:
    """No decision to make: these tools cannot fail, so this is bookkeeping."""
    return dispatch_and_record(call, trace, lambda name, args, _span: TOOLS[name].invoke(args))


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    return run_scenarios(
        "ch06_two_tools",
        model_kind,
        SCENARIOS,
        partial(run_scenario, system_prompt=SYSTEM_PROMPT, execute_tool=execute_tool),
        turn_cap,
    )
