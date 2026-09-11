# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The toolbox lets you down in three ways, and only one of them looks like it.

Every chapter so far had tools that worked. Here they do not, in three
different ways, run as three scenarios of the same loop so they can be read
against each other:

    no_tool_available   the tool that would answer does not exist
    tool_exception      a tool runs and raises
    partial_failure     two calls in one reply, one raises, one does not

`no_tool_available` names the tool you would need, not the toolbox: there are
tools, they are declared, they run. None of them answers the question.

All three runs end the same way -- `no tool_calls`, `finish_reason: stop`,
every span opened and closed. Three different failures, one ending, and the
summary cannot tell them apart. That is the reason they share a page.

**no_tool_available.** The question asks when the next delivery is; nothing
knows. The model does not refuse and does not invent a tool -- the declaration
is enforced by the model provider before a call exists, so a name we never
sent cannot come back. It reaches for the nearest thing it has, calls it, and
explains in prose that it cannot answer.

This is the one scenario where the mock and the live columns disagree, and
the disagreement is the point: the model *may* reach for the nearest tool, and
may instead answer directly with no call at all. Roughly two runs in three
substitute. The mock shows the substitution because it is the more interesting
half; the live column shows whichever happened. Neither is wrong, and a
harness cannot count on the wasted call being there to see.

Read that scenario's trace as a harness would: the tool call succeeded and
returned a real number, the loop terminated normally, there is no error
anywhere and no status set to anything but success. A dashboard counting tool
calls and completions scores it clean. The only evidence is a sentence of
English in the final message, and prose is not a field.

**tool_exception.** The premise is sound and the world is broken, which is the
ordinary case. Nothing above this file catches it and nothing below will, so
the program chooses: raise and stop, or report and let the loop survive. This
reports, and the cost of that choice is the second half of the lesson.

One failure produces two texts. The exception goes in the trace, in full, for
us -- an exception is written for a developer reading a stack trace and can
carry a path, a host or a credential. A sentence we wrote goes to the model,
saying what happened and what to do instead. LangGraph's default does neither:
it formats the exception with `repr` and appends "Please fix your mistakes", so
the raw exception reaches the model provider and the model is told to retry.

**partial_failure.** The model asks for price and stock in one reply. One
raises, one does not, and the loop cannot bail on the first exception: an
assistant message carrying two `tool_call_id`s must be followed by a
`ToolMessage` for each of them, or the model provider rejects the next request
outright. So the fan-out has to complete and report a mixed result.

That is a constraint, not a preference, and it is the first thing in this
primer that a naive `try` around the whole loop would break.
"""

from support.scenario import Scenario, run_scenario, run_scenarios
from support.trace import ModelKind, Trace

from functools import partial
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

type GroceryItem = Literal["bread", "butter", "chips", "milk"]
# A test asserts these two agree; a Literal cannot be built from a dict.

TURN_CAP = 6


class PriceServiceError(RuntimeError):
    """The price service is unreachable. Raised for real, caught by us."""


@tool
def stock_on_hand(item: GroceryItem) -> int:
    """How many of an item are currently in stock."""
    return STOCK_ON_HAND[item]


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item."""
    # Deterministic, so the chapter fails the same way on every run, mock and
    # live alike. A tool that failed intermittently would teach the same lesson
    # and make a bad chapter.
    raise PriceServiceError(f"pricing service unreachable for ['{item}']")


TOOLS: dict[str, Any] = {price_of.name: price_of, stock_on_hand.name: stock_on_hand}

SCENARIOS = [
    Scenario(
        name="no_tool_available",
        question="When is the next delivery of milk expected?",
        # `price_of` is withheld here. Declared, it would raise when the model
        # reached for it, and this scenario's whole point is that nothing goes
        # wrong. That is why a scenario carries its own toolbox.
        tools=(stock_on_hand,),
        mock_model_replies=[
            AIMessage(
                "", tool_calls=[{"name": "stock_on_hand", "args": {"item": "milk"}, "id": "c1"}]
            ),
            AIMessage(
                "I don't have access to delivery schedules. There are 2 units of milk in stock."
            ),
        ],
    ),
    Scenario(
        name="tool_exception",
        question="How much does milk cost right now?",
        tools=(price_of, stock_on_hand),
        mock_model_replies=[
            AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}]),
            AIMessage("I cannot get the price of milk -- the pricing service is unreachable."),
        ],
    ),
    Scenario(
        name="partial_failure",
        question="How much is milk, and how many do we have?",
        tools=(price_of, stock_on_hand),
        mock_model_replies=[
            AIMessage(
                "",
                tool_calls=[
                    {"name": "price_of", "args": {"item": "milk"}, "id": "c1"},
                    {"name": "stock_on_hand", "args": {"item": "milk"}, "id": "c2"},
                ],
            ),
            AIMessage("There are 2 milk in stock. The price is unavailable right now."),
        ],
    ),
]


def execute_tool(
    call: ToolCall, _scenario: Scenario, _model_kind: ModelKind, trace: Trace
) -> ToolMessage:
    """One tool call, and the decision to make when it raises."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        try:
            result = TOOLS[call["name"]].invoke(call["args"])
        except PriceServiceError as failure:
            # The exception in full, for us. It stays in the trace and never
            # reaches the model provider.
            span.add_note(
                "failed",
                exception=type(failure).__name__,
                message=str(failure),
                decision="report to the model; this chapter has no retry",
            )
            # And a sentence for the model, written on purpose: what happened,
            # and what to do about it. It does not say "try again".
            return ToolMessage(
                content=(
                    f"['{call['name']}'] failed and will not succeed on a retry now. "
                    "Answer without it, and say plainly that you could not get it."
                ),
                tool_call_id=call["id"],
                status="error",
            )
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    # The scenarios are the chapter; running them and recording the run
    # around them is bookkeeping, and lives in support/scenario.py.
    return run_scenarios(
        "ch04_tool_failures",
        model_kind,
        SCENARIOS,
        partial(run_scenario, system_prompt=SYSTEM_PROMPT, execute_tool=execute_tool),
        turn_cap,
    )
