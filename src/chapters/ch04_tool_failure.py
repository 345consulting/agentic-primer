# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The tool runs and fails, and we choose what the model is told.

Chapter 3's tools all worked; what was missing was a tool for the job. Here
the right tool exists, is declared correctly, is called correctly -- and
raises. The premise is sound and the world is broken, which is the ordinary
case and the first one in this primer where something actually goes wrong.

Nothing above this file catches it and nothing below it will, so the program
has to choose. Two answers, both defensible:

- **Raise.** The run stops. The failure reaches whoever ran it, at full
  fidelity, and no model gets a chance to talk around it.
- **Report.** The failure becomes a `ToolMessage` and re-enters the context.
  The loop survives, and the model answers with what it has.

This chapter reports, and then shows what that costs. Two things are worth
watching in the trace, and neither is obvious until you see it:

1. **The error text is a prompt.** It is written by us, read by the model, and
   nothing else ever sees it. LangGraph's default is literally
   `"Error: {error}\\n Please fix your mistakes."` -- an instruction to try
   again, shipped as a default. Ours says what happened and what to do
   instead, because those are the only two things the model can act on.

2. **`repr(e)` goes on the wire.** LangGraph's template formats the exception
   with `repr`, so whatever a tool put in its message -- a path, a host, a
   connection string -- is sent to the model provider. We send a sentence we
   wrote instead. The exception's own text stays in the trace, where it is
   for us.

What this chapter does not do is retry. It cannot: with two written turns
there is nowhere to put a second attempt, and no counter to stop it if there
were. Chapters 7 and 8 are that, once chapter 5 has a loop to hold it.
"""

from support.models import build_model
from support.trace import ModelKind, Trace

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

USER_PROMPT = "How much does milk cost right now?"

STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

type GroceryItem = Literal["bread", "butter", "chips", "milk"]
# A test asserts these two agree; a Literal cannot be built from a dict.


class PriceServiceError(RuntimeError):
    """The price service is unreachable. Raised for real, caught by us."""


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item."""
    # Deterministic, so the chapter fails the same way on every run, mock and
    # live alike. A tool that fails intermittently would teach the same lesson
    # and would make a bad chapter.
    raise PriceServiceError(f"pricing service unreachable for ['{item}']")


@tool
def stock_on_hand(item: GroceryItem) -> int:
    """How many of an item are currently in stock."""
    return STOCK_ON_HAND[item]


TOOLS: dict[str, Any] = {price_of.name: price_of, stock_on_hand.name: stock_on_hand}

DECLARED_TOOLS = list(TOOLS.values())

MOCK_MODEL_ASKS_FOR_A_PRICE = [
    AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "call_1"}])
]

MOCK_MODEL_ANSWERS = [
    AIMessage("I cannot get the price of milk right now -- the pricing service is unreachable.")
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
    """One tool call, and the decision to make when it raises."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        try:
            result = TOOLS[call["name"]].invoke(call["args"])
        except Exception as failure:
            # The exception in full, for us. It stays in the trace and never
            # goes to the model provider -- an exception's text is written for
            # a developer reading a stack trace, and can carry a host, a path
            # or a credential.
            span.add_note(
                "failed",
                exception=type(failure).__name__,
                message=str(failure),
                decision="report to the model; this chapter has no retry",
            )
            # And a sentence for the model, written on purpose. It says what
            # happened and what to do about it, because those are the only two
            # things the model can act on. It does not say "try again".
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


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch04_tool_failure", model_kind=model_kind)
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]

    with trace.span("turn", number=1):
        reply = ask_model(model_kind, MOCK_MODEL_ASKS_FOR_A_PRICE, messages, trace)
        messages.append(reply)
        for call in reply.tool_calls:
            messages.append(execute_tool(call, trace))

    with trace.span("turn", number=2):
        reply = ask_model(model_kind, MOCK_MODEL_ANSWERS, messages, trace)
        messages.append(reply)

    # Unlike chapter 3, this run knows something went wrong -- the tool span
    # carries a `failed` note. What it still cannot say is whether the answer
    # was any good, and `ended` reports a clean stop either way.
    ended = "no tool_calls" if not reply.tool_calls else "out of written turns"
    trace.close(turns=2, messages=len(messages), ended=ended)
    return trace
