# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The agent's three verbs, once, after the chapters that teach them.

    an agent is a loop that sends the whole context to a model     ask_model
    executes whatever the model requests, appends the results      execute_tool
    and repeats until the model asks for nothing further           run_turns

`ch02_tool_call` teaches the first two by hand and `ch03_the_loop` teaches the
third; both keep their own copies, because a chapter that imports its own
subject teaches nothing. Everything after them composes these instead of
copying them -- `ask_model` had gone four chapters byte-identical, and the loop
was on its way to the same.

**Composition, not a base class.** These are functions taking what varies,
not methods to override: a reader of a later chapter can see the whole of its
control flow without following an inheritance chain, which is the rule that
made the copies tolerable in the first place.

**And what varies stays out of here.** These do the bookkeeping -- open a
span, record the context, record the reply, record the arguments and the
result. They make no decisions. A chapter that catches a failure and chooses
what the model is told keeps that in the chapter, because in `ch04`,
`ch07` and `ch08` the choice is the lesson.
"""

from support.models import build_model
from support.trace import ModelKind, Span, Trace

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

# What a chapter provides so a tool can be run: the name the model asked for,
# the arguments it chose, and the span to record into. `ch07_tool_http` and
# `ch08_tool_program` need that span, which is why dispatch is a callable
# rather than a lookup table.
type Dispatch = Callable[[str, dict[str, Any], Span], Any]

# One invocation, and what to do with whatever it asked for.
type Ask = Callable[[int, list[BaseMessage]], AIMessage]
type Execute = Callable[[ToolCall], ToolMessage]


def ask_model(
    trace: Trace,
    model_kind: ModelKind,
    mock_model_replies: Sequence[AIMessage],
    messages: list[BaseMessage],
    tools: Sequence[BaseTool],
    max_tokens: int | None = None,
) -> AIMessage:
    """One invocation, recorded: the exact context in, the reply out.

    `ch02_tool_call` writes this out by hand, with the comments explaining why
    the whole list goes every time and what `bind_tools` does and does not do.
    """
    with trace.span("model", model_kind=model_kind) as span:
        span.add_context(messages)
        model = build_model(model_kind, mock_model_replies, span, max_tokens)
        reply = model.bind_tools(list(tools)).invoke(messages) if tools else model.invoke(messages)
        span.add_reply(reply)
    # invoke() is typed as returning BaseMessage. Narrowing is for the type
    # checker, not for correctness.
    assert isinstance(reply, AIMessage)
    return reply


def execute_tool(call: ToolCall, trace: Trace, dispatch: Dispatch) -> ToolMessage:
    """One tool call, dispatched and recorded, with no opinion about failure.

    A tool that raises raises through this. Catching it is a decision, and the
    chapters that make one make it themselves.
    """
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        result = dispatch(call["name"], call["args"], span)
        span.add_note("result", value=result)
    # `tool_call_id` is the only thing tying this result to the request that
    # asked for it. The model provider matches on that, not on position.
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run_turns(
    messages: list[BaseMessage], trace: Trace, turn_cap: int, ask: Ask, execute: Execute
) -> tuple[int, int, str]:
    """`ch03_the_loop`'s loop, moved here after that chapter taught it.

    Returns how many turns ran, how long the list got, and how it ended. The
    two endings are the model asking for nothing and us stopping it, and a run
    that finished looks exactly like a run that was stopped unless the summary
    says which.
    """
    turns = 0
    ended = "turns_exhausted"

    while turns < turn_cap:
        turns += 1
        with trace.span("turn", number=turns):
            reply = ask(turns, messages)
            messages.append(reply)
            if not reply.tool_calls:
                ended = "no_tool_calls"
                break
            for call in reply.tool_calls:
                messages.append(execute(call))

    return turns, len(messages), ended
