# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 11 asserts a value that survives between turns, and who moves it.

`increment_order_total` is tested directly -- a read, a failed write, a successful write --
because those three cases are the whole of "who may write it," and a trace
can only ever show that they happened, not why the rule holds in general.
"""

from chapters import ch11_state as chapter
from support.trace import Span

import inspect

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.tools import StructuredTool


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def state_after(turn: Span) -> int:
    value: int = turn.require("state").payload["total_ordered"]
    return value


def test_a_read_leaves_the_total_alone() -> None:
    state = chapter.State(total_ordered=5)
    call = tool_call(name="price_of", args={"item": "milk"}, id="c1")
    reply = ToolMessage(content="1.20 for milk", tool_call_id="c1")
    chapter.increment_order_total(state, call, reply)
    assert state.total_ordered == 5


def test_a_failed_write_leaves_the_total_alone() -> None:
    state = chapter.State(total_ordered=5)
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 3}, id="c1")
    failed = ToolMessage(content="failed", tool_call_id="c1", status="error")
    chapter.increment_order_total(state, call, failed)
    assert state.total_ordered == 5


def test_a_successful_write_moves_the_total() -> None:
    state = chapter.State(total_ordered=5)
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 3}, id="c1")
    reply = ToolMessage(content="ordered 3 x milk", tool_call_id="c1")
    chapter.increment_order_total(state, call, reply)
    assert state.total_ordered == 8


def test_submit_order_never_receives_state() -> None:
    # Structural, not a convention: the tool's own signature has no place to
    # put it, so there is nothing in the call that could write the total.
    assert isinstance(chapter.place_order, StructuredTool)
    func = chapter.place_order.func
    assert func is not None
    assert "state" not in inspect.signature(func).parameters


def test_the_total_survives_between_turns() -> None:
    span = scenario_span("a_running_total_survives_between_calls")
    turn_one, turn_two, turn_three = span.children
    assert state_after(turn_one) == 5
    assert state_after(turn_two) == 8
    assert state_after(turn_three) == 8


def test_a_read_between_writes_does_not_reset_or_change_it() -> None:
    span = scenario_span("reads_do_not_touch_the_total")
    turn_one, turn_two, turn_three = span.children
    assert state_after(turn_one) == 0
    assert state_after(turn_two) == 4
    assert state_after(turn_three) == 4


def test_the_total_is_never_written_into_a_message() -> None:
    # The model only ever sees `messages`. If the total leaked into one, a
    # reader would find it there -- it should not be findable anywhere.
    trace = chapter.run()
    for scenario in trace.find_spans("scenario"):
        for turn in scenario.children:
            for model in [c for c in turn.children if c.name == "model"]:
                context = model.find("context")
                if context is not None:
                    for message in context.payload["messages"]:
                        assert "total_ordered" not in str(message.get("content", ""))
