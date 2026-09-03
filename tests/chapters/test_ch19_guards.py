# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 19 asserts a veto reading what already happened, not just the
call in front of it.
"""

from chapters import ch19_guards as chapter
from support.trace import Span

from langchain_core.messages.tool import tool_call


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_a_call_within_budget_is_unaffected_by_an_empty_state() -> None:
    state = chapter.State()
    guard = chapter.make_budget_guard(state, budget=12)
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 8}, id="c1")
    assert guard(call).veto is None


def test_the_same_call_is_refused_once_the_budget_is_already_spent() -> None:
    # The call itself never changes -- only what state already holds does.
    state = chapter.State(total_ordered=8)
    guard = chapter.make_budget_guard(state, budget=12)
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 8}, id="c1")
    assert guard(call).veto is not None


def test_the_guard_ignores_calls_it_was_not_built_to_watch() -> None:
    state = chapter.State(total_ordered=100)
    guard = chapter.make_budget_guard(state, budget=12)
    call = tool_call(name="price_of", args={"item": "milk"}, id="c1")
    assert guard(call).veto is None


def test_only_the_recorder_writes_total_ordered() -> None:
    state = chapter.State()
    recorder = chapter.make_order_recorder(state)
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 5}, id="c1")
    recorder(call, "ordered 5 x milk")
    assert state.total_ordered == 5


def test_a_read_does_not_move_the_total() -> None:
    state = chapter.State()
    recorder = chapter.make_order_recorder(state)
    call = tool_call(name="price_of", args={"item": "milk"}, id="c1")
    recorder(call, "1.20 for milk")
    assert state.total_ordered == 0


def test_a_single_order_within_budget_succeeds() -> None:
    span = scenario_span("an_order_within_budget")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    assert tool_span.require("result").payload["value"] == "ordered 8 x milk"


def test_the_second_of_a_split_order_is_the_one_refused() -> None:
    span = scenario_span("a_split_order_exceeds_the_budget")
    tool_spans = [c for c in span.children if c.name == "tool"]
    assert len(tool_spans) == 2
    first, second = tool_spans
    assert first.require("result").payload["value"] == "ordered 8 x milk"
    assert first.find("hook") is None or all(
        n.payload["veto"] is None for n in first.notes if n.label == "hook"
    )
    assert second.find("result") is None
    assert second.find("dispatched") is None


def test_the_first_call_alone_would_have_been_fine() -> None:
    # The point of the chapter: the refused call is not malformed. It is the
    # same shape as the one that just succeeded, refused only because of
    # what the run already holds.
    span = scenario_span("a_split_order_exceeds_the_budget")
    tool_spans = [c for c in span.children if c.name == "tool"]
    first_args = tool_spans[0].require("args").payload
    second_args = tool_spans[1].require("args").payload
    assert first_args == second_args == {"item": "milk", "quantity": 8}
