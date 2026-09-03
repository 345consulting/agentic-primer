# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 20 asserts a third ending -- "vetoed", distinct from both ways
the loop already knew how to stop -- and that reaching it is a choice the
loop makes, not something dispatch_with_hooks forces on it.
"""

from chapters import ch20_loop_veto as chapter
from support.hooks import dispatch_with_hooks
from support.trace import Span, Trace

import inspect

from langchain_core.messages.tool import tool_call


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_hard_stop_ends_vetoed_not_turns_exhausted_or_no_tool_calls() -> None:
    span = scenario_span("hard_stop_ends_the_run_immediately")
    assert span.require("ended").payload["reason"] == "vetoed"


def test_hard_stop_never_asks_a_third_time() -> None:
    span = scenario_span("hard_stop_ends_the_run_immediately")
    turns = [c for c in span.children if c.name == "turn"]
    assert len(turns) == 2


def test_hard_stop_still_records_the_refused_call() -> None:
    # The run ended immediately, but the attempt is not erased from the trace.
    span = scenario_span("hard_stop_ends_the_run_immediately")
    tools = [c for turn in span.children for c in turn.children if c.name == "tool"]
    assert len(tools) == 2
    assert tools[1].find("dispatched") is None


def test_soft_continue_ends_no_tool_calls_the_same_as_any_other_run() -> None:
    span = scenario_span("soft_continue_lets_the_model_try_again")
    assert span.require("ended").payload["reason"] == "no_tool_calls"


def test_soft_continue_gets_a_turn_after_the_refusal() -> None:
    span = scenario_span("soft_continue_lets_the_model_try_again")
    turns = [c for c in span.children if c.name == "turn"]
    assert len(turns) == 4


def test_the_smaller_retry_fits_the_budget_and_dispatches() -> None:
    span = scenario_span("soft_continue_lets_the_model_try_again")
    tools = [c for turn in span.children for c in turn.children if c.name == "tool"]
    assert len(tools) == 3
    assert tools[2].require("args").payload == {"item": "milk", "quantity": 4}
    assert tools[2].require("dispatched").payload["value"] == "ordered 4 x milk"


def test_the_total_never_exceeds_the_budget_even_after_the_retry() -> None:
    # The model got what it wanted eventually, but the constraint the guard
    # actually enforces -- the running total -- was never violated to get it.
    span = scenario_span("soft_continue_lets_the_model_try_again")
    tools = [c for turn in span.children for c in turn.children if c.name == "tool"]
    dispatched = [t.require("dispatched").payload["value"] for t in tools if t.find("dispatched")]
    assert dispatched == ["ordered 8 x milk", "ordered 4 x milk"]


def test_run_soft_continue_is_run_turns_with_no_veto_specific_code() -> None:
    source = inspect.getsource(chapter.run_soft_continue)
    # The variable name from unpacking dispatch_with_hooks's tuple is fine --
    # what must not appear is a branch that reads it.
    assert '"vetoed"' not in source
    assert "if _vetoed" not in source and "if vetoed" not in source
    assert "run_turns(" in source


def test_dispatch_with_hooks_reports_the_veto_structurally() -> None:
    # ch20's whole reason to exist: a loop cannot tell a veto from an
    # ordinary failure by reading the ToolMessage's text.
    state = chapter.State(total_ordered=8)
    before = [chapter.make_budget_guard(state, budget=12)]
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 8}, id="c1")
    trace = Trace(chapter="test", model_kind="mock")
    with trace.span("scenario", name="probe"):
        _message, vetoed = dispatch_with_hooks(call, trace, before, [], chapter.TOOLS)
    assert vetoed is True
