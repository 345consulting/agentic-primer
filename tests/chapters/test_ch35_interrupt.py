# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""interrupt and Command(resume=...) as an approval gate. Every scenario
wraps its invoke() calls in explicit "pause"/"resume" spans, ch34's own
"checkpoint" span pattern, so the boundary is visible, not just a single
summary note at the end.
"""

from chapters import ch35_interrupt as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def child_span(parent: Span, name: str) -> Span:
    (span,) = [c for c in parent.children if c.name == name]
    return span


def test_invoke_returns_normally_with_the_interrupt_value() -> None:
    span = scenario_span("interrupt_pauses_the_run_and_returns_immediately")
    note = span.require("the_call_returned_instead_of_raising")
    assert note.payload["raised"] is False
    assert note.payload["interrupt_present"] is True
    assert note.payload["interrupt_value"] == {"question": "approve this order?", "total": 500}


def test_the_pause_checkpoint_shows_place_order_pending() -> None:
    span = scenario_span("command_resume_dispatches_straight_to_the_pending_node")
    pause = child_span(span, "pause")
    note = pause.require("checkpoint_after_pause")
    assert note.payload["pending_node"] == ["place_order"]
    assert note.payload["step_one_run_count"] == 1


def test_only_the_pending_node_reruns_on_resume() -> None:
    span = scenario_span("command_resume_dispatches_straight_to_the_pending_node")
    resume = child_span(span, "resume")
    note = resume.require("checkpoint_after_resume")
    assert note.payload["pending_node"] == []
    assert note.payload["step_one_run_count"] == 1

    summary = span.require("only_the_pending_node_reran")
    assert summary.payload["step_one_run_count"] == 1
    assert summary.payload["final_approved"] is True


def test_code_before_interrupt_runs_exactly_twice() -> None:
    span = scenario_span("code_before_the_interrupt_call_runs_twice_on_resume")
    pause = child_span(span, "pause")
    resume = child_span(span, "resume")
    assert pause.require("before_interrupt_runs_so_far").payload["count"] == 1
    assert resume.require("before_interrupt_runs_so_far").payload["count"] == 2

    summary = span.require("the_pre_interrupt_line_ran_twice")
    assert summary.payload["before_interrupt_runs"] == 2


def test_the_gate_actually_gates_the_order() -> None:
    span = scenario_span("an_approval_gate_actually_gates_the_action")
    note = span.require("the_gate_actually_gated_the_order")
    assert note.payload["orders_placed"] == ["ordered 500 x widget"]
    assert note.payload["approved_thread_final"] == "ordered 500 x widget"
    assert note.payload["denied_thread_final"] == "denied by approver"
