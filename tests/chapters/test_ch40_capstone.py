# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""everything since graph, composed into one real scenario. `sse_frame_sent`
notes are recorded on whichever span the phase actually belongs to --
`items` for `checking_items`, `scenario` for the run-level frames -- so
these tests check structural position (tool nested in turn, the frame
sitting before the subtree it announces) as well as the phase sequence
and the cross-check against the real callback.
"""

from chapters import ch40_capstone as chapter
from support.trace import Note, Span, Trace

import asyncio

import pytest
from langchain_core.messages import AIMessage


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def entry_names(span: Span) -> list[str]:
    """The span's own `_entries` in true recorded order -- notes and
    children interleaved, a note's label standing in for its own name.
    """
    return [e.label if isinstance(e, Note) else e.name for e in span._entries]


def test_a_small_order_never_pauses_and_streams_five_phases() -> None:
    span = scenario_span("a_small_order_streams_straight_through_no_pause")
    note = span.require("the_small_order_never_paused")
    assert note.payload["events"] == [
        "run_started",
        "checking_items",
        "pricing_started",
        "order_placed",
        "run_ended",
    ]
    assert note.payload["approved_without_a_gate"] is True
    assert note.payload["total"] == 30.0
    assert note.payload["callback_events"] == ["model_start", "model_end"]


def test_a_large_order_pauses_then_a_separate_request_resumes_it() -> None:
    span = scenario_span("a_large_order_pauses_for_approval_then_resumes_from_checkpoint")
    note = span.require("the_pause_and_resume_were_two_separate_requests")
    assert note.payload["events"] == [
        "run_started",
        "checking_items",
        "pricing_started",
        "awaiting_approval",
        "resuming",
        "order_placed",
        "run_ended",
    ]
    assert note.payload["paused_before_resuming"] is True
    assert note.payload["approved_after_resume"] is True
    # 20 units at $10 = $200 subtotal; over 150 triggers the pricing
    # subgraph's discount, so the total that actually needed approval is
    # the post-discount figure, not the raw subtotal.
    assert note.payload["total"] == 180.0


def test_the_callback_and_the_trace_agree_on_model_calls_but_not_tool_calls() -> None:
    span = scenario_span("the_real_callback_trace_agrees_with_our_own")
    note = span.require("the_callback_sees_every_model_call_but_no_tool_call")
    assert note.payload["model_counts_agree"] is True
    assert note.payload["model_calls_the_trace_saw"] == 2
    assert note.payload["tool_calls_the_trace_saw"] == 2
    # The honest divergence this scenario exists to show: dispatch_with_hooks
    # never goes through a real LangChain Tool.invoke(), so the callback has
    # no path into it at all -- not a bug in either mechanism.
    assert note.payload["tool_calls_the_callback_could_have_seen"] == 0


def test_the_tool_call_is_nested_inside_the_turn_that_made_it() -> None:
    span = scenario_span("a_small_order_streams_straight_through_no_pause")
    (items,) = [c for c in span.children if c.name == "order"]
    (branch,) = [c for c in items.children if c.name == "branch"]
    (turn,) = [c for c in branch.children if c.name == "turn"]
    assert [c.name for c in turn.children] == ["model", "tool"]


def test_checking_items_is_announced_before_the_items_it_describes() -> None:
    span = scenario_span("a_small_order_streams_straight_through_no_pause")
    (items,) = [c for c in span.children if c.name == "order"]
    # The frame is the first entry inside items -- announcing the phase
    # before its subtree runs, not trailing after it once it is done.
    assert entry_names(items)[0] == "sse_frame_sent"
    assert entry_names(items)[1] == "branch"


def test_a_fully_declined_order_never_places_itself_as_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real bug this locks in: every item declining left `total=0.0`,
    indistinguishable from a genuinely free order, and the workflow
    placed it anyway. `declined` is what has to gate the outcome, not
    `total` -- a zero computed from zero successful checks is not a price.
    """

    def declining_ask_model(*_a: object, **_k: object) -> AIMessage:
        return AIMessage("I can't check that right now.")

    monkeypatch.setattr(chapter, "ask_model", declining_ask_model)

    trace = Trace(chapter="ch40_capstone", model_kind="mock")
    callback = chapter.TracingCallback()
    send, frames = chapter._collector()
    with trace.span("scenario", name="probe") as span:
        compiled, set_items_span = chapter._build_order_graph(trace, "mock", callback, send)
        result = asyncio.run(
            chapter._start_order_over_sse(
                send,
                trace=trace,
                scenario_span=span,
                compiled=compiled,
                set_items_span=set_items_span,
                thread=chapter._thread("declined-order"),
                items=[{"sku": "widget", "quantity": 1}],
            )
        )

    assert [f["event"] for f in frames] == [
        "run_started",
        "checking_items",
        "pricing_started",
        "order_incomplete",
        "run_ended",
    ]
    assert result["declined"] == ["widget"]
    assert result["total"] == 0.0
