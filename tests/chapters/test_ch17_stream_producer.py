# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""the harness becomes the server. `sse_frame_sent` notes are recorded
on the span each frame actually belongs to (scenario for run-level
events, turn for everything in between), so these tests check position
in the trace, not just the final summary note.
"""

from chapters import ch17_stream_producer as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def frame_events(span: Span) -> list[str]:
    return [n.payload["event"] for n in span.notes if n.label == "sse_frame_sent"]


def test_five_frames_arrive_in_the_order_the_run_actually_progresses() -> None:
    span = scenario_span("the_harness_streams_lifecycle_events_as_an_asgi_app")
    note = span.require("five_frames_arrived_in_order")
    assert note.payload["events"] == [
        "run_started",
        "turn_started",
        "model_replied",
        "turn_ended",
        "run_ended",
    ]
    assert note.payload["turns"] == 1
    assert note.payload["ended"] == "no_tool_calls"


def test_run_level_frames_are_recorded_on_the_scenario_span_not_a_turn() -> None:
    span = scenario_span("the_harness_streams_lifecycle_events_as_an_asgi_app")
    assert frame_events(span) == ["run_started", "run_ended"]
    (turn,) = [c for c in span.children if c.name == "turn"]
    assert frame_events(turn) == ["turn_started", "model_replied", "turn_ended"]


def test_the_event_stream_is_smaller_than_the_trace_it_came_from() -> None:
    span = scenario_span("the_event_stream_is_smaller_than_the_trace")
    note = span.require("the_stream_is_curated_not_mirrored")
    assert note.payload["frames_are_fewer"] is True
    assert note.payload["frame_count"] < note.payload["trace_entries_added"]


def test_a_streamed_reply_never_leaks_a_partial_frame() -> None:
    span = scenario_span("a_streamed_reply_arrives_as_one_complete_frame_not_pieces")
    note = span.require("exactly_one_complete_frame")
    assert note.payload["model_replied_frame_count"] == 1
    assert note.payload["content"] == "Milk is one pound twenty, and butter is two pounds."


def test_tool_dispatch_and_completion_are_two_distinct_frames() -> None:
    span = scenario_span("tool_dispatched_and_tool_completed_are_two_frames_not_one")
    note = span.require("dispatch_and_completion_are_distinct")
    assert note.payload["tool_dispatched_count"] == 1
    assert note.payload["tool_completed_count"] == 1
    assert note.payload["order"].index("tool_dispatched") < note.payload["order"].index(
        "tool_completed"
    )


def test_a_disconnected_client_does_not_stop_the_loop_from_finishing() -> None:
    span = scenario_span("a_client_that_disconnects_mid_run_does_not_crash_the_loop")
    note = span.require("the_loop_finished_though_nobody_was_listening")
    assert note.payload["loop_ran_to_completion"] is True
    assert note.payload["turns_completed"] == 2
    assert note.payload["ended"] == "no_tool_calls"
    # The client vanished partway through -- fewer frames were actually
    # delivered than the run's own six-per-turn-plus-two lifecycle, even
    # though the run itself did not stop early.
    assert note.payload["frames_actually_delivered"] < 10
