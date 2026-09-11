# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 1 asserts the smallest set of facts: one call, and why it stopped."""

from chapters import ch01_single_call


def test_one_invocation_is_one_turn() -> None:
    trace = ch01_single_call.run()
    assert len(trace.find_spans("turn")) == 1
    assert trace.summary["turns"] == 1


def test_the_model_call_happens_inside_the_turn() -> None:
    # One shape for every chapter: a run holds scenarios, a scenario holds
    # turns, a turn holds the model call and whatever tools it asked for.
    trace = ch01_single_call.run()
    depths = {span.name: depth for depth, span in trace.walk()}
    assert depths["scenario"] == 0
    assert depths["turn"] == 1
    assert depths["model"] == 2


def test_the_context_sent_is_exactly_what_we_assembled() -> None:
    trace = ch01_single_call.run()
    assert [m["role"] for m in trace.find_spans("model")[0].context] == ["system", "human"]


def test_the_loop_would_end_because_the_reply_asks_for_no_tools() -> None:
    trace = ch01_single_call.run()
    assert "tool_calls" not in trace.find_spans("model")[0].reply
    assert trace.summary["ended"] == "no_tool_calls"


def test_the_caller_appended_the_reply_to_the_history() -> None:
    # Two messages went in, three exist after. The provider appended nothing.
    trace = ch01_single_call.run()
    assert len(trace.find_spans("model")[0].context) == 2
    assert trace.summary["messages"] == 3


def test_every_span_that_was_entered_was_exited() -> None:
    trace = ch01_single_call.run()
    assert all(span.exited_at is not None for _, span in trace.walk())
