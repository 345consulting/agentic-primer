# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 3 asserts that nothing in the trace notices the question was unanswered.

The mechanics are chapter 2's and are already tested there. What is worth
asserting here is the absence: no error, no failed call, no non-standard stop
reason -- and an answer to a question nobody asked.
"""

from chapters import ch03_missing_tool

from typing import get_args


def test_no_tool_covers_the_question_that_was_asked() -> None:
    # The premise. Nothing here knows about deliveries, and nothing says so.
    assert "delivery" in ch03_missing_tool.USER_PROMPT
    assert [tool.name for tool in ch03_missing_tool.DECLARED_TOOLS] == ["stock_on_hand"]


def test_the_model_reaches_for_the_nearest_tool_it_has() -> None:
    # It does not invent a tool: the declaration is enforced by the model
    # provider before a call exists. It calls the one thing it was offered.
    trace = ch03_missing_tool.run()
    asked = [call["name"] for call in trace.find_spans("model")[0].reply["tool_calls"]]
    assert asked == ["stock_on_hand"]


def test_the_tool_call_succeeded_and_answered_a_different_question() -> None:
    (tool,) = ch03_missing_tool.run().find_spans("tool")
    assert tool.require("result").payload["value"] == 2
    assert tool.find("error") is None


def test_nothing_in_the_trace_records_a_failure() -> None:
    """The lesson. A harness counting tool calls and completions sees a clean run.

    Every span opened and closed, the call returned a real number, and the run
    ended the way a finished conversation ends. The gap between what was asked
    and what the tools cover is not represented anywhere.
    """
    trace = ch03_missing_tool.run()
    assert trace.summary["ended"] == "no tool_calls"
    assert all(span.exited_at is not None for _, span in trace.walk())
    labels = {note.label for _, span in trace.walk() for note in span.notes}
    assert not {label for label in labels if "error" in label or "fail" in label}


def test_the_answer_admits_what_the_run_does_not() -> None:
    # Only the prose says anything went wrong, and prose is not a field.
    trace = ch03_missing_tool.run()
    answer = trace.find_spans("model")[1].reply["content"]
    assert "don't have access" in answer
    assert not trace.find_spans("model")[1].reply.get("tool_calls")


def test_the_schema_and_the_lookup_allow_exactly_the_same_items() -> None:
    permitted = set(get_args(ch03_missing_tool.GroceryItem.__value__))
    assert permitted == set(ch03_missing_tool.STOCK_ON_HAND)
