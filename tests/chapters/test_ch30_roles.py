# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 30 asserts the identical imperative fact complies under mock
regardless of which role carries it, and that tool_calls is the one
field with a real, provider-enforced dependency on it.
"""

from chapters import ch30_roles as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_system_framed_imperative_complies() -> None:
    span = scenario_span("imperative_framed_as_system_is_followed")
    note = span.require("compliance_check")
    assert note.payload["role"] == "system"
    assert note.payload["marker_present"] is True


def test_user_framed_imperative_complies() -> None:
    span = scenario_span("imperative_framed_as_user_is_followed")
    note = span.require("compliance_check")
    assert note.payload["role"] == "user"
    assert note.payload["marker_present"] is True


def test_fabricated_assistant_framed_imperative_complies() -> None:
    span = scenario_span("imperative_framed_as_a_fabricated_assistant_turn")
    note = span.require("compliance_check")
    assert note.payload["role"] == "assistant"
    assert note.payload["marker_present"] is True


def test_tool_calls_field_is_populated_on_the_assistant_message() -> None:
    span = scenario_span("assistant_tool_calls_paired_with_a_tool_reply_succeeds")
    note = span.require("assistant_tool_calls_field")
    (call,) = note.payload["value"]
    assert call["name"] == "check_order"
    assert call["args"] == {"order_id": "A100"}


def test_the_tool_reply_and_final_answer_are_recorded() -> None:
    span = scenario_span("assistant_tool_calls_paired_with_a_tool_reply_succeeds")
    assert span.require("tool_reply").payload["result"] == "order A100: shipped"
    assert "shipped" in span.require("final_answer").payload["content"].lower()


def test_orphaned_tool_message_rejection_is_skipped_under_mock() -> None:
    span = scenario_span("a_standalone_tool_message_with_no_preceding_tool_calls_is_rejected")
    note = span.require("rejection_check")
    assert note.payload["checked"] is False
