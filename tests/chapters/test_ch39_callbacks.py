# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The audit trail and the hook are the same mechanism in LangChain --
tested against langchain_core's real BaseCallbackHandler, not asserted.
"""

from chapters import ch39_callbacks as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_a_tracer_sees_the_full_chain_tool_and_model_edges() -> None:
    span = scenario_span("a_tracer_observes_without_participating")
    note = span.require("the_tracer_saw_every_edge")
    assert note.payload["saw_chain"] is True
    assert note.payload["saw_tool"] is True
    assert note.payload["saw_model"] is True


def test_the_recorder_fires_without_being_passed_a_config() -> None:
    span = scenario_span("callbacks_propagate_ambiently_without_being_passed")
    note = span.require("the_node_never_mentioned_the_recorder")
    assert note.payload["recorder_fired_anyway"] is True


def test_a_default_broken_handler_does_not_crash_the_call() -> None:
    span = scenario_span("a_default_callback_error_is_isolated")
    note = span.require("the_call_survived_the_broken_handler")
    assert note.payload["raised"] is False
    assert note.payload["tracer_still_fired"] is True


def test_raise_error_erases_a_completed_answers_own_audit_trail() -> None:
    span = scenario_span("a_tracer_can_turn_a_completed_answer_into_a_crash")
    note = span.require("a_completed_answer_became_an_unrecorded_crash")
    assert note.payload["raised"] is True
    assert note.payload["faulty_fired"] is True
    assert note.payload["tracer_suppressed"] is True


def test_a_callback_cannot_mutate_what_the_model_sees() -> None:
    span = scenario_span("a_callback_can_only_observe_or_abort_never_modify")
    note = span.require("the_mutation_attempt_changed_nothing")
    assert "reply" in note.payload


def test_a_default_broken_on_tool_error_handler_is_isolated() -> None:
    span = scenario_span("a_broken_handler_can_replace_the_real_tool_error_with_its_own")
    note = span.require("default_is_isolated")
    assert note.payload["tracer_still_fired"] is True
    assert note.payload["caller_saw_the_real_tool_error"] is True


def test_raise_error_substitutes_the_wrong_exception_for_a_tool_failure() -> None:
    span = scenario_span("a_broken_handler_can_replace_the_real_tool_error_with_its_own")
    note = span.require("raise_error_substitutes_the_wrong_exception")
    assert note.payload["tracer_suppressed"] is True
    assert note.payload["caller_saw_the_real_tool_error"] is False
    assert note.payload["caller_saw_the_handlers_own_error"] is True
