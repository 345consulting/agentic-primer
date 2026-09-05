# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""InMemorySaver is ch31's snapshot-and-resume, reimplemented -- with no
explicit save call, and a finer checkpoint grain than a turn.
"""

from chapters import ch34_checkpoint as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_second_call_sees_the_first_calls_history() -> None:
    span = scenario_span("a_checkpointer_resumes_a_thread_across_separate_invoke_calls")
    note = span.require("second_call_saw_the_first")
    assert note.payload["second_call_input_message_count"] == 1
    assert note.payload["full_history_length"] == 5


def test_thread_b_never_sees_thread_as_order_number() -> None:
    span = scenario_span("two_thread_ids_never_see_each_others_history")
    note = span.require("thread_b_starts_empty")
    assert note.payload["thread_b_never_saw_a100"] is True


def test_the_checkpoint_already_holds_the_model_reply_before_the_crash() -> None:
    span = scenario_span("the_checkpoint_granularity_is_finer_than_our_snapshot")
    note = span.require("resumed_without_reasking_the_model")
    assert note.payload["crashed"] is True
    assert note.payload["checkpoint_captured_the_model_reply_before_the_crash"] is True


def test_resuming_reruns_only_the_failed_node_not_the_model() -> None:
    span = scenario_span("the_checkpoint_granularity_is_finer_than_our_snapshot")
    note = span.require("resumed_without_reasking_the_model")
    assert note.payload["call_model_invocations"] == 1
    assert note.payload["call_tool_invocations"] == 2
    assert note.payload["final_answer_present"] is True


def test_a_fresh_thread_has_no_checkpointed_state_at_all() -> None:
    span = scenario_span("a_checkpoint_is_not_memory")
    note = span.require("checkpoint_scope_vs_memory_scope")
    assert note.payload["fresh_thread_has_any_checkpointed_state"] is False


def test_the_cross_run_memory_store_finds_the_fact_regardless() -> None:
    span = scenario_span("a_checkpoint_is_not_memory")
    note = span.require("checkpoint_scope_vs_memory_scope")
    assert note.payload["cross_run_memory_still_finds_the_fact"] is True


def test_a_fresh_process_has_no_state_until_seeded_from_disk() -> None:
    span = scenario_span("we_can_bolt_our_own_disk_persistence_onto_it")
    note = span.require("resumed_after_a_simulated_process_restart")
    assert note.payload["fresh_process_had_no_state_before_seeding"] is True
    assert note.payload["seeding_from_disk_restored_state"] is True


def test_the_resumed_thread_carries_the_full_history() -> None:
    span = scenario_span("we_can_bolt_our_own_disk_persistence_onto_it")
    note = span.require("resumed_after_a_simulated_process_restart")
    assert note.payload["final_history_length"] == 5
