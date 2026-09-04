# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 27 asserts what a compression pass actually costs, against a
fixed three-round seed conversation every scenario starts from.
"""

from chapters import ch27_compression as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_naive_truncation_orphans_a_tool_message() -> None:
    span = scenario_span("truncating_by_position_breaks_the_pairing")
    note = span.require("truncated")
    assert note.payload["orphaned_tool_message"] is True


def test_naive_truncation_is_not_checked_against_the_provider_under_mock() -> None:
    span = scenario_span("truncating_by_position_breaks_the_pairing")
    note = span.require("rejected")
    assert note.payload["accepted"] is None


def test_turn_truncation_keeps_the_dropped_rounds_fact_out_of_recall() -> None:
    span = scenario_span("truncating_by_turn_preserves_pairing_but_still_loses_information")
    note = span.require("recall_check")
    assert note.payload["fact_present"] is False


def test_only_tool_results_shrink_the_system_prompt_does_not() -> None:
    span = scenario_span("tool_results_are_compressed_the_system_prompt_never_is")
    note = span.require("compressed")
    assert note.payload["system_unchanged"] is True
    assert note.payload["tool_chars_after"] < note.payload["tool_chars_before"]


def test_recency_based_compression_shrinks_the_oldest_round_only() -> None:
    span = scenario_span("recent_turns_stay_verbatim_older_ones_shrink")
    note = span.require("shrunk")
    assert note.payload["oldest_round_kept_verbatim"] is False
    assert note.payload["newest_reply_kept_verbatim"] is True
    assert note.payload["message_count_after"] < note.payload["message_count_before"]


def test_summarization_is_a_real_extra_model_call() -> None:
    span = scenario_span("summarization_costs_a_model_call_and_still_loses_detail")
    note = span.require("summarized")
    assert note.payload["extra_model_calls"] == 1


def test_summarization_drops_the_exact_tracking_detail() -> None:
    span = scenario_span("summarization_costs_a_model_call_and_still_loses_detail")
    note = span.require("detail_check")
    assert note.payload["detail_present"] is False


def test_aggressive_compression_breaks_immediate_recall() -> None:
    span = scenario_span("compression_too_aggressive_breaks_the_very_next_turn")
    note = span.require("immediate_recall_check")
    assert note.payload["fact_present"] is False


def test_reactive_compression_leaves_a_short_conversation_untouched() -> None:
    span = scenario_span("reactive_compression_only_fires_when_actually_needed")
    note = span.require("policy_comparison")
    assert note.payload["reactive_did_nothing"] is True
    assert note.payload["proactive_compressed_anyway"] is True


def test_cache_check_is_skipped_under_mock() -> None:
    span = scenario_span("compression_invalidates_the_cache_prefix")
    note = span.require("cache_check")
    assert note.payload["checked"] is False


def test_pinned_result_survives_while_others_are_compressed() -> None:
    span = scenario_span("a_pinned_message_is_never_compressed_even_when_eligible")
    note = span.require("result")
    assert note.payload["pinned_survived"] is True
    assert note.payload["others_compressed"] == 2


def test_tool_filtering_reduces_declared_characters() -> None:
    span = scenario_span("only_relevant_tools_are_declared_this_turn")
    note = span.require("filtered")
    assert note.payload["selected"] == ["check_order"]
    assert note.payload["declared_selected_chars"] < note.payload["declared_full_chars"]


def test_protect_pinned_only_vetoes_the_pinned_id() -> None:
    from langchain_core.messages import ToolMessage

    guard = chapter._protect_pinned("pinned-id")
    assert guard(ToolMessage(content="x", tool_call_id="pinned-id")).veto is not None
    assert guard(ToolMessage(content="x", tool_call_id="other-id")).veto is None


def test_select_relevant_tools_falls_back_to_the_first_entry_when_nothing_matches() -> None:
    catalog = {"check_order": "x", "check_inventory": "y"}
    selected = chapter._select_relevant_tools("what time is it", catalog)
    assert selected == ["check_order"]
