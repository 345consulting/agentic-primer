# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Instrumentation that observes and never participates -- checked
against a real opentelemetry-sdk TracerProvider and SpanExporter, not
asserted about our own hand-built Trace.
"""

from chapters import ch16_observability as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_a_real_otel_span_is_exported_to_disk() -> None:
    span = scenario_span("a_real_otel_span_carries_the_gen_ai_attributes")
    note = span.require("otel_span_exported")
    assert note.payload["path"].endswith("spans.jsonl")


def test_the_attribute_keys_are_the_real_gen_ai_constants() -> None:
    span = scenario_span("a_real_otel_span_carries_the_gen_ai_attributes")
    note = span.require("attributes_used_real_constants")
    assert note.payload["attribute_keys"] == ["gen_ai.system", "gen_ai.request.model"]


def test_the_exported_span_chain_matches_our_own_stacks_nesting() -> None:
    span = scenario_span("otel_nests_spans_via_contextvars_the_same_way_our_stack_does")
    note = span.require("the_chain_matches_our_own_stacks_nesting")
    assert note.payload["tool_parent_is_model"] is True
    assert note.payload["model_parent_is_scenario"] is True
    assert note.payload["scenario_has_no_parent"] is True
    assert note.payload["all_spans_share_one_trace_id"] is True


def test_a_broken_exporter_never_reaches_the_actual_call() -> None:
    span = scenario_span("a_broken_span_exporter_cannot_break_the_call")
    note = span.require("the_call_completed_despite_the_broken_exporter")
    assert note.payload["raised"] is False
    assert note.payload["reply"]


def test_usage_mapping_is_skipped_honestly_under_mock() -> None:
    span = scenario_span("usage_and_reasoning_tokens_map_to_the_spec")
    note = span.require("usage_mapped_under_mock_only_when_present")
    assert note.payload["usage_present"] is False
    assert note.payload["usage"] is None
