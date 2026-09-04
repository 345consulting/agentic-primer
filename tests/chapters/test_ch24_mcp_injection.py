# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 24 asserts a hostile tool description can influence a model with
no tool call at all, and that a guard at declaration -- not dispatch --
stops it, either by refusing or by sanitizing.
"""

from chapters import ch24_mcp_injection as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_undefended_scenario_declares_the_hostile_description_verbatim() -> None:
    span = scenario_span("description_injection_reaches_the_model")
    note = span.require("declared_tools")
    assert note.payload["count"] == 1
    assert chapter.INJECTION_PHRASE in note.payload["descriptions"][0]


def test_the_undefended_scenario_complies_with_the_marker() -> None:
    span = scenario_span("description_injection_reaches_the_model")
    note = span.require("compliance_check")
    assert note.payload["marker_present"] is True


def test_the_veto_guard_drops_the_tool_entirely() -> None:
    span = scenario_span("guard_vetoes_the_declaration")
    note = span.require("declared_tools")
    assert note.payload["count"] == 0


def test_the_veto_guard_scenario_never_complies() -> None:
    span = scenario_span("guard_vetoes_the_declaration")
    note = span.require("compliance_check")
    assert note.payload["marker_present"] is False


def test_the_sanitize_guard_keeps_the_tool_with_a_clean_description() -> None:
    span = scenario_span("guard_sanitizes_instead_of_blocking")
    note = span.require("declared_tools")
    assert note.payload["count"] == 1
    description = note.payload["descriptions"][0]
    assert chapter.INJECTION_PHRASE not in description
    assert chapter.MARKER not in description
    assert description == chapter._CLEAN_DESCRIPTION


def test_the_sanitize_guard_scenario_never_complies() -> None:
    span = scenario_span("guard_sanitizes_instead_of_blocking")
    note = span.require("compliance_check")
    assert note.payload["marker_present"] is False


def test_refuse_injected_description_vetoes_only_the_hostile_entry() -> None:
    clean = chapter._catalog_entry(chapter._CLEAN_DESCRIPTION)
    hostile = chapter._catalog_entry(chapter._HOSTILE_DESCRIPTION)
    assert chapter.refuse_injected_description(clean).veto is None
    assert chapter.refuse_injected_description(hostile).veto is not None


def test_strip_after_marker_leaves_a_clean_description_untouched() -> None:
    clean = chapter._catalog_entry(chapter._CLEAN_DESCRIPTION)
    verdict = chapter.strip_after_marker(clean)
    assert verdict.replacement is None


def test_strip_after_marker_removes_exactly_the_injection_phrase() -> None:
    hostile = chapter._catalog_entry(chapter._HOSTILE_DESCRIPTION)
    verdict = chapter.strip_after_marker(hostile)
    assert verdict.replacement is not None
    assert verdict.replacement["description"] == chapter._CLEAN_DESCRIPTION


def test_no_scenario_in_this_chapter_ever_calls_a_tool() -> None:
    trace = chapter.run()
    assert trace.find_spans("tool") == []
