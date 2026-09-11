# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 25 asserts a document can out-rank the legitimate answer on the
query's own terms while also carrying an attack, and that a guard between
retrieval and the splice -- not inside the retriever -- catches it.
"""

from chapters import ch25_rag_injection as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_hostile_document_scores_higher_than_the_legitimate_one() -> None:
    legitimate_score = chapter._score(chapter.QUERY, chapter._LEGITIMATE)
    hostile_score = chapter._score(chapter.QUERY, chapter._HOSTILE)
    assert hostile_score > legitimate_score


def test_retrieval_on_the_benign_corpus_returns_the_legitimate_document() -> None:
    span = scenario_span("retrieval_returns_the_relevant_document")
    note = span.require("retrieved")
    assert note.payload["winner"] == "help-1"


def test_retrieval_on_the_poisoned_corpus_returns_the_hostile_document() -> None:
    span = scenario_span("a_well_crafted_document_wins_and_carries_an_injection")
    note = span.require("retrieved")
    assert note.payload["winner"] == "help-2"


def test_the_undefended_scenario_complies_with_the_marker() -> None:
    span = scenario_span("a_well_crafted_document_wins_and_carries_an_injection")
    note = span.require("compliance_check")
    assert note.payload["marker_present"] is True


def test_the_baseline_scenario_never_complies() -> None:
    span = scenario_span("retrieval_returns_the_relevant_document")
    note = span.require("compliance_check")
    assert note.payload["marker_present"] is False


def test_the_veto_guard_splices_in_no_document_at_all() -> None:
    span = scenario_span("guard_vetoes_the_retrieved_document")
    note = span.require("spliced_document")
    assert note.payload["present"] is False
    assert note.payload["text"] is None


def test_the_veto_guard_scenario_never_complies() -> None:
    span = scenario_span("guard_vetoes_the_retrieved_document")
    note = span.require("compliance_check")
    assert note.payload["marker_present"] is False


def test_the_sanitize_guard_keeps_a_clean_document_spliced_in() -> None:
    span = scenario_span("guard_sanitizes_the_retrieved_document")
    note = span.require("spliced_document")
    assert note.payload["present"] is True
    assert chapter.INJECTION_PHRASE not in note.payload["text"]
    assert chapter.MARKER not in note.payload["text"]


def test_the_sanitize_guard_scenario_never_complies() -> None:
    span = scenario_span("guard_sanitizes_the_retrieved_document")
    note = span.require("compliance_check")
    assert note.payload["marker_present"] is False


def test_refuse_injected_document_vetoes_only_the_hostile_one() -> None:
    legitimate = {"id": "x", "text": chapter._LEGITIMATE}
    hostile = {"id": "y", "text": chapter._HOSTILE}
    assert chapter.refuse_injected_document(legitimate).veto is None
    assert chapter.refuse_injected_document(hostile).veto is not None


def test_strip_after_marker_removes_exactly_the_injection_phrase() -> None:
    hostile = {"id": "y", "text": chapter._HOSTILE}
    verdict = chapter.strip_after_marker(hostile)
    assert verdict.replacement is not None
    assert chapter.INJECTION_PHRASE not in verdict.replacement["text"]
    assert verdict.replacement["text"] in chapter._HOSTILE


def test_no_scenario_in_this_chapter_ever_calls_a_tool() -> None:
    trace = chapter.run()
    assert trace.find_spans("tool") == []


def test_the_image_scenario_content_is_a_list_not_a_flattened_string() -> None:
    span = scenario_span("non_text_content_is_a_typed_block")
    note = span.require("spliced_content_shape")
    assert note.payload["content_type"] == "list"


def test_every_text_scenario_content_is_a_plain_string_by_contrast() -> None:
    for name in (
        "retrieval_returns_the_relevant_document",
        "a_well_crafted_document_wins_and_carries_an_injection",
    ):
        span = scenario_span(name)
        (model,) = [c for turn in span.children for c in turn.children if c.name == "model"]
        for m in model.context:
            assert isinstance(m["content"], str)


def test_the_pdf_scenario_content_is_also_a_typed_block() -> None:
    span = scenario_span("pdf_or_docx_uses_the_generic_file_block")
    note = span.require("spliced_content_shape")
    assert note.payload["content_type"] == "list"


def test_the_pdf_scenario_skips_the_rejection_probe_under_mock() -> None:
    span = scenario_span("pdf_or_docx_uses_the_generic_file_block")
    note = span.require("rejected_variant")
    assert note.payload["accepted"] is None
    assert note.payload["error"] == "not applicable under mock"
