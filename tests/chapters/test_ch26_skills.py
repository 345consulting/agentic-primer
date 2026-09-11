# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 26 asserts a skill's result is read as an instruction, not
data -- and that the boundary is grammatical mood in theory, weaker than
that in practice, since live testing showed a purely descriptive sentence
still gets imitated most of the time.
"""

from chapters import ch26_skills as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_imperative_phrasing_is_complied_with() -> None:
    span = scenario_span("imperative_phrasing_is_treated_as_a_directive")
    note = span.require("compliance_check")
    assert note.payload["done_present"] is True


def test_declarative_phrasing_is_not_complied_with_under_mock() -> None:
    # Mock scripts the naive hypothesis deterministically -- the live
    # finding that this mostly does NOT hold for a real model is
    # documented in the chapter docstring, not asserted here.
    span = scenario_span("declarative_phrasing_of_the_same_fact")
    note = span.require("compliance_check")
    assert note.payload["done_present"] is False


def test_the_skill_persists_into_an_unrelated_second_turn() -> None:
    span = scenario_span("once_admitted_it_persists_across_turns")
    note = span.require("compliance_check")
    assert note.payload["done_present"] is True


def test_the_unlabeled_variant_carries_no_skill_name_in_its_result() -> None:
    span = scenario_span("a_provenance_label_helps_a_reader_not_the_model")
    (unlabeled, _labeled) = [n for n in span.notes if n.label == "loaded"]
    assert unlabeled.payload["variant"] == "imperative"
    assert unlabeled.payload["name_appears_in_result"] is False


def test_the_labeled_variant_carries_the_skill_name_in_its_result() -> None:
    span = scenario_span("a_provenance_label_helps_a_reader_not_the_model")
    (_unlabeled, labeled) = [n for n in span.notes if n.label == "loaded"]
    assert labeled.payload["variant"] == "labeled"
    assert labeled.payload["name_appears_in_result"] is True


def test_compliance_is_identical_regardless_of_the_label() -> None:
    span = scenario_span("a_provenance_label_helps_a_reader_not_the_model")
    checks = [n for n in span.notes if n.label == "compliance_check"]
    assert all(c.payload["done_present"] for c in checks)


def test_the_declarative_and_imperative_skills_state_the_same_fact() -> None:
    assert "DONE" in chapter._IMPERATIVE
    assert "DONE" in chapter._DECLARATIVE
    assert chapter._IMPERATIVE != chapter._DECLARATIVE
