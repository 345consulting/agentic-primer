# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""the same task, two models -- what a complexity signal actually has to
get right. The mock model ignores max_tokens entirely, so the scripted
reply itself has to carry the finish_reason each tier would actually
produce -- these tests check that the scripted outcome matches the tier
the routing function actually chose, not a hardcoded assumption.
"""

from chapters import ch42_model_tiers as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_route_by_length_is_fooled_by_the_hard_short_question() -> None:
    assert chapter.route_by_length(chapter.HARD_SHORT_QUESTION) == "cheap"


def test_the_length_heuristic_truncates_the_hard_question() -> None:
    span = scenario_span("a_length_based_heuristic_gets_fooled")
    note = span.require("the_short_question_was_actually_hard")
    assert note.payload["tier_chosen"] == "cheap"
    assert note.payload["was_truncated"] is True


def test_structural_heuristic_catches_math_but_misses_the_syllogism() -> None:
    span = scenario_span("a_structural_heuristic_does_better_but_still_guesses")
    note = span.require("better_at_one_thing_blind_to_another")
    assert note.payload["math_question_routed_generous"] is True
    assert note.payload["math_question_finished_clean"] is True
    assert note.payload["hard_question_still_routed_cheap"] is True
    assert note.payload["hard_question_still_truncated"] is True


def test_the_cascade_escalates_only_after_checking_the_result() -> None:
    span = scenario_span("a_cheap_models_own_uncertainty_escalates_correctly")
    note = span.require("checked_the_result_instead_of_guessing_the_question")
    assert note.payload["first_attempt_finish_reason"] == "length"
    assert note.payload["escalated"] is True
    assert note.payload["final_finish_reason"] == "stop"


def test_the_simulated_router_gets_both_decisions_right() -> None:
    span = scenario_span("a_simulated_learned_router_decides_with_no_visible_reason")
    note = span.require("correct_decisions_with_no_readable_rationale")
    assert note.payload["hard_question_routed_generous"] is True
    assert note.payload["hard_question_finished_clean"] is True
    assert note.payload["easy_question_routed_cheap"] is True
    assert note.payload["easy_question_finished_clean"] is True


def test_the_simulated_router_has_no_rule_relating_input_to_output() -> None:
    """The one thing this scenario is actually about: the decision is a
    dict lookup, not a function of anything observable in the question.
    """
    assert chapter.simulated_learned_router("a question never seen before") == "cheap"
    assert chapter.SIMULATED_ROUTER_DECISIONS[chapter.HARD_SHORT_QUESTION] == "generous"
