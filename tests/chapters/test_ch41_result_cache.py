# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""a cache of answers, not of attention -- exact match, then what a
paraphrase needs to hit it. `embed_all` computes one shared IDF over
every question compared in a scenario -- these tests check that the
similarity numbers match what the module docstring claims, computed
once and not tuned backward from the story.
"""

from chapters import ch41_result_cache as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_an_exact_repeat_never_calls_the_model_a_second_time() -> None:
    span = scenario_span("an_exact_match_hits_and_skips_the_call_entirely")
    note = span.require("the_second_call_never_happened")
    assert note.payload["skipped_entirely"] is True
    assert note.payload["model_spans_after_first_question"] == 1
    assert note.payload["model_spans_after_second_question"] == 1


def test_a_real_paraphrase_misses_exact_match_but_clears_the_high_threshold() -> None:
    span = scenario_span("a_paraphrase_misses_exact_match_but_hits_semantically")
    note = span.require("exact_missed_semantic_caught_it")
    assert note.payload["exact_match_hit"] is False
    assert note.payload["similarity"] == 0.57
    assert note.payload["served_from_cache"] is True


def test_a_naive_threshold_serves_the_wrong_cached_answer() -> None:
    span = scenario_span("a_naive_single_threshold_serves_a_confidently_wrong_answer")
    note = span.require("a_different_question_got_the_wrong_cached_answer")
    assert note.payload["similarity"] == 0.385
    assert note.payload["served_from_cache"] is True


def test_ttl_overrides_a_high_confidence_match() -> None:
    span = scenario_span("a_confident_match_past_its_sources_ttl_still_asks")
    note = span.require("ttl_overrode_an_otherwise_confident_match")
    assert note.payload["high_confidence"] is True
    assert note.payload["source_was_fresh"] is False
    assert note.payload["asked_for_confirmation"] is True
    # An "ask" with no confirmation available still has to end in an
    # answer -- the real gap caught by "on a cache miss, should it not
    # call the model again?"
    assert note.payload["fresh_answer"] is not None
    assert note.payload["fresh_answer_differs_from_stale_cache"] is True
    (turn_two,) = [c for c in span.children if c.attributes.get("number") == 2]
    assert [m.name for m in turn_two.children] == ["model"]


def test_the_ambiguous_middle_asks_rather_than_serving_or_missing() -> None:
    span = scenario_span("a_weak_match_within_ttl_also_asks_instead_of_guessing")
    note = span.require("the_ambiguous_middle_asks_rather_than_guesses")
    assert note.payload["zone"] == "ask"
    assert note.payload["neither_served_nor_missed"] is True
    assert note.payload["fresh_answer"] is not None


def test_the_structural_pair_scores_higher_than_the_true_paraphrase() -> None:
    """The finding named in the module docstring: no single threshold
    accepts the real paraphrase (0.57) and rejects this pair (0.669) --
    it is TF-IDF's actual ceiling, not a tuning mistake.
    """
    original_vector, structural_pair_vector = chapter.embed_all(
        ["What is our return policy?", "What is our shipping policy?"]
    )
    similarity = chapter._cosine_similarity(original_vector, structural_pair_vector)
    assert round(similarity, 3) == 0.669
    assert similarity > 0.57


def test_embed_all_uses_one_shared_idf_not_one_per_question() -> None:
    """Two vectors from the same embed_all call must be built against
    the same IDF -- comparing vectors from different IDF spaces was the
    real bug this chapter's numbers were caught drifting from.
    """
    a, b = chapter.embed_all(["reset my password", "reset my password"])
    assert a == b
