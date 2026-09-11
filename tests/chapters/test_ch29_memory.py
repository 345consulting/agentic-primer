# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 29 asserts a fact written by one conversation is read by a
completely separate one sharing zero messages, and that retrieval,
overwriting, and admission all behave the way this primer's own
throughline (everything is a prompt, everything else is a decision) says
they should.
"""

from chapters import ch29_memory as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_a_fact_written_in_one_run_is_recalled_in_a_separate_one() -> None:
    span = scenario_span("a_fact_written_in_one_run_is_read_in_a_completely_separate_run")
    note = span.require("result")
    assert note.payload["recalled_fact_used"] is True


def test_an_unwritten_fact_is_not_found_by_search() -> None:
    span = scenario_span("an_unwritten_fact_is_not_hallucinated_as_remembered")
    note = span.require("nothing_relevant_found")
    assert note.payload["found_none"] is True


def test_retrieval_finds_only_the_relevant_fact_among_three() -> None:
    span = scenario_span("retrieval_pulls_only_the_relevant_memory_not_everything_stored")
    note = span.require("result")
    assert note.payload["stored_fact_count"] == 3
    assert note.payload["retrieved_key"] == "shipping_preference"
    assert note.payload["other_facts_not_spliced"] is True


def test_overwriting_a_key_leaves_only_the_latest_value() -> None:
    span = scenario_span("overwriting_a_memory_replaces_the_old_value")
    note = span.require("result")
    assert note.payload["only_one_value_ever_stored"] is True
    assert "SMS" in note.payload["stored_value"] or "sms" in note.payload["stored_value"].lower()


def test_only_the_explicitly_flagged_statement_is_admitted() -> None:
    span = scenario_span("admission_is_a_decision_a_harness_makes")
    note = span.require("admission")
    assert note.payload["admitted_keys"] == ["contact_preference"]
    assert note.payload["stored_keys"] == ["contact_preference"]


def test_search_memory_returns_none_for_an_empty_store() -> None:
    store = chapter.MemoryStore()
    assert chapter.search_memory(store, "anything at all") is None


def test_corpus_stopwords_are_empty_for_a_single_fact_store() -> None:
    store = chapter.MemoryStore()
    store.write("only", "the customer likes tea")
    assert chapter._corpus_stopwords(store) == set()


def test_corpus_stopwords_exclude_words_shared_by_every_fact() -> None:
    store = chapter.MemoryStore()
    store.write("a", "the customer likes tea")
    store.write("b", "the customer dislikes coffee")
    stopwords = chapter._corpus_stopwords(store)
    assert "customer" in stopwords
    assert "tea" not in stopwords
    assert "coffee" not in stopwords


def test_admit_requires_the_remember_prefix() -> None:
    assert chapter._admit("Remember: I like tea.") is True
    assert chapter._admit("I like tea.") is False
