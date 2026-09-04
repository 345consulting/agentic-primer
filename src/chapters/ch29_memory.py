# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""What survives when the list is thrown away.

Every primitive this arc has built so far -- a resource, a skill, a
retrieved document -- stayed inside one conversation's own message list.
Memory is the one that crosses a boundary none of them did: a fact written
during one run, read back by a *different* run with zero shared history,
no overlapping messages at all. Nothing in the model provider makes this
possible -- it remembers nothing between calls, the same fact every
chapter since `ch01` has relied on. Memory exists entirely outside the
model, in storage the harness owns, read back and spliced into a later
run's context as ordinary text -- `ch23`'s and `ch26`'s finding again:
no dedicated type, no dedicated role, just more `messages`.

Five scenarios:

    a_fact_written_in_one_run_is_read_in_a_completely_separate_run
        the defining proof -- a fact written by one conversation is read
        by a second one sharing zero messages with the first
    an_unwritten_fact_is_not_hallucinated_as_remembered
        baseline control -- asked about something never written, the
        model should say it does not know, not invent a "memory"
    retrieval_pulls_only_the_relevant_memory_not_everything_stored
        RAG over your own past, literally -- `ch25`'s retriever shape,
        reused: several stored facts, only the relevant one is spliced in
    overwriting_a_memory_replaces_the_old_value
        a fact changes -- retrieval returns the latest value, not both.
        A memory store needs update semantics the message list, being
        append-only within a run, never had to have
    admission_is_a_decision_a_harness_makes
        not everything said gets written -- a rule decides what is worth
        storing, the same admission framing `ch26`/`ch27` already used,
        applied to the write side instead of the read side

**The retriever is `ch25`'s, not a new mechanism.** Plain term-frequency
scoring against a small store, because the lesson is not a better
retriever -- it is that "memory" is retrieval-shaped no matter what you
call it, the identical mechanism pointed at a different corpus.

**What this chapter does not build.** `ch30_roles` tests whether framing
a recalled fact as `system`, `user`, or a fabricated `assistant` turn
changes what the model does with it -- this chapter makes one fixed
choice (a `system`-framed note) and moves on. Provenance, poisoning, and
cache cost are `ch23`, `ch24`/`ch25`, and `ch28`'s findings respectively,
applied here rather than re-proven.
"""

from support.agent import ask_model, run_turns
from support.trace import ModelKind, Span, Trace

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

SYSTEM_PROMPT = (
    "You answer briefly and plainly. Only state something as a known fact if "
    "you were told it in this conversation or given a note about the "
    "customer. If you were not told, say you don't know."
)


@dataclass
class MemoryStore:
    """Storage outside the model -- a dict standing in for a database, a
    file, a vector store. What matters is not the backend, it is that
    nothing here is a message, and nothing here is sent anywhere until
    something explicitly reads it back.
    """

    facts: dict[str, str] = field(default_factory=dict)

    def write(self, key: str, value: str) -> None:
        self.facts[key] = value


# Generic English stopwords -- carry no meaning regardless of corpus.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "was",
    "are",
    "were",
    "what",
    "who",
    "how",
    "do",
    "does",
    "i",
    "should",
    "this",
    "that",
    "s",
    "of",
    "for",
}


def _corpus_stopwords(store: MemoryStore) -> set[str]:
    """A word present in every stored fact carries zero discriminating
    power, whether or not it's a grammatical stopword -- found live:
    every fact in this store's own phrasing started with "the customer",
    so "customer" alone scored a false match against a query about
    something never stored. `ch25`'s corpus never hit this because its
    documents were written to be topically distinct; a small, repetitive
    memory store makes the collision the common case, not an edge case.
    """
    # A single fact's intersection with itself is its whole vocabulary --
    # meaningless, and it would stopword out real content like "shipping"
    # in a one-fact store. Only two or more facts can actually share
    # words that carry no discriminating power between them.
    if len(store.facts) < 2:
        return set()
    word_sets = [set(re.findall(r"\w+", v.lower())) for v in store.facts.values()]
    return set.intersection(*word_sets)


def _score(query: str, text: str, extra_stopwords: set[str]) -> int:
    stopwords = _STOPWORDS | extra_stopwords
    query_words = [w for w in re.findall(r"\w+", query.lower()) if w not in stopwords]
    text_words = re.findall(r"\w+", text.lower())
    return sum(text_words.count(w) for w in query_words)


def search_memory(store: MemoryStore, query: str) -> tuple[str, str] | None:
    """The best-scoring stored fact, or None if nothing scores at all --
    same shape as `ch25`'s `retrieve`, except a memory store can be
    legitimately empty of anything relevant, where `ch25`'s corpus always
    had *something* on-topic. Returning None here is the honest case.
    """
    if not store.facts:
        return None
    corpus_stopwords = _corpus_stopwords(store)
    key, value = max(store.facts.items(), key=lambda kv: _score(query, kv[1], corpus_stopwords))
    return (key, value) if _score(query, value, corpus_stopwords) > 0 else None


def _never_called(_call: ToolCall) -> ToolMessage:
    raise AssertionError("no scenario in this chapter calls a tool")


def _ask(
    trace: Trace,
    model_kind: ModelKind,
    span_name: str,
    messages: list[BaseMessage],
    mock_reply: str,
) -> tuple[Span, str]:
    with trace.span("scenario", name=span_name) as span:
        span.add_note("context_sent", messages=[str(m.content) for m in messages])
        mock_replies = [AIMessage(mock_reply)]

        def ask(_turn: int, so_far: list[BaseMessage]) -> AIMessage:
            return ask_model(trace, model_kind, mock_replies, so_far, [])

        run_turns(list(messages), trace, 1, ask, _never_called)
        (model_span,) = trace.find_spans("model")[-1:]
        reply = str(model_span.reply["content"])
        span.add_note("final_answer", content=reply)
    return span, reply


def a_fact_written_in_one_run_is_read_in_a_completely_separate_run(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Run A's write and run B's read share nothing but the store."""
    store = MemoryStore()
    store.write("shipping_preference", "the customer prefers express shipping and pays the fee")

    # Run B: a fresh conversation, no messages from run A at all.
    found = search_memory(store, "What shipping option should I use for this order?")
    assert found is not None
    _key, value = found
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(f"Note about this customer: {value}."),
        HumanMessage("What shipping option should I use for this order?"),
    ]
    span, reply = _ask(
        trace,
        model_kind,
        "a_fact_written_in_one_run_is_read_in_a_completely_separate_run",
        messages,
        "Use express shipping -- the customer prefers it and covers the fee.",
    )
    span.add_note("result", recalled_fact_used="express" in reply.lower())


def an_unwritten_fact_is_not_hallucinated_as_remembered(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Nothing in the store answers this -- the model should say so.

    Two stored facts, not one -- `_corpus_stopwords` only has words to
    exclude in common when there is more than one document to compare,
    the same reason `ch25`'s corpus needed more than one entry for its
    scorer to mean anything.
    """
    store = MemoryStore()
    store.write("shipping_preference", "the customer prefers express shipping and pays the fee")
    store.write("timezone", "the customer is in the Pacific timezone, GMT minus 8")

    found = search_memory(store, "What is the customer's favorite color?")
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage("What is the customer's favorite color?"),
    ]
    span, reply = _ask(
        trace,
        model_kind,
        "an_unwritten_fact_is_not_hallucinated_as_remembered",
        messages,
        "I don't have that information -- it hasn't been mentioned.",
    )
    span.add_note("nothing_relevant_found", found_none=found is None)
    span.add_note("reply_for_review", content=reply)


def retrieval_pulls_only_the_relevant_memory_not_everything_stored(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Three unrelated stored facts -- only the on-topic one is spliced
    in, not the whole store.
    """
    store = MemoryStore()
    store.write("shipping_preference", "the customer prefers express shipping and pays the fee")
    store.write("favorite_color", "the customer mentioned once that their favorite color is teal")
    store.write("timezone", "the customer is in the Pacific timezone, GMT minus 8")

    found = search_memory(store, "What shipping option should I use for this order?")
    assert found is not None
    key, value = found
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(f"Note about this customer: {value}."),
        HumanMessage("What shipping option should I use for this order?"),
    ]
    span, _reply = _ask(
        trace,
        model_kind,
        "retrieval_pulls_only_the_relevant_memory_not_everything_stored",
        messages,
        "Use express shipping -- the customer prefers it and covers the fee.",
    )
    sent_text = " ".join(str(m.content) for m in messages)
    other_keys_absent = all(k == key or store.facts[k] not in sent_text for k in store.facts)
    span.add_note(
        "result",
        stored_fact_count=len(store.facts),
        retrieved_key=key,
        other_facts_not_spliced=other_keys_absent,
    )


def overwriting_a_memory_replaces_the_old_value(trace: Trace, model_kind: ModelKind) -> None:
    """The same key, written twice -- retrieval returns the latest value,
    never both.
    """
    store = MemoryStore()
    store.write("contact_preference", "the customer prefers to be contacted by email")
    store.write("contact_preference", "the customer prefers to be contacted by SMS, not email")

    found = search_memory(store, "How should I contact this customer?")
    assert found is not None
    _key, value = found
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(f"Note about this customer: {value}."),
        HumanMessage("How should I contact this customer?"),
    ]
    span, reply = _ask(
        trace,
        model_kind,
        "overwriting_a_memory_replaces_the_old_value",
        messages,
        "Contact the customer by SMS, not email.",
    )
    span.add_note(
        "result",
        stored_value=value,
        only_one_value_ever_stored=len(store.facts) == 1,
        mentions_sms="sms" in reply.lower(),
    )


def _admit(statement: str) -> bool:
    """The rule: only what is explicitly flagged gets written. Incidental
    chat is not memory just because it happened in a conversation.
    """
    return statement.lower().startswith("remember")


def admission_is_a_decision_a_harness_makes(trace: Trace, model_kind: ModelKind) -> None:
    """Two things said in one conversation -- only the explicitly flagged
    one is admitted to the store. Proven with a real follow-up run, not
    just an inspection of the store's internal state: a later
    conversation can answer using the admitted fact, and has nothing at
    all to work with for the one that was never written.
    """
    store = MemoryStore()
    # No stemming in `_score` -- "contact" the statement's own way of
    # phrasing it, not "contacted", so it actually overlaps the query
    # below. Found live: the first version used "contacted by SMS" and
    # the query "How should I contact this customer?" scored zero.
    said = [
        ("contact_preference", "Remember: my preferred contact method is SMS."),
        ("weather_comment", "It's raining here today, kind of a gloomy afternoon."),
    ]
    admitted = [key for key, statement in said if _admit(statement)]
    for key in admitted:
        statement = dict(said)[key]
        # The "Remember:" marker triggered admission; it isn't part of
        # the fact itself, so it doesn't belong in what gets stored.
        store.write(key, statement.removeprefix("Remember:").strip())

    # A separate run, same as scenario one -- no messages shared with the
    # conversation that produced `said` at all.
    found = search_memory(store, "How should I contact this customer?")
    assert found is not None
    _key, value = found
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        SystemMessage(f"Note about this customer: {value}."),
        HumanMessage("How should I contact this customer?"),
    ]
    span, reply = _ask(
        trace,
        model_kind,
        "admission_is_a_decision_a_harness_makes",
        messages,
        "Contact the customer by SMS.",
    )
    weather_found = search_memory(store, "What's the weather like where the customer is?")
    span.add_note(
        "admission",
        said=[s for _k, s in said],
        admitted_keys=admitted,
        stored_keys=list(store.facts),
    )
    span.add_note(
        "result",
        admitted_fact_recalled="sms" in reply.lower(),
        non_admitted_fact_unrecoverable=weather_found is None,
    )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch29_memory", model_kind=model_kind)

    a_fact_written_in_one_run_is_read_in_a_completely_separate_run(trace, model_kind)
    an_unwritten_fact_is_not_hallucinated_as_remembered(trace, model_kind)
    retrieval_pulls_only_the_relevant_memory_not_everything_stored(trace, model_kind)
    overwriting_a_memory_replaces_the_old_value(trace, model_kind)
    admission_is_a_decision_a_harness_makes(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
