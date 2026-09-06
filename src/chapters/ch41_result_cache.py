# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""a cache of answers, not of attention -- exact match, then what a
paraphrase needs to hit it.

`ch28_caching` is the model provider's own prefix cache: byte-exact,
saving compute *inside* a call that still happens. This chapter is a
different layer entirely -- an application-level cache that skips the
call to the model *completely* when it has genuinely seen the question
before, exact or close enough.

**No neural embedding here, and that absence is itself a finding.**
DeepSeek's API -- checked against its own docs, not assumed -- offers
exactly three models (`deepseek-v4-flash`, `deepseek-v4-pro`,
`deepseek-v4-flash-vision-exp`), none of them an embeddings endpoint.
The obvious local alternative, `sentence-transformers`, was checked
against its own PyPI metadata and turned out to need `torch`,
`transformers`, `scikit-learn`, and `scipy` unconditionally -- the exact
dependency weight `ch42` already declined once for `routellm`, wearing a
different package name. Every real embedding option here is either a new
API key this project does not have, or a repeat of a dependency decision
already made the other way. So `embed()` below is hand-rolled TF-IDF
cosine similarity -- real, live, zero new dependencies -- and it is
honestly weaker than a learned embedding: it catches a paraphrase that
shares vocabulary with the original question, and nothing else. A true
synonym rephrasing with no shared words ("how do I reset my password"
vs. "I forgot how to log in") scores 0.385 here, below even this
chapter's most permissive threshold -- a real embedding model would
likely catch that case; TF-IDF cannot see past the missing overlap.

**Five scenarios, using real similarity scores computed once, up front,
not tuned backward from the story they were meant to tell:**

    an_exact_match_hits_and_skips_the_call_entirely
        the baseline `ch28` cannot make: an identical repeat never
        reaches the model at all, checked by comparing model-span
        counts before and after, not asserted
    a_paraphrase_misses_exact_match_but_hits_semantically
        a real paraphrase (0.570 similarity) misses the exact-match
        hash and clears this chapter's high-confidence threshold (0.55)
    a_naive_single_threshold_serves_a_confidently_wrong_answer
        a single cutoff set too low (0.35) lets a different, only
        adjacently-related question (0.385) through -- served the
        wrong cached answer, not a miss, not slow
    a_confident_match_past_its_sources_ttl_still_asks
        the same 0.570 paraphrase, but its source's TTL (backdated on
        purpose -- this is metadata under this chapter's own control,
        not a live timing claim) has expired. Surfaces for confirmation
        despite high similarity, proving TTL is a gate similarity
        cannot override
    a_weak_match_within_ttl_also_asks_instead_of_guessing
        a genuinely different question (0.450) that happens to sit in
        the ambiguous middle band -- too close to ignore, not close
        enough to trust. Surfaces for confirmation instead of guessing
        either way

**Found while computing the thresholds, not designed in from the
start: a structurally similar but substantively different pair --
"what is our return policy" vs. "what is our shipping policy" -- scores
0.669, *higher* than the genuine paraphrase's 0.570.** No single
threshold accepts the real paraphrase and rejects that pair; any cutoff
permissive enough for 0.570 already lets 0.669 through. That is not a
tuning mistake to fix -- it is TF-IDF's actual ceiling: shared sentence
scaffolding ("what is our ___ policy") outweighs the one word that
actually distinguishes the two questions. A learned embedding, trained
on meaning rather than word overlap, is the real fix for this specific
failure -- named here rather than papered over.

**What production actually reaches for, checked rather than assumed.**
GPTCache (`zilliztech/GPTCache`, real and actively maintained) is a
dedicated semantic-cache library: pluggable embedding providers (OpenAI,
HuggingFace, Cohere, ONNX) and pluggable vector backends (Milvus, FAISS,
Redis, Qdrant), with LangChain and llama_index integrations already
built. Closer to home, `langchain_redis.cache.RedisSemanticCache` plugs
directly into LangChain's own `set_llm_cache` with an embedding provider
and a `distance_threshold` -- this project already depends on
`langchain-core`, so that is a real graduation path already sitting in
the ecosystem this ladder lives in, not a hypothetical. Both replace
this chapter's hand-rolled dict-and-loop with a real vector index and a
real embedding call once the scale or the false-hit risk justifies it;
neither changes the two things this chapter actually teaches --
threshold placement and TTL as an independent gate -- which are
configuration decisions, not implementation ones.
"""

from support.agent import ask_model
from support.trace import ModelKind, Span, Trace

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage

type Vector = dict[str, float]

# Computed once, from real TF-IDF cosine similarity over this chapter's
# own question pairs -- see the module docstring for the actual numbers.
# Below LOW: too different to consider. LOW to HIGH: too close to
# ignore, not close enough to trust -- ask. At or above HIGH: serve.
SIMILARITY_LOW = 0.40

SIMILARITY_HIGH = 0.55

# A single naive cutoff, deliberately too permissive -- scenario three's
# subject, not the design used anywhere else in this chapter.
NAIVE_BAD_THRESHOLD = 0.35

SOURCE_TTLS: dict[str, timedelta] = {
    "policy_doc": timedelta(days=90),
    "inventory_system": timedelta(minutes=5),
}


@dataclass
class CachedAnswer:
    question: str
    vector: Vector
    answer: str
    source: str
    cached_at: datetime


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _term_frequencies(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _inverse_document_frequencies(corpus_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(corpus_tokens)
    document_frequency: Counter[str] = Counter()
    for tokens in corpus_tokens:
        for term in set(tokens):
            document_frequency[term] += 1
    return {
        term: math.log((1 + n_docs) / (1 + freq)) + 1 for term, freq in document_frequency.items()
    }


def _cosine_similarity(a: Vector, b: Vector) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[term] * b[term] for term in shared)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_all(questions: list[str]) -> list[Vector]:
    """Hand-rolled TF-IDF, standing in for a real embedding call -- see
    the module docstring for exactly why no neural embedding is used.

    One shared IDF over every question passed in, not one computed
    fresh per question -- two vectors built from different IDF spaces
    are not a principled comparison, even though cosine similarity
    would still return *a* number for them. Every question being
    compared in one scenario goes through this one call together.
    """
    idf = _inverse_document_frequencies([_tokenize(q) for q in questions])
    return [
        {term: freq * idf.get(term, 0.0) for term, freq in _term_frequencies(_tokenize(q)).items()}
        for q in questions
    ]


def _is_fresh(entry: CachedAnswer, now: datetime) -> bool:
    return now - entry.cached_at < SOURCE_TTLS[entry.source]


def _count_model_spans(span: Span) -> int:
    return sum(1 for c in span.children if c.name == "model") + sum(
        _count_model_spans(c) for c in span.children
    )


def an_exact_match_hits_and_skips_the_call_entirely(trace: Trace, model_kind: ModelKind) -> None:
    """The claim `ch28` cannot make: the second call never happens at
    all, checked by counting model spans, not asserted from the outside.
    """
    name = "an_exact_match_hits_and_skips_the_call_entirely"
    with trace.span("scenario", name=name) as span:
        exact_cache: dict[str, str] = {}
        question = "What is our return policy?"

        with trace.span("turn", number=1):
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage("Returns are accepted within 30 days of purchase.")],
                [HumanMessage(question)],
                [],
            )
        exact_cache[question] = str(reply.content)
        spans_after_first_question = _count_model_spans(span)

        with trace.span("decision", mechanism="exact_match") as decision_span:
            cached_answer = exact_cache.get(question)
            decision_span.add_note(
                "decided_before_calling_the_model",
                hit=cached_answer is not None,
                reason="identical question already in the exact-match cache",
            )
        spans_after_second_question = _count_model_spans(span)

        span.add_note(
            "the_second_call_never_happened",
            served_from_cache=cached_answer,
            model_spans_after_first_question=spans_after_first_question,
            model_spans_after_second_question=spans_after_second_question,
            skipped_entirely=spans_after_first_question == spans_after_second_question,
        )


def a_paraphrase_misses_exact_match_but_hits_semantically(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A real paraphrase misses the hash and clears the high-confidence
    threshold -- 0.570, computed once against this exact pair, not tuned
    backward from wanting a hit.
    """
    name = "a_paraphrase_misses_exact_match_but_hits_semantically"
    with trace.span("scenario", name=name) as span:
        original = "How do I reset my password?"
        paraphrase = "How can I reset my password if I forgot it?"

        with trace.span("turn", number=1):
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage("Click 'forgot password' on the login page.")],
                [HumanMessage(original)],
                [],
            )
        answer = str(reply.content)
        exact_cache = {original: answer}

        with trace.span("decision", mechanism="exact_match") as d1:
            exact_hit = paraphrase in exact_cache
            d1.add_note("decided_before_calling_the_model", hit=exact_hit)

        # One shared IDF over both questions -- two vectors built from
        # different IDF spaces (one computed when caching the original,
        # a second recomputed later once the query is known) would not
        # be a principled comparison, even though cosine similarity
        # would still return *a* number for them regardless.
        with trace.span("embed", questions=[original, paraphrase]) as embed_span:
            original_vector, query_vector = embed_all([original, paraphrase])
            embed_span.add_note(
                "embedded_together", term_counts=[len(original_vector), len(query_vector)]
            )
        cache = [CachedAnswer(original, original_vector, answer, "policy_doc", datetime.now(UTC))]

        with trace.span("decision", mechanism="semantic_match_three_zone") as d2:
            similarity = _cosine_similarity(cache[0].vector, query_vector)
            served = similarity >= SIMILARITY_HIGH
            d2.add_note(
                "decided_before_calling_the_model",
                similarity=round(similarity, 3),
                zone="serve" if served else ("ask" if similarity >= SIMILARITY_LOW else "miss"),
            )

        span.add_note(
            "exact_missed_semantic_caught_it",
            exact_match_hit=exact_hit,
            similarity=round(similarity, 3),
            served_from_cache=served,
            cached_answer=cache[0].answer if served else None,
        )


def a_naive_single_threshold_serves_a_confidently_wrong_answer(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A single cutoff, set too low, lets a genuinely different
    question through -- served, not missed, not slow, just wrong.
    """
    name = "a_naive_single_threshold_serves_a_confidently_wrong_answer"
    with trace.span("scenario", name=name) as span:
        original = "How do I reset my password?"
        different_question = "I forgot how to log in, what do I do?"

        with trace.span("turn", number=1):
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage("Click 'forgot password' on the login page.")],
                [HumanMessage(original)],
                [],
            )
        answer = str(reply.content)
        with trace.span("embed", questions=[original, different_question]) as embed_span:
            original_vector, query_vector = embed_all([original, different_question])
            embed_span.add_note(
                "embedded_together", term_counts=[len(original_vector), len(query_vector)]
            )
        cache = [CachedAnswer(original, original_vector, answer, "policy_doc", datetime.now(UTC))]

        with trace.span("decision", mechanism="naive_single_threshold") as decision_span:
            similarity = _cosine_similarity(cache[0].vector, query_vector)
            served = similarity >= NAIVE_BAD_THRESHOLD
            decision_span.add_note(
                "decided_before_calling_the_model",
                similarity=round(similarity, 3),
                threshold=NAIVE_BAD_THRESHOLD,
                served=served,
                would_have_been_correctly_rejected_at=SIMILARITY_LOW,
            )

        span.add_note(
            "a_different_question_got_the_wrong_cached_answer",
            similarity=round(similarity, 3),
            served_from_cache=served,
            served_answer_was_actually_about="resetting a password",
            asked_question_was_actually_about="being unable to log in at all",
        )


def a_confident_match_past_its_sources_ttl_still_asks(trace: Trace, model_kind: ModelKind) -> None:
    """A high-confidence match, from a source whose TTL has expired --
    backdated on purpose. Proves TTL is a gate similarity cannot
    override, not a second, redundant miss condition.
    """
    name = "a_confident_match_past_its_sources_ttl_still_asks"
    with trace.span("scenario", name=name) as span:
        original = "How do I reset my password?"
        paraphrase = "How can I reset my password if I forgot it?"

        with trace.span("turn", number=1):
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage("Click 'forgot password' on the login page.")],
                [HumanMessage(original)],
                [],
            )
        answer = str(reply.content)
        with trace.span("embed", questions=[original, paraphrase]) as embed_span:
            original_vector, query_vector = embed_all([original, paraphrase])
            embed_span.add_note(
                "embedded_together", term_counts=[len(original_vector), len(query_vector)]
            )

        # Sourced from inventory_system's short TTL and backdated well
        # past it -- this is metadata this chapter controls directly,
        # not a claim about real elapsed wall-clock time.
        stale_cache_entry = CachedAnswer(
            original,
            original_vector,
            answer,
            "inventory_system",
            datetime.now(UTC) - timedelta(hours=1),
        )

        with trace.span("decision", mechanism="semantic_match_plus_ttl") as decision_span:
            similarity = _cosine_similarity(stale_cache_entry.vector, query_vector)
            fresh = _is_fresh(stale_cache_entry, datetime.now(UTC))
            high_confidence = similarity >= SIMILARITY_HIGH
            asked_for_confirmation = high_confidence and not fresh
            decision_span.add_note(
                "decided_before_calling_the_model",
                similarity=round(similarity, 3),
                high_confidence=high_confidence,
                source=stale_cache_entry.source,
                ttl=str(SOURCE_TTLS[stale_cache_entry.source]),
                fresh=fresh,
                outcome="ask_despite_high_confidence" if asked_for_confirmation else "serve",
            )

        # A miss and an unconfirmed "ask" are the same fact from the
        # model's point of view: nothing was served, so an answer still
        # has to come from somewhere. Asking a human is this chapter's
        # own UX layer, not a model call it can await -- absent that
        # confirmation, the honest fallback is the same one a clean miss
        # already takes: call the model, get a real answer.
        fresh_reply = None
        if asked_for_confirmation:
            with trace.span("turn", number=2) as turn_span:
                fresh_reply = ask_model(
                    trace,
                    model_kind,
                    [
                        AIMessage(
                            "Click 'Forgot password?' on the sign-in page and check "
                            "your email for a reset link."
                        )
                    ],
                    [HumanMessage(paraphrase)],
                    [],
                )
                turn_span.add_note("called_fresh_because_nothing_was_served", question=paraphrase)

        span.add_note(
            "ttl_overrode_an_otherwise_confident_match",
            similarity=round(similarity, 3),
            high_confidence=high_confidence,
            source_was_fresh=fresh,
            asked_for_confirmation=asked_for_confirmation,
            fresh_answer=str(fresh_reply.content) if fresh_reply else None,
            fresh_answer_differs_from_stale_cache=(
                fresh_reply is not None and str(fresh_reply.content) != stale_cache_entry.answer
            ),
        )


def a_weak_match_within_ttl_also_asks_instead_of_guessing(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A genuinely different question, sitting in the ambiguous middle
    band -- too close to ignore, not close enough to trust. Surfaces for
    confirmation, the same outcome as the TTL scenario, from the other
    axis entirely.
    """
    name = "a_weak_match_within_ttl_also_asks_instead_of_guessing"
    with trace.span("scenario", name=name) as span:
        original = "How do I reset my password?"
        different_question = "How do I check my order status?"

        with trace.span("turn", number=1):
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage("Click 'forgot password' on the login page.")],
                [HumanMessage(original)],
                [],
            )
        answer = str(reply.content)
        with trace.span("embed", questions=[original, different_question]) as embed_span:
            original_vector, query_vector = embed_all([original, different_question])
            embed_span.add_note(
                "embedded_together", term_counts=[len(original_vector), len(query_vector)]
            )
        fresh_entry = CachedAnswer(
            original, original_vector, answer, "policy_doc", datetime.now(UTC)
        )

        with trace.span("decision", mechanism="semantic_match_three_zone") as decision_span:
            similarity = _cosine_similarity(fresh_entry.vector, query_vector)
            fresh = _is_fresh(fresh_entry, datetime.now(UTC))
            zone = (
                "serve"
                if similarity >= SIMILARITY_HIGH
                else ("ask" if similarity >= SIMILARITY_LOW else "miss")
            )
            decision_span.add_note(
                "decided_before_calling_the_model",
                similarity=round(similarity, 3),
                zone=zone,
                source_fresh=fresh,
            )

        # Same as the TTL scenario: "ask" with no confirmation available
        # still has to end in an answer, so it falls through to the same
        # call a clean miss would make.
        fresh_reply = None
        if zone == "ask":
            with trace.span("turn", number=2) as turn_span:
                fresh_reply = ask_model(
                    trace,
                    model_kind,
                    [AIMessage("You can check your order status from the 'My Orders' page.")],
                    [HumanMessage(different_question)],
                    [],
                )
                turn_span.add_note(
                    "called_fresh_because_nothing_was_served", question=different_question
                )

        span.add_note(
            "the_ambiguous_middle_asks_rather_than_guesses",
            similarity=round(similarity, 3),
            zone=zone,
            neither_served_nor_missed=zone == "ask",
            fresh_answer=str(fresh_reply.content) if fresh_reply else None,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch41_result_cache", model_kind=model_kind)

    an_exact_match_hits_and_skips_the_call_entirely(trace, model_kind)
    a_paraphrase_misses_exact_match_but_hits_semantically(trace, model_kind)
    a_naive_single_threshold_serves_a_confidently_wrong_answer(trace, model_kind)
    a_confident_match_past_its_sources_ttl_still_asks(trace, model_kind)
    a_weak_match_within_ttl_also_asks_instead_of_guessing(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
