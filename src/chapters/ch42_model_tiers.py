# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""the same task, two models -- getting a complexity signal right.

This project has exactly one live model, chosen for price since `ch01`.
There is no second, larger provider configured, and there will not be
one here: the "tier" below is a `max_tokens` budget on that single
model, not a real model swap. That is a real, defensible cost lever on
its own -- constraining response length genuinely bounds cost -- but
this chapter does not claim to prove that a smaller *model* answers as
well as a larger one. It claims something narrower and fully provable
here: that guessing a budget from the question's surface form fails in
a specific, findable way, and that checking the result you actually got
is a different, better-grounded strategy than guessing ever was.

Four scenarios, in increasing sophistication:

    a_length_based_heuristic_gets_fooled
        the article's own heuristic (`len(question) < N`), tried
        honestly -- a short, conceptually hard question routes to the
        cheap tier and comes back cut off mid-answer. finish_reason ==
        "length", found live, not assumed
    a_structural_heuristic_does_better_but_still_guesses
        regex for math notation, code fences, multi-part phrasing --
        catches a question length alone would miss, and is still fooled
        by the exact same hard question as above, because nothing about
        its grammar is unusual. A sharper instrument pointed at the same
        wrong target
    a_cheap_models_own_uncertainty_escalates_correctly
        no upfront guess at all -- call the cheap tier, read what
        actually came back (finish_reason), escalate to the generous
        tier only when that signal says the first attempt fell short.
        The cascade this ladder's own ch09_who_retries matrix already
        argued for, applied to a new failure class: succeeded, but not
        well enough
    a_simulated_learned_router_decides_with_no_visible_reason
        a stand-in for what a real trained router -- RouteLLM, Not
        Diamond, Martian -- returns: a label, scripted exactly the way
        MockModel scripts a reply, explicitly not a trained model.
        There is nothing to point at in this scenario's own source and
        say "this is why" the way there is in the two heuristics above --
        that is the honest cost of routing this way, not a mock
        limitation this chapter happens to have

**Why scenario four is simulated, not real, for two separate reasons.**
The real `routellm` package pulls in `torch`, `transformers`, `datasets`,
`scikit-learn`, and `litellm` as unconditional base dependencies --
checked directly against its `pyproject.toml`, not assumed -- against a
project whose entire dependency list today is `langchain-core`,
`langchain-deepseek`, `httpx`, and `opentelemetry-*`. That alone would be
a real, written-down deviation. But installing it would not actually
answer this chapter's question even if the weight were free: RouteLLM's
published checkpoints (`routellm/mf_gpt4_augmented` and siblings) are
trained on Chatbot Arena preference data -- general chit-chat and
open-domain Q&A. Their own claim is that a router generalizes across
*model pairs* it was not trained on; there is no claim it generalizes
across *query distributions* it was not trained on. Complexity is not a
property of a sentence -- it is a property of the relationship between a
specific query and a specific weak model's specific failure history on a
specific business's actual traffic. A retail company's own SKU codes
mean nothing to a generically-pretrained router; only that company's own
labeled outcomes could teach one. This project has no business and no
outcome history to train on, so the honest move is to simulate the
*shape* of what a trained router returns -- a decision, and nothing that
explains it -- not to pretend a generic checkpoint would answer a
question it was never trained to answer.

**Structural heuristics were checked against NLTK and rejected on
purpose, not by default.** POS tagging, a real NLTK strength, detects
grammatical mood -- useful for a genuinely different problem, catching
imperative sentences ("ignore previous instructions") for an injection
heuristic in `ch24`/`ch25`'s territory. It would not help here: the flaw
in a structural complexity signal is not measurement precision, it is
that the whole category of signal is weak regardless of how precisely
it's measured. A sharper parse still cannot see that a plain,
grammatically unremarkable question is a classic syllogism fallacy in
disguise. Plain regex proves the same limitation as a heavier parser would, and
keeps the heuristic fully readable -- which is the exact property
scenario four's simulated router does not have, and the contrast is
the point.
"""

from support.agent import ask_model
from support.trace import ModelKind, Trace

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

type Tier = Literal["cheap", "generous"]

CHEAP_MAX_TOKENS = 20

GENEROUS_MAX_TOKENS = 300

# Short enough to fool a length-based heuristic; nothing in its grammar
# marks it as hard -- no operators, no code, one plain clause. A classic
# syllogism-fallacy question, chosen after a first candidate (a
# self-referential consistency question) was found live to trigger a
# genuine runaway reasoning loop -- 8000 reasoning tokens at an 8000
# token budget, never converging at any size tested. That is a real,
# reproducible model behavior, not a routing question, and not this
# chapter's subject; this question needs real reasoning without falling
# into that trap.
HARD_SHORT_QUESTION = "If some cats are pets, are all pets cats?"

EASY_SHORT_QUESTION = "What is 2 + 2?"

HARD_QUESTION_ANSWER = (
    "No. “Some cats are pets” only tells us that at least one cat is a "
    "pet -- it says nothing about every pet being a cat. Dogs, goldfish, "
    "and parakeets are all pets too, and none of them are cats."
)

HARD_QUESTION_TRUNCATED = "No. “Some cats are pets” only tells us that at"

MATH_QUESTION = "Solve for x: 2x + 5 = 15"

MATH_PATTERN = re.compile(r"[+\-*/^=]|\bsolve\b|\bderivative\b|\bintegral\b", re.IGNORECASE)

CODE_PATTERN = re.compile(r"```|\bdef \b|\bfunction\b")

MULTI_PART_PATTERN = re.compile(r"\?.*\?|\bfirst\b.*\bthen\b", re.IGNORECASE)

# A stand-in for a real trained router's own output, scripted exactly
# the way MockModel scripts a reply -- not derived from the question by
# any rule visible in this file, because a real one would not be either.
SIMULATED_ROUTER_DECISIONS: dict[str, Tier] = {
    HARD_SHORT_QUESTION: "generous",
    EASY_SHORT_QUESTION: "cheap",
}


def route_by_length(question: str, threshold: int = 50) -> Tier:
    """The article's own heuristic, tried honestly rather than dismissed."""
    return "cheap" if len(question) < threshold else "generous"


def route_by_structure(question: str) -> Tier:
    """Math notation, code, or multi-part phrasing -- a better guess,
    still a guess.
    """
    if (
        MATH_PATTERN.search(question)
        or CODE_PATTERN.search(question)
        or MULTI_PART_PATTERN.search(question)
    ):
        return "generous"
    return "cheap"


def simulated_learned_router(question: str) -> Tier:
    """What a real trained router's interface looks like from the
    outside: a label in, nothing in this function's own body that
    explains why. See the module docstring for why this is simulated
    rather than `routellm` itself.
    """
    return SIMULATED_ROUTER_DECISIONS.get(question, "cheap")


def _max_tokens_for(tier: Tier) -> int:
    return CHEAP_MAX_TOKENS if tier == "cheap" else GENEROUS_MAX_TOKENS


def _structural_signal(question: str) -> str:
    """Which pattern, if any, `route_by_structure` actually matched --
    the "why" a `decision` span needs to be more than a bare label.
    """
    if MATH_PATTERN.search(question):
        return "math_pattern"
    if CODE_PATTERN.search(question):
        return "code_pattern"
    if MULTI_PART_PATTERN.search(question):
        return "multi_part_pattern"
    return "no_pattern_matched"


def a_length_based_heuristic_gets_fooled(trace: Trace, model_kind: ModelKind) -> None:
    """A short, hard question -- routed cheap by length alone, and cut
    off mid-answer.
    """
    name = "a_length_based_heuristic_gets_fooled"
    with trace.span("scenario", name=name) as span:
        with trace.span("decision", mechanism="length_heuristic") as decision_span:
            tier = route_by_length(HARD_SHORT_QUESTION)
            decision_span.add_note(
                "decided_before_calling_the_model",
                tier=tier,
                question_length=len(HARD_SHORT_QUESTION),
                threshold=50,
                reason=f"{len(HARD_SHORT_QUESTION)} characters is under the 50-character threshold",
            )
        with trace.span("turn", number=1) as turn_span:
            # The mock script itself has to reflect the tier actually
            # computed, not a hardcoded outcome -- otherwise the trace
            # would show "truncated" regardless of whether the heuristic
            # got it right, proving nothing about the routing decision.
            script = (
                AIMessage(
                    HARD_QUESTION_TRUNCATED,
                    response_metadata={"finish_reason": "length"},
                )
                if tier == "cheap"
                else AIMessage(HARD_QUESTION_ANSWER, response_metadata={"finish_reason": "stop"})
            )
            reply = ask_model(
                trace,
                model_kind,
                [script],
                [HumanMessage(HARD_SHORT_QUESTION)],
                [],
                max_tokens=_max_tokens_for(tier),
            )
            finish_reason = reply.response_metadata.get("finish_reason")
            turn_span.add_note("heuristic_routed_here", tier=tier, max_tokens=_max_tokens_for(tier))
        span.add_note(
            "the_short_question_was_actually_hard",
            tier_chosen=tier,
            finish_reason=finish_reason,
            was_truncated=finish_reason == "length",
            content=str(reply.content),
        )


def a_structural_heuristic_does_better_but_still_guesses(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Catches the math question length alone would not have flagged --
    and is fooled by the exact same hard question scenario one used,
    because nothing about its grammar looks unusual.
    """
    name = "a_structural_heuristic_does_better_but_still_guesses"
    with trace.span("scenario", name=name) as span:
        with trace.span("decision", mechanism="structural_heuristic", question=MATH_QUESTION) as d1:
            math_tier = route_by_structure(MATH_QUESTION)
            d1.add_note(
                "decided_before_calling_the_model",
                tier=math_tier,
                matched=_structural_signal(MATH_QUESTION),
            )
        with trace.span("turn", number=1) as math_turn:
            math_reply = ask_model(
                trace,
                model_kind,
                [AIMessage("x = 5", response_metadata={"finish_reason": "stop"})],
                [HumanMessage(MATH_QUESTION)],
                [],
                max_tokens=_max_tokens_for(math_tier),
            )
            math_turn.add_note("structural_signal_matched", question=MATH_QUESTION)

        with trace.span(
            "decision", mechanism="structural_heuristic", question=HARD_SHORT_QUESTION
        ) as d2:
            hard_tier = route_by_structure(HARD_SHORT_QUESTION)
            d2.add_note(
                "decided_before_calling_the_model",
                tier=hard_tier,
                matched=_structural_signal(HARD_SHORT_QUESTION),
            )
        with trace.span("turn", number=2) as hard_turn:
            truncated = AIMessage(
                HARD_QUESTION_TRUNCATED,
                response_metadata={"finish_reason": "length"},
            )
            hard_reply = ask_model(
                trace,
                model_kind,
                [truncated],
                [HumanMessage(HARD_SHORT_QUESTION)],
                [],
                max_tokens=_max_tokens_for(hard_tier),
            )
            hard_turn.add_note("structural_signal_found_nothing", question=HARD_SHORT_QUESTION)

        span.add_note(
            "better_at_one_thing_blind_to_another",
            math_question_routed_generous=math_tier == "generous",
            math_question_finished_clean=math_reply.response_metadata.get("finish_reason")
            != "length",
            hard_question_still_routed_cheap=hard_tier == "cheap",
            hard_question_still_truncated=hard_reply.response_metadata.get("finish_reason")
            == "length",
        )


def a_cheap_models_own_uncertainty_escalates_correctly(trace: Trace, model_kind: ModelKind) -> None:
    """No upfront guess -- call the cheap tier, read what actually came
    back, escalate only when that signal says the first attempt fell
    short.
    """
    name = "a_cheap_models_own_uncertainty_escalates_correctly"
    with trace.span("scenario", name=name) as span:
        with trace.span("turn", number=1) as first_turn:
            truncated = AIMessage(
                HARD_QUESTION_TRUNCATED,
                response_metadata={"finish_reason": "length"},
            )
            first_reply = ask_model(
                trace,
                model_kind,
                [truncated],
                [HumanMessage(HARD_SHORT_QUESTION)],
                [],
                max_tokens=CHEAP_MAX_TOKENS,
            )
            first_finish = first_reply.response_metadata.get("finish_reason")
            first_turn.add_note("cheap_tier_attempt", finish_reason=first_finish)

        with trace.span("decision", mechanism="check_the_result_after_the_fact") as decision_span:
            escalated = first_finish == "length"
            decision_span.add_note(
                "decided_after_calling_the_model_not_before",
                escalate=escalated,
                reason=f"first attempt's finish_reason was {first_finish!r}",
            )
        if escalated:
            with trace.span("turn", number=2) as second_turn:
                complete = AIMessage(
                    HARD_QUESTION_ANSWER, response_metadata={"finish_reason": "stop"}
                )
                second_reply = ask_model(
                    trace,
                    model_kind,
                    [complete],
                    [HumanMessage(HARD_SHORT_QUESTION)],
                    [],
                    max_tokens=GENEROUS_MAX_TOKENS,
                )
                second_turn.add_note(
                    "escalated_to_the_generous_tier",
                    finish_reason=second_reply.response_metadata.get("finish_reason"),
                )
            final = second_reply
        else:
            final = first_reply

        span.add_note(
            "checked_the_result_instead_of_guessing_the_question",
            first_attempt_finish_reason=first_finish,
            escalated=escalated,
            final_finish_reason=final.response_metadata.get("finish_reason"),
            final_content=str(final.content),
        )


def a_simulated_learned_router_decides_with_no_visible_reason(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A scripted stand-in for a real trained router's decision --
    correct on both questions, and unlike the two heuristics above,
    nothing in `simulated_learned_router`'s own body explains why.
    """
    name = "a_simulated_learned_router_decides_with_no_visible_reason"
    with trace.span("scenario", name=name) as span:
        with trace.span(
            "decision", mechanism="simulated_learned_router", question=HARD_SHORT_QUESTION
        ) as d1:
            hard_tier = simulated_learned_router(HARD_SHORT_QUESTION)
            d1.add_note(
                "decided_before_calling_the_model",
                tier=hard_tier,
                reason=None,
                explainable=False,
            )

        with trace.span(
            "decision", mechanism="simulated_learned_router", question=EASY_SHORT_QUESTION
        ) as d2:
            easy_tier = simulated_learned_router(EASY_SHORT_QUESTION)
            d2.add_note(
                "decided_before_calling_the_model",
                tier=easy_tier,
                reason=None,
                explainable=False,
            )

        with trace.span("turn", number=1):
            complete = AIMessage(HARD_QUESTION_ANSWER, response_metadata={"finish_reason": "stop"})
            hard_reply = ask_model(
                trace,
                model_kind,
                [complete],
                [HumanMessage(HARD_SHORT_QUESTION)],
                [],
                max_tokens=_max_tokens_for(hard_tier),
            )

        with trace.span("turn", number=2):
            easy_reply = ask_model(
                trace,
                model_kind,
                [AIMessage("4", response_metadata={"finish_reason": "stop"})],
                [HumanMessage(EASY_SHORT_QUESTION)],
                [],
                max_tokens=_max_tokens_for(easy_tier),
            )

        span.add_note(
            "correct_decisions_with_no_readable_rationale",
            hard_question_routed_generous=hard_tier == "generous",
            hard_question_finished_clean=hard_reply.response_metadata.get("finish_reason")
            != "length",
            easy_question_routed_cheap=easy_tier == "cheap",
            easy_question_finished_clean=easy_reply.response_metadata.get("finish_reason")
            != "length",
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch42_model_tiers", model_kind=model_kind)

    a_length_based_heuristic_gets_fooled(trace, model_kind)
    a_structural_heuristic_does_better_but_still_guesses(trace, model_kind)
    a_cheap_models_own_uncertainty_escalates_correctly(trace, model_kind)
    a_simulated_learned_router_decides_with_no_visible_reason(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
