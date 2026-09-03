# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 21 asserts a verdict with nowhere to put a rewrite, and a
rejection that retries the model the way ch09 retried a tool.
"""

from chapters import ch21_judge as chapter
from support.trace import Span, Trace

import dataclasses

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def judged_notes(span: Span) -> list[dict[str, object]]:
    (turn,) = [c for c in span.children if c.name == "turn"]
    return [n.payload for n in turn.notes if n.label == "judged"]


def test_verdict_has_no_replacement_field() -> None:
    # Structural, not a convention: there is nowhere on this type to put a
    # rewrite even by mistake.
    fields = {f.name for f in dataclasses.fields(chapter.Verdict)}
    assert fields == {"passed", "reason"}


def test_a_schema_valid_quantity_can_still_be_rejected() -> None:
    call_reply = chapter.TOOL_CALL_ATTEMPTS[0]
    verdict = chapter.judge_tool_call_is_reasonable(call_reply)
    assert verdict.passed is False
    assert "500" in str(verdict.reason)


def test_a_reasonable_quantity_passes() -> None:
    call_reply = chapter.TOOL_CALL_ATTEMPTS[1]
    assert chapter.judge_tool_call_is_reasonable(call_reply).passed is True


def test_groundedness_judge_rejects_an_answer_missing_the_topic() -> None:
    judge = chapter.make_groundedness_judge("milk")
    assert judge(chapter.FINAL_ANSWER_ATTEMPTS[0]).passed is False
    assert judge(chapter.FINAL_ANSWER_ATTEMPTS[1]).passed is True


def test_tool_call_scenario_rejects_once_then_accepts() -> None:
    notes = judged_notes(scenario_span("judge_rejects_a_tool_call"))
    assert [n["passed"] for n in notes] == [False, True]


def test_the_rejected_attempt_never_reaches_the_dispatched_tool() -> None:
    span = scenario_span("judge_rejects_a_tool_call")
    (tool_span,) = [c for c in span.children[0].children if c.name == "tool"]
    assert tool_span.require("args").payload == {"item": "milk", "quantity": 2}


def test_final_answer_scenario_rejects_once_then_accepts() -> None:
    notes = judged_notes(scenario_span("judge_rejects_a_final_answer"))
    assert [n["passed"] for n in notes] == [False, True]


def test_give_up_scenario_exhausts_every_attempt_and_none_pass() -> None:
    notes = judged_notes(scenario_span("judge_gives_up_after_max_attempts"))
    assert len(notes) == chapter.MAX_ATTEMPTS
    assert all(n["passed"] is False for n in notes)


def test_give_up_scenario_never_dispatches_a_tool() -> None:
    span = scenario_span("judge_gives_up_after_max_attempts")
    turn = span.children[0]
    assert [c.name for c in turn.children] == ["model"] * chapter.MAX_ATTEMPTS


def test_the_ending_names_how_many_attempts_it_took() -> None:
    trace = chapter.run()
    assert "accepted on attempt 2" in trace.summary["ended"]
    assert f"gave up after {chapter.MAX_ATTEMPTS} attempts" in trace.summary["ended"]


def test_a_rejected_tool_call_gets_a_correction_before_retrying() -> None:
    # The bug the live run found: without this, every attempt re-asks the
    # identical question with no memory of the failure.
    trace = Trace(chapter="test", model_kind="mock")
    messages: list[BaseMessage] = [
        SystemMessage(chapter.SYSTEM_PROMPT),
        HumanMessage("Order a couple bottles."),
    ]
    with trace.span("turn", number=1) as span:
        chapter.ask_and_judge(
            span,
            trace,
            "mock",
            chapter.TOOL_CALL_ATTEMPTS,
            messages,
            chapter.DECLARED_TOOLS,
            chapter.judge_tool_call_is_reasonable,
        )
    # The rejected AIMessage and a correction naming the reason both landed
    # in the context the second attempt actually saw.
    contents = [str(m.content) for m in messages]
    assert any("rejected" in c and "different arguments" in c for c in contents)
