# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 9 asserts the matrix: who tried again, how often, and who was told.

One assertion per cell, plus the two that hold across all of them -- that the
retried attempts never reach the message list, and that the sentence handed
back names the cell.
"""

from chapters import ch09_who_retries as chapter
from support.trace import Span

import pytest


def tool_spans(scenario_name: str) -> list[Span]:
    trace = chapter.run()
    (scenario,) = [
        span for span in trace.find_spans("scenario") if span.attributes["name"] == scenario_name
    ]
    return [span for turn in scenario.children for span in turn.children if span.name == "tool"]


def test_the_matrix_has_one_situation_per_cell() -> None:
    assert [scenario.name for scenario in chapter.SCENARIOS] == [
        "service_is_busy",
        "order_may_have_landed",
        "arguments_are_wrong",
        "service_says_no",
    ]


def test_transient_and_safe_is_retried_by_the_harness() -> None:
    # 429 on a read: the only cell where the harness tries again.
    (tool,) = tool_spans("service_is_busy")
    assert [note.label for note in tool.notes].count("attempt") == 2
    assert tool.find("backoff") is not None
    assert tool.require("result").payload["value"] == "1.20"


def test_the_model_is_never_told_there_was_a_retry() -> None:
    """Two service calls, one tool result, and one turn.

    The counter lives outside the loop because from inside the context there
    is nothing to count -- a failed attempt of a retried call never enters the
    message list at all.
    """
    (tool,) = tool_spans("service_is_busy")
    assert len([note for note in tool.notes if note.label == "wire request"]) == 2
    assert tool.find("failed") is None


def test_transient_and_unsafe_is_retried_by_nobody() -> None:
    # A timeout on a write. The failure is retryable and the call is not, so
    # `retryable and idempotent` stops it -- one attempt, and the model is
    # told the outcome is unknown rather than that it failed.
    (tool,) = tool_spans("order_may_have_landed")
    attempt = tool.require("attempt").payload
    assert attempt["retryable"] and not attempt["idempotent"]
    assert [note.label for note in tool.notes].count("attempt") == 1
    said = chapter.sentence("place_order", _failure(tool))
    assert "may have taken effect" in said
    assert "Do not call it again" in said


def test_bad_arguments_are_fixed_by_the_model_and_cost_a_turn() -> None:
    # The one cell where the model retrying is right. Two tool calls, in two
    # different turns, and the second one carries different arguments.
    rejected, accepted = tool_spans("arguments_are_wrong")
    assert rejected.require("args").payload["quantity"] == 20
    assert rejected.require("failed").payload["caller_can_fix"] is True
    assert accepted.require("args").payload["quantity"] == chapter.MAX_PER_ORDER
    assert accepted.require("result").payload["value"] == "ordered 12 x milk"


def test_a_rejected_argument_never_reaches_the_service() -> None:
    # The bound is the tool's, checked before anything leaves the process,
    # which is where most bad arguments are actually caught.
    rejected, _ = tool_spans("arguments_are_wrong")
    assert rejected.find("wire request") is None


def test_permanent_is_retried_by_nobody_and_says_so() -> None:
    (tool,) = tool_spans("service_says_no")
    failed = tool.require("failed").payload
    assert failed["retryable"] is False
    assert "404" in failed["message"]
    assert "will not succeed" in chapter.sentence("price_of", _failure(tool))


def test_a_transient_read_gives_up_when_the_policy_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    # Policy is the harness's, and it is two numbers in one place. Turn the
    # cap down to one and the same situation stops retrying -- no tool
    # changed, and no tool could have prevented it.
    monkeypatch.setattr(chapter, "MAX_ATTEMPTS", 1)
    (tool,) = tool_spans("service_is_busy")
    assert [note.label for note in tool.notes].count("attempt") == 1
    assert tool.require("failed").payload["retryable"] is True


def test_the_declared_tools_are_not_the_dispatched_ones() -> None:
    assert [tool.name for tool in chapter.DECLARED_TOOLS] == ["place_order", "price_of"]
    assert chapter.TOOLS["price_of"] is chapter.fetch_price
    assert chapter.TOOLS["place_order"] is chapter.submit_order
    with pytest.raises(AssertionError):
        chapter.price_of.invoke({"item": "milk"})


def _failure(tool_span: Span) -> chapter.ToolCallError:
    """Rebuild the failure from what the span recorded, to read the sentence."""
    payload = tool_span.require("failed").payload
    return chapter.ToolCallError(
        payload["message"],
        retryable=payload["retryable"],
        idempotent=payload["idempotent"],
        caller_can_fix=payload["caller_can_fix"],
    )
