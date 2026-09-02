# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 4 asserts that three different failures produce one ending.

Each scenario is asserted for what only it can show, and then the chapter's
claim is asserted across all three: same ending, same shape, different truth.
"""

from chapters import ch04_tool_failures as chapter
from support.trace import ENDED

from typing import get_args

import pytest


def test_all_three_scenarios_end_the_same_way() -> None:
    """The chapter. Three failures, one ending, and the summary cannot help.

    Each scenario records its own ending, so this asserts the claim directly
    rather than by parsing a sentence: three scenarios, one distinct value
    between them.
    """
    trace = chapter.run()
    endings = [span.require(ENDED).payload["reason"] for span in trace.find_spans("scenario")]
    assert endings == ["no_tool_calls"] * 3
    assert trace.summary["ended"] == "no_tool_calls"


def test_a_scenario_can_withhold_a_tool_from_the_model() -> None:
    # price_of raises, so declaring it in no_tool_available would trip the
    # next scenario's failure inside this one.
    withheld, *rest = chapter.SCENARIOS
    assert [t.name for t in withheld.tools] == ["stock_on_hand"]
    assert all("price_of" in [t.name for t in s.tools] for s in rest)


def test_nothing_fails_when_no_tool_is_available() -> None:
    trace = chapter.run()
    scenario = trace.find_spans("scenario")[0]
    tools = [span for span in scenario.children for span in span.children if span.name == "tool"]
    assert all(tool.find("failed") is None for tool in tools)
    assert all(tool.find("result") is not None for tool in tools)


def test_the_exception_stays_in_the_trace_and_never_reaches_the_model() -> None:
    """An exception is written for a developer, and can carry a host or a path.

    LangGraph's default formats it with repr and sends that to the model
    provider. This sends a sentence we wrote instead.
    """
    trace = chapter.run()
    (failed,) = [
        span for span in trace.find_spans("tool") if span.attributes["name"] == "price_of"
    ][:1]
    recorded = failed.require("failed").payload["message"]
    assert "pricing service unreachable" in recorded

    scenario = trace.find_spans("scenario")[1]
    sent_onward = scenario.children[-1].children[0].context[-1]["content"]
    assert recorded not in sent_onward
    assert "will not succeed on a retry" in sent_onward


def test_both_calls_are_answered_even_though_one_raised() -> None:
    """The constraint, not a preference.

    An assistant message carrying two tool_call_ids must be followed by a
    ToolMessage for each, or the model provider rejects the next request. So
    the loop cannot bail on the first exception.
    """
    trace = chapter.run()
    scenario = trace.find_spans("scenario")[2]
    asked = scenario.children[0].children[0].reply["tool_calls"]
    assert [call["name"] for call in asked] == ["price_of", "stock_on_hand"]

    answered = [m for m in scenario.children[-1].children[0].context if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in answered] == [c["id"] for c in asked]


def test_the_tool_really_raises() -> None:
    with pytest.raises(chapter.PriceServiceError):
        chapter.price_of.invoke({"item": "milk"})


def test_the_schema_and_the_lookup_allow_exactly_the_same_items() -> None:
    permitted = set(get_args(chapter.GroceryItem.__value__))
    assert permitted == set(chapter.STOCK_ON_HAND)
