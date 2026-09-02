# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 6 asserts that an empty tool_calls list is not a claim of success.

Two scenarios end with a reply carrying no tool calls. One of them finished.
Everything here is about the difference, and about the field that states it.
"""

from chapters import ch06_loop_finish_reason as chapter
from support.trace import ENDED

from typing import get_args


def test_both_scenarios_run_the_same_loop() -> None:
    trace = chapter.run()
    scenarios = trace.find_spans("scenario")
    assert [span.attributes["name"] for span in scenarios] == [
        "within_token_limit",
        "tokens_exhausted",
    ]
    assert [span.attributes["max_tokens"] for span in scenarios] == [None, 24]


def test_both_scenarios_end_with_no_tool_calls() -> None:
    # The premise: the loop's own condition fires identically in both, so it
    # cannot be what tells them apart.
    trace = chapter.run()
    for scenario in trace.find_spans("scenario"):
        last_turn = scenario.children[-1]
        assert not last_turn.children[0].reply.get("tool_calls")


def test_only_the_providers_own_account_distinguishes_them() -> None:
    trace = chapter.run()
    stopped, cut_off = trace.find_spans("scenario")
    assert stopped.children[-1].children[0].reply["finish_reason"] == "stop"
    assert cut_off.children[-1].children[0].reply["finish_reason"] == "length"


def test_each_scenario_records_its_own_ending() -> None:
    trace = chapter.run()
    stopped, cut_off = trace.find_spans("scenario")
    # `answer_received`, not `no_tool_calls`: this loop read the field, so it
    # can make the stronger claim. ch01 to ch05 only knew the list was empty.
    assert stopped.require(ENDED).payload["reason"] == "answer_received (finish_reason = stop)"
    assert cut_off.require(ENDED).payload["reason"] == "tokens_exhausted (finish_reason = length)"
    # The run summary carries the distinct endings, not one per scenario.
    assert trace.summary["ended"] == (
        "answer_received (finish_reason = stop); tokens_exhausted (finish_reason = length)"
    )


def test_the_exhausted_reply_produced_nothing_at_all() -> None:
    """The sharpest form of the failure, and the live transcript.

    A budget of 24 is spent before the model emits anything: no content, no
    tool calls. A loop checking only tool_calls would report a clean run that
    produced nothing, and would be within its rights.
    """
    trace = chapter.run()
    _, cut_off = trace.find_spans("scenario")
    assert len(cut_off.children) == 1
    reply = cut_off.children[0].children[0].reply
    assert reply["content"] == ""
    assert "tool_calls" not in reply


def test_max_tokens_is_the_first_parameter_this_primer_sets() -> None:
    # Five chapters reported all seven of CONFIGURABLE as defaulted by the
    # model provider. This is the first one we take back.
    from support.models import CONFIGURABLE

    assert "max_tokens" in CONFIGURABLE
    assert [s.max_tokens for s in chapter.SCENARIOS] == [None, 24]


def test_the_schema_the_lookup_and_the_prices_allow_the_same_items() -> None:
    permitted = set(get_args(chapter.GroceryItem.__value__))
    assert permitted == set(chapter.STOCK_ON_HAND) == set(chapter.PRICE)
