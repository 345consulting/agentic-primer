# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 6 asserts that an empty tool_calls list is not a claim of success.

Two scenarios end with a reply carrying no tool calls. One of them finished.
Everything here is about the difference, and about the field that states it.
"""

from chapters import ch06_loop_finish_reason as chapter

from typing import get_args


def test_both_scenarios_run_the_same_loop() -> None:
    trace = chapter.run()
    scenarios = trace.find_spans("scenario")
    assert [span.attributes["name"] for span in scenarios] == ["answers", "truncated"]
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
    answers, truncated = trace.find_spans("scenario")
    assert answers.children[-1].children[0].reply["finish_reason"] == "stop"
    assert truncated.children[-1].children[0].reply["finish_reason"] == "length"


def test_the_summary_reports_the_truncated_run_as_cut_off() -> None:
    trace = chapter.run()
    assert trace.summary["ended"] == "answers: no tool_calls; truncated: cut off: length"


def test_the_truncated_reply_produced_nothing_at_all() -> None:
    """The sharpest form of the failure, and the live transcript.

    A budget of 24 is spent before the model emits anything: no content, no
    tool calls. A loop checking only tool_calls would report a clean run that
    produced nothing, and would be within its rights.
    """
    trace = chapter.run()
    _, truncated = trace.find_spans("scenario")
    assert len(truncated.children) == 1
    reply = truncated.children[0].children[0].reply
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
