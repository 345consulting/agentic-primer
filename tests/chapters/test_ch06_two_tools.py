# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 6 asserts that two calls in one reply are still one turn.

And that the shape of a run comes from the question rather than the agent:
the same two tool calls cost one turn or two depending on whether the second
argument was already known.
"""

from chapters import ch06_two_tools as chapter

from typing import get_args


def test_two_calls_in_one_reply_belong_to_one_turn() -> None:
    # The calls are one turn's worth. The run is longer than that: it takes
    # another invocation to turn their results into an answer.
    trace = chapter.run()
    fanned_out, _ = trace.find_spans("scenario")
    first_turn = fanned_out.children[0]
    tools = [span for span in first_turn.children if span.name == "tool"]
    assert [span.attributes["name"] for span in tools] == ["price_of", "stock_on_hand"]
    assert len(first_turn.children) == 3  # one model call, two tools


def test_both_scenarios_make_the_same_two_calls() -> None:
    # The work is identical. Only the shape differs, which is the chapter.
    trace = chapter.run()
    counted = [
        sum(1 for turn in scenario.children for span in turn.children if span.name == "tool")
        for scenario in trace.find_spans("scenario")
    ]
    assert counted == [2, 2]


def test_the_chain_costs_a_turn_the_fan_out_does_not() -> None:
    trace = chapter.run()
    fanned_out, chained = trace.find_spans("scenario")
    assert len(fanned_out.children) == 2
    assert len(chained.children) == 3


def test_the_second_call_could_not_have_been_sent_first() -> None:
    """Why the chain is a chain: the argument does not exist until turn one."""
    trace = chapter.run()
    _, chained = trace.find_spans("scenario")
    (first,) = [s for s in chained.children[0].children if s.name == "tool"]
    (second,) = [s for s in chained.children[1].children if s.name == "tool"]
    assert first.attributes["name"] == "lowest_stock_item"
    assert first.require("result").payload["value"] == "butter"
    assert second.attributes["name"] == "price_of"
    assert dict(second.notes[0].payload) == {"item": "butter"}


def test_the_two_calls_run_in_order_on_one_thread() -> None:
    # Ours is sequential. A framework dispatching a turn's calls on a pool
    # would show two threads here, and the tree would look identical.
    trace = chapter.run()
    fanned_out, _ = trace.find_spans("scenario")
    tools = [span for span in fanned_out.children[0].children if span.name == "tool"]
    assert [span.seq for span in tools] == sorted(span.seq for span in tools)
    assert len({span.thread for span in tools}) == 1


def test_the_schema_the_lookup_and_the_prices_allow_the_same_items() -> None:
    permitted = set(get_args(chapter.GroceryItem.__value__))
    assert permitted == set(chapter.STOCK_ON_HAND) == set(chapter.PRICE)
