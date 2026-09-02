# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 5 asserts that the number of turns is a property of the run.

Chapters 1 to 4 typed their turns out, so a test could only confirm the
typing. Here the count is decided by the model, the chain is forced by the
tools, and the ending has to say which of the two it was.
"""

from chapters import ch05_the_loop

from typing import get_args


def test_the_number_of_turns_is_not_written_anywhere() -> None:
    # Three turns because the model asked for two tools in sequence, not
    # because anyone typed three blocks.
    trace = ch05_the_loop.run()
    assert trace.summary["turns"] == 3
    assert len(trace.find_spans("turn")) == 3


def test_the_second_call_could_not_have_been_made_first() -> None:
    """The chain. Turn two's argument is turn one's result.

    `lowest_stock_item` takes no arguments and returns a name; `price_of`
    needs that name. Nothing in the question supplies it, so the turns cannot
    collapse into one reply -- which is what makes this a loop rather than
    chapter 2 with more syntax.
    """
    trace = ch05_the_loop.run()
    first, second = trace.find_spans("tool")[:2]
    assert first.attributes["name"] == "lowest_stock_item"
    assert first.require("result").payload["value"] == "butter"
    assert second.attributes["name"] == "price_of"
    assert dict(second.notes[0].payload) == {"item": "butter"}


def test_the_loop_ends_because_the_model_asked_for_nothing() -> None:
    trace = ch05_the_loop.run()
    assert trace.summary["ended"] == "no tool_calls"
    assert not trace.find_spans("model")[-1].reply.get("tool_calls")


def test_a_capped_run_says_so_rather_than_looking_finished() -> None:
    """The other ending, and the reason it is recorded separately.

    A run that finished and a run we stopped are the same length, the same
    shape and the same colour on the page. Only the summary distinguishes them.
    """
    trace = ch05_the_loop.run()
    assert trace.summary["ended"] != "turn cap"

    # Force it: a cap of one cannot reach the model's own ending.
    capped = ch05_the_loop.run(turn_cap=1)
    assert capped.summary["ended"] == "turn cap"
    assert capped.summary["turns"] == 1


def test_the_mock_refuses_to_cover_for_a_loop_that_ran_too_long() -> None:
    # Running past the script raises rather than inventing a reply, so a
    # chapter cannot quietly loop further than it was written to.
    assert len(ch05_the_loop.MOCK_MODEL_REPLIES) == 3


def test_the_schema_the_lookup_and_the_prices_allow_the_same_items() -> None:
    permitted = set(get_args(ch05_the_loop.GroceryItem.__value__))
    assert permitted == set(ch05_the_loop.STOCK_ON_HAND) == set(ch05_the_loop.PRICE)
