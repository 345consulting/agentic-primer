# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 10 asserts three places for the same `if`: before, after, on state.

Each scenario gets one test on the routing function itself -- unit, no trace
needed -- and one on what the chosen branch left behind in the run.
"""

from chapters import ch10_routing as chapter
from support.trace import Span

import pytest


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_three_places_have_one_scenario_each() -> None:
    assert [scenario.name for scenario in chapter.SCENARIOS] == [
        "the_question_needs_no_model",
        "a_write_takes_a_longer_path",
        "the_turns_are_running_out",
    ]


def test_route_before_the_model_reads_the_question_not_a_name() -> None:
    # Grounded in the actual prompt, the way a real router would see it --
    # there is no scenario identifier at this point, only text.
    assert chapter.route_before_the_model("What items can you order?") == "no_model_needed"
    assert chapter.route_before_the_model("what items can you order") == "no_model_needed"
    assert chapter.route_before_the_model("Order 2 bottles of milk.") == "ask_the_model"


def test_the_question_needs_no_model_never_calls_the_model() -> None:
    span = scenario_span("the_question_needs_no_model")
    assert span.require("ended").payload["reason"] == "answered_without_a_model"
    (turn,) = span.children
    # No model span nested inside: the branch made visible as an absence.
    assert turn.children == []
    answer = turn.require("answered directly").payload["value"]
    assert answer == "bread, butter, chips, milk"


def test_route_write_and_read_to_different_destinations() -> None:
    assert chapter.route("place_order") == "confirm_before_dispatch"
    assert chapter.route("price_of") == "dispatch_directly"


def test_a_write_is_routed_through_confirmation_before_it_runs() -> None:
    span = scenario_span("a_write_takes_a_longer_path")
    (turn_one, _turn_two) = span.children
    (tool,) = [child for child in turn_one.children if child.name == "tool"]
    assert tool.require("routed").payload["to"] == "confirm_before_dispatch"
    assert tool.require("confirmed").payload["summary"] == "2 x milk"
    # Routed before the call, not after: the note precedes the result.
    labels = [note.label for note in tool.notes]
    assert labels.index("routed") < labels.index("result")


def test_route_on_turns_remaining_wraps_up_only_on_the_last_turn() -> None:
    assert chapter.route_on_turns_remaining(0, budget=2) == "ask_the_model"
    assert chapter.route_on_turns_remaining(1, budget=2) == "wrap_up_now"


def test_the_turn_budget_stops_the_second_question_from_being_asked() -> None:
    span = scenario_span("the_turns_are_running_out")
    assert span.require("ended").payload["reason"] == "wrapped_up_early"
    turn_one, turn_two = span.children
    # Milk was looked up; chips was not -- the budget spent the last turn
    # wrapping up rather than asking for it.
    (tool,) = [child for child in turn_one.children if child.name == "tool"]
    assert tool.require("args").payload["item"] == "milk"
    assert turn_two.children == []
    said = turn_two.require("wrapped up").payload["value"]
    assert "1.20 for milk" in said
    assert "chips" not in said


def test_wrapped_up_early_is_not_turns_exhausted() -> None:
    # ch05_loop_endings' ending for hitting the cap by accident. This chapter
    # never lets that happen: the budget is checked before it is crossed.
    span = scenario_span("the_turns_are_running_out")
    assert span.require("ended").payload["reason"] != "turns_exhausted"


def test_wrap_up_reports_what_was_found_and_says_so_when_nothing_was() -> None:
    assert chapter.wrap_up([]) == "I ran out of turns before finding an answer."
    assert "1.20 for milk" in chapter.wrap_up(["1.20 for milk"])


@pytest.mark.parametrize(
    ("name", "value"),
    [("the_question_needs_no_model", 1), ("a_write_takes_a_longer_path", 2)],
)
def test_the_shared_scenarios_still_run_the_ordinary_loop(name: str, value: int) -> None:
    # Only `the_turns_are_running_out` keeps its own loop; the other two run
    # through the shared machinery, one of them by skipping straight past it.
    span = scenario_span(name)
    assert len(span.children) == value
