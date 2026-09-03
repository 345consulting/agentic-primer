# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 13 asserts a sequence with nothing deciding it at run time --
and one step, of five, whose own body still has something deciding.
"""

from chapters import ch13_workflow as chapter
from support.trace import Span

import inspect


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def step_names(span: Span) -> list[str]:
    return [c.attributes["name"] for c in span.children if c.name == "step"]


def test_messages_is_always_zero_even_around_the_agent_kind_step() -> None:
    trace = chapter.run()
    assert trace.summary["messages"] == 0


def test_single_step_runs_exactly_one_step() -> None:
    assert step_names(scenario_span("single_step")) == ["check_stock"]


def test_dual_in_sequence_always_runs_both_in_order() -> None:
    assert step_names(scenario_span("dual_in_sequence")) == ["get_price", "apply_discount"]


def test_route_after_check_is_deterministic_given_its_inputs() -> None:
    assert chapter.route_after_check(available=6, requested=2) == "reserve"
    assert chapter.route_after_check(available=2, requested=5) == "backorder"
    assert chapter.route_after_check(available=2, requested=2) == "reserve"


def test_enough_stock_takes_the_reserve_step_and_skips_backorder() -> None:
    span = scenario_span("three_steps_reserves")
    assert step_names(span) == ["check_stock", "reserve_stock"]
    assert "create_backorder" not in step_names(span)


def test_not_enough_stock_takes_the_backorder_step_and_skips_reserve() -> None:
    span = scenario_span("three_steps_backorders")
    assert step_names(span) == ["check_stock", "create_backorder"]
    assert "reserve_stock" not in step_names(span)


def test_the_same_check_step_runs_in_both_three_step_scenarios() -> None:
    reserves = scenario_span("three_steps_reserves")
    backorders = scenario_span("three_steps_backorders")
    assert step_names(reserves)[0] == step_names(backorders)[0] == "check_stock"


def test_the_four_plain_scenarios_have_no_way_to_read_model_kind() -> None:
    # Structural, not a convention: none of these functions even accept the
    # parameter, so a mock and a live run cannot differ inside them.
    for fn in (
        chapter.run_single_step,
        chapter.run_dual_in_sequence,
        chapter.run_three_steps_reserves,
        chapter.run_three_steps_backorders,
    ):
        assert "model_kind" not in inspect.signature(fn).parameters


def test_every_step_names_what_actually_ran_it() -> None:
    trace = chapter.run()
    steps = trace.find_spans("step")
    kinds = {s.attributes["name"]: s.attributes["kind"] for s in steps}
    assert kinds["check_stock"] == "function"
    assert kinds["get_price"] == "function"
    assert kinds["apply_discount"] == "function"
    assert kinds["check_price_via_agent"] == "agent"


def test_the_agent_kind_step_is_called_by_fixed_code_not_a_model() -> None:
    # Nothing decided whether to run this step; it always runs. What is not
    # fixed is what happens once it does.
    span = scenario_span("a_step_can_be_an_agent")
    (step_span,) = [c for c in span.children if c.name == "step"]
    assert step_span.attributes["kind"] == "agent"


def test_the_agent_kind_steps_own_turns_nest_inside_its_step_span() -> None:
    span = scenario_span("a_step_can_be_an_agent")
    (step_span,) = [c for c in span.children if c.name == "step"]
    turns = [c for c in step_span.children if c.name == "turn"]
    assert len(turns) == 2
    (tool,) = [c for c in turns[0].children if c.name == "tool"]
    assert tool.attributes["name"] == "price_of"
