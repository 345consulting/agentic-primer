# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""recursion_limit is ch03's turn cap, reimplemented by LangGraph -- and it
fails differently when it binds: the hand-built cap ends the run and
reports why; this one raises out of invoke() itself.
"""

from chapters import ch33_limits as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_a_run_within_the_limit_completes_normally() -> None:
    span = scenario_span("a_run_within_the_limit_completes_normally")
    note = span.require("completed")
    assert note.payload["raised"] is False
    assert note.payload["final_answer"] == "Order A100 has shipped."


def test_a_run_that_never_stops_hits_the_limit_and_raises() -> None:
    span = scenario_span("a_run_that_never_stops_hits_the_limit_and_raises")
    note = span.require("hit_the_limit")
    assert note.payload["raised"] is True
    assert note.payload["error_type"] == "GraphRecursionError"


def test_the_limit_is_set_per_call_not_per_graph() -> None:
    span = scenario_span("the_limit_is_set_per_call_not_per_graph")
    note = span.require("same_graph_two_budgets")
    assert note.payload["raised_with_limit_2"] is True
    assert note.payload["raised_with_limit_50"] is False


def test_a_nested_graph_shares_the_parents_budget() -> None:
    span = scenario_span("a_nested_graph_spends_the_parents_budget")
    note = span.require("nested_vs_isolated")
    assert note.payload["nested_as_node_raised"] is True


def test_invoking_the_child_separately_gives_it_its_own_budget() -> None:
    span = scenario_span("a_nested_graph_spends_the_parents_budget")
    note = span.require("nested_vs_isolated")
    assert note.payload["invoked_separately_raised"] is False
    assert note.payload["wrapper_final_message"] == "child hit its own limit, parent continues"
