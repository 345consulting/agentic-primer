# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The supervisor as a graph node -- what did it buy? Same-schema sharing,
narrower-schema mapping, the context bill, and ch33's recursion-limit
finding reproduced on a real two-node child.
"""

from chapters import ch36_subgraph as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_childs_whole_transcript_merges_into_the_parent() -> None:
    span = scenario_span("a_subgraph_with_the_same_schema_shares_the_parents_state")
    note = span.require("the_childs_whole_transcript_is_now_the_parents")
    assert note.payload["roles"] == ["system", "human", "ai", "tool", "ai"]
    assert note.payload["message_count"] == 5


def test_the_undeclared_child_field_does_not_survive() -> None:
    span = scenario_span("a_subgraph_with_a_narrower_schema_only_exposes_what_it_maps")
    note = span.require("only_the_declared_keys_survive")
    assert note.payload["scratch_present"] is False


def test_the_declared_field_maps_back_correctly() -> None:
    span = scenario_span("a_subgraph_with_a_narrower_schema_only_exposes_what_it_maps")
    note = span.require("only_the_declared_keys_survive")
    assert note.payload["status_mapped_back"] == "order A100: shipped"


def test_the_subgraph_path_costs_more_messages_than_the_supervisor_path() -> None:
    span = scenario_span("the_context_bill_differs_even_for_the_same_work")
    note = span.require("same_work_different_bill")
    assert (
        note.payload["subgraph_parent_message_count"]
        > note.payload["supervisor_parent_message_count"]
    )


def test_the_parents_recursion_limit_binds_on_the_childs_steps() -> None:
    span = scenario_span("the_childs_super_steps_count_against_the_parents_recursion_limit")
    note = span.require("parents_limit_binds_on_the_childs_steps")
    assert note.payload["raised"] is True
