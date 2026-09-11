# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Two updates to one field: append, or replace. No model calls anywhere
in this chapter -- these are pure state-merge mechanics, identical under
mock and live, so there is nothing for a live run to add.
"""

from chapters import ch38_reducers as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_second_write_is_all_that_survives_with_no_reducer() -> None:
    span = scenario_span("no_reducer_means_replace")
    note = span.require("second_write_is_all_that_survives")
    assert note.payload["final_note"] == "from second"


def test_messages_accumulates_while_the_plain_field_replaces() -> None:
    span = scenario_span("add_messages_is_the_one_field_that_appends")
    note = span.require("one_field_accumulates_the_other_replaces")
    assert note.payload["message_count"] == 3
    assert note.payload["final_note"] == "from turn two"


def test_operator_add_accumulates_across_turns() -> None:
    span = scenario_span("a_custom_reducer_is_just_a_function")
    note = span.require("operator_add_accumulates_across_turns")
    assert note.payload["final_total"] == 8


def test_the_first_order_id_is_silently_lost() -> None:
    span = scenario_span("the_wrong_reducer_choice_silently_loses_data")
    note = span.require("a100_never_recorded")
    assert note.payload["a100_survived"] is False
    assert note.payload["final_order_ids_checked"] == ["B200"]


def test_both_simultaneous_writes_survive_in_one_merge() -> None:
    span = scenario_span("two_nodes_writing_the_same_field_in_one_super_step_both_survive")
    note = span.require("both_simultaneous_writes_survived")
    assert note.payload["both_branches_present"] is True
    assert note.payload["total"] == 11
