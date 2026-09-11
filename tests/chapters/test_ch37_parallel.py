# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""fan-out, Send, join, and the order things merge in. Branch spans carry
an explicit `parent` because a worker thread never has anything of its
own already open -- these tests check the tree those spans land in, not
just the final summary note.
"""

from chapters import ch37_parallel as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def child_span(parent: Span, name: str) -> Span:
    (span,) = [c for c in parent.children if c.name == name]
    return span


def test_branch_count_matches_each_lists_length() -> None:
    span = scenario_span("send_fans_out_dynamically_based_on_runtime_data")
    note = span.require("branch_count_matched_each_lists_length")
    assert note.payload["two_item_run"] == {
        "input_count": 2,
        "branches_ran": ["A100", "A200"],
    }
    assert note.payload["four_item_run"] == {
        "input_count": 4,
        "branches_ran": ["B100", "B200", "B300", "B400"],
    }


def test_each_round_holds_its_own_branches_as_siblings() -> None:
    span = scenario_span("send_fans_out_dynamically_based_on_runtime_data")
    two_items = span.children[0]
    assert two_items.name == "round"
    branches = [c for c in two_items.children if c.name == "branch"]
    assert {b.attributes["item"] for b in branches} == {"A100", "A200"}
    # Siblings, not nested inside one another -- the exact corruption a
    # single shared open-span stack produced under real thread interleaving.
    assert all(not b.children for b in branches)


def test_elapsed_time_matches_the_slowest_branch_not_the_sum() -> None:
    span = scenario_span("branches_run_concurrently_not_sequentially")
    note = span.require("elapsed_matches_the_slowest_branch_not_the_sum")
    assert note.payload["faster_than_the_sum"] is True

    branches = [c for c in span.children if c.name == "branch"]
    assert {b.attributes["item"] for b in branches} == {"slow", "fast", "medium"}
    assert all(not b.children for b in branches)


def test_merge_order_follows_dispatch_order_not_completion_order() -> None:
    span = scenario_span("merge_order_follows_dispatch_order_not_completion_order")
    note = span.require("merged_order_matched_dispatch_not_completion")
    assert note.payload["dispatch_order"] == ["slow", "fast", "medium"]
    assert note.payload["completion_order"] == ["fast", "medium", "slow"]
    assert note.payload["matches_dispatch_order"] is True


def test_fan_out_over_tool_calls_reduces_to_one_answer() -> None:
    span = scenario_span("fan_out_over_real_tool_calls_then_reduces_to_one_answer")
    note = span.require("each_order_checked_then_reduced_to_one_answer")
    assert note.payload["orders_checked"] == [
        "order A100: shipped",
        "order A200: shipped",
        "order A300: shipped",
    ]

    branches = [c for c in span.children if c.name == "branch"]
    assert len(branches) == 3
    for branch in branches:
        tool = child_span(branch, "tool")
        assert tool.require("result").payload["value"].startswith("order")

    turn = child_span(span, "turn")
    model = child_span(turn, "model")
    assert "All checked" in model.reply["content"]


def test_a_recovering_batch_stops_early_once_the_judge_passes_it() -> None:
    name = "a_single_judgment_over_the_whole_batch_retries_the_round_until_three_strikes"
    span = scenario_span(name)
    note = span.require("quality_improved_so_it_stopped_early")
    assert note.payload["rounds_run"] == 3
    assert note.payload["final_verdict"] == "pass"


def test_a_batch_that_never_improves_escalates_at_three_strikes() -> None:
    name = "a_single_judgment_over_the_whole_batch_retries_the_round_until_three_strikes"
    span = scenario_span(name)
    note = span.require("quality_never_improved_so_it_escalated_at_three_strikes")
    assert note.payload["rounds_run"] == 3
    assert note.payload["final_verdict"] == "retry"


def test_the_judge_verdict_covers_the_whole_batch_not_one_branch() -> None:
    name = "a_single_judgment_over_the_whole_batch_retries_the_round_until_three_strikes"
    span = scenario_span(name)
    judges = [c for c in span.children if c.name == "judge"]
    first_judgment = judges[0].require("batch_judged")
    assert len(first_judgment.payload["batch"]) == 3
