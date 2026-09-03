# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 12 asserts nesting, and the cost it hides.

The mechanism is `Trace.span`'s own stack -- no new API, so the tests check
that the stack did what a stack does, not that something new was built.
"""

from chapters import ch12_supervisor as chapter
from support.trace import Span

import pytest


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_supervisor_has_two_experts_to_choose_between() -> None:
    assert [tool.name for tool in chapter.DECLARED_TOOLS] == [
        "consult_inventory_expert",
        "consult_orders_expert",
    ]


def test_the_outer_turn_count_is_its_own_not_the_inner_agents() -> None:
    span = scenario_span("the_supervisor_delegates")
    assert len(span.children) == 2


def test_the_inner_agent_nests_inside_the_outer_tool_span() -> None:
    span = scenario_span("the_supervisor_delegates")
    outer_turn_one = span.children[0]
    (tool,) = [c for c in outer_turn_one.children if c.name == "tool"]
    assert tool.attributes["name"] == "consult_inventory_expert"
    # The inner agent's own turns, nested where nothing but the stack put them.
    inner_turns = [c for c in tool.children if c.name == "turn"]
    assert len(inner_turns) == 2
    (inner_tool,) = [c for c in inner_turns[0].children if c.name == "tool"]
    assert inner_tool.attributes["name"] == "price_of"


def test_exactly_one_string_crosses_back_into_the_outer_messages() -> None:
    span = scenario_span("the_supervisor_delegates")
    outer_turn_one = span.children[0]
    (tool,) = [c for c in outer_turn_one.children if c.name == "tool"]
    assert tool.require("args").payload == {"question": "How much is milk?"}
    assert tool.require("result").payload["value"] == "Milk costs 1.20 per bottle."


def test_one_reply_can_delegate_to_both_experts_at_once() -> None:
    # ch06_two_tools' fan-out, one level up: two tool spans, siblings under
    # the same turn, each opening its own nested agent underneath it.
    span = scenario_span("the_supervisor_delegates_to_both")
    outer_turn_one = span.children[0]
    tools = [c for c in outer_turn_one.children if c.name == "tool"]
    assert [t.attributes["name"] for t in tools] == [
        "consult_inventory_expert",
        "consult_orders_expert",
    ]


def test_the_experts_run_one_after_the_other_not_at_once() -> None:
    # run_turns dispatches a reply's calls with a plain for loop. Ordering
    # and concurrency are ch33_parallel's question, not this chapter's.
    span = scenario_span("the_supervisor_delegates_to_both")
    outer_turn_one = span.children[0]
    tools = [c for c in outer_turn_one.children if c.name == "tool"]
    assert tools[0].seq < tools[1].seq


def test_the_two_experts_see_different_tools() -> None:
    span = scenario_span("the_supervisor_delegates_to_both")
    outer_turn_one = span.children[0]
    inventory, orders = [c for c in outer_turn_one.children if c.name == "tool"]
    (inventory_call,) = [c for c in inventory.children[0].children if c.name == "tool"]
    (orders_call,) = [c for c in orders.children[0].children if c.name == "tool"]
    assert inventory_call.attributes["name"] == "price_of"
    assert orders_call.attributes["name"] == "place_order"


def test_neither_declared_tool_is_the_dispatched_one() -> None:
    with pytest.raises(AssertionError):
        chapter.consult_inventory_expert.invoke({"question": "How much is milk?"})
    with pytest.raises(AssertionError):
        chapter.consult_orders_expert.invoke({"item": "milk", "quantity": 5})
