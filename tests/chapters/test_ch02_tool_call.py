# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 2 asserts the round trip: the model asks, we execute, turn two knows.

The facts here that chapter 1 could not have: a turn that contains work, a
result carried back as a message, and an id tying the two together.
"""

from chapters import ch02_tool_call

from typing import get_args


def test_a_turn_is_one_invocation_plus_the_tools_it_asked_for() -> None:
    trace = ch02_tool_call.run()
    turns = trace.find_spans("turn")
    assert len(turns) == 2
    # The tool ran inside turn one, not between the turns. Executing a tool is
    # the consequence of an invocation, not a turn of its own.
    assert [child.name for child in turns[0].children] == ["model", "tool"]
    assert [child.name for child in turns[1].children] == ["model"]


def test_the_first_reply_is_a_request_and_carries_no_answer() -> None:
    trace = ch02_tool_call.run()
    first = trace.find_spans("model")[0].reply
    assert first["content"] == ""
    assert [call["name"] for call in first["tool_calls"]] == ["stock_on_hand"]


def test_we_dispatched_the_tool_ourselves() -> None:
    trace = ch02_tool_call.run()
    (tool,) = trace.find_spans("tool")
    assert tool.attributes["name"] == "stock_on_hand"
    assert dict(tool.notes[0].payload) == {"item": "milk"}
    assert tool.notes[1].payload["value"] == 2


def test_the_result_re_enters_the_context_as_a_message() -> None:
    # The provider is stateless. The only way turn two learns what the tool
    # said is that the result is in the list turn two sends.
    trace = ch02_tool_call.run()
    sent = trace.find_spans("model")[1].context
    assert [m["role"] for m in sent] == ["system", "human", "ai", "tool"]
    assert sent[-1]["content"] == "2"


def test_the_id_ties_the_result_to_the_request_that_asked_for_it() -> None:
    trace = ch02_tool_call.run()
    asked = trace.find_spans("model")[0].reply["tool_calls"][0]["id"]
    answered = trace.find_spans("model")[1].context[-1]["tool_call_id"]
    assert answered == asked


def test_turn_two_asks_for_nothing_further_and_that_is_why_it_ends() -> None:
    trace = ch02_tool_call.run()
    assert "tool_calls" not in trace.find_spans("model")[1].reply
    assert trace.summary["ended"] == "no_tool_calls"


def test_every_span_that_was_entered_was_exited() -> None:
    trace = ch02_tool_call.run()
    assert all(span.exited_at is not None for _, span in trace.walk())


def test_the_schema_and_the_lookup_allow_exactly_the_same_parts() -> None:
    """The pairing that produced the finding of 2026-09-01, guarded.

    `GroceryItem` is what the request body permits the model to ask for; STOCK_ON_HAND
    is what the tool can answer. A Literal cannot be built from a dict, so the
    two are written by hand and this is what keeps them honest. Drift one way
    hides an item the model can never reach; the other way advertises one
    that raises when asked for.
    """
    permitted = set(get_args(ch02_tool_call.GroceryItem.__value__))
    assert permitted == set(ch02_tool_call.STOCK_ON_HAND)
