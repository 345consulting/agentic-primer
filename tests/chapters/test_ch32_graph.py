# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 32 builds LangGraph's core objects up one clause at a time --
ch01, ch02, ch03 as a graph -- then asks what any of it actually bought.
"""

from chapters import ch32_graph as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_a_single_model_node_answers_with_no_tools_declared() -> None:
    span = scenario_span("a_single_model_node_is_ch01_as_a_graph")
    note = span.require("final_answer")
    assert note.payload["message_count"] == 3


def test_a_fixed_edge_always_dispatches_the_tool() -> None:
    span = scenario_span("a_fixed_edge_from_model_to_tool_is_ch02_as_a_graph")
    note = span.require("fixed_edge_always_dispatched")
    assert note.payload["tool_call_present"] is True
    assert note.payload["roles"] == ["system", "human", "ai", "tool"]


def test_the_conditional_edge_routes_more_than_once() -> None:
    span = scenario_span("a_conditional_edge_turns_it_into_the_loop")
    note = span.require("loop_ran_more_than_once")
    assert note.payload["tool_dispatches"] == 2
    assert note.payload["total_turns"] >= 2


def test_route_on_tool_calls_returns_the_tool_node_when_present() -> None:
    from langchain_core.messages import AIMessage, BaseMessage

    messages: list[BaseMessage] = [
        AIMessage("", tool_calls=[{"name": "check_order", "args": {}, "id": "c1"}])
    ]
    state: chapter.State = {"messages": messages}
    assert chapter.route_on_tool_calls(state) == "call_tool"


def test_route_on_tool_calls_returns_end_when_absent() -> None:
    from langchain_core.messages import AIMessage, BaseMessage
    from langgraph.graph import END

    messages: list[BaseMessage] = [AIMessage("done")]
    state: chapter.State = {"messages": messages}
    assert chapter.route_on_tool_calls(state) == END


def test_the_graph_and_the_hand_built_loop_agree() -> None:
    span = scenario_span("the_graph_and_the_hand_built_loop_produce_the_same_final_state")
    note = span.require("comparison")
    assert note.payload["same_final_answer"] is True
    assert note.payload["hand_built_message_count"] == note.payload["graph_message_count"]


def test_the_well_behaved_node_result_is_untouched() -> None:
    span = scenario_span("the_declared_state_schema_is_or_is_not_enforced_at_runtime")
    note = span.require("well_behaved_node")
    assert note.payload["last_message_type"] == "AIMessage"
    assert note.payload["last_message_content"] == "fine."


def test_the_misbehaved_node_string_is_coerced_not_rejected() -> None:
    span = scenario_span("the_declared_state_schema_is_or_is_not_enforced_at_runtime")
    note = span.require("misbehaved_node")
    assert note.payload["raised"] is False
    assert note.payload["last_message_type"] == "HumanMessage"
    assert note.payload["last_message_content"] == "not a list of messages"


def test_an_undeclared_extra_key_does_not_survive() -> None:
    span = scenario_span("the_declared_state_schema_is_or_is_not_enforced_at_runtime")
    note = span.require("misbehaved_node")
    assert note.payload["extra_key_survived"] is False
