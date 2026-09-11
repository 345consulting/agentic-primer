# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The supervisor as a graph node -- what did it buy?

`ch12_supervisor` built multi-agent the cheap way: a tool whose body is a
whole second `run_turns`. Its own docstring names the other way to build
it -- a worker's whole transcript merging into the parent's context
instead of one string coming back. This chapter builds that second way
and asks what it actually costs and buys.

Four scenarios:

    a_subgraph_with_the_same_schema_shares_the_parents_state
        a child graph sharing the parent's exact State, added as a node
        with add_node("child", compiled_child) -- no wrapper function, no
        mapping. The child's whole sub-conversation merges straight into
        the parent's messages list. ch12's supervisor never let this
        happen; only the sub-agent's final string ever crossed the line
    a_subgraph_with_a_narrower_schema_only_exposes_what_it_maps
        found live: a child with its own schema, added the same direct
        way -- LangGraph maps by shared key name, both ways, silently.
        A field the parent's schema never declared is dropped on the way
        back, even though the child wrote it -- ch32's own finding about
        undeclared keys, reproduced at a graph boundary instead of inside
        one node's return
    the_context_bill_differs_even_for_the_same_work
        the identical sub-task, run once ch12's way and once as a
        same-schema subgraph -- the parent's final message count is the
        direct answer to "what did it buy": supervisor costs one string,
        subgraph costs the whole child transcript, for equal work done
    the_childs_super_steps_count_against_the_parents_recursion_limit
        ch33's nested-budget finding, reproduced on a real two-node child
        instead of the minimal stand-in -- confirms nothing about that
        finding depended on the toy shape it was first shown in

**Nesting was already free in `ch12`.** `Trace.span` pushes onto a stack
regardless of what is running inside it, so a subgraph's turns nest under
its parent's tool span the same way `ch12`'s inner `run_turns` did. What a
graph adds is not nesting -- it is a second way to draw the boundary
between them, with a different default (share everything) and an opt-in
narrower one, instead of `ch12`'s only option (share nothing but a
string).
"""

from support.agent import ask_model, execute_tool, run_turns
from support.trace import ModelKind, Trace

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def route_on_tool_calls(state: State) -> str:
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    return "call_tool" if last.tool_calls else END


def _limit(n: int) -> RunnableConfig:
    return {"recursion_limit": n}


def _child_graph(
    trace: Trace, model_kind: ModelKind, replies: list[AIMessage]
) -> CompiledStateGraph[State, None, State, State]:
    """A model node and a tool node, the same shape as ch32's loop
    scenario -- reused here as a subgraph's body instead of the whole
    program's.
    """
    turn = [0]

    def call_model(state: State) -> State:
        turn[0] += 1
        with trace.span("turn", number=turn[0]):
            reply = ask_model(
                trace, model_kind, [replies[turn[0] - 1]], state["messages"], [check_order]
            )
        return {"messages": [reply]}

    def call_tool(state: State) -> State:
        with trace.span("turn", number=turn[0]):
            last = state["messages"][-1]
            assert isinstance(last, AIMessage)
            return {
                "messages": [
                    execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))
                    for call in last.tool_calls
                ]
            }

    graph = StateGraph(State)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tool", call_tool)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model", route_on_tool_calls, {"call_tool": "call_tool", END: END}
    )
    graph.add_edge("call_tool", "call_model")
    return graph.compile()


def a_subgraph_with_the_same_schema_shares_the_parents_state(
    trace: Trace, model_kind: ModelKind
) -> None:
    """No wrapper, no mapping -- add_node("child", compiled_child) with an
    identical State. The child's own turns become the parent's own
    messages, indistinguishable from a node the parent wrote itself.
    """
    name = "a_subgraph_with_the_same_schema_shares_the_parents_state"
    with trace.span("scenario", name=name) as span:
        replies = [
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
            AIMessage("Order A100 has shipped."),
        ]
        child = _child_graph(trace, model_kind, replies)

        parent = StateGraph(State)
        parent.add_node("child", child)
        parent.add_edge(START, "child")
        parent.add_edge("child", END)
        compiled_parent = parent.compile()

        result = compiled_parent.invoke(
            {
                "messages": [
                    SystemMessage("Answer briefly."),
                    HumanMessage("What's the status of order A100?"),
                ]
            }
        )
        span.add_note(
            "the_childs_whole_transcript_is_now_the_parents",
            roles=[m.type for m in result["messages"]],
            message_count=len(result["messages"]),
        )


class ParentState(TypedDict):
    order_id: str
    note: str
    status: str


class ChildState(TypedDict):
    order_id: str
    status: str
    scratch: str


def a_subgraph_with_a_narrower_schema_only_exposes_what_it_maps(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """The child writes a field the parent's schema never declared.
    LangGraph maps parent -> child -> parent by shared key name -- found
    live: the undeclared field does not raise, does not warn, it is just
    gone from what the parent ever sees. ch32's own finding, at a graph
    boundary instead of inside one node's return.
    """
    name = "a_subgraph_with_a_narrower_schema_only_exposes_what_it_maps"
    with trace.span("scenario", name=name) as span:

        def check(state: ChildState) -> ChildState:
            return {
                "order_id": state["order_id"],
                "status": f"order {state['order_id']}: shipped",
                "scratch": "internal working note, never declared by the parent",
            }

        child = StateGraph(ChildState)
        child.add_node("check", check)
        child.add_edge(START, "check")
        child.add_edge("check", END)
        compiled_child = child.compile()

        parent = StateGraph(ParentState)
        parent.add_node("child", compiled_child)
        parent.add_edge(START, "child")
        parent.add_edge("child", END)
        compiled_parent = parent.compile()

        result = compiled_parent.invoke({"order_id": "A100", "note": "hi", "status": ""})

        span.add_note(
            "only_the_declared_keys_survive",
            result_keys=sorted(result.keys()),
            status_mapped_back=result.get("status"),
            scratch_present="scratch" in result,
        )


def the_context_bill_differs_even_for_the_same_work(trace: Trace, model_kind: ModelKind) -> None:
    """The identical sub-task -- ask about order A100, one tool call, one
    answer -- run once ch12's way (a tool whose body is run_turns) and once
    as a same-schema subgraph node. Same work, different bill in the
    parent's own context.
    """
    name = "the_context_bill_differs_even_for_the_same_work"
    with trace.span("scenario", name=name) as span:
        script = [
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
            AIMessage("Order A100 has shipped."),
        ]

        # ch12's way: a tool whose body is a whole run_turns, called from
        # inside execute_tool. Only its final string crosses into the
        # supervisor's own messages.
        supervisor_messages: list[BaseMessage] = [
            SystemMessage("Answer briefly."),
            HumanMessage("What's the status of order A100?"),
        ]

        def run_sub_agent_as_a_tool(_question: str) -> str:
            sub_messages: list[BaseMessage] = [
                SystemMessage("Answer briefly."),
                HumanMessage("What's the status of order A100?"),
            ]

            def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
                return ask_model(trace, model_kind, script[turn - 1 :], so_far, [check_order])

            def execute(call: ToolCall) -> ToolMessage:
                return execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))

            run_turns(sub_messages, trace, 2, ask, execute)
            return str(sub_messages[-1].content)

        call: ToolCall = {"name": "consult_order_expert", "args": {}, "id": "outer1"}
        supervisor_messages.append(
            AIMessage("", tool_calls=[{"name": "consult_order_expert", "args": {}, "id": "outer1"}])
        )
        with trace.span("turn", number=1):
            supervisor_messages.append(
                execute_tool(call, trace, lambda _n, _a, _s: run_sub_agent_as_a_tool(""))
            )

        # The subgraph way: same script, same tool, added directly as a
        # node with the same State the parent already uses.
        child = _child_graph(trace, model_kind, script)
        parent = StateGraph(State)
        parent.add_node("child", child)
        parent.add_edge(START, "child")
        parent.add_edge("child", END)
        compiled_parent = parent.compile()
        subgraph_result = compiled_parent.invoke(
            {
                "messages": [
                    SystemMessage("Answer briefly."),
                    HumanMessage("What's the status of order A100?"),
                ]
            }
        )

        span.add_note(
            "same_work_different_bill",
            supervisor_parent_message_count=len(supervisor_messages),
            subgraph_parent_message_count=len(subgraph_result["messages"]),
        )


def the_childs_super_steps_count_against_the_parents_recursion_limit(
    trace: Trace, model_kind: ModelKind
) -> None:
    """ch33's finding, shown again on a real two-node child instead of
    the minimal single-node stand-in: nesting it as a node spends the
    parent's own recursion_limit, not a budget of its own.

    Under mock only, for ch33's own reason: the child is scripted to
    always re-request the tool, and only a scripted model cooperates with
    that. Live, the real model answers once it has the tool result rather
    than looping, so the limit never binds and nothing here contradicts
    the finding -- there is just nothing live that needs the budget.
    """
    name = "the_childs_super_steps_count_against_the_parents_recursion_limit"
    with trace.span("scenario", name=name) as span:
        looping_replies = [
            AIMessage(
                "",
                tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": f"c{i}"}],
            )
            for i in range(1, 10)
        ]
        child = _child_graph(trace, model_kind, looping_replies)

        parent = StateGraph(State)
        parent.add_node("child", child)
        parent.add_edge(START, "child")
        parent.add_edge("child", END)
        compiled_parent = parent.compile()

        raised = False
        try:
            compiled_parent.invoke(
                {
                    "messages": [
                        SystemMessage("Answer briefly."),
                        HumanMessage("What's the status of order A100?"),
                    ]
                },
                config=_limit(4),
            )
        except GraphRecursionError:
            raised = True

        span.add_note("parents_limit_binds_on_the_childs_steps", raised=raised)


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch36_subgraph", model_kind=model_kind)

    a_subgraph_with_the_same_schema_shares_the_parents_state(trace, model_kind)
    a_subgraph_with_a_narrower_schema_only_exposes_what_it_maps(trace, model_kind)
    the_context_bill_differs_even_for_the_same_work(trace, model_kind)
    the_childs_super_steps_count_against_the_parents_recursion_limit(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
