# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The same behaviour as a StateGraph -- what did it buy?

Everything up to here has stayed in plain Python on purpose, so each of a
framework's answers can be asked what it bought instead of taken on faith.
This chapter builds LangGraph's core objects up the same way `ch01`
through `ch03` built the atomic loop -- one clause at a time -- and asks,
at each step, whether anything actually changed.

Five scenarios:

    a_single_model_node_is_ch01_as_a_graph
        the minimum possible graph -- START, one model node, END. No
        tools, no branching, one fixed edge. ch01_single_call, as a graph
    a_fixed_edge_from_model_to_tool_is_ch02_as_a_graph
        a model node and a tool node, connected by a plain edge -- always
        dispatches, never decides whether to. ch02_tool_call's "the model
        asks; we execute," before there is any question of when
    a_conditional_edge_turns_it_into_the_loop
        the fixed edge replaced with add_conditional_edges -- routes to
        the tool node when tool_calls is present, straight to END when it
        is not. ch03_the_loop, and also ch10's "if," now expressed as
        graph structure. Scripted for two tool calls across three turns,
        so the routing function is exercised on more than one pass -- live
        found the real model batching both requests into one turn instead
        (the same fan-out this primer has seen since ch06), needing only
        two. The graph routed both shapes correctly either way; it does
        not care how many tool_calls one turn carries, only whether the
        list is empty
    the_graph_and_the_hand_built_loop_produce_the_same_final_state
        the identical mock script, run two ways -- through run_turns, and
        through the compiled graph above -- same final message list, same
        answer. The direct "what did it buy" comparison, not just "does
        it work"
    the_declared_state_schema_is_or_is_not_enforced_at_runtime
        a node returns a bare string where the schema declares a list of
        messages -- found live: `add_messages` does not reject it, it
        *coerces* it into a legitimate-looking `HumanMessage`, silently,
        with nothing marking it as having come from a malformed update.
        `ch22` through `ch24`'s own finding, reproduced inside the
        framework meant to add structure around exactly this

**Nodes call the same functions every earlier chapter already built.**
`ask_model` and `execute_tool` do not change; they are called from inside
a node instead of inside a `while` loop. What LangGraph adds is structure
around that call, not a new way of making it.

**What this chapter does not test.** `recursion_limit` is `ch33`'s own
subject. A checkpointer, `interrupt`, subgraphs, `Send`, and reducers
belong to `ch34` through `ch38`. This chapter is the loop itself, nothing
past it yet.
"""

from support.agent import ask_model, execute_tool
from support.trace import ModelKind, Trace

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


@tool
def check_inventory(item: str) -> str:
    """Check how many units of an item are in stock."""
    return f"{item}: 12 in stock"


class State(TypedDict):
    """The declared shape of what flows through the graph -- `ch11_state`
    formalized: instead of local variables threading through a `while`
    loop by hand, the shape is written down once, upfront. `add_messages`
    is the reducer -- `ch38`'s own subject -- deciding that two updates to
    `messages` append rather than replace.
    """

    messages: Annotated[list[BaseMessage], add_messages]


def a_single_model_node_is_ch01_as_a_graph(trace: Trace, model_kind: ModelKind) -> None:
    """START, one model node, END -- no tools, no branching, one edge."""
    with trace.span("scenario", name="a_single_model_node_is_ch01_as_a_graph") as span:

        def call_model(state: State) -> State:
            with trace.span("turn", number=1):
                reply = ask_model(
                    trace, model_kind, [AIMessage("Hello there.")], state["messages"], []
                )
            return {"messages": [reply]}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", END)
        compiled = graph.compile()

        result = compiled.invoke(
            {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage("Say hello, briefly.")]}
        )
        final = result["messages"][-1]
        span.add_note(
            "final_answer", content=str(final.content), message_count=len(result["messages"])
        )


def a_fixed_edge_from_model_to_tool_is_ch02_as_a_graph(trace: Trace, model_kind: ModelKind) -> None:
    """A model node and a tool node, joined by a plain edge -- always
    dispatches, never decides whether to.
    """
    name = "a_fixed_edge_from_model_to_tool_is_ch02_as_a_graph"
    with trace.span("scenario", name=name) as span:

        def call_model(state: State) -> State:
            with trace.span("turn", number=1):
                reply = ask_model(
                    trace,
                    model_kind,
                    [
                        AIMessage(
                            "",
                            tool_calls=[
                                {"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}
                            ],
                        )
                    ],
                    state["messages"],
                    [check_order],
                )
            return {"messages": [reply]}

        def call_tool(state: State) -> State:
            last = state["messages"][-1]
            assert isinstance(last, AIMessage)
            outs: list[BaseMessage] = []
            for call in last.tool_calls:
                outs.append(
                    execute_tool(
                        call, trace, lambda n, a, _s: {"check_order": check_order}[n].invoke(a)
                    )
                )
            return {"messages": outs}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_node("call_tool", call_tool)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", "call_tool")
        graph.add_edge("call_tool", END)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "messages": [
                    SystemMessage(SYSTEM_PROMPT),
                    HumanMessage("What's the status of order A100?"),
                ]
            }
        )
        roles = [m.type for m in result["messages"]]
        span.add_note(
            "fixed_edge_always_dispatched", roles=roles, tool_call_present="tool" in roles
        )


def route_on_tool_calls(state: State) -> str:
    """The conditional edge function -- `ch10`'s `if`, expressed as graph
    structure instead of inline code. Returns the name of whichever node
    should run next, based on nothing but the last message.
    """
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    return "call_tool" if last.tool_calls else END


def a_conditional_edge_turns_it_into_the_loop(trace: Trace, model_kind: ModelKind) -> None:
    """Scripted for three turns -- two tool calls before the model asks
    for nothing further -- though live may batch both into one turn and
    finish in two. Either way, the routing function gets exercised more
    than once.
    """
    name = "a_conditional_edge_turns_it_into_the_loop"
    with trace.span("scenario", name=name) as span:
        mock_replies = [
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
            AIMessage(
                "", tool_calls=[{"name": "check_inventory", "args": {"item": "widget"}, "id": "c2"}]
            ),
            AIMessage("Order A100 has shipped, and we have 12 widgets in stock."),
        ]
        turn = [0]
        tools = {"check_order": check_order, "check_inventory": check_inventory}

        def call_model(state: State) -> State:
            turn[0] += 1
            with trace.span("turn", number=turn[0]):
                reply = ask_model(
                    trace,
                    model_kind,
                    [mock_replies[turn[0] - 1]],
                    state["messages"],
                    list(tools.values()),
                )
            return {"messages": [reply]}

        def call_tool(state: State) -> State:
            last = state["messages"][-1]
            assert isinstance(last, AIMessage)
            outs: list[BaseMessage] = []
            for call in last.tool_calls:
                outs.append(execute_tool(call, trace, lambda n, a, _s: tools[n].invoke(a)))
            return {"messages": outs}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_node("call_tool", call_tool)
        graph.add_edge(START, "call_model")
        graph.add_conditional_edges(
            "call_model", route_on_tool_calls, {"call_tool": "call_tool", END: END}
        )
        graph.add_edge("call_tool", "call_model")
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "messages": [
                    SystemMessage(SYSTEM_PROMPT),
                    HumanMessage(
                        "What's the status of order A100, and do we have widgets in stock?"
                    ),
                ]
            }
        )
        tool_turns = sum(1 for m in result["messages"] if m.type == "tool")
        span.add_note(
            "loop_ran_more_than_once",
            total_turns=turn[0],
            tool_dispatches=tool_turns,
            final_answer=str(result["messages"][-1].content),
        )


def the_graph_and_the_hand_built_loop_produce_the_same_final_state(
    trace: Trace, model_kind: ModelKind
) -> None:
    """The identical mock script, run through run_turns and through a
    compiled graph -- same final message list, same answer.
    """
    from support.agent import run_turns

    name = "the_graph_and_the_hand_built_loop_produce_the_same_final_state"
    with trace.span("scenario", name=name) as span:
        script = [
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
            AIMessage("Order A100 has shipped."),
        ]

        # The hand-built path -- ch03's own loop, composed.
        hand_built_messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("What's the status of order A100?"),
        ]

        def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
            return ask_model(trace, model_kind, script[turn - 1 :], so_far, [check_order])

        def execute(call: ToolCall) -> ToolMessage:
            return execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))

        run_turns(hand_built_messages, trace, 2, ask, execute)

        # The graph path -- identical script, same tool.
        graph_turn = [0]

        def call_model(state: State) -> State:
            graph_turn[0] += 1
            reply = ask_model(
                trace, model_kind, [script[graph_turn[0] - 1]], state["messages"], [check_order]
            )
            return {"messages": [reply]}

        def call_tool(state: State) -> State:
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
        compiled = graph.compile()
        graph_result = compiled.invoke(
            {
                "messages": [
                    SystemMessage(SYSTEM_PROMPT),
                    HumanMessage("What's the status of order A100?"),
                ]
            }
        )

        hand_built_final = str(hand_built_messages[-1].content)
        graph_final = str(graph_result["messages"][-1].content)
        span.add_note(
            "comparison",
            hand_built_final_answer=hand_built_final,
            graph_final_answer=graph_final,
            hand_built_message_count=len(hand_built_messages),
            graph_message_count=len(graph_result["messages"]),
            same_final_answer=hand_built_final == graph_final,
        )


def the_declared_state_schema_is_or_is_not_enforced_at_runtime(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """A node returns something outside the declared schema -- not
    rejected, not passed through unchanged either. `add_messages` coerces
    a bare string into a real `HumanMessage`, and an undeclared extra key
    is silently dropped rather than raising.
    """
    name = "the_declared_state_schema_is_or_is_not_enforced_at_runtime"
    with trace.span("scenario", name=name) as span:
        # Both typed loosely on purpose -- `misbehaved` exists specifically
        # to return something the declared `State` schema forbids, so
        # pretending either one is type-checked here would misrepresent
        # what this scenario tests: runtime behavior, not static typing.
        def well_behaved(_state: State) -> dict[str, object]:
            return {"messages": [AIMessage("fine.")]}

        def misbehaved(_state: State) -> dict[str, object]:
            # Not a list of messages at all -- a bare string, and an
            # extra key the schema never declared.
            return {"messages": "not a list of messages", "extra_undeclared_key": 123}

        for label, node_fn in [("well_behaved", well_behaved), ("misbehaved", misbehaved)]:
            graph = StateGraph(State)
            # This scenario's whole point is a node that violates the
            # declared schema at runtime -- mypy correctly rejects that
            # statically, which the ignore below is honestly admitting,
            # not working around.
            graph.add_node("n", node_fn)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
            graph.add_edge(START, "n")
            graph.add_edge("n", END)
            compiled = graph.compile()
            try:
                result = compiled.invoke({"messages": [HumanMessage("hi")]})
                last = result["messages"][-1]
                span.add_note(
                    f"{label}_node",
                    raised=False,
                    error=None,
                    last_message_type=type(last).__name__,
                    last_message_content=str(last.content),
                    extra_key_survived="extra_undeclared_key" in result,
                )
            except Exception as exc:  # the point is whatever LangGraph actually raises, if anything
                span.add_note(
                    f"{label}_node", raised=True, error=f"{type(exc).__name__}: {exc}"[:300]
                )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch32_graph", model_kind=model_kind)

    a_single_model_node_is_ch01_as_a_graph(trace, model_kind)
    a_fixed_edge_from_model_to_tool_is_ch02_as_a_graph(trace, model_kind)
    a_conditional_edge_turns_it_into_the_loop(trace, model_kind)
    the_graph_and_the_hand_built_loop_produce_the_same_final_state(trace, model_kind)
    the_declared_state_schema_is_or_is_not_enforced_at_runtime(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
