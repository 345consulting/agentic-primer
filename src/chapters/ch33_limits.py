# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""recursion_limit at the boundary.

`ch32` built the loop as a graph and set its own cap aside as this
chapter's subject. `recursion_limit` is `ch03`'s turn cap, reimplemented
by LangGraph itself rather than hand-written -- but it counts something
subtly different, and fails differently when it binds.

Four scenarios:

    a_run_within_the_limit_completes_normally
        the limit is set, and never binds -- baseline
    a_run_that_never_stops_hits_the_limit_and_raises
        a model that always asks for the same tool, never a plain answer.
        ch03's cap ends the run and reports why; this ends the program --
        GraphRecursionError propagates out of invoke() unless the caller
        catches it
    the_limit_is_set_per_call_not_per_graph
        the same compiled graph, invoked twice with two different
        recursion_limit values -- one raises, one does not. The limit is
        a config passed to invoke(), not a property compile() fixed
    a_nested_graph_spends_the_parents_budget
        a compiled child graph, added as a node inside a parent graph,
        spends the parent's recursion_limit rather than getting one of
        its own -- unless the child is invoked as its own separate
        invoke() call, with its own separate budget

**"Recursion" is a misnomer.** There is no call stack. `recursion_limit`
counts super-steps -- one synchronized round of whatever the graph has
scheduled next, which can hold more than one node once fan-out exists
(`ch37`). It happens to equal "how many node calls has this run made"
only because nothing here fans out.

**Found live: the infinite loop is the mock's cooperation, not a property
of the graph.** Scenarios 2 and 4 rely on `_looping_graph`'s mock script
always re-requesting the same tool. Under mock, `ask_model` plays that
script back verbatim, so the loop truly never ends on its own and only
`recursion_limit` can stop it. Live, `ask_model` hands the real messages
and tools to the model and lets it decide -- and once it has the tool's
answer, it has no reason to ask again, so it just answers. Both scenarios
complete normally live; the limit never binds, because nothing live ever
tries to loop forever in the first place. `recursion_limit` itself is
still proven live, just by scenario 3, whose two budgets bracket a graph
that stops on its own either way. The standing warning that mock and live
do not prove the same things applies here exactly as it did to
`ch03`'s cap.
"""

from support.agent import ask_model, execute_tool
from support.trace import ModelKind, Trace

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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


def _limit(n: int) -> RunnableConfig:
    return {"recursion_limit": n}


def route_on_tool_calls(state: State) -> str:
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    return "call_tool" if last.tool_calls else END


def _looping_graph(
    trace: Trace, model_kind: ModelKind, steps: list[int], limit: int
) -> CompiledStateGraph[State, None, State, State]:
    """A graph that never routes to END on its own -- every reply asks for
    the same tool again. Only a recursion_limit can stop it. `steps` is
    incremented once per node call -- both call_model and call_tool -- so
    a scenario can read off how many super-steps actually ran, alongside
    whatever limit it set. `turn` is separate: it names which model call a
    given tool dispatch belongs to, so call_tool's span nests under the
    same turn number as the call_model that requested it -- the two nodes
    are one turn in this primer's sense, even though they are two
    super-steps in LangGraph's. `limit` is recorded in every turn's own
    span, not just the scenario's summary note -- so the budget a turn is
    running against is visible next to it, not only after the fact.
    """
    turn = [0]

    def call_model(state: State) -> State:
        steps[0] += 1
        turn[0] += 1
        with trace.span("turn", number=turn[0]) as span:
            span.add_note("budget", limit_set=limit, step_number=steps[0])
            script = AIMessage(
                "",
                tool_calls=[
                    {"name": "check_order", "args": {"order_id": "A100"}, "id": f"c{turn[0]}"}
                ],
            )
            reply = ask_model(trace, model_kind, [script], state["messages"], [check_order])
        return {"messages": [reply]}

    def call_tool(state: State) -> State:
        steps[0] += 1
        with trace.span("turn", number=turn[0]) as span:
            span.add_note("budget", limit_set=limit, step_number=steps[0])
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


def _finite_graph(
    trace: Trace, model_kind: ModelKind, steps: list[int], limit: int
) -> CompiledStateGraph[State, None, State, State]:
    """A graph that does stop on its own -- one tool call, then a plain
    answer. Contrast to _looping_graph, which never routes to END. Same
    `steps` and `limit` convention as _looping_graph.
    """
    turn = [0]
    script = [
        AIMessage(
            "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
        ),
        AIMessage("Order A100 has shipped."),
    ]

    def call_model(state: State) -> State:
        turn[0] += 1
        steps[0] += 1
        with trace.span("turn", number=turn[0]) as span:
            span.add_note("budget", limit_set=limit, step_number=steps[0])
            reply = ask_model(
                trace, model_kind, [script[turn[0] - 1]], state["messages"], [check_order]
            )
        return {"messages": [reply]}

    def call_tool(state: State) -> State:
        steps[0] += 1
        with trace.span("turn", number=turn[0]) as span:
            span.add_note("budget", limit_set=limit, step_number=steps[0])
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


def a_run_within_the_limit_completes_normally(trace: Trace, model_kind: ModelKind) -> None:
    """The limit is set comfortably above what the loop needs -- it never
    binds, and the run finishes the way ch32's loop scenario did.
    """
    name = "a_run_within_the_limit_completes_normally"
    with trace.span("scenario", name=name) as span:
        steps = [0]
        limit = 10
        compiled = _finite_graph(trace, model_kind, steps, limit)
        result = compiled.invoke(
            {
                "messages": [
                    SystemMessage("Answer briefly."),
                    HumanMessage("What's the status of order A100?"),
                ]
            },
            config=_limit(limit),
        )
        span.add_note(
            "completed",
            raised=False,
            limit_set=limit,
            steps_taken=steps[0],
            final_answer=str(result["messages"][-1].content),
        )


def a_run_that_never_stops_hits_the_limit_and_raises(trace: Trace, model_kind: ModelKind) -> None:
    """ch03's cap ends the run and returns a state with an explanation.
    This ends the program -- GraphRecursionError propagates out of
    invoke() itself; there is no state to inspect unless the caller wraps
    the call.

    Under mock only -- live, the real model answers once it has the tool
    result rather than asking again, so the limit never binds. See the
    module docstring.
    """
    name = "a_run_that_never_stops_hits_the_limit_and_raises"
    with trace.span("scenario", name=name) as span:
        steps = [0]
        limit = 4
        compiled = _looping_graph(trace, model_kind, steps, limit)
        try:
            compiled.invoke(
                {
                    "messages": [
                        SystemMessage("Answer briefly."),
                        HumanMessage("What's the status of order A100?"),
                    ]
                },
                config=_limit(limit),
            )
            span.add_note(
                "unexpectedly_completed", raised=False, limit_set=limit, steps_taken=steps[0]
            )
        except GraphRecursionError as exc:
            span.add_note(
                "hit_the_limit",
                raised=True,
                limit_set=limit,
                steps_taken=steps[0],
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )


def the_limit_is_set_per_call_not_per_graph(trace: Trace, model_kind: ModelKind) -> None:
    """The same graph, built and invoked twice -- one call's config gives
    it enough budget, the other does not. compile() fixed nothing about
    how many steps a run gets; invoke()'s caller decides that, every time,
    the same way ch03's cap was always a parameter to run().
    """
    name = "the_limit_is_set_per_call_not_per_graph"
    with trace.span("scenario", name=name) as span:
        # A fresh compile per call: turn counters close over their own
        # graph, and this scenario's point is the config, not the state.
        inputs: State = {
            "messages": [
                SystemMessage("Answer briefly."),
                HumanMessage("What's the status of order A100?"),
            ]
        }

        steps_low = [0]
        raised_low = False
        try:
            _finite_graph(trace, model_kind, steps_low, 2).invoke(inputs, config=_limit(2))
        except GraphRecursionError:
            raised_low = True

        steps_high = [0]
        raised_high = False
        try:
            _finite_graph(trace, model_kind, steps_high, 50).invoke(inputs, config=_limit(50))
        except GraphRecursionError:
            raised_high = True

        span.add_note(
            "same_graph_two_budgets",
            limit_set_low=2,
            raised_with_limit_2=raised_low,
            steps_taken_with_limit_2=steps_low[0],
            limit_set_high=50,
            raised_with_limit_50=raised_high,
            steps_taken_with_limit_50=steps_high[0],
        )


def a_nested_graph_spends_the_parents_budget(trace: Trace, model_kind: ModelKind) -> None:
    """A compiled child graph, added as a node inside a parent -- its
    super-steps count against the parent's recursion_limit, not a budget
    of its own. The contrast: a node that calls the same child through its
    own separate invoke() gets a separate budget, because that call is its
    own top-level run.

    Reuses _looping_graph, so the same mock-only caveat applies: live, the
    child finishes on its own and neither path raises. The budget-sharing
    claim is what the mock proves; nothing here contradicts it, there is
    just nothing live that needs the budget.
    """
    name = "a_nested_graph_spends_the_parents_budget"
    with trace.span("scenario", name=name) as span:
        inputs: State = {
            "messages": [
                SystemMessage("Answer briefly."),
                HumanMessage("What's the status of order A100?"),
            ]
        }

        limit = 4

        # Nested as a node -- shares the parent's step count.
        nested_steps = [0]
        child = _looping_graph(trace, model_kind, nested_steps, limit)
        nested_parent = StateGraph(State)
        nested_parent.add_node("child", child)
        nested_parent.add_edge(START, "child")
        nested_parent.add_edge("child", END)
        compiled_nested = nested_parent.compile()

        nested_raised = False
        try:
            compiled_nested.invoke(inputs, config=_limit(limit))
        except GraphRecursionError:
            nested_raised = True

        # Invoked from inside a plain node -- its own top-level call, its
        # own budget, caught locally rather than propagating.
        isolated_steps = [0]
        isolated_child = _looping_graph(trace, model_kind, isolated_steps, limit)

        def run_isolated_child(state: State) -> State:
            try:
                isolated_child.invoke(state, config=_limit(limit))
                return {"messages": [AIMessage("child finished")]}
            except GraphRecursionError:
                return {"messages": [AIMessage("child hit its own limit, parent continues")]}

        wrapper = StateGraph(State)
        wrapper.add_node("run_isolated_child", run_isolated_child)
        wrapper.add_edge(START, "run_isolated_child")
        wrapper.add_edge("run_isolated_child", END)
        compiled_wrapper = wrapper.compile()

        wrapper_raised = False
        try:
            result = compiled_wrapper.invoke(inputs, config=_limit(limit))
        except GraphRecursionError:
            wrapper_raised = True
            result = None

        span.add_note(
            "nested_vs_isolated",
            limit_set=limit,
            nested_as_node_raised=nested_raised,
            nested_steps_taken=nested_steps[0],
            invoked_separately_raised=wrapper_raised,
            isolated_child_steps_taken=isolated_steps[0],
            wrapper_final_message=str(result["messages"][-1].content) if result else None,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch33_limits", model_kind=model_kind)

    a_run_within_the_limit_completes_normally(trace, model_kind)
    a_run_that_never_stops_hits_the_limit_and_raises(trace, model_kind)
    the_limit_is_set_per_call_not_per_graph(trace, model_kind)
    a_nested_graph_spends_the_parents_budget(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
