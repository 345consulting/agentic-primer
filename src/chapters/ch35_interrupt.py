# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""interrupt and Command(resume=...) as an approval gate.

`ch19_guards` built a hook that can say no, in code, instantly, before
dispatch. `interrupt()` is a different kind of gate: it does not decide
anything itself -- it hands the decision to something outside the
process entirely, a human or an external system, and the run simply
stops there. Not blocked in memory: `.invoke()` returns, completely, the
moment `interrupt()` fires. What is paused is the checkpoint, not a
thread -- `ch34`'s own mechanism, extended.

Four scenarios, each wrapped in explicit "pause" and "resume" spans --
`ch34`'s own "checkpoint" span pattern, so the boundary between the two
invoke() calls is visible, not just a single summary note at the end:

    interrupt_pauses_the_run_and_returns_immediately
        a node calls interrupt() -- invoke() returns normally, no
        exception, carrying an __interrupt__ key describing what is
        being asked. A clean pause, not an error path
    command_resume_dispatches_straight_to_the_pending_node
        found live: a completed node never re-runs on resume -- only
        the node that called interrupt() does, dispatched straight from
        the checkpoint's own pending-task list (snap.next, snap.tasks),
        never re-derived from START's edges
    code_before_the_interrupt_call_runs_twice_on_resume
        the gotcha this chapter exists to surface: a counter incremented
        before the interrupt() call inside the same node increments
        again on resume, because the whole node reruns from its own top
        and only the interrupt() call itself is short-circuited by a
        recorded answer
    an_approval_gate_actually_gates_the_action
        the real shape: call_model proposes a place_order tool call,
        approval_gate calls interrupt() with that proposal *before*
        execute_tool ever dispatches it. Command(resume=False) means the
        order is never placed; Command(resume=True) means it is. Contrast
        to ch19's guards: a guard decides in-code, instantly; this blocks
        for however long an answer takes, with nothing running while it
        waits

**Why this is not ch34 again.** `ch34`'s resume picked an *unplanned*
crash back up -- the checkpoint existed because something failed.
`interrupt()`'s pause is deliberate, part of the design, chosen by the
node itself before it does something that needs approval. The mechanism
underneath -- a checkpoint, `Command(resume=...)`, dispatch from
`snap.next` -- is the same one `ch34` already proved; `interrupt()` is
what a node calls to make a pause happen on purpose instead of by
accident.
"""

from support.agent import ask_model, execute_tool
from support.trace import ModelKind, Trace

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, StateSnapshot, interrupt


class ApprovalState(TypedDict):
    order_total: int
    approved: bool


class GateState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def place_order(item: str, quantity: int) -> str:
    """Place an order for a quantity of an item."""
    return f"ordered {quantity} x {item}"


def _thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _json_safe_values(values: dict[str, object]) -> dict[str, object]:
    """Same rule ch34's own checkpoint recorder used: a message list is
    not JSON on its own -- record role/content, not the raw objects.
    """
    safe: dict[str, object] = {}
    for key, value in values.items():
        if key == "messages" and isinstance(value, list):
            safe[key] = [
                {"role": m.type, "content": str(m.content)} if isinstance(m, BaseMessage) else m
                for m in value
            ]
        else:
            safe[key] = value
    return safe


def _describe(snap: StateSnapshot) -> dict[str, object]:
    """What a checkpoint holds at this moment -- the state, the pending
    node(s), and any interrupt value attached to a pending task. The
    same information ch34's own `_record_checkpoint` read, plus the
    scheduling fields interrupt() adds.
    """
    pending_interrupts = [i.value for task in snap.tasks for i in task.interrupts]
    return {
        "values": _json_safe_values(dict(snap.values)),
        "pending_node": list(snap.next),
        "pending_interrupt_value": pending_interrupts[0] if pending_interrupts else None,
    }


def interrupt_pauses_the_run_and_returns_immediately(trace: Trace, _model_kind: ModelKind) -> None:
    """A node calls interrupt() -- invoke() returns normally, carrying
    what was asked, not an exception.
    """
    name = "interrupt_pauses_the_run_and_returns_immediately"
    with trace.span("scenario", name=name) as span:

        def place_order(state: ApprovalState) -> dict[str, object]:
            interrupt({"question": "approve this order?", "total": state["order_total"]})
            return {"approved": True}  # never reached on this pass

        graph = StateGraph(ApprovalState)
        graph.add_node("place_order", place_order)
        graph.add_edge(START, "place_order")
        graph.add_edge("place_order", END)
        compiled = graph.compile(checkpointer=InMemorySaver())
        thread = _thread("gate-1")

        raised = False
        with trace.span("pause") as pause_span:
            try:
                result = compiled.invoke({"order_total": 500, "approved": False}, config=thread)
            except Exception:
                raised = True
                result = {}
            pause_span.add_note("checkpoint_after_pause", **_describe(compiled.get_state(thread)))

        interrupts = result.get("__interrupt__", [])
        first: Interrupt | None = interrupts[0] if interrupts else None

        span.add_note(
            "the_call_returned_instead_of_raising",
            raised=raised,
            interrupt_present=first is not None,
            interrupt_value=first.value if first else None,
            approved_field_untouched=result.get("approved"),
        )


def command_resume_dispatches_straight_to_the_pending_node(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """step_one completes and is never re-scheduled; only place_order,
    the node that actually called interrupt(), runs again on resume --
    dispatched from the checkpoint's own pending-task list.
    """
    name = "command_resume_dispatches_straight_to_the_pending_node"
    with trace.span("scenario", name=name) as span:
        step_one_runs = [0]

        def step_one(
            state: ApprovalState,  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
        ) -> dict[str, object]:
            step_one_runs[0] += 1
            return {}

        def place_order(state: ApprovalState) -> dict[str, object]:
            decision = interrupt({"question": "approve?", "total": state["order_total"]})
            return {"approved": bool(decision)}

        graph = StateGraph(ApprovalState)
        graph.add_node("step_one", step_one)
        graph.add_node("place_order", place_order)
        graph.add_edge(START, "step_one")
        graph.add_edge("step_one", "place_order")
        graph.add_edge("place_order", END)
        compiled = graph.compile(checkpointer=InMemorySaver())
        thread = _thread("gate-2")

        with trace.span("pause") as pause_span:
            compiled.invoke({"order_total": 500, "approved": False}, config=thread)
            pause_span.add_note(
                "checkpoint_after_pause",
                step_one_run_count=step_one_runs[0],
                **_describe(compiled.get_state(thread)),
            )

        with trace.span("resume") as resume_span:
            result = compiled.invoke(Command(resume=True), config=thread)
            resume_span.add_note(
                "checkpoint_after_resume",
                step_one_run_count=step_one_runs[0],
                **_describe(compiled.get_state(thread)),
            )

        span.add_note(
            "only_the_pending_node_reran",
            step_one_run_count=step_one_runs[0],
            final_approved=result["approved"],
        )


def code_before_the_interrupt_call_runs_twice_on_resume(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """A counter incremented before interrupt() -- it counts twice, once
    per pass through the node, because resume reruns the whole node from
    its own top and only the interrupt() call itself is short-circuited.
    """
    name = "code_before_the_interrupt_call_runs_twice_on_resume"
    with trace.span("scenario", name=name) as span:
        before_interrupt_runs = [0]

        def place_order(state: ApprovalState) -> dict[str, object]:
            before_interrupt_runs[0] += 1
            decision = interrupt({"question": "approve?", "total": state["order_total"]})
            return {"approved": bool(decision)}

        graph = StateGraph(ApprovalState)
        graph.add_node("place_order", place_order)
        graph.add_edge(START, "place_order")
        graph.add_edge("place_order", END)
        compiled = graph.compile(checkpointer=InMemorySaver())
        thread = _thread("gate-3")

        with trace.span("pause") as pause_span:
            compiled.invoke({"order_total": 500, "approved": False}, config=thread)
            pause_span.add_note("before_interrupt_runs_so_far", count=before_interrupt_runs[0])

        with trace.span("resume") as resume_span:
            compiled.invoke(Command(resume=True), config=thread)
            resume_span.add_note("before_interrupt_runs_so_far", count=before_interrupt_runs[0])

        span.add_note(
            "the_pre_interrupt_line_ran_twice",
            before_interrupt_runs=before_interrupt_runs[0],
        )


def an_approval_gate_actually_gates_the_action(trace: Trace, model_kind: ModelKind) -> None:
    """The real use case: an order is only placed if the resumed
    decision says so. Contrast to ch19's guards -- this decides nothing
    itself, it blocks until something outside the process answers. The
    gate sits exactly between the two: after the model decides what it
    wants to do, before the tool actually does it.
    """
    name = "an_approval_gate_actually_gates_the_action"
    with trace.span("scenario", name=name) as span:
        orders_placed: list[str] = []

        script = AIMessage(
            "",
            tool_calls=[
                {
                    "name": "place_order",
                    "args": {"item": "widget", "quantity": 500},
                    "id": "c1",
                }
            ],
        )

        def call_model(state: GateState) -> dict[str, object]:
            with trace.span("turn", number=1):
                reply = ask_model(trace, model_kind, [script], state["messages"], [place_order])
            return {"messages": [reply]}

        def approval_gate(state: GateState) -> dict[str, object]:
            last = state["messages"][-1]
            assert isinstance(last, AIMessage)
            call = last.tool_calls[0]
            # interrupt() sits here -- after the model proposed the tool
            # call, before execute_tool ever dispatches it.
            decision = interrupt({"tool": call["name"], "args": call["args"]})
            if decision:
                message = execute_tool(call, trace, lambda _n, a, _s: place_order.invoke(a))
                orders_placed.append(str(message.content))
            else:
                message = ToolMessage(content="denied by approver", tool_call_id=call["id"])
            return {"messages": [message]}

        def build_graph() -> CompiledStateGraph[GateState, None, GateState, GateState]:
            graph = StateGraph(GateState)
            graph.add_node("call_model", call_model)
            graph.add_node("approval_gate", approval_gate)
            graph.add_edge(START, "call_model")
            graph.add_edge("call_model", "approval_gate")
            graph.add_edge("approval_gate", END)
            return graph.compile(checkpointer=InMemorySaver())

        approved_compiled = build_graph()
        approved_thread = _thread("gate-approved")
        with trace.span("pause", thread="approved") as pause_span:
            approved_compiled.invoke(
                {"messages": [HumanMessage("Order 500 widgets.")]}, config=approved_thread
            )
            pause_span.add_note(
                "checkpoint_after_pause", **_describe(approved_compiled.get_state(approved_thread))
            )
        with trace.span("resume", thread="approved") as resume_span:
            approved_result = approved_compiled.invoke(Command(resume=True), config=approved_thread)
            resume_span.add_note("orders_placed_so_far", orders=list(orders_placed))

        denied_compiled = build_graph()
        denied_thread = _thread("gate-denied")
        with trace.span("pause", thread="denied") as pause_span:
            denied_compiled.invoke(
                {"messages": [HumanMessage("Order 500 widgets.")]}, config=denied_thread
            )
            pause_span.add_note(
                "checkpoint_after_pause", **_describe(denied_compiled.get_state(denied_thread))
            )
        with trace.span("resume", thread="denied") as resume_span:
            denied_result = denied_compiled.invoke(Command(resume=False), config=denied_thread)
            resume_span.add_note("orders_placed_so_far", orders=list(orders_placed))

        span.add_note(
            "the_gate_actually_gated_the_order",
            orders_placed=orders_placed,
            approved_thread_final=str(approved_result["messages"][-1].content),
            denied_thread_final=str(denied_result["messages"][-1].content),
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch35_interrupt", model_kind=model_kind)

    interrupt_pauses_the_run_and_returns_immediately(trace, model_kind)
    command_resume_dispatches_straight_to_the_pending_node(trace, model_kind)
    code_before_the_interrupt_call_runs_twice_on_resume(trace, model_kind)
    an_approval_gate_actually_gates_the_action(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
