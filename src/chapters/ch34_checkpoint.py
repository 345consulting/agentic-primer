# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""MemorySaver, thread_id, resume -- and why that is not memory.

`ch31_stop_resume` picked up one stopped run from a jsonl snapshot written
once per turn -- deserialize the last line, keep calling ask_model like
nothing happened. `InMemorySaver` (LangGraph's `MemorySaver`) is the same
idea, reimplemented, with two differences the comparison exists to find:
it needs no explicit save call at all, and it checkpoints at a finer
grain than a turn.

Five scenarios:

    a_checkpointer_resumes_a_thread_across_separate_invoke_calls
        two invoke() calls, one thread_id, the second call passes only
        the new message -- the reply proves the full prior history was
        there anyway, restored by the checkpointer with nothing asked for
    two_thread_ids_never_see_each_others_history
        the same compiled graph, two thread_ids -- ch31's "two run_ids
        never see each other's history," with the checkpointer doing the
        separating instead of two jsonl files
    the_checkpoint_granularity_is_finer_than_our_snapshot
        found live in mock: a crash between call_model and call_tool --
        the checkpoint already holds the model's tool-call request, and
        resuming reruns only call_tool. ch31's snapshot was per turn, so
        the same crash there would have re-asked the model; this does not
    a_checkpoint_is_not_memory
        a fresh thread_id has no access to a fact from another thread, not
        even a relevant one -- no search, no recall, nothing. ch29's
        memory store answered a query across any past run; a checkpoint
        resumes one run's own state. The same word would be wrong for both
    we_can_bolt_our_own_disk_persistence_onto_it
        InMemorySaver keeps every checkpoint in a plain dict in this
        process -- kill the process and it is gone, unlike ch31's jsonl
        file. LangGraph ships no first-party file-based saver, only
        SQLite and Postgres. Nothing stops us writing our own: serialize
        get_state().values with ch31's own message format, discard the
        checkpointer entirely, build a brand new one from a fresh
        process's worth of state, and seed it with update_state() before
        the first invoke() -- the thread resumes anyway

**What "resume" bought here that ch31 could not, until scenario five.**
`ch31`'s resume was a deserialize step we wrote -- read the last line,
rebuild message objects, call ask_model. `invoke(None, config)` needs no
such step, because the checkpointer already holds real objects and every
node's return commits automatically -- but only within one process.
Scenario five shows the deserialize step was never obsolete; it just
moved from "every turn" to "only when the checkpointer itself would not
have survived."
"""

from support.agent import execute_tool
from support.trace import Json, ModelKind, Trace

import json
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

OUT_ROOT = Path("out")


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


def _thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _serialize_message(m: BaseMessage) -> Json:
    """ch31's own format -- unchanged, because the shape of a message on
    disk does not depend on what checkpoints it.
    """
    d: Json = {"role": m.type, "content": str(m.content)}
    if isinstance(m, AIMessage) and m.tool_calls:
        d["tool_calls"] = [dict(c) for c in m.tool_calls]
    return d


def _deserialize_message(d: Json) -> BaseMessage:
    role = d["role"]
    if role == "system":
        return SystemMessage(d["content"])
    if role == "human":
        return HumanMessage(d["content"])
    if role == "ai":
        return AIMessage(d["content"], tool_calls=d.get("tool_calls") or [])
    raise ValueError(f"unknown role in disk snapshot: {role!r}")


def _write_disk_snapshot(path: Path, messages: list[BaseMessage]) -> None:
    path.write_text(json.dumps({"messages": [_serialize_message(m) for m in messages]}))


def _read_disk_snapshot(path: Path) -> list[BaseMessage]:
    line = json.loads(path.read_text())
    return [_deserialize_message(d) for d in line["messages"]]


def _record_checkpoint(
    trace: Trace,
    compiled: CompiledStateGraph[State, None, State, State],
    thread: RunnableConfig,
    label: str,
) -> None:
    """A read of whatever the checkpointer holds for this thread, right
    now. This is us asking InMemorySaver what it has, not an intercepted
    write -- LangGraph commits a checkpoint after every node returns,
    invisibly to us, the same way a provider's own cache write in ch28
    was never something we could watch happen either.
    """
    configurable = thread.get("configurable")
    assert configurable is not None
    with trace.span("checkpoint", label=label, thread_id=configurable["thread_id"]) as span:
        values = compiled.get_state(thread).values
        messages = values.get("messages", []) if values else []
        span.add_note(
            "held_by_the_checkpointer",
            message_count=len(messages),
            roles=[m.type for m in messages],
        )


def a_checkpointer_resumes_a_thread_across_separate_invoke_calls(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Two invoke() calls, one thread_id. The second passes only the new
    human message -- no system prompt, no prior turn -- and the reply
    still reflects the full history, restored by the checkpointer before
    call_model ever ran.
    """
    from support.agent import ask_model

    name = "a_checkpointer_resumes_a_thread_across_separate_invoke_calls"
    with trace.span("scenario", name=name) as span:
        replies = [
            AIMessage("This is turn one."),
            AIMessage("You already asked me something, in turn one."),
        ]
        turn = [0]

        def call_model(state: State) -> State:
            turn[0] += 1
            with trace.span("turn", number=turn[0]):
                reply = ask_model(trace, model_kind, [replies[turn[0] - 1]], state["messages"], [])
            return {"messages": [reply]}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", END)
        compiled = graph.compile(checkpointer=InMemorySaver())

        thread = _thread("resume-demo")
        compiled.invoke(
            {"messages": [SystemMessage("Answer briefly."), HumanMessage("Say something.")]},
            config=thread,
        )
        _record_checkpoint(trace, compiled, thread, "saved after turn 1")
        _record_checkpoint(trace, compiled, thread, "read back before turn 2 -- what it will load")
        result = compiled.invoke(
            {"messages": [HumanMessage("What did I just say?")]}, config=thread
        )
        _record_checkpoint(trace, compiled, thread, "saved after turn 2")

        span.add_note(
            "second_call_saw_the_first",
            second_call_input_message_count=1,
            full_history_length=len(result["messages"]),
            final_reply=str(result["messages"][-1].content),
        )


def two_thread_ids_never_see_each_others_history(trace: Trace, model_kind: ModelKind) -> None:
    """The same compiled graph, two thread_ids -- ch31's isolation, with
    the checkpointer keying by thread_id instead of a jsonl filename.
    """
    from support.agent import ask_model

    name = "two_thread_ids_never_see_each_others_history"
    with trace.span("scenario", name=name) as span:

        def call_model(state: State) -> State:
            with trace.span("turn", number=len(state["messages"])):
                reply = ask_model(trace, model_kind, [AIMessage("noted.")], state["messages"], [])
            return {"messages": [reply]}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", END)
        compiled = graph.compile(checkpointer=InMemorySaver())

        thread_a, thread_b = _thread("thread-a"), _thread("thread-b")
        compiled.invoke(
            {"messages": [SystemMessage("Answer briefly."), HumanMessage("My order is A100.")]},
            config=thread_a,
        )
        _record_checkpoint(trace, compiled, thread_a, "saved for thread-a")
        _record_checkpoint(trace, compiled, thread_b, "thread-b before its own first call -- empty")
        result_b = compiled.invoke(
            {"messages": [HumanMessage("What's my order number?")]}, config=thread_b
        )

        span.add_note(
            "thread_b_starts_empty",
            thread_b_message_count_before_reply=1,
            thread_b_never_saw_a100="A100" not in str(result_b["messages"]),
        )


def the_checkpoint_granularity_is_finer_than_our_snapshot(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A crash between call_model and call_tool. ch31 snapshotted once
    per complete turn, so a crash at this exact point would have left
    nothing newer than the previous turn -- resuming would re-ask the
    model. Here the checkpoint already holds the model's tool-call
    request; resuming reruns only call_tool.
    """
    name = "the_checkpoint_granularity_is_finer_than_our_snapshot"
    with trace.span("scenario", name=name) as span:
        from support.agent import ask_model

        model_calls = [0]
        tool_calls = [0]
        should_fail = [True]

        def call_model(state: State) -> State:
            model_calls[0] += 1
            with trace.span("turn", number=model_calls[0]):
                script = AIMessage(
                    "",
                    tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}],
                )
                reply = ask_model(trace, model_kind, [script], state["messages"], [check_order])
            return {"messages": [reply]}

        def call_tool(state: State) -> State:
            tool_calls[0] += 1
            with trace.span("turn", number=model_calls[0]):
                if should_fail[0]:
                    should_fail[0] = False
                    raise RuntimeError("simulated crash mid-turn")
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
        graph.add_edge("call_tool", END)
        compiled = graph.compile(checkpointer=InMemorySaver())

        thread = _thread("crash-demo")
        crashed = False
        try:
            compiled.invoke(
                {
                    "messages": [
                        SystemMessage("Answer briefly."),
                        HumanMessage("What's the status of order A100?"),
                    ]
                },
                config=thread,
            )
        except RuntimeError:
            crashed = True

        _record_checkpoint(trace, compiled, thread, "saved mid-turn, right after the crash")
        snapshot = compiled.get_state(thread)
        checkpoint_has_tool_call_already = any(
            isinstance(m, AIMessage) and m.tool_calls for m in snapshot.values["messages"]
        )

        result = compiled.invoke(None, config=thread)
        _record_checkpoint(trace, compiled, thread, "saved after the resumed turn completed")

        span.add_note(
            "resumed_without_reasking_the_model",
            crashed=crashed,
            checkpoint_captured_the_model_reply_before_the_crash=checkpoint_has_tool_call_already,
            call_model_invocations=model_calls[0],
            call_tool_invocations=tool_calls[0],
            final_answer_present="tool" in [m.type for m in result["messages"]],
        )


def a_checkpoint_is_not_memory(trace: Trace, model_kind: ModelKind) -> None:
    """A fresh thread_id has no access to a fact from another thread --
    not a fuzzy match, not a partial one, nothing. ch29's memory store
    answered a query across any past run by relevance; a checkpoint
    resumes one run's exact state and nothing else.
    """
    from support.agent import ask_model

    name = "a_checkpoint_is_not_memory"
    with trace.span("scenario", name=name) as span:
        # A minimal stand-in for ch29's store: one dict, written once,
        # found by exact key regardless of which thread asks.
        cross_run_memory = {"favorite_color": "teal"}

        def call_model(state: State) -> State:
            with trace.span("turn", number=1):
                reply = ask_model(
                    trace, model_kind, [AIMessage("I don't know that.")], state["messages"], []
                )
            return {"messages": [reply]}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", END)
        compiled = graph.compile(checkpointer=InMemorySaver())

        color_thread = _thread("color-thread")
        compiled.invoke(
            {
                "messages": [
                    SystemMessage("Answer briefly."),
                    HumanMessage("My favorite color is teal."),
                ]
            },
            config=color_thread,
        )
        _record_checkpoint(trace, compiled, color_thread, "saved -- scoped to color-thread only")

        fresh_thread = _thread("unrelated-thread")
        _record_checkpoint(trace, compiled, fresh_thread, "a fresh thread -- nothing to load")
        fresh_thread_has_state = bool(compiled.get_state(fresh_thread).values)

        span.add_note(
            "checkpoint_scope_vs_memory_scope",
            fresh_thread_has_any_checkpointed_state=fresh_thread_has_state,
            cross_run_memory_still_finds_the_fact=cross_run_memory.get("favorite_color") == "teal",
        )


def we_can_bolt_our_own_disk_persistence_onto_it(trace: Trace, model_kind: ModelKind) -> None:
    """InMemorySaver survives separate invoke() calls, but not a new
    process -- a brand new InMemorySaver() has no idea the old one ever
    existed. Write what it held to disk, in ch31's own format, and a
    fresh checkpointer can be seeded with update_state() before its first
    invoke() -- the thread resumes anyway.
    """
    from support.agent import ask_model

    name = "we_can_bolt_our_own_disk_persistence_onto_it"
    with trace.span("scenario", name=name) as span:

        def build_graph() -> CompiledStateGraph[State, None, State, State]:
            turn = [0]

            def call_model(state: State) -> State:
                turn[0] += 1
                with trace.span("turn", number=turn[0]):
                    reply = ask_model(
                        trace, model_kind, [AIMessage("noted, on disk now.")], state["messages"], []
                    )
                return {"messages": [reply]}

            graph = StateGraph(State)
            graph.add_node("call_model", call_model)
            graph.add_edge(START, "call_model")
            graph.add_edge("call_model", END)
            return graph.compile(checkpointer=InMemorySaver())

        thread = _thread("disk-demo")
        first_process = build_graph()
        first_process.invoke(
            {"messages": [SystemMessage("Answer briefly."), HumanMessage("Remember: A100.")]},
            config=thread,
        )
        saved_messages = first_process.get_state(thread).values["messages"]

        snapshot_path = OUT_ROOT / "ch34_checkpoint" / name / "disk-demo.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        _write_disk_snapshot(snapshot_path, saved_messages)
        span.add_note("snapshot_written", path=str(snapshot_path))

        # first_process and its InMemorySaver go out of scope here -- a
        # new process would start with neither, the same way ch31's did.
        second_process = build_graph()
        fresh_has_state_before_seeding = bool(second_process.get_state(thread).values)

        recovered_messages = _read_disk_snapshot(snapshot_path)
        second_process.update_state(thread, {"messages": recovered_messages})
        seeded_has_state = bool(second_process.get_state(thread).values)

        result = second_process.invoke(
            {"messages": [HumanMessage("What did I ask you?")]}, config=thread
        )

        span.add_note(
            "resumed_after_a_simulated_process_restart",
            fresh_process_had_no_state_before_seeding=not fresh_has_state_before_seeding,
            seeding_from_disk_restored_state=seeded_has_state,
            final_history_length=len(result["messages"]),
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch34_checkpoint", model_kind=model_kind)

    a_checkpointer_resumes_a_thread_across_separate_invoke_calls(trace, model_kind)
    two_thread_ids_never_see_each_others_history(trace, model_kind)
    the_checkpoint_granularity_is_finer_than_our_snapshot(trace, model_kind)
    a_checkpoint_is_not_memory(trace, model_kind)
    we_can_bolt_our_own_disk_persistence_onto_it(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
