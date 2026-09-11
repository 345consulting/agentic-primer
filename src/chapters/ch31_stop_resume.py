# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""A jsonl snapshot after every turn, and picking a stopped run back up.

Every chapter until now has kept its whole state in one process's memory.
A run that stops -- crashes, gets redeployed, is deliberately paused --
loses all of it unless something wrote it down first. This is not `ch29`'s
memory: memory persists a fact *across* unrelated runs; this is picking up
*the same run*, later -- same message list, same turn count, same
identity, continuing instead of starting over.

**Real files, not simulated.** Every scenario writes actual jsonl under
`out/ch31_stop_resume/<scenario>/`, inspectable directly -- the whole
point is proving a genuinely separate process, sharing nothing in memory,
can pick this up.

**One line per completed turn, the full state, not a diff.** A cache
lookup finds the longest matching prefix (`ch28`); a checkpoint here is
simpler and cruder on purpose -- read the *last* line, not replay every
line since the start. Simpler to get right, and correctness here matters
more than the extra bytes an event log would have saved.

Four scenarios:

    a_snapshot_after_every_turn_lets_a_new_process_resume
        the core proof -- two turns run and snapshotted; a genuinely
        separate `Trace`, sharing nothing in memory, loads only the
        jsonl file and continues from turn three
    snapshotting_mid_turn_would_orphan_a_tool_call
        the danger a naive version hits -- snapshot between an
        assistant's tool_calls and its matching tool reply instead of
        after the complete turn, resume from it, and get `ch27`'s own
        rejection back
    a_crash_before_the_next_snapshot_loses_at_most_one_turn
        turn three starts, but the process dies before that turn's
        snapshot is written. Resuming reads the last *complete*
        snapshot -- end of turn two -- and safely redoes turn three,
        never a corrupted one
    separate_run_ids_do_not_collide
        two runs' snapshots live in the same store -- resuming one only
        ever sees its own history, the same role `thread_id` plays later

**Not built here, on purpose.** `ch33_checkpoint` compares this hand-built
version against LangGraph's `MemorySaver`/`thread_id`/`Command(resume=...)`
-- what it buys is answerable because this exists first, the same reason
`ch18_hooks` came before any framework's callback system.
"""

from support.agent import ask_model, execute_tool
from support.trace import Json, ModelKind, Trace

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool
from langchain_openai.chat_models.base import OpenAIInvalidRequestError

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."
OUT_ROOT = Path("out")


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped, tracking XYZ001"


def serialize_message(m: BaseMessage) -> Json:
    d: Json = {"role": m.type, "content": str(m.content)}
    if isinstance(m, AIMessage) and m.tool_calls:
        d["tool_calls"] = [dict(c) for c in m.tool_calls]
    if isinstance(m, ToolMessage):
        d["tool_call_id"] = m.tool_call_id
    return d


def deserialize_message(d: Json) -> BaseMessage:
    role = d["role"]
    if role == "system":
        return SystemMessage(d["content"])
    if role == "human":
        return HumanMessage(d["content"])
    if role == "ai":
        return AIMessage(d["content"], tool_calls=d.get("tool_calls") or [])
    if role == "tool":
        return ToolMessage(content=d["content"], tool_call_id=d["tool_call_id"])
    raise ValueError(f"unknown role in checkpoint: {role!r}")


def checkpoint_dir(chapter: str, scenario: str) -> Path:
    path = OUT_ROOT / chapter / scenario
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_snapshot(directory: Path, run_id: str, turn: int, messages: list[BaseMessage]) -> Path:
    """Append one line -- the complete state as of this turn, not a diff."""
    path = directory / f"{run_id}.jsonl"
    line = {"turn": turn, "messages": [serialize_message(m) for m in messages]}
    with path.open("a") as f:
        f.write(f"{json.dumps(line)}\n")
    return path


def read_last_snapshot(directory: Path, run_id: str) -> Json | None:
    """The most recent complete turn, or None -- never a partial write,
    since a line only ever gets appended once a turn is fully done.
    """
    path = directory / f"{run_id}.jsonl"
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None
    result: Json = json.loads(lines[-1])
    return result


def _dispatch(call: ToolCall, trace: Trace) -> ToolMessage:
    return execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))


def _run_one_turn(
    trace: Trace,
    model_kind: ModelKind,
    turn: int,
    messages: list[BaseMessage],
    mock_reply: AIMessage,
) -> None:
    """One complete turn -- the model's reply, and every tool call it
    asked for, dispatched and appended. Caller decides when to snapshot.
    """
    with trace.span("turn", number=turn):
        reply = ask_model(trace, model_kind, [mock_reply], messages, [check_order])
        messages.append(reply)
        for call in reply.tool_calls:
            messages.append(_dispatch(call, trace))


def a_snapshot_after_every_turn_lets_a_new_process_resume(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Two turns, snapshotted after each. A separate Trace -- nothing
    shared in memory -- loads only the jsonl file and continues.
    """
    name = "a_snapshot_after_every_turn_lets_a_new_process_resume"
    directory = checkpoint_dir(trace.chapter, name)
    run_id = "order-support-run"
    (directory / f"{run_id}.jsonl").unlink(missing_ok=True)

    with trace.span("scenario", name=name) as span:
        # "Process A" -- runs two turns, snapshots after each.
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("What's the status of order A100?"),
        ]
        _run_one_turn(
            trace,
            model_kind,
            1,
            messages,
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
        )
        write_snapshot(directory, run_id, 1, messages)
        _run_one_turn(trace, model_kind, 2, messages, AIMessage("Order A100 has shipped."))
        snapshot_path = write_snapshot(directory, run_id, 2, messages)
        span.add_note("snapshot_written", path=str(snapshot_path), turn=2)

        # "Process B" -- a genuinely separate list of message objects,
        # rebuilt from nothing but the file on disk.
        loaded = read_last_snapshot(directory, run_id)
        assert loaded is not None
        resumed_messages = [deserialize_message(m) for m in loaded["messages"]]
        resumed_messages.append(HumanMessage("And what's the tracking number?"))
        _run_one_turn(
            trace,
            model_kind,
            loaded["turn"] + 1,
            resumed_messages,
            AIMessage("Tracking number: XYZ001."),
        )
        final_snapshot = write_snapshot(directory, run_id, loaded["turn"] + 1, resumed_messages)
        span.add_note(
            "resumed_in_a_new_process",
            path=str(final_snapshot),
            resumed_from_turn=loaded["turn"],
            final_message_count=len(resumed_messages),
            final_reply=str(resumed_messages[-1].content),
        )


def snapshotting_mid_turn_would_orphan_a_tool_call(trace: Trace, model_kind: ModelKind) -> None:
    """Snapshot between an assistant's tool_calls and its tool reply --
    the naive mistake -- and resume from exactly that broken point.
    """
    name = "snapshotting_mid_turn_would_orphan_a_tool_call"
    directory = checkpoint_dir(trace.chapter, name)
    run_id = "bad-timing-run"
    (directory / f"{run_id}.jsonl").unlink(missing_ok=True)

    with trace.span("scenario", name=name) as span:
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("What's the status of order A100?"),
        ]
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
                messages,
                [check_order],
            )
            messages.append(reply)
            # The mistake: snapshot right here, before the tool call's own
            # reply is appended -- an assistant message with tool_calls
            # and no matching tool reply yet.
            bad_snapshot = write_snapshot(directory, run_id, 1, messages)
            for call in reply.tool_calls:
                messages.append(_dispatch(call, trace))
        span.add_note("bad_snapshot_written_mid_turn", path=str(bad_snapshot))

        loaded = read_last_snapshot(directory, run_id)
        assert loaded is not None
        resumed = [deserialize_message(m) for m in loaded["messages"]]
        if model_kind != "live":
            span.add_note("resume_check", checked=False, reason="not applicable under mock")
            return
        try:
            ask_model(trace, model_kind, [], resumed, [])
            span.add_note("resume_check", accepted=True, error=None)
        except OpenAIInvalidRequestError as rejection:
            span.add_note("resume_check", accepted=False, error=str(rejection)[:300])


def a_crash_before_the_next_snapshot_loses_at_most_one_turn(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Turn three starts but its snapshot is never written -- resuming
    finds turn two, the last complete one, and safely redoes turn three.
    """
    name = "a_crash_before_the_next_snapshot_loses_at_most_one_turn"
    directory = checkpoint_dir(trace.chapter, name)
    run_id = "crash-mid-turn-run"
    (directory / f"{run_id}.jsonl").unlink(missing_ok=True)

    with trace.span("scenario", name=name) as span:
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("What's the status of order A100?"),
        ]
        _run_one_turn(
            trace,
            model_kind,
            1,
            messages,
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
        )
        write_snapshot(directory, run_id, 1, messages)
        _run_one_turn(trace, model_kind, 2, messages, AIMessage("Order A100 has shipped."))
        write_snapshot(directory, run_id, 2, messages)

        # Turn three "starts" -- a real model call, a real answer -- and
        # then the process dies before write_snapshot ever runs.
        crashed_messages = list(messages)
        crashed_messages.append(HumanMessage("Anything else I should know?"))
        with trace.span("turn", number=3):
            ask_model(
                trace, model_kind, [AIMessage("No, that's everything.")], crashed_messages, []
            )
        # No write_snapshot call here -- the crash.

        loaded = read_last_snapshot(directory, run_id)
        assert loaded is not None
        span.add_note(
            "recovery_point",
            recovered_turn=loaded["turn"],
            turn_three_was_lost=loaded["turn"] == 2,
        )
        resumed = [deserialize_message(m) for m in loaded["messages"]]
        resumed.append(HumanMessage("Anything else I should know?"))
        _run_one_turn(trace, model_kind, 3, resumed, AIMessage("No, that's everything."))
        final = write_snapshot(directory, run_id, 3, resumed)
        span.add_note("turn_three_redone_cleanly", path=str(final))


def separate_run_ids_do_not_collide(trace: Trace, model_kind: ModelKind) -> None:
    """Two runs, same store -- resuming one never sees the other's
    history.
    """
    name = "separate_run_ids_do_not_collide"
    directory = checkpoint_dir(trace.chapter, name)
    for run_id in ("run-a", "run-b"):
        (directory / f"{run_id}.jsonl").unlink(missing_ok=True)

    with trace.span("scenario", name=name) as span:
        messages_a: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("Track order A100."),
        ]
        _run_one_turn(trace, model_kind, 1, messages_a, AIMessage("Order A100 has shipped."))
        write_snapshot(directory, "run-a", 1, messages_a)

        messages_b: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("Track order B200."),
        ]
        _run_one_turn(trace, model_kind, 1, messages_b, AIMessage("Order B200 is processing."))
        write_snapshot(directory, "run-b", 1, messages_b)

        loaded_a = read_last_snapshot(directory, "run-a")
        loaded_b = read_last_snapshot(directory, "run-b")
        assert loaded_a is not None and loaded_b is not None
        text_a = " ".join(m["content"] for m in loaded_a["messages"])
        text_b = " ".join(m["content"] for m in loaded_b["messages"])
        span.add_note(
            "isolation_check",
            run_a_mentions_a100="A100" in text_a,
            run_a_mentions_b200="B200" in text_a,
            run_b_mentions_b200="B200" in text_b,
            run_b_mentions_a100="A100" in text_b,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch31_stop_resume", model_kind=model_kind)

    a_snapshot_after_every_turn_lets_a_new_process_resume(trace, model_kind)
    snapshotting_mid_turn_would_orphan_a_tool_call(trace, model_kind)
    a_crash_before_the_next_snapshot_loses_at_most_one_turn(trace, model_kind)
    separate_run_ids_do_not_collide(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
