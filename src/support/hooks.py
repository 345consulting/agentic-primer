# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The hook mechanism, once, after the chapter that teaches it.

`ch18_hooks` built this by hand first -- `HookVerdict`, the two hook types,
and `dispatch_with_hooks` -- because a chapter that imports its own subject
teaches nothing. `ch19_guards` is the first to reuse it, the graduation
trigger this primer has used every other time: `ask_model`, `execute_tool`
and `run_turns` moved here the same way once `ch04` needed them, and `ch18`
now composes this file instead of keeping the copy it was built with, the
same as every chapter after `ch03` does for those three.

**What travels here is the shell; what varies stays in the chapter.** Running
a list of hooks, applying the first veto or replacement, recording what
happened -- that is bookkeeping, identical in every chapter that will ever
use it. Which hooks are attached, and what condition each one checks, is the
lesson, every time: `ch18`'s catalog check, `ch19`'s cross-call budget,
whatever comes after. Nothing about that belongs here.

**A hook needing state beyond the call reaches it by closure, not by a wider
signature.** `BeforeHook`/`AfterHook` take only what every hook has --
`ch18`'s single-call checks need nothing else, and forcing a state parameter
onto them for `ch19`'s sake would make the common case carry the rare one's
plumbing. `ch11_state`'s own pattern already relies on closures (`ask`,
`execute` capturing `scenario` and `trace`); a guard capturing a `state`
object it was built with is the same idiom, not a new one.
"""

from support.trace import Trace

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall, tool_call


@dataclass
class HookVerdict:
    """What one hook decided. Independent fields -- a hook may leave a note
    regardless of whether it also modifies or vetoes, and the fields being
    separate is what makes `observe` provably powerless: it can fill `note`
    and nothing else, and nothing else is what changes the run.
    """

    note: str | None = None
    veto: str | None = None
    replacement: dict[str, Any] | None = None


type BeforeHook = Callable[[ToolCall], HookVerdict]
type AfterHook = Callable[[ToolCall, Any], HookVerdict]


def dispatch_with_hooks(
    call: ToolCall,
    trace: Trace,
    before_hooks: Sequence[BeforeHook],
    after_hooks: Sequence[AfterHook],
    tools: dict[str, Callable[..., Any]],
) -> tuple[ToolMessage, bool]:
    """One call, run through a named point's hooks before and after dispatch.

    `pre` and `post` do not offer the same veto. Before dispatch, nothing has
    happened, and a veto refuses the action. After, the action already
    occurred -- a veto there stops the consequence instead, and the
    `dispatched` note recorded before any `after_hooks` run is what lets the
    trace prove that difference rather than merely claim it.

    Returns the message and whether a veto produced it. `ch20_loop_veto`
    is why the second value exists: a `ToolMessage` with `status="error"`
    looks the same whether a tool raised or a hook refused, and a loop
    deciding whether to stop outright needs to tell those apart without
    guessing from the text of a message meant for the model, not for it.
    """
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        for before_hook in before_hooks:
            verdict = before_hook(call)
            span.add_note(
                "hook",
                when="pre",
                by=before_hook.__name__,
                note=verdict.note,
                veto=verdict.veto,
                replacement=verdict.replacement,
            )
            if verdict.veto is not None:
                return (
                    ToolMessage(
                        content=f"['{call['name']}'] blocked before it ran: {verdict.veto}",
                        tool_call_id=call["id"],
                        status="error",
                    ),
                    True,
                )
            if verdict.replacement is not None:
                call = tool_call(name=call["name"], args=verdict.replacement, id=call["id"])

        result: Any = tools[call["name"]](**call["args"])
        span.add_note("dispatched", value=result)

        for after_hook in after_hooks:
            verdict = after_hook(call, result)
            span.add_note(
                "hook",
                when="post",
                by=after_hook.__name__,
                note=verdict.note,
                veto=verdict.veto,
                replacement=verdict.replacement,
            )
            if verdict.veto is not None:
                return (
                    ToolMessage(
                        content=(
                            f"['{call['name']}'] ran, but the result was withheld: {verdict.veto}"
                        ),
                        tool_call_id=call["id"],
                        status="error",
                    ),
                    True,
                )
            if verdict.replacement is not None:
                result = verdict.replacement["value"]

        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"]), False
