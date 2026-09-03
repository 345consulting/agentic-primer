# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""A hook that can say no, before dispatch -- with something real to refuse.

`ch18_hooks`' catalog check was deliberately thin: single-call, nothing
carried between one dispatch and the next. This chapter's veto is the one
`ch18` left for it -- `ch11_state`'s `total_ordered`, read across calls
rather than reset for each one, which is the exact gap the live run of
`ch09_who_retries` found and could not close: asked for twenty bottles
against a limit of twelve, the model did not correct its argument, it called
`place_order` twice, twelve and eight, and satisfied its instructions by
routing around a bound that only ever looked at one call at a time.

**`refuse_if_over_budget` looks at what already happened this run, not just
what is being asked now.** The guard is a closure over a `State`, the same
shape `ch11` built and the same rule: only `record_the_order`, an `after`
hook, ever writes `total_ordered`, and only once a call has actually
dispatched. The guard only reads it.

**Two scenarios, and the second is the fix.**

    an_order_within_budget          one call, comfortably under the limit
    a_split_order_exceeds_the_budget the exact ch09 shape -- eight, then
                                     eight again -- and the second call is
                                     refused, because eight was already fine
                                     on its own and the budget is not about
                                     any one call

The second call in that scenario is not malformed, not outside a catalog,
not anything a schema could have caught -- `place_order(item="milk",
quantity=8)` is a perfectly ordinary request. What makes it refusable is
`state.total_ordered` already being 8 when it arrives, and nothing before
this chapter could read that.
"""

from support.hooks import AfterHook, BeforeHook, HookVerdict, dispatch_with_hooks
from support.trace import ENDED, ModelKind, Trace

from dataclasses import dataclass
from typing import Any

from langchain_core.messages.tool import ToolCall, tool_call

BUDGET = 12


def place_order(item: str, quantity: int) -> str:
    """Order a quantity of an item. The guard decides whether this runs at
    all; once it does, the tool has no opinion about how much was ordered
    before -- it never receives `state`, the same discipline `ch11` used.
    """
    return f"ordered {quantity} x {item}"


TOOLS = {"place_order": place_order}


@dataclass
class State:
    """What travels beside the calls in one scenario. Never sent anywhere,
    read by the guard, written by nothing but `record_the_order`.
    """

    total_ordered: int = 0


def make_budget_guard(state: State, budget: int) -> BeforeHook:
    """The veto `ch18` left simple. Reads what this run has already ordered,
    not just the call in front of it -- the thing a tool's own argument
    check can never do, because a tool never sees the calls beside it.
    """

    def refuse_if_over_budget(call: ToolCall) -> HookVerdict:
        if call["name"] != "place_order":
            return HookVerdict()
        quantity = call["args"].get("quantity", 0)
        projected = state.total_ordered + quantity
        if projected > budget:
            return HookVerdict(
                veto=(
                    f"{state.total_ordered} already ordered this run; "
                    f"{quantity} more would be {projected}, over the budget of {budget}"
                )
            )
        return HookVerdict()

    return refuse_if_over_budget


def make_order_recorder(state: State) -> AfterHook:
    """The only place `total_ordered` is written -- after dispatch, and only
    for the tool whose calls the guard is counting.
    """

    def record_the_order(call: ToolCall, _result: Any) -> HookVerdict:
        if call["name"] == "place_order":
            state.total_ordered += call["args"]["quantity"]
        return HookVerdict(note=f"total_ordered now {state.total_ordered}")

    return record_the_order


def run(model_kind: ModelKind = "mock") -> Trace:
    """`model_kind` is accepted and never read -- nothing here calls a model,
    the same honest no-op `ch13_workflow` and `ch18_hooks` used.
    """
    trace = Trace(chapter="ch19_guards", model_kind=model_kind)
    endings: list[str] = []

    scenarios: list[tuple[str, list[ToolCall]]] = [
        (
            "an_order_within_budget",
            [tool_call(name="place_order", args={"item": "milk", "quantity": 8}, id="c1")],
        ),
        (
            "a_split_order_exceeds_the_budget",
            [
                tool_call(name="place_order", args={"item": "milk", "quantity": 8}, id="c2"),
                tool_call(name="place_order", args={"item": "milk", "quantity": 8}, id="c3"),
            ],
        ),
    ]

    for name, calls in scenarios:
        with trace.span("scenario", name=name) as span:
            state = State()
            before = [make_budget_guard(state, BUDGET)]
            after = [make_order_recorder(state)]
            results = [dispatch_with_hooks(call, trace, before, after, TOOLS)[0] for call in calls]
            ended = f"total_ordered={state.total_ordered}; " + "; ".join(
                f"{r.status}: {r.content}" for r in results
            )
            span.add_note(ENDED, reason=ended)
        endings.append(ended)

    trace.close(turns=len(scenarios), messages=0, ended="; ".join(endings))
    return trace
