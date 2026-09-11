# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Two updates to one field: append, or replace.

`ch32` used `add_messages` without asking what the alternative even looks
like. Every field this ladder has ever declared without `Annotated[...]`
has been replacing silently, the whole time -- `messages` was never the
default, it was the one field anyone bothered to choose a reducer for.

Five scenarios:

    no_reducer_means_replace
        a plain field, no reducer -- the second node's write is the only
        one that survives. This has been the default since ch32; nothing
        until now made it the subject
    add_messages_is_the_one_field_that_appends
        the same graph, two fields side by side across two turns --
        messages (add_messages) accumulates, a plain field replaces.
        The contrast ch32 never drew explicitly
    a_custom_reducer_is_just_a_function
        Annotated[int, operator.add] -- a reducer is any (old, new) ->
        merged callable. add_messages is the one LangGraph ships for the
        message case, not the only shape a reducer can take
    the_wrong_reducer_choice_silently_loses_data
        a field that should have accumulated, declared with no reducer by
        mistake -- the practical cost of scenario one, framed as the bug
        it would actually be
    two_nodes_writing_the_same_field_in_one_super_step_both_survive
        found live: a reducer resolves every simultaneous write within
        one super-step in a single merge, not just two sequential ones --
        fan out to two siblings, both write messages and a custom summed
        field, and both survive. The order they land in is not this
        chapter's claim -- that is ch37_parallel's subject, currently
        skipped
"""

from support.trace import ModelKind, Trace

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ReplaceOnlyState(TypedDict):
    note: str


def no_reducer_means_replace(trace: Trace, _model_kind: ModelKind) -> None:
    """No Annotated[...] on `note` -- LangGraph's default merge is
    overwrite. The second node's return is the only one that survives.
    """
    name = "no_reducer_means_replace"
    with trace.span("scenario", name=name) as span:

        def first(state: ReplaceOnlyState) -> ReplaceOnlyState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="first") as node:
                write: ReplaceOnlyState = {"note": "from first"}
                node.add_note("wrote", **write)
                return write

        def second(state: ReplaceOnlyState) -> ReplaceOnlyState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="second") as node:
                write: ReplaceOnlyState = {"note": "from second"}
                node.add_note("wrote", **write)
                return write

        graph = StateGraph(ReplaceOnlyState)
        graph.add_node("first", first)
        graph.add_node("second", second)
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        compiled = graph.compile()

        result = compiled.invoke({"note": "initial"})
        span.add_note("second_write_is_all_that_survives", final_note=result["note"])


class MixedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    note: str


def add_messages_is_the_one_field_that_appends(trace: Trace, _model_kind: ModelKind) -> None:
    """Two fields, two turns, one graph -- messages accumulates because it
    is annotated with add_messages; note replaces because it is not. The
    contrast ch32 never drew side by side.
    """
    name = "add_messages_is_the_one_field_that_appends"
    with trace.span("scenario", name=name) as span:

        def turn_one(state: MixedState) -> MixedState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="turn_one") as node:
                reply = AIMessage("turn one reply")
                node.add_note("wrote", message=str(reply.content), note="from turn one")
                return {"messages": [reply], "note": "from turn one"}

        def turn_two(state: MixedState) -> MixedState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="turn_two") as node:
                reply = AIMessage("turn two reply")
                node.add_note("wrote", message=str(reply.content), note="from turn two")
                return {"messages": [reply], "note": "from turn two"}

        graph = StateGraph(MixedState)
        graph.add_node("turn_one", turn_one)
        graph.add_node("turn_two", turn_two)
        graph.add_edge(START, "turn_one")
        graph.add_edge("turn_one", "turn_two")
        graph.add_edge("turn_two", END)
        compiled = graph.compile()

        seed: MixedState = {"messages": [HumanMessage("go")], "note": "initial"}
        span.add_note("seed_passed_to_invoke", message="go", note="initial")
        result = compiled.invoke(seed)
        span.add_note(
            "one_field_accumulates_the_other_replaces",
            message_count=len(result["messages"]),
            messages=[str(m.content) for m in result["messages"]],
            final_note=result["note"],
        )


class CountingState(TypedDict):
    total: Annotated[int, operator.add]


def a_custom_reducer_is_just_a_function(trace: Trace, _model_kind: ModelKind) -> None:
    """operator.add attached to Annotated[int, ...] -- a reducer is any
    (old, new) -> merged callable, and this one was never written by
    LangGraph or by us for this purpose; it is the standard library's
    own addition operator, pressed into service unchanged.
    """
    name = "a_custom_reducer_is_just_a_function"
    with trace.span("scenario", name=name) as span:

        def add_five(state: CountingState) -> CountingState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="add_five") as node:
                node.add_note("wrote", total=5)
                return {"total": 5}

        def add_three(state: CountingState) -> CountingState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="add_three") as node:
                node.add_note("wrote", total=3)
                return {"total": 3}

        graph = StateGraph(CountingState)
        graph.add_node("add_five", add_five)
        graph.add_node("add_three", add_three)
        graph.add_edge(START, "add_five")
        graph.add_edge("add_five", "add_three")
        graph.add_edge("add_three", END)
        compiled = graph.compile()

        result = compiled.invoke({"total": 0})
        span.add_note("operator_add_accumulates_across_turns", final_total=result["total"])


class OrderTrackingState(TypedDict):
    order_ids_checked: list[str]


def the_wrong_reducer_choice_silently_loses_data(trace: Trace, _model_kind: ModelKind) -> None:
    """order_ids_checked should have accumulated one entry per turn -- it
    was declared with no reducer, by mistake. The second turn's write
    replaces the first's outright. No error, no warning: scenario one's
    finding, as the bug it would actually be in a real tool-call log.
    """
    name = "the_wrong_reducer_choice_silently_loses_data"
    with trace.span("scenario", name=name) as span:

        def check_first_order(state: OrderTrackingState) -> OrderTrackingState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="check_first_order") as node:
                node.add_note("wrote", order_ids_checked=["A100"])
                return {"order_ids_checked": ["A100"]}

        def check_second_order(state: OrderTrackingState) -> OrderTrackingState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="check_second_order") as node:
                node.add_note("wrote", order_ids_checked=["B200"])
                return {"order_ids_checked": ["B200"]}

        graph = StateGraph(OrderTrackingState)
        graph.add_node("check_first_order", check_first_order)
        graph.add_node("check_second_order", check_second_order)
        graph.add_edge(START, "check_first_order")
        graph.add_edge("check_first_order", "check_second_order")
        graph.add_edge("check_second_order", END)
        compiled = graph.compile()

        result = compiled.invoke({"order_ids_checked": []})
        span.add_note(
            "a100_never_recorded",
            final_order_ids_checked=result["order_ids_checked"],
            a100_survived="A100" in result["order_ids_checked"],
        )


class FanState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    total: Annotated[int, operator.add]


def two_nodes_writing_the_same_field_in_one_super_step_both_survive(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """Found live: fan out from one node to two siblings, both writing to
    messages and to total in the same super-step. Both writes to each
    field survive, merged in one reducer application rather than two
    sequential ones -- a reducer resolves however many simultaneous
    contributions a super-step produces, not just an old value and a new
    one. Which sibling's write lands first in the merged list is not
    claimed here -- that is ch37_parallel's subject, currently skipped.
    """
    name = "two_nodes_writing_the_same_field_in_one_super_step_both_survive"
    with trace.span("scenario", name=name) as span:

        def fan_out(state: FanState) -> dict[str, object]:  # pyright: ignore[reportUnusedParameter]
            # A pass-through node purely to give the fan-out somewhere to
            # start from -- it updates nothing, so its return is partial
            # by design, unlike every other node here returning a full
            # (if single-key) State.
            return {}

        def left(state: FanState) -> FanState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="left") as node:
                node.add_note("wrote", message="left branch", total=1)
                return {"messages": [AIMessage("left branch")], "total": 1}

        def right(state: FanState) -> FanState:  # pyright: ignore[reportUnusedParameter]
            with trace.span("node", name="right") as node:
                node.add_note("wrote", message="right branch", total=10)
                return {"messages": [AIMessage("right branch")], "total": 10}

        graph = StateGraph(FanState)
        graph.add_node("fan_out", fan_out)
        graph.add_node("left", left)
        graph.add_node("right", right)
        graph.add_edge(START, "fan_out")
        graph.add_edge("fan_out", "left")
        graph.add_edge("fan_out", "right")
        graph.add_edge("left", END)
        graph.add_edge("right", END)
        compiled = graph.compile()

        span.add_note("seed_passed_to_invoke", message="go", total=0)
        result = compiled.invoke({"messages": [HumanMessage("go")], "total": 0})
        contents = [str(m.content) for m in result["messages"]]
        span.add_note(
            "both_simultaneous_writes_survived",
            messages=contents,
            both_branches_present="left branch" in contents and "right branch" in contents,
            total=result["total"],
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch38_reducers", model_kind=model_kind)

    no_reducer_means_replace(trace, model_kind)
    add_messages_is_the_one_field_that_appends(trace, model_kind)
    a_custom_reducer_is_just_a_function(trace, model_kind)
    the_wrong_reducer_choice_silently_loses_data(trace, model_kind)
    two_nodes_writing_the_same_field_in_one_super_step_both_survive(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
