# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""fan-out, Send, join, and the order things merge in.

`ch38_reducers`' own fifth scenario proved a reducer resolves every
simultaneous write within one super-step in a single merge -- and named
its own limit: "which sibling's write lands first is not claimed here."
That claim is this chapter's subject. `Send` is also the one thing
nothing in this ladder has built by hand or seen a static edge do:
dynamic fan-out, a number of parallel branches decided at runtime, not
declared at graph-build time.

Five scenarios:

    send_fans_out_dynamically_based_on_runtime_data
        a router returns one Send per item in a list whose length is not
        known until the graph runs -- the same graph, invoked twice with
        different-length lists, dispatches a different number of
        branches each time
    branches_run_concurrently_not_sequentially
        found live: three branches with different delays finish in
        roughly their slowest branch's time, not the sum of all three --
        real concurrency, not a for loop wearing Send's syntax
    merge_order_follows_dispatch_order_not_completion_order
        found live: the fastest branch finishes first and the merged
        result still reflects dispatch order, not completion order --
        the direct closure of ch38's own deferred question
    fan_out_over_real_tool_calls_then_reduces_to_one_answer
        the practical shape: fan out to real check_order dispatches, one
        per order, then one real model call folds the joined results
        into a single answer
    a_single_judgment_over_the_whole_batch_retries_the_round_until_three_strikes
        one judge verdict over all fanned-out results together, not one
        per branch -- a failing verdict re-dispatches the whole round;
        three failed rounds escalate rather than loop forever, matching
        the "three strikes, then escalate" policy this primer has
        followed since ch09

**Why the judge sees the batch, not the branches.** A judge per branch
would be five scenarios of `ch21_judge` wearing `Send`'s syntax. The
actual new question is whether a verdict spanning everything fan-out
just produced can gate whether the *whole round* happened at all --
which needs the join to have already happened, which needs the reducer,
which is why this chapter comes after `ch38` and not before it.
"""

from support.agent import ask_model, execute_tool
from support.trace import ModelKind, Span, Trace

import operator
import time
from typing import Annotated, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


def send_fans_out_dynamically_based_on_runtime_data(trace: Trace, _model_kind: ModelKind) -> None:
    """The same graph, invoked with two different-length lists -- the
    number of branches that actually run matches each list's length,
    decided only once the graph is running, not when it was built.
    """

    class State(TypedDict):
        items: list[str]
        branches_ran: Annotated[list[str], operator.add]

    class BranchState(TypedDict):
        item: str

    def dispatch(
        state: State,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, object]:
        return {}

    def fan_out(state: State) -> list[Send]:
        with trace.span("dispatch", branch_count=len(state["items"])) as dspan:
            dspan.add_note("send_objects_created", items=list(state["items"]))
        return [Send("check_one", {"item": item}) for item in state["items"]]

    round_span: Span | None

    def check_one(state: BranchState) -> dict[str, object]:
        with trace.span("branch", parent=round_span, item=state["item"]):
            return {"branches_ran": [state["item"]]}

    graph = StateGraph(State)
    graph.add_node("dispatch", dispatch)
    graph.add_node("check_one", check_one)
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", fan_out, ["check_one"])
    graph.add_edge("check_one", END)
    compiled = graph.compile()

    name = "send_fans_out_dynamically_based_on_runtime_data"
    with trace.span("scenario", name=name) as span:
        # `check_one` runs on a worker thread that never had a parent on its
        # own stack, so the round it belongs to is handed to it explicitly
        # through this closure variable rather than left to infer.
        with trace.span("round", label="two items") as round_span:
            two_items = compiled.invoke({"items": ["A100", "A200"], "branches_ran": []})
        with trace.span("round", label="four items") as round_span:
            four_items = compiled.invoke(
                {"items": ["B100", "B200", "B300", "B400"], "branches_ran": []}
            )

        span.add_note(
            "branch_count_matched_each_lists_length",
            two_item_run={
                "input_count": 2,
                "branches_ran": sorted(two_items["branches_ran"]),
            },
            four_item_run={
                "input_count": 4,
                "branches_ran": sorted(four_items["branches_ran"]),
            },
        )


def branches_run_concurrently_not_sequentially(trace: Trace, _model_kind: ModelKind) -> None:
    """Three branches, three different delays -- total wall-clock time
    checked against the slowest branch, not the sum of all three.
    """

    class State(TypedDict):
        delays: dict[str, float]
        finished: Annotated[list[str], operator.add]

    class BranchState(TypedDict):
        item: str
        delay: float

    def dispatch(
        state: State,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, object]:
        return {}

    def fan_out(state: State) -> list[Send]:
        return [
            Send("wait_one", {"item": item, "delay": delay})
            for item, delay in state["delays"].items()
        ]

    scenario_span: Span | None

    def wait_one(state: BranchState) -> dict[str, object]:
        with trace.span("branch", parent=scenario_span, item=state["item"], delay=state["delay"]):
            time.sleep(state["delay"])
            return {"finished": [state["item"]]}

    graph = StateGraph(State)
    graph.add_node("dispatch", dispatch)
    graph.add_node("wait_one", wait_one)
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", fan_out, ["wait_one"])
    graph.add_edge("wait_one", END)
    compiled = graph.compile()

    name = "branches_run_concurrently_not_sequentially"
    # `wait_one` runs on a worker thread with nothing of its own already
    # open, so this scenario span is handed to it explicitly, the same
    # closure pattern as scenario 1's round span.
    with trace.span("scenario", name=name) as scenario_span:
        span = scenario_span
        delays = {"slow": 0.3, "fast": 0.05, "medium": 0.15}
        start = time.monotonic()
        compiled.invoke({"delays": delays, "finished": []})
        elapsed = time.monotonic() - start

        span.add_note(
            "elapsed_matches_the_slowest_branch_not_the_sum",
            elapsed_seconds=round(elapsed, 3),
            slowest_branch_seconds=max(delays.values()),
            sum_of_all_branches_seconds=round(sum(delays.values()), 3),
            faster_than_the_sum=elapsed < sum(delays.values()),
        )


def merge_order_follows_dispatch_order_not_completion_order(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """The fastest branch finishes first -- the merged reducer result
    still reflects dispatch order, the direct closure of ch38's own
    deferred question about which sibling's write lands first.
    """

    class State(TypedDict):
        order: list[str]
        delays: dict[str, float]
        results: Annotated[list[str], operator.add]

    class BranchState(TypedDict):
        item: str
        delay: float

    def dispatch(
        state: State,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, object]:
        return {}

    def fan_out(state: State) -> list[Send]:
        return [
            Send("check_one", {"item": item, "delay": state["delays"][item]})
            for item in state["order"]
        ]

    scenario_span: Span | None

    def check_one(state: BranchState) -> dict[str, object]:
        with trace.span("branch", parent=scenario_span, item=state["item"], delay=state["delay"]):
            time.sleep(state["delay"])
            return {"results": [state["item"]]}

    graph = StateGraph(State)
    graph.add_node("dispatch", dispatch)
    graph.add_node("check_one", check_one)
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", fan_out, ["check_one"])
    graph.add_edge("check_one", END)
    compiled = graph.compile()

    name = "merge_order_follows_dispatch_order_not_completion_order"
    with trace.span("scenario", name=name) as scenario_span:
        span = scenario_span
        # "fast" is dispatched second but finishes well before "slow",
        # dispatched first.
        dispatch_order = ["slow", "fast", "medium"]
        delays = {"slow": 0.2, "fast": 0.02, "medium": 0.1}

        result = compiled.invoke({"order": dispatch_order, "delays": delays, "results": []})

        span.add_note(
            "merged_order_matched_dispatch_not_completion",
            dispatch_order=dispatch_order,
            completion_order=sorted(delays, key=lambda k: delays[k]),
            merged_result_order=result["results"],
            matches_dispatch_order=result["results"] == dispatch_order,
        )


def fan_out_over_real_tool_calls_then_reduces_to_one_answer(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Fan out to a real check_order dispatch per order id, then one
    real model call folds the joined results into a single answer.
    """

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        order_ids: list[str]
        checked: Annotated[list[str], operator.add]

    class BranchState(TypedDict):
        order_id: str

    def dispatch(
        state: State,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, object]:
        return {}

    def fan_out(state: State) -> list[Send]:
        return [Send("check_one", {"order_id": oid}) for oid in state["order_ids"]]

    scenario_span: Span | None

    def check_one(state: BranchState) -> dict[str, object]:
        call: ToolCall = {
            "name": "check_order",
            "args": {"order_id": state["order_id"]},
            "id": "c1",
        }
        with trace.span("branch", parent=scenario_span, order_id=state["order_id"]):
            message = execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))
        return {"checked": [str(message.content)]}

    def summarize(state: State) -> dict[str, object]:
        joined = "; ".join(sorted(state["checked"]))
        with trace.span("turn", parent=scenario_span, number=1):
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage(f"All checked: {joined}.")],
                [*state["messages"], HumanMessage(f"Results so far: {joined}")],
                [],
            )
        return {"messages": [reply]}

    graph = StateGraph(State)
    graph.add_node("dispatch", dispatch)
    graph.add_node("check_one", check_one)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", fan_out, ["check_one"])
    graph.add_edge("check_one", "summarize")
    graph.add_edge("summarize", END)
    compiled = graph.compile()

    name = "fan_out_over_real_tool_calls_then_reduces_to_one_answer"
    with trace.span("scenario", name=name) as scenario_span:
        span = scenario_span
        order_ids = ["A100", "A200", "A300"]
        result = compiled.invoke(
            {
                "messages": [HumanMessage("Check the status of orders A100, A200, A300.")],
                "order_ids": order_ids,
                "checked": [],
            }
        )

        span.add_note(
            "each_order_checked_then_reduced_to_one_answer",
            orders_checked=sorted(result["checked"]),
            final_answer=str(result["messages"][-1].content),
        )


def a_single_judgment_over_the_whole_batch_retries_the_round_until_three_strikes(
    trace: Trace, model_kind: ModelKind
) -> None:
    """One judge verdict over every fanned-out result together -- a
    failing verdict re-dispatches the whole round, not one branch. Three
    failed rounds escalate; a round that actually improves stops early
    because it passed, not because it ran out of strikes.

    Found live, not assumed: the "recovering" round only recovers under
    mock. `quality_by_attempt` is a scripted reply the mock model returns
    verbatim; a live model is asked only "Summarize order A100." with no
    signal for what "good" means, so it never produces a reply ending in
    "good" and every round -- recovering and non-recovering alike --
    escalates at three strikes. The mock's cooperation was the thing
    proving early-stop worked, not a property of the judge itself.
    """

    class State(TypedDict):
        items: list[str]
        attempt: int
        verdict: str
        results: Annotated[list[str], operator.add]

    class BranchState(TypedDict):
        item: str
        attempt: int

    def dispatch(
        state: State,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, object]:
        return {}

    def fan_out(state: State) -> list[Send]:
        return [
            Send("check_one", {"item": item, "attempt": state["attempt"]})
            for item in state["items"]
        ]

    scenario_span: Span | None

    def _run_round(quality_by_attempt: dict[int, str]) -> dict[str, object]:
        def check_one(state: BranchState) -> dict[str, object]:
            quality = quality_by_attempt.get(state["attempt"], "bad")
            with trace.span("turn", parent=scenario_span, number=state["attempt"] + 1):
                reply = ask_model(
                    trace,
                    model_kind,
                    [AIMessage(f"order {state['item']}: {quality}")],
                    [HumanMessage(f"Summarize order {state['item']}.")],
                    [],
                )
            return {"results": [str(reply.content)]}

        def judge(state: State) -> dict[str, object]:
            recent = state["results"][-len(state["items"]) :]
            all_good = all(r.endswith("good") for r in recent)
            verdict = "pass" if all_good else "retry"
            with trace.span("judge", round=state["attempt"] + 1) as jspan:
                jspan.add_note("batch_judged", batch=recent, verdict=verdict)
            return {"attempt": state["attempt"] + 1, "verdict": verdict}

        def route_after_judge(state: State) -> str:
            if state["verdict"] == "pass" or state["attempt"] >= 3:
                return END
            return "dispatch"

        graph = StateGraph(State)
        graph.add_node("dispatch", dispatch)
        graph.add_node("check_one", check_one)
        graph.add_node("judge", judge)
        graph.add_edge(START, "dispatch")
        graph.add_conditional_edges("dispatch", fan_out, ["check_one"])
        graph.add_edge("check_one", "judge")
        graph.add_conditional_edges("judge", route_after_judge, ["dispatch", END])
        compiled = graph.compile()

        return compiled.invoke(
            {"items": ["A100", "A200", "A300"], "attempt": 0, "verdict": "", "results": []}
        )

    name = "a_single_judgment_over_the_whole_batch_retries_the_round_until_three_strikes"
    with trace.span("scenario", name=name) as scenario_span:
        span = scenario_span
        recovers = _run_round({2: "good"})
        recovers_results = cast(list[str], recovers["results"])
        span.add_note(
            "quality_improved_so_it_stopped_early",
            rounds_run=recovers["attempt"],
            final_verdict=recovers["verdict"],
            total_results=len(recovers_results),
        )

        never_improves = _run_round({})
        never_improves_results = cast(list[str], never_improves["results"])
        span.add_note(
            "quality_never_improved_so_it_escalated_at_three_strikes",
            rounds_run=never_improves["attempt"],
            final_verdict=never_improves["verdict"],
            total_results=len(never_improves_results),
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch37_parallel", model_kind=model_kind)

    send_fans_out_dynamically_based_on_runtime_data(trace, model_kind)
    branches_run_concurrently_not_sequentially(trace, model_kind)
    merge_order_follows_dispatch_order_not_completion_order(trace, model_kind)
    fan_out_over_real_tool_calls_then_reduces_to_one_answer(trace, model_kind)
    a_single_judgment_over_the_whole_batch_retries_the_round_until_three_strikes(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
