# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""everything since graph, composed into one real scenario.

Not one more primitive tested in isolation -- whether the eight already
proven separately (`ch32`-`ch39`) actually compose into one working thing.
The domain: an order with several line items, checked, priced, and placed
-- or paused for approval if it is large enough -- streamed live the whole
way through, `ch17`'s own mechanism.

One workflow, three scenarios -- not nine touches forced onto one graph,
three different paths an order actually takes through it:

    a_small_order_streams_straight_through_no_pause
        items fan out via Send and a reducer (ch37, ch38), pricing runs
        as a same-schema subgraph (ch36), a real callback traces the
        whole thing (ch39) -- and no interrupt, because nothing about
        this order needs approval
    a_large_order_pauses_for_approval_then_resumes_from_checkpoint
        the same graph, an order over threshold -- interrupt() pauses it
        (ch35) and the SSE response actually ends there; resume is a
        second, separate request reading the paused state back from a
        real checkpoint (ch34), not a variable still sitting in memory
    the_real_callback_trace_agrees_with_our_own
        ch39's own open question, asked one more time on the most
        demanding graph this ladder has built: a real LangChain callback
        attached to the same run Trace already recorded -- and where the
        two actually diverge

**The graph, once, shared by all three:**

    dispatch -[Send, one per line item]-> check_item
        a turn: ask_model decides to call check_stock, dispatched
        through ch18's own hook mechanism (dispatch_with_hooks) rather
        than execute_tool -- an observe hook logs every dispatch, so a
        hook firing is visible in the trace beside the model call and
        the tool call, not just implied by the ladder having built one.
        checked/line_totals accumulate by ch38's reducer as branches join
    check_item -> price_order
        a same-schema subgraph (ch36's first scenario) -- sum_subtotal,
        then apply_discount, sharing the parent's exact OrderState
    price_order -[conditional]-> approval_gate | END
        over threshold, the order needs a human; under it, it is done
    approval_gate
        interrupt()s with the total -- Command(resume=True/False) is
        what a second request answers with

**Why the SSE frames are coarser than in ch17.** LangGraph's own
`.invoke()`/`.ainvoke()` is the one call driving this graph -- there is
no hand-rolled turn loop here to interleave `send()` calls into the way
`ch17`'s own loop could, for most of it. The frames this chapter emits
are phase boundaries (`checking_items`, `pricing_started`,
`awaiting_approval`, `order_placed`) rather than a frame per node, which
is `ch17`'s own finding about curation applied one level coarser: a
client watching this order does not need to know which line item
resolved first, only what phase the order is in. `price_order` is the
one node given its own access to `send` -- async, so it can emit
`pricing_started` before doing its work rather than narrating the
aftermath once `ainvoke()` already returned, the same live-ness
`checking_items` gets and `order_placed`/`run_ended` structurally
cannot. Turn-level and tool-level detail -- the model call per item, the
hook that dispatched it, the real callback firing -- all still land in
`Trace`, at full granularity, the same audit `ch17` already proved the
SSE stream is not.

**Concurrent branches recorded with an explicit `parent=`, not the
implicit stack.** `ch37`'s own finding: `Send`-dispatched branches run on
a real thread pool even from a synchronous graph, and a shared,
thread-local span stack has no way to know which branch's frame belongs
under which round unless told. `items_span` is threaded into `check_item`
through a closure for exactly that reason, the identical fix built there.

**`recursion_limit` is sized to what this graph actually needs
(`ch33`).** dispatch (1) + the fan-out's join (1) + `price_order`'s two
internal steps (2, counted against the parent's own budget, `ch33`'s
nested-budget finding) + `approval_gate` (1) is five; `RECURSION_LIMIT`
below is set to eight, room for LangGraph's own bookkeeping steps without
being an arbitrary round number.

**Found live, not assumed: the same prompt does not always call the
tool.** `check_item`'s prompt is identical every time -- "Check stock for
Nx sku." -- and the mock model always calls `check_stock`, because it is
scripted to. A live model occasionally answers without calling it at
all, confirmed by replaying the exact same prompt in isolation and
getting the tool call that time. This is the primer's own oldest
gotcha (`README`'s "the mock and the live model do not prove the same
things") surfacing here as a real decision a live model makes, not a
framework guarantee -- `check_item` reports a decline as its own line
item rather than crashing the run or silently pricing it at zero.

**A decline is not free -- `declined` says so, because `total` cannot.**
The first fix (report, don't crash) still let a fully-declined order
finish with `total=0.0` and place itself, indistinguishable from a
genuinely free order -- caught by actually seeing `pricing`/`order_placed`
frames both carrying `total: 0` in a live run and asking what that even
meant. `OrderState.declined` is the field that answers it: a non-empty
list routes to `order_incomplete` instead of `order_placed` regardless
of what `total` says, because a number computed from zero successful
checks is not a price, and the workflow does not get to place an order
it never actually priced.
"""

from support.agent import ask_model
from support.hooks import HookVerdict, dispatch_with_hooks
from support.trace import ModelKind, Span, Trace

import asyncio
import json
import operator
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict, override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.tool import ToolCall, tool_call
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send

type Json = dict[str, Any]
type SendFn = Callable[[Json], Awaitable[None]]
type ReceiveFn = Callable[[], Awaitable[Json]]

APPROVAL_THRESHOLD = 100.0

PRICE_PER_UNIT = 10.0

RECURSION_LIMIT = 8


@tool
def check_stock(sku: str, quantity: int) -> str:
    """Check stock and price for a quantity of a sku."""
    return f"{sku}: {quantity} in stock at ${PRICE_PER_UNIT:.2f} each"


def note_the_dispatch(call: ToolCall) -> HookVerdict:
    """observe, pre -- ch18's own hook, reused rather than rebuilt. This
    is the one place this chapter's trace shows a hook firing, distinct
    from the real LangChain callback firing beside it.
    """
    return HookVerdict(note=f"dispatching {call['name']} for {call['args'].get('sku')}")


def _dispatch_check_stock(sku: str, quantity: int) -> str:
    """`dispatch_with_hooks` wants a plain callable, not a `BaseTool` --
    `ch18`'s own dispatch table shape. `check_stock` still needs to be a
    real `@tool` so the model can see its schema; this is the bridge.
    """
    return str(check_stock.invoke({"sku": sku, "quantity": quantity}))


class LineItem(TypedDict):
    sku: str
    quantity: int


class OrderState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    items: list[LineItem]
    checked: Annotated[list[str], operator.add]
    line_totals: Annotated[list[float], operator.add]
    # Which skus the model declined to check -- distinct from `checked`,
    # because a $0 total from every item being declined must never look
    # like a $0 total from a genuinely free order. Empty is the only
    # value that means "every item was actually priced."
    declined: Annotated[list[str], operator.add]
    subtotal: float
    total: float
    approved: bool | None


class BranchState(TypedDict):
    sku: str
    quantity: int


class PricingState(TypedDict):
    """Only what pricing needs to read and write -- no reducer field
    here ever crosses back out to double against the parent's own copy.
    """

    line_totals: list[float]
    subtotal: float
    total: float


class TracingCallback(BaseCallbackHandler):
    """A real LangChain callback, attached the same way `ch39` attached
    one -- through `config`, not through anything this ladder built.
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    @override
    def on_chat_model_start(self, *_a: object, **_k: object) -> None:
        self.events.append("model_start")

    @override
    def on_llm_end(self, *_a: object, **_k: object) -> None:
        self.events.append("model_end")


def _thread(order_id: str) -> RunnableConfig:
    return {
        "configurable": {"thread_id": order_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def _build_order_graph(
    trace: Trace,
    model_kind: ModelKind,
    callback: TracingCallback,
    send: SendFn,
) -> tuple[CompiledStateGraph[OrderState, None, OrderState, OrderState], Callable[[Span], None]]:
    items_span: Span | None = None

    def set_items_span(span: Span) -> None:
        nonlocal items_span
        items_span = span

    def dispatch(
        state: OrderState,  # pyright: ignore[reportUnusedParameter]
    ) -> dict[str, object]:
        return {}

    def fan_out_items(state: OrderState) -> list[Send]:
        return [Send("check_item", item) for item in state["items"]]

    def check_item(state: BranchState) -> dict[str, object]:
        with (
            trace.span("branch", parent=items_span, sku=state["sku"]),
            trace.span("turn", number=1) as turn_span,
        ):
            script = tool_call(
                name="check_stock",
                args={"sku": state["sku"], "quantity": state["quantity"]},
                id=f"c-{state['sku']}",
            )
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage("", tool_calls=[script])],
                [HumanMessage(f"Check stock for {state['quantity']}x {state['sku']}.")],
                [check_stock],
                config={"callbacks": [callback]},
            )
            # Found live, not assumed: the identical prompt calls the tool
            # every time under mock (scripted) but occasionally does not
            # live -- a real model's own decision, not a bug in the
            # prompt or the binding. Declining is reported rather than
            # crashing the run; it does not get to silently mean $0 owed.
            if not reply.tool_calls:
                turn_span.add_note("model_declined_the_tool_call", sku=state["sku"])
                return {
                    "checked": [f"{state['sku']}: declined by the model"],
                    "line_totals": [],
                    "declined": [state["sku"]],
                }
            # The tool dispatch stays inside this same "turn" span -- a
            # turn is one model call plus whatever it triggers, the
            # convention `support/agent.py`'s `run_turns` has followed
            # since `ch03`, not a sibling recorded after the fact.
            message, _vetoed = dispatch_with_hooks(
                reply.tool_calls[0],
                trace,
                [note_the_dispatch],
                [],
                {"check_stock": _dispatch_check_stock},
            )
        return {
            "checked": [str(message.content)],
            "line_totals": [PRICE_PER_UNIT * state["quantity"]],
        }

    def sum_subtotal(state: PricingState) -> dict[str, object]:
        return {"subtotal": sum(state["line_totals"])}

    def apply_discount(state: PricingState) -> dict[str, object]:
        discounted = state["subtotal"] * 0.9 if state["subtotal"] > 150 else state["subtotal"]
        return {"total": discounted}

    pricing = StateGraph(PricingState)
    pricing.add_node("sum_subtotal", sum_subtotal)
    pricing.add_node("apply_discount", apply_discount)
    pricing.add_edge(START, "sum_subtotal")
    pricing.add_edge("sum_subtotal", "apply_discount")
    pricing.add_edge("apply_discount", END)
    compiled_pricing = pricing.compile()

    async def price_order(state: OrderState) -> dict[str, object]:
        # A wrapper, not `add_node("price_order", compiled_pricing)` --
        # `ch36`'s own contrast between the two, needed here for a
        # reason that chapter never hit. A narrower schema alone was not
        # enough: `line_totals` still has to cross in by shared key name
        # for the child to read it, and add_node's automatic mapping
        # merges the child's *entire* final state back through the
        # parent's reducer regardless -- doubling it even though nothing
        # inside the subgraph ever touched that field. A wrapper controls
        # exactly what crosses each way: `line_totals` goes in as a
        # plain read, and only `subtotal`/`total` -- the two fields
        # pricing actually computed -- come back out.
        #
        # Async, unlike check_item -- not because pricing runs
        # concurrently with anything (it never does; check_item's own
        # fan-out has already joined by the time this node runs), but
        # because emitting a frame *before* the work starts needs an
        # `await`, and a sync node has no way to reach one. Making only
        # this node async, in a graph invoked with `ainvoke`, still runs
        # check_item's sync branches on the same thread pool `ch37`
        # already proved safe -- LangGraph schedules sync and async
        # nodes differently, not sync nodes differently depending on
        # whether some other node in the graph happens to be async.
        with trace.span("pricing") as pricing_span:
            # Recorded on pricing_span itself, as its own first entry --
            # the same fix "checking_items" already needed. scenario_span
            # was already open and had entries when this fires; attaching
            # there would land the frame after this span's own entry.
            await _emit(send, pricing_span, "pricing_started")
            priced = compiled_pricing.invoke(
                {"line_totals": state["line_totals"], "subtotal": 0.0, "total": 0.0}
            )
            pricing_span.add_note(
                "subgraph_result", subtotal=priced["subtotal"], total=priced["total"]
            )
        return {"subtotal": priced["subtotal"], "total": priced["total"]}

    def route_after_pricing(state: OrderState) -> str:
        return "approval_gate" if state["total"] > APPROVAL_THRESHOLD else END

    def approval_gate(state: OrderState) -> dict[str, object]:
        from langgraph.types import interrupt

        decision = interrupt({"total": state["total"]})
        return {"approved": bool(decision)}

    graph = StateGraph(OrderState)
    graph.add_node("dispatch", dispatch)
    graph.add_node("check_item", check_item)
    graph.add_node("price_order", price_order)
    graph.add_node("approval_gate", approval_gate)
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges("dispatch", fan_out_items, ["check_item"])
    graph.add_edge("check_item", "price_order")
    graph.add_conditional_edges("price_order", route_after_pricing, ["approval_gate", END])
    graph.add_edge("approval_gate", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    return compiled, set_items_span


def _count_descendants(span: Span, name: str) -> int:
    """Spans named `name` under this one span, at any depth -- `Trace`'s
    own `find_spans` walks the whole run, and this scenario's own subtree
    is what has to be compared against a callback scoped to one call.
    """
    return sum(1 for c in span.children if c.name == name) + sum(
        _count_descendants(c, name) for c in span.children
    )


def _frame(event: str, **fields: Any) -> bytes:
    return f"data: {json.dumps({'event': event, **fields})}\n\n".encode()


async def _emit(send: SendFn, record_on: Span, event: str, **fields: Any) -> None:
    """Observe only, same as `ch17`'s own `_emit` -- recorded on whichever
    span this phase boundary actually belongs to.
    """
    record_on.add_note("sse_frame_sent", event=event, **fields)
    await send({"type": "http.response.body", "body": _frame(event, **fields), "more_body": True})


def _collector() -> tuple[SendFn, list[Json]]:
    frames: list[Json] = []

    async def send(message: Json) -> None:
        if message["type"] == "http.response.body" and message["body"]:
            for line in message["body"].decode().strip().split("\n\n"):
                frames.append(json.loads(line.removeprefix("data: ")))

    return send, frames


async def _start_order_over_sse(
    send: SendFn,
    *,
    trace: Trace,
    scenario_span: Span,
    compiled: CompiledStateGraph[OrderState, None, OrderState, OrderState],
    set_items_span: Callable[[Span], None],
    thread: RunnableConfig,
    items: list[LineItem],
) -> dict[str, object]:
    """One request: submit the order. Ends either at `order_placed` (no
    approval needed) or at `awaiting_approval` -- the response genuinely
    ends there, the same way `ch35`'s own pause returns from `.invoke()`
    instead of blocking inside it.
    """
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    await _emit(send, scenario_span, "run_started")

    # Named "order", not "items" -- this span wraps the entire ainvoke()
    # call, not just item-checking. It has to: there is no boundary
    # visible from outside the graph between check_item's fan-in and
    # price_order starting, both happen inside this one call. pricing
    # nesting under it is honest, not a mistake -- pricing genuinely
    # runs while this call is still in flight.
    with trace.span("order", count=len(items)) as items_span:
        set_items_span(items_span)
        # Recorded on items_span itself, not scenario_span -- this frame
        # belongs to the item-checking phase, and attaching it to a span
        # opened earlier and still open would land it after this span's
        # own entry in that span's list, rendering the frame as if it
        # were sent once item-checking had already finished.
        await _emit(send, items_span, "checking_items", count=len(items))
        result = await compiled.ainvoke(
            {
                "messages": [HumanMessage("Place this order.")],
                "items": items,
                "checked": [],
                "line_totals": [],
                "declined": [],
                "subtotal": 0.0,
                "total": 0.0,
                "approved": None,
            },
            config=thread,
        )
    if declined := result.get("declined"):
        # A total of zero here could mean a genuinely free order or every
        # item being declined -- `declined` is the only thing that tells
        # them apart, and the order does not get to place itself either
        # way just because nothing raised.
        await _emit(send, scenario_span, "order_incomplete", declined_skus=declined)
        await _emit(send, scenario_span, "run_ended", approved=False)
    elif "__interrupt__" in result:
        await _emit(send, scenario_span, "awaiting_approval", total=result["total"])
    else:
        await _emit(send, scenario_span, "order_placed", total=result["total"])
        await _emit(send, scenario_span, "run_ended", approved=True)
    await send({"type": "http.response.body", "body": b"", "more_body": False})
    return result


async def _resume_order_over_sse(
    send: SendFn,
    *,
    scenario_span: Span,
    compiled: CompiledStateGraph[OrderState, None, OrderState, OrderState],
    thread: RunnableConfig,
    decision: bool,
) -> dict[str, object]:
    """A second, separate request -- a fresh call into the same compiled
    graph, keyed only by `thread_id`. Nothing from `_start_order_over_sse`
    is still in scope here; what it reads back comes entirely from the
    checkpointer, `ch34`'s own claim, tested on this graph instead of the
    minimal one it was first shown on.
    """
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    await _emit(send, scenario_span, "resuming", decision=decision)
    result = await compiled.ainvoke(Command(resume=decision), config=thread)
    await _emit(send, scenario_span, "order_placed" if decision else "order_rejected")
    await _emit(send, scenario_span, "run_ended", approved=decision)
    await send({"type": "http.response.body", "body": b"", "more_body": False})
    return result


def a_small_order_streams_straight_through_no_pause(trace: Trace, model_kind: ModelKind) -> None:
    """One item, well under threshold -- fan-out, reducer, subgraph
    pricing, a real callback, all streamed live, and no interrupt.
    """
    name = "a_small_order_streams_straight_through_no_pause"
    send, frames = _collector()
    callback = TracingCallback()
    with trace.span("scenario", name=name) as span:
        compiled, set_items_span = _build_order_graph(trace, model_kind, callback, send)
        result = asyncio.run(
            _start_order_over_sse(
                send,
                trace=trace,
                scenario_span=span,
                compiled=compiled,
                set_items_span=set_items_span,
                thread=_thread("order-small"),
                items=[{"sku": "widget", "quantity": 3}],
            )
        )
        span.add_note(
            "the_small_order_never_paused",
            events=[f["event"] for f in frames],
            total=result["total"],
            approved_without_a_gate="__interrupt__" not in result,
            callback_events=callback.events,
        )


def a_large_order_pauses_for_approval_then_resumes_from_checkpoint(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Over threshold -- the first request's SSE response genuinely ends
    at `awaiting_approval`; the second request resumes a graph object
    that never ran the first half, reading the pause back from the
    checkpointer instead of from anything still in memory.
    """
    name = "a_large_order_pauses_for_approval_then_resumes_from_checkpoint"
    send, frames = _collector()
    callback = TracingCallback()
    with trace.span("scenario", name=name) as span:
        compiled, set_items_span = _build_order_graph(trace, model_kind, callback, send)
        thread = _thread("order-large")
        first = asyncio.run(
            _start_order_over_sse(
                send,
                trace=trace,
                scenario_span=span,
                compiled=compiled,
                set_items_span=set_items_span,
                thread=thread,
                items=[{"sku": "widget", "quantity": 20}],
            )
        )
        paused_here = "__interrupt__" in first

        second = asyncio.run(
            _resume_order_over_sse(
                send,
                scenario_span=span,
                compiled=compiled,
                thread=thread,
                decision=True,
            )
        )
        span.add_note(
            "the_pause_and_resume_were_two_separate_requests",
            events=[f["event"] for f in frames],
            paused_before_resuming=paused_here,
            approved_after_resume=second.get("approved"),
            total=first.get("total"),
        )


def the_real_callback_trace_agrees_with_our_own(trace: Trace, model_kind: ModelKind) -> None:
    """ch39's own open question, asked again on this graph: does a real
    callback's own record of the run agree with what Trace recorded --
    and where, honestly, does it not.
    """
    name = "the_real_callback_trace_agrees_with_our_own"
    send, _frames = _collector()
    callback = TracingCallback()
    with trace.span("scenario", name=name) as span:
        compiled, set_items_span = _build_order_graph(trace, model_kind, callback, send)
        asyncio.run(
            _start_order_over_sse(
                send,
                trace=trace,
                scenario_span=span,
                compiled=compiled,
                set_items_span=set_items_span,
                thread=_thread("order-callback-check"),
                items=[{"sku": "widget", "quantity": 1}, {"sku": "gadget", "quantity": 2}],
            )
        )
        model_spans_in_trace = _count_descendants(span, "model")
        model_starts_in_callback = callback.events.count("model_start")
        tool_spans_in_trace = _count_descendants(span, "tool")

        span.add_note(
            "the_callback_sees_every_model_call_but_no_tool_call",
            model_calls_the_callback_saw=model_starts_in_callback,
            model_calls_the_trace_saw=model_spans_in_trace,
            model_counts_agree=model_starts_in_callback == model_spans_in_trace,
            tool_calls_the_trace_saw=tool_spans_in_trace,
            # dispatch_with_hooks calls a plain dict lookup, never a real
            # LangChain Tool.invoke() -- the callback has no path into it
            # at all, which is the honest divergence this scenario exists
            # to show, not a bug in either mechanism.
            tool_calls_the_callback_could_have_seen=0,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch40_capstone", model_kind=model_kind)

    a_small_order_streams_straight_through_no_pause(trace, model_kind)
    a_large_order_pauses_for_approval_then_resumes_from_checkpoint(trace, model_kind)
    the_real_callback_trace_agrees_with_our_own(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
