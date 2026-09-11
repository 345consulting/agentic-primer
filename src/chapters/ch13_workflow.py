# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The same job with nothing deciding -- is the loop worth it?

No model decides the *sequence* here. Not mocked, not live, not behind a
tool -- the rule for what runs next was fixed when this file was written,
not read from a reply at run time. That is the whole of what "workflow"
means, and it is easy to overstate into "no model anywhere," which is wrong:
a step's own body can still be one.

**`step` plays a `tool`'s role, with no model asking for it.** It opens a
span, records what went in and what came out -- the same shape
`execute_tool` has kept since `ch04_tool_failures`. What is missing is the
thing that decided to call it: no `tool_calls`, no name the model chose.
Here, the code that runs `step(...)` decided, and it decided in advance.

**`kind` says what actually ran, so a reader does not have to guess from an
`args`/`result` pair that looks the same either way.** A row that reads
`given: item=milk · returned: 2` could be a dict lookup, an API call, or a
whole nested agent -- indistinguishable at that level of detail, the same
ambiguity `ch12_supervisor`'s `tool` rows had until you expanded one. Four of
these five steps are `kind="function"`: plain code, nothing underneath to
find. The fifth is `kind="agent"`, and for that one there is something
underneath, visible without opening anything first.

**Five scenarios:**

    single_step             one step, nothing to sequence
    dual_in_sequence         two steps, always both, fixed order
    three_steps_reserves     check, then reserve -- enough is in stock
    three_steps_backorders   check, then backorder -- the same check, skipped
    a_step_can_be_an_agent   fixed code calls it; its own next move is not

The middle two are the same three-step pipeline, `check_stock` first in
both, and then one of two steps runs -- never both. `route_after_check` is
the `if`, and it is deterministic given its inputs: a count compared to a
request, nothing read from a reply, because there is no reply here to read.
Which step ran is not narrated in a note -- it is visible as which span
exists at all, the same absence `ch10_routing` used for the branch a model
never got asked to take.

**The fifth is where the earlier claim gets proven rather than asserted.**
`check_price_via_agent` is called the same way every other step is -- fixed
code, unconditionally, no `tool_calls` involved in the decision to run it at
all. What happens once it is running is not fixed: it holds its own
`system_prompt`, its own tool, its own `run_turns`, and its own model call
decides whether to use that tool. `Trace.span`'s stack nests its turns inside
the step's span for free, the same mechanism `ch12_supervisor` used one
level further out. `kind="supervisor"` is the third legal value and is not
built here -- a step whose body is a supervisor looks, from a workflow's own
trace, identical to one whose body is a plain agent; the distinguishing
shape lives one level down, inside the step, and `ch12` already showed it.

**`turns` counts steps; `messages` stays zero even here.** The agent-kind
step holds a real message list -- system prompt, question, reply, tool
result -- and it is real cost, real tokens, a real model call. None of it
is counted at the workflow's level, for the same reason `ch12_supervisor`'s
outer count stayed its own: the expense is real and stays invisible unless
the step's span gets opened. This chapter's zero was honest for four
scenarios by having nothing to count; for the fifth it is honest by the same
rule `ch12` used -- count only what is yours.

**`model_kind` matters for exactly one scenario now, and only that one.**
The four function-kind steps have no parameter to read it through --
checkable directly on their own signatures, not merely claimed -- so a mock
and a live run of this chapter are identical everywhere except inside
`a_step_can_be_an_agent`, where the difference is the same one every other
chapter has always had.

**Why this keeps none of `support/scenario.py`.** `Scenario` carries
`mock_model_replies` and assumes every scenario wants a model; four of these
five do not. This is the first chapter whose scenarios differ in shape
rather than in data -- one step, two, three with a branch, one with a whole
agent inside it -- so each stays its own function rather than one runner
parameterized over a table, the same reason `ch01`-`ch03` keep their own
copies of what they teach.

`route_after_check` is `add_conditional_edges`'s function, same as
`ch10_routing`'s were -- the graph does not know or care that this one reads
a stock count instead of a model's reply.
"""

from support.agent import ask_model, execute_tool, run_turns
from support.trace import ENDED, ModelKind, Trace

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

# Reused from ch04_tool_failures and ch09_who_retries: the same shop, so a
# reader is not asked to learn a new domain to read a new mechanism.
STOCK_ON_HAND = {"bread": 2, "butter": 1, "chips": 6, "milk": 2}

AGENT_STEP_SYSTEM_PROMPT = "You answer briefly and plainly. Use the tool you are given."

AGENT_STEP_TURN_CAP = 4

AGENT_STEP_MOCK_REPLIES = [
    AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "a1"}]),
    AIMessage("Milk is 1.20 per bottle."),
]


def check_stock(item: str) -> int:
    """How many of an item are on hand. A read, same as ever."""
    return STOCK_ON_HAND[item]


def get_price(item: str) -> str:
    """The shelf price of an item, before any discount."""
    del item  # every item costs the same here; kept so `step`'s args note names it
    return "1.20"


def apply_discount(price: str) -> str:
    """A fixed 10% off. Always runs after `get_price`, never on its own."""
    return f"{float(price) * 0.9:.2f}"


def reserve_stock(item: str, quantity: int) -> str:
    """Take the quantity out of stock. Only reachable when enough is on hand."""
    return f"reserved {quantity} x {item}"


def create_backorder(item: str, quantity: int) -> str:
    """Record the shortfall. Only reachable when not enough is on hand."""
    return f"backordered {quantity} x {item}"


def route_after_check(available: int, requested: int) -> str:
    """The workflow's own `if`. Deterministic given its inputs -- a count
    compared to a request, nothing read from a reply that does not exist.
    """
    return "reserve" if available >= requested else "backorder"


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item, in pounds. The agent-kind step's only tool."""
    return f"1.20 for {item}"


def check_price_via_agent(item: str, model_kind: ModelKind, trace: Trace) -> str:
    """The step whose body is a whole agent. Fixed code below decided to run
    this, unconditionally -- but once it is running, its own next move is
    the model's, exactly as it has been since `ch02_tool_call`.
    """
    messages: list[BaseMessage] = [
        SystemMessage(AGENT_STEP_SYSTEM_PROMPT),
        HumanMessage(f"How much is {item}?"),
    ]

    def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(trace, model_kind, AGENT_STEP_MOCK_REPLIES[turn - 1 :], so_far, [price_of])

    def execute(call: ToolCall) -> ToolMessage:
        return execute_tool(call, trace, lambda _name, args, _span: price_of.invoke(args))

    run_turns(messages, trace, AGENT_STEP_TURN_CAP, ask, execute)
    return str(messages[-1].content)


def step(trace: Trace, name: str, kind: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
    """One step, recorded the way a tool call has been since `ch04`. `kind`
    says what actually ran, so a reader does not have to guess from
    `args`/`result` alone -- and does not have to open the span to find out.
    """
    with trace.span("step", name=name, kind=kind) as span:
        span.add_note("args", **kwargs)
        result = fn(**kwargs)
        span.add_note("result", value=result)
    return result


def run_single_step(trace: Trace) -> tuple[int, str]:
    with trace.span("scenario", name="single_step") as span:
        available = step(trace, "check_stock", "function", check_stock, item="chips")
        ended = f"{available} in stock"
        span.add_note(ENDED, reason=ended)
    return 1, ended


def run_dual_in_sequence(trace: Trace) -> tuple[int, str]:
    with trace.span("scenario", name="dual_in_sequence") as span:
        price = step(trace, "get_price", "function", get_price, item="milk")
        discounted = step(trace, "apply_discount", "function", apply_discount, price=price)
        ended = f"discounted to {discounted}"
        span.add_note(ENDED, reason=ended)
    return 2, ended


def run_three_steps_reserves(trace: Trace) -> tuple[int, str]:
    with trace.span("scenario", name="three_steps_reserves") as span:
        available = step(trace, "check_stock", "function", check_stock, item="milk")
        destination = route_after_check(available, requested=2)
        assert destination == "reserve"
        result = step(trace, "reserve_stock", "function", reserve_stock, item="milk", quantity=2)
        span.add_note(ENDED, reason=result)
    return 2, result


def run_three_steps_backorders(trace: Trace) -> tuple[int, str]:
    with trace.span("scenario", name="three_steps_backorders") as span:
        available = step(trace, "check_stock", "function", check_stock, item="milk")
        destination = route_after_check(available, requested=5)
        assert destination == "backorder"
        result = step(
            trace, "create_backorder", "function", create_backorder, item="milk", quantity=5
        )
        span.add_note(ENDED, reason=result)
    return 2, result


def run_a_step_can_be_an_agent(trace: Trace, model_kind: ModelKind) -> tuple[int, str]:
    """Not routed through `step()`: `trace` and `model_kind` are plumbing
    this agent-kind step needs to run, not business arguments a reader of
    the trace should see recorded as if they were `item` or `quantity`.
    """
    with trace.span("scenario", name="a_step_can_be_an_agent") as span:
        with trace.span("step", name="check_price_via_agent", kind="agent") as step_span:
            step_span.add_note("args", item="milk")
            answer = check_price_via_agent("milk", model_kind, trace)
            step_span.add_note("result", value=answer)
        span.add_note(ENDED, reason=answer)
    return 1, answer


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch13_workflow", model_kind=model_kind)

    results = [
        run_single_step(trace),
        run_dual_in_sequence(trace),
        run_three_steps_reserves(trace),
        run_three_steps_backorders(trace),
        run_a_step_can_be_an_agent(trace, model_kind),
    ]
    steps = sum(taken for taken, _ in results)
    endings = {ended for _, ended in results}

    trace.close(turns=steps, messages=0, ended="; ".join(sorted(endings)))
    return trace
