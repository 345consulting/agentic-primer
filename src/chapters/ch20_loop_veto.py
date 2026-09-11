# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Stopped because forbidden, which is not stopped because done.

`ch18_hooks` and `ch19_guards` never ran inside a loop -- every scenario was
one isolated call, so nothing has ever asked what happens *next* when a veto
fires. `run_turns` has exactly two endings, `no_tool_calls` and
`turns_exhausted`, and neither one is "the model asked for something and was
refused." This chapter is where that third ending has to exist, because it
is the first one with an actual multi-turn loop for it to happen inside.

**`soft_continue` needs nothing new, and that is itself the finding.**
`execute`'s contract in `support/agent.py` is `Callable[[ToolCall],
ToolMessage]` -- `run_turns` appends whatever comes back and asks again,
never inspecting it. Wire `dispatch_with_hooks` in as `execute` and a veto is
indistinguishable, to the loop, from any other tool error: the model sees a
refusal in its next turn's context and gets to try something else, exactly
`ch09_who_retries`' third sentence -- "correct them and call it again" --
except the correction here is invented by the model, not offered by us.

**`hard_stop` needs a loop of its own, because `run_turns` cannot express
it.** Nothing in its shape lets a call stop the whole run before asking
again -- `execute`'s return value is opaque to it. `dispatch_with_hooks` now
returns whether a veto produced the message, exactly so a loop can act on
that without guessing from the text of a sentence written for the model, not
for it. `run_hard_stop` is `run_turns` with one added line: if the call was
vetoed, return `"vetoed"` instead of asking again.

**The budget from `ch19` returns, and `soft_continue` tries the exact
adversarial thing `ch09`'s live model did.** Refused at a second order of
eight -- 8 + 8 = 16, over a budget of 12 -- the mock script does not give
up. It tries four instead. `8 + 4 = 12`, exactly at the limit, and it
succeeds. The model routed around the refusal, same as `ch09`'s split order
did -- but this time the total never went over twelve, because the guard
recomputed the real total on every attempt rather than remembering that this
tool had already said no once. Routing around a refusal is not the same as
routing around the constraint; only the first one is available here.

**The live run tested a sharper case than the mock scripted, twice.** The
live model put both `place_order` calls in one reply -- `ch06_two_tools`'
shape, not the two separate turns the mock was written as -- so `hard_stop`'s
veto fired inside turn one's own call loop, never reaching a second `ask` at
all. The mechanism does not care: it checks each call as it is processed,
turn boundary or not, and this is the run that actually proved that rather
than assuming it. And `soft_continue`'s live model did not route around the
refusal the way `ch09`'s did with a schema bound -- it stopped and asked:
*"Would you like me to order 4 more instead (to reach 12)?"* No silent
retry, no smaller call. Explained, then handed the decision back.
"""

from support.agent import ask_model, run_turns
from support.hooks import AfterHook, BeforeHook, HookVerdict, dispatch_with_hooks
from support.trace import ENDED, ModelKind, Trace

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

QUESTION = "Order 8 bottles of milk, then order 8 more."

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

TURN_CAP = 4

BUDGET = 12


@tool
def place_order(item: GroceryItem, quantity: int) -> str:
    """Order a quantity of an item."""
    raise AssertionError(
        f"place_order ['{quantity}' x '{item}'] is declared for its schema; "
        "submit_order below runs, once the guard allows it"
    )


def submit_order(item: str, quantity: int) -> str:
    """What actually dispatches. The guard decides whether this runs at all;
    it never receives `state`, the same discipline `ch11`/`ch19` used.
    """
    return f"ordered {quantity} x {item}"


TOOLS = {"place_order": submit_order}

DECLARED_TOOLS = [place_order]


@dataclass
class State:
    """What travels beside the calls in one scenario. Written only by
    `make_order_recorder`'s hook, read only by `make_budget_guard`'s.
    """

    total_ordered: int = 0


def make_budget_guard(state: State, budget: int) -> BeforeHook:
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
    def record_the_order(call: ToolCall, _result: Any) -> HookVerdict:
        if call["name"] == "place_order":
            state.total_ordered += call["args"]["quantity"]
        return HookVerdict(note=f"total_ordered now {state.total_ordered}")

    return record_the_order


HARD_STOP_MOCK_REPLIES = [
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 8}, "id": "c1"}],
    ),
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 8}, "id": "c2"}],
    ),
]

SOFT_CONTINUE_MOCK_REPLIES = [
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 8}, "id": "c1"}],
    ),
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 8}, "id": "c2"}],
    ),
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 4}, "id": "c3"}],
    ),
    AIMessage(
        "Ordered 8, then 4 more after the second 8 was refused -- 12 in total, at the limit."
    ),
]


def run_hard_stop(model_kind: ModelKind, trace: Trace, turn_cap: int) -> tuple[int, int, str]:
    """`run_turns`, with the one thing it cannot express: a call that ends
    the run instead of letting the loop ask again.
    """
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(QUESTION)]
    state = State()
    before = [make_budget_guard(state, BUDGET)]
    after = [make_order_recorder(state)]

    turns = 0
    while turns < turn_cap:
        turns += 1
        with trace.span("turn", number=turns):
            reply = ask_model(
                trace, model_kind, HARD_STOP_MOCK_REPLIES[turns - 1 :], messages, DECLARED_TOOLS
            )
            messages.append(reply)
            if not reply.tool_calls:
                return turns, len(messages), "no_tool_calls"
            for call in reply.tool_calls:
                result, vetoed = dispatch_with_hooks(call, trace, before, after, TOOLS)
                messages.append(result)
                if vetoed:
                    return turns, len(messages), "vetoed"
    return turns, len(messages), "turns_exhausted"


def run_soft_continue(model_kind: ModelKind, trace: Trace, turn_cap: int) -> tuple[int, int, str]:
    """Exactly `run_turns`. A veto is just what `dispatch_with_hooks`
    returned this time -- the loop was never told to treat it differently.
    """
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(QUESTION)]
    state = State()
    before = [make_budget_guard(state, BUDGET)]
    after = [make_order_recorder(state)]

    def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(
            trace, model_kind, SOFT_CONTINUE_MOCK_REPLIES[turn - 1 :], so_far, DECLARED_TOOLS
        )

    def execute(call: ToolCall) -> ToolMessage:
        result, _vetoed = dispatch_with_hooks(call, trace, before, after, TOOLS)
        return result

    return run_turns(messages, trace, turn_cap, ask, execute)


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    trace = Trace(chapter="ch20_loop_veto", model_kind=model_kind)

    with trace.span("scenario", name="hard_stop_ends_the_run_immediately") as span:
        hard_turns, hard_messages, hard_ended = run_hard_stop(model_kind, trace, turn_cap)
        span.add_note(ENDED, reason=hard_ended)

    with trace.span("scenario", name="soft_continue_lets_the_model_try_again") as span:
        soft_turns, soft_messages, soft_ended = run_soft_continue(model_kind, trace, turn_cap)
        span.add_note(ENDED, reason=soft_ended)

    trace.close(
        turns=hard_turns + soft_turns,
        messages=hard_messages + soft_messages,
        ended=f"{hard_ended}; {soft_ended}",
    )
    return trace
