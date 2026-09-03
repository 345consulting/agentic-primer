# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""One loop calls another: a tool whose body is an agent.

Nothing new gets built. `consult_inventory_expert` is a tool, declared the
same way every tool since `ch04_tool_failures` has been -- but its
implementation is not a lookup, a request, or a subprocess. It is a whole
second `run_turns`, with its own system prompt, its own tool, and its own
turn cap, called from inside the first one's `execute_tool`.

**Nesting is free.** `Trace.span` already pushes onto a stack -- that is how
a `tool` span ends up holding `args`/`result` notes today. Open a `tool` span
for `consult_inventory_expert` and, while it is still open, call the inner
loop: every `turn`/`model`/`tool` span the inner loop opens nests inside it,
because the stack does not know or care that the code calling `trace.span`
this time is a whole other agent rather than a dict lookup. No change to
`support/trace.py` was needed to make this true.

**The cost hides in the transcript.** The outer agent's `messages` list grows
by exactly one `ToolMessage` -- the inner agent's final answer, a single
string. Everything spent producing that string -- two more model calls, two
more sets of tokens, a whole second conversation -- happened and was billed,
and is invisible from the outer run's turn count unless you open that one
tool span and look inside it. `run(model_kind).summary["turns"]` counts only
the outer loop's own turns, on purpose: that asymmetry -- a cheap-looking
number hiding an expensive thing that happened underneath it -- is real, not
a bug in how this chapter counts.

`ch32_subgraph` is the other way to build this: a worker's whole transcript
merges into the parent's context instead of one string coming back. This
chapter is the cheap mechanism multi-agent needs to exist in plain Python at
all; `subgraph` is what a framework's alternative buys and costs.

**A second expert makes the choice visible.** With one tool, the supervisor's
`tool_calls` decision has only one way to go, and nothing shows that a choice
was ever made. `consult_orders_expert` gives it a second -- its own system
prompt, its own tool (`place_order`, a write, where the inventory expert only
ever sees a read), so the split is a real division of responsibility, not two
names for the same thing.

`the_supervisor_delegates_to_both` asks one question that needs both, in one
reply -- the `ch06_two_tools` fan-out shape, one level up: two `tool` spans,
siblings under the same turn, each opening its own nested agent underneath
it. `run_turns` dispatches a reply's calls with a plain `for` loop, so the two
experts run one after the other, not at once -- ordering and concurrency are
`ch33_parallel`'s question, where a framework runs the same shape on a pool
and `seq`/`thread` are what change.

**"Supervising" oversells what actually happens.** The model cannot call
another agent, only ever a tool -- `consult_inventory_expert` is
indistinguishable from `price_of` from where the model sits, a name and
nothing more, the same finding `ch02_tool_call` made about every tool there
has ever been. And the supervisor itself does not watch, correct, or see any
of the sub-agent's own turns while they run -- it asks a question and reads
back one string, blind to whatever produced it. That is the same asymmetry
as the cost that hides in the transcript, from the other side: not just that
the expense is invisible, but that there is no oversight to have caught it
even if someone were looking. A toolbox with some plain tools and some
agent-shaped ones is not "supervised" uniformly either -- the label only
describes the relationship with whichever tools happen to be agents; a plain
tool sitting beside them is an ordinary call, `ch04`-shaped, nothing
supervisor-flavored about it. If none of a supervisor's tools were agents, it
would not be a supervisor at all -- just an agent, the same as any other.
"""

from support.agent import ask_model, execute_tool, run_turns
from support.scenario import Outcome, RunOne, Scenario, run_scenarios
from support.trace import ModelKind, Trace

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SUPERVISOR_SYSTEM_PROMPT = (
    "You answer briefly and plainly. You have no pricing or ordering knowledge "
    "of your own -- consult the inventory expert for price or stock, and the "
    "orders expert to place an order."
)

INVENTORY_SYSTEM_PROMPT = "You answer briefly and plainly. Use the tool you are given."

ORDERS_SYSTEM_PROMPT = "You answer briefly and plainly. Use the tool you are given."

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

TURN_CAP = 4

INVENTORY_TURN_CAP = 4

ORDERS_TURN_CAP = 4


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item, in pounds. The expert's only tool."""
    return f"1.20 for {item}"


INVENTORY_TOOLS: dict[str, Any] = {price_of.name: price_of}


@tool
def place_order(item: GroceryItem, quantity: int) -> str:
    """Order a quantity of an item, in the shop. The orders expert's only tool."""
    return f"ordered {quantity} x {item}"


ORDERS_TOOLS: dict[str, Any] = {place_order.name: place_order}

# What the inner agent says, scripted independently of the outer one -- the
# same way any two `ask_model` calls in this primer have always been able to
# carry different replies, nested or not.
INVENTORY_MOCK_REPLIES = [
    AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "i1"}]),
    AIMessage("Milk costs 1.20 per bottle."),
]


def run_inventory_expert(question: str, model_kind: ModelKind, trace: Trace) -> str:
    """A whole `run_turns`, called from inside the outer tool's span.

    Reuses `ask_model`, `execute_tool` and `run_turns` exactly as they are --
    a supervisor needs no loop of its own, only a tool whose body happens to
    contain one.
    """
    messages: list[BaseMessage] = [
        SystemMessage(INVENTORY_SYSTEM_PROMPT),
        HumanMessage(question),
    ]

    def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(trace, model_kind, INVENTORY_MOCK_REPLIES[turn - 1 :], so_far, [price_of])

    def execute(call: ToolCall) -> ToolMessage:
        return execute_tool(
            call, trace, lambda name, args, _span: INVENTORY_TOOLS[name].invoke(args)
        )

    run_turns(messages, trace, INVENTORY_TURN_CAP, ask, execute)
    # The one string that crosses back out. Everything else the inner loop
    # did -- its own turns, its own spans -- stays inside the tool span this
    # was called from and never becomes part of the outer `messages` list.
    return str(messages[-1].content)


@tool
def consult_inventory_expert(question: str) -> str:
    """Ask the inventory expert about price or stock. It has its own tools."""
    raise AssertionError(
        f"consult_inventory_expert ['{question}'] is declared for its schema; "
        "execute_supervisor_tool below runs the inner agent"
    )


ORDERS_MOCK_REPLIES = [
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 5}, "id": "o1"}],
    ),
    AIMessage("Ordered 5 bottles of milk."),
]


def run_orders_expert(item: str, quantity: int, model_kind: ModelKind, trace: Trace) -> str:
    """The second expert, wired exactly like the first -- a write instead of
    a read, and nothing else different about how it nests or how it is called.
    """
    messages: list[BaseMessage] = [
        SystemMessage(ORDERS_SYSTEM_PROMPT),
        HumanMessage(f"Order {quantity} of {item}."),
    ]

    def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(trace, model_kind, ORDERS_MOCK_REPLIES[turn - 1 :], so_far, [place_order])

    def execute(call: ToolCall) -> ToolMessage:
        return execute_tool(call, trace, lambda name, args, _span: ORDERS_TOOLS[name].invoke(args))

    run_turns(messages, trace, ORDERS_TURN_CAP, ask, execute)
    return str(messages[-1].content)


@tool
def consult_orders_expert(item: GroceryItem, quantity: int) -> str:
    """Ask the orders expert to place an order. It has its own tools."""
    raise AssertionError(
        f"consult_orders_expert ['{quantity}' x '{item}'] is declared for its schema; "
        "execute_supervisor_tool below runs the inner agent"
    )


DECLARED_TOOLS = [consult_inventory_expert, consult_orders_expert]

# Which expert a call goes to, and how to pull its arguments out of the
# model's own -- the same DECLARED-vs-dispatched split every chapter since
# ch04_tool_failures has kept, one level up.
EXPERTS: dict[str, Callable[[dict[str, Any], ModelKind, Trace], str]] = {
    consult_inventory_expert.name: (
        lambda args, kind, trace: run_inventory_expert(args["question"], kind, trace)
    ),
    consult_orders_expert.name: (
        lambda args, kind, trace: run_orders_expert(args["item"], args["quantity"], kind, trace)
    ),
}

SCENARIOS = [
    Scenario(
        name="the_supervisor_delegates",
        question="How much is milk?",
        tools=DECLARED_TOOLS,
        mock_model_replies=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "consult_inventory_expert",
                        "args": {"question": "How much is milk?"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage("Milk is 1.20."),
        ],
    ),
    Scenario(
        name="the_supervisor_delegates_to_both",
        question="How much is milk, and please order 5 bottles.",
        tools=DECLARED_TOOLS,
        mock_model_replies=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "consult_inventory_expert",
                        "args": {"question": "How much is milk?"},
                        "id": "c1",
                    },
                    {
                        "name": "consult_orders_expert",
                        "args": {"item": "milk", "quantity": 5},
                        "id": "c2",
                    },
                ],
            ),
            AIMessage("Milk is 1.20 each; ordered 5 bottles."),
        ],
    ),
]


def execute_supervisor_tool(
    call: ToolCall, _scenario: Scenario, model_kind: ModelKind, trace: Trace
) -> ToolMessage:
    """Every call this supervisor can make, and each one is another agent."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        answer = EXPERTS[call["name"]](call["args"], model_kind, trace)
        span.add_note("result", value=answer)
    return ToolMessage(content=answer, tool_call_id=call["id"])


def run_one(model_kind: ModelKind, scenario: Scenario, trace: Trace, turn_cap: int) -> Outcome:
    messages: list[BaseMessage] = [
        SystemMessage(SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(scenario.question),
    ]

    def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(
            trace, model_kind, scenario.mock_model_replies[turn - 1 :], so_far, scenario.tools
        )

    def execute(call: ToolCall) -> ToolMessage:
        return execute_supervisor_tool(call, scenario, model_kind, trace)

    return run_turns(messages, trace, turn_cap, ask, execute)


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    typed_run_one: RunOne = run_one
    return run_scenarios("ch12_supervisor", model_kind, SCENARIOS, typed_run_one, turn_cap)
