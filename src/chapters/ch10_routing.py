# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The loop branches on its own, not on the name the model sent.

**`the_question_needs_no_model` is the first branch, and it runs before the
model is ever called.** Some questions do not need one at all -- the catalog
of items this shop sells is a fact we already hold, not something to ask a
model to guess at. `route_before_the_model` reads the question itself, the
way a real router would: there is no scenario name at that point, only the
prompt a user typed, matched against what is already known. When it matches,
`ask_model` is never called: there is no request, no reply, and no `model`
span. On the page this scenario's one turn has a note and no children, which
is the branch made visible as an absence rather than asserted in prose.

`run_scenario` cannot express this -- it calls `ask_model` unconditionally,
because every chapter through `ch09` always wants the model called. So this
scenario keeps its own `run_one`, the way `ch05_loop_endings` kept its own
loop: the decision not to call the model is the lesson, and a mechanism that
already assumes the model gets called cannot make it.

**`a_write_takes_a_longer_path` is the second, and it runs after.** Every
chapter so far had exactly one `if` after a reply: does it carry a tool call.
`ch04_tool_failures` added a second lookup -- `TOOLS[call["name"]]` -- but
that one is still the model's choice; we only look up what it asked for.
This `if` is ours: `route()` decides where a call goes based on *what kind*
of call it is, not on dispatching whatever name arrived. The model never
asked to be routed -- it asked for `place_order`, and the decision to send a
write through an extra step before dispatch is made after the reply and
before the call runs.

**`the_turns_are_running_out` is the third, and it branches on neither.** Not
the reply, not the question -- a number the loop already carries. Given two
turns to answer "how much is milk, and how much are chips", asking a second
time is the obvious move and the wrong one: it spends the only turn left on
a request that might come back with another tool call and nothing to show
for it. `route_on_turns_remaining` checks the turn about to be spent against
the budget before asking, and on the last one wraps up with what has already
been found rather than gambling it on one more round trip.

This is `ch05_loop_endings`'s `turns_exhausted` turned from something that
happens to the loop into something the loop decides. Hitting the cap with no
answer and choosing to stop with a partial one are different endings, so this
scenario ends `wrapped_up_early`, never `turns_exhausted` -- reaching the cap
was never let to happen.

`set_conditional_entry_point` is LangGraph's name for the first branch;
`add_conditional_edges` is the second and the third -- the same call handles
routing on state, because from where the graph decides, a reply already
appended and a counter already held are both just state to read.
"""

from support.agent import ask_model
from support.scenario import Outcome, RunOne, Scenario, run_scenario, run_scenarios
from support.trace import ModelKind, Trace

from typing import Any, Literal, get_args

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

TURN_CAP = 4

# The routing decision, named rather than inferred: a tool is a write if it
# is in this set. Nothing about the call itself says so -- an author has to
# declare it, the same way `ch09_who_retries` declared `idempotent`.
WRITE_TOOLS = frozenset({"place_order"})


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item, in pounds. A read: routed directly."""
    return f"1.20 for {item}"


@tool
def place_order(item: GroceryItem, quantity: int) -> str:
    """Order a quantity of an item. A write: routed through confirmation first."""
    return f"ordered {quantity} x {item}"


TOOLS: dict[str, Any] = {price_of.name: price_of, place_order.name: place_order}

DECLARED_TOOLS = [place_order, price_of]


# scenario: a_write_takes_a_longer_path
def route(name: str) -> str:
    """The `if`. Not a lookup of what the model asked for -- a decision about it."""
    return "confirm_before_dispatch" if name in WRITE_TOOLS else "dispatch_directly"


# Answerable from what we already know, with nothing to ask a model to guess
# at. The catalog itself is the fact -- `GroceryItem` names it once, and
# asking a model to repeat it back would be a call spent on nothing.
#
# Keyed by the question a user would actually type, not by a scenario name --
# a real router has no scenario names to read, only the prompt. Matched after
# normalizing case and trailing punctuation, the way a cache lookup would.
FAQ: dict[str, str] = {"what items can you order": ", ".join(get_args(GroceryItem.__value__))}


def normalized(question: str) -> str:
    return question.strip().rstrip("?.").lower()


SCENARIOS = [
    Scenario(
        name="the_question_needs_no_model",
        question="What items can you order?",
        # Never read: this scenario never calls a model, mock or live.
        mock_model_replies=(),
    ),
    Scenario(
        name="a_write_takes_a_longer_path",
        question="Order 2 bottles of milk.",
        tools=DECLARED_TOOLS,
        mock_model_replies=[
            AIMessage(
                "",
                tool_calls=[
                    {"name": "place_order", "args": {"item": "milk", "quantity": 2}, "id": "c1"}
                ],
            ),
            AIMessage("Ordered 2 bottles of milk."),
        ],
    ),
    Scenario(
        name="the_turns_are_running_out",
        question="How much is milk, and how much are chips?",
        tools=DECLARED_TOOLS,
        # Only the first is ever read. The second would ask for chips next --
        # the budget check below stops that turn from being spent asking.
        mock_model_replies=[
            AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}]),
            AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "chips"}, "id": "c2"}]),
        ],
    ),
]


def execute_tool(
    call: ToolCall, _scenario: Scenario, _model_kind: ModelKind, trace: Trace
) -> ToolMessage:
    """One tool call, routed before it runs rather than dispatched on sight."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        destination = route(call["name"])
        span.add_note("routed", to=destination)
        if destination == "confirm_before_dispatch":
            # The extra step a write takes and a read never does. A real one
            # might hold for approval; here it is a fact recorded in the
            # trace, which is enough to show the branch was taken.
            args = call["args"]
            span.add_note("confirmed", summary=f"{args['quantity']} x {args['item']}")
        result = TOOLS[call["name"]].invoke(call["args"])
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


# scenario: the_question_needs_no_model
def answer_without_a_model(
    _model_kind: ModelKind, scenario: Scenario, trace: Trace, _turn_cap: int
) -> Outcome:
    """The one turn this scenario runs, with no model inside it.

    Two messages would normally be held before the first call: the system
    prompt and the question. Neither goes anywhere, because nothing is asked.
    """
    with trace.span("turn", number=1) as span:
        answer = FAQ[normalized(scenario.question)]
        span.add_note("answered directly", because="the catalog is a known fact", value=answer)
    return 1, 2, "answered_without_a_model"


# scenario: the_question_needs_no_model
def route_before_the_model(question: str) -> str:
    """The second `if`. Runs before any call to `ask_model` is made.

    On the actual question, the way a real router would see it -- there is
    no scenario name to read at this point, only the prompt a user typed.
    """
    return "no_model_needed" if normalized(question) in FAQ else "ask_the_model"


# How many turns this scenario is willing to spend, independent of the
# chapter's TURN_CAP -- a real budget is set by what the caller can afford,
# not by the harness's own safety limit.
TURNS_BUDGET = 2


# scenario: the_turns_are_running_out
def route_on_turns_remaining(turns_spent: int, budget: int) -> str:
    """The third `if`. On a number the loop already carries, not on a call.

    `turns_spent` is turns already taken; one more would be `turns_spent + 1`.
    If that would meet or pass the budget, this is the last turn there is
    room for, and it must not be spent asking -- there would be nowhere left
    to report what came back.
    """
    return "wrap_up_now" if turns_spent + 1 >= budget else "ask_the_model"


def wrap_up(collected: list[str]) -> str:
    if not collected:
        return "I ran out of turns before finding an answer."
    return "Here's what I found before running out of turns: " + "; ".join(collected)


# scenario: the_turns_are_running_out
def running_out_of_turns(model_kind: ModelKind, scenario: Scenario, trace: Trace) -> Outcome:
    """Its own loop, like `answer_without_a_model` -- the budget check has to
    run between turns, and `run_turns` has no place for a check that is not
    `reply.tool_calls`.
    """
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(scenario.question)]
    collected: list[str] = []
    turns = 0
    while turns < TURNS_BUDGET:
        if route_on_turns_remaining(turns, TURNS_BUDGET) == "wrap_up_now":
            turns += 1
            with trace.span("turn", number=turns) as span:
                answer = wrap_up(collected)
                span.add_note("wrapped up", because="no turns left to ask again", value=answer)
            return turns, len(messages), "wrapped_up_early"
        turns += 1
        with trace.span("turn", number=turns):
            reply = ask_model(
                trace,
                model_kind,
                scenario.mock_model_replies[turns - 1 :],
                messages,
                scenario.tools,
                scenario.max_tokens,
            )
            messages.append(reply)
            if not reply.tool_calls:
                return turns, len(messages), "no_tool_calls"
            for call in reply.tool_calls:
                result = execute_tool(call, scenario, model_kind, trace)
                messages.append(result)
                collected.append(str(result.content))
    return turns, len(messages), "turns_exhausted"


def run_one(model_kind: ModelKind, scenario: Scenario, trace: Trace, turn_cap: int) -> Outcome:
    if route_before_the_model(scenario.question) == "no_model_needed":
        return answer_without_a_model(model_kind, scenario, trace, turn_cap)
    if scenario.name == "the_turns_are_running_out":
        return running_out_of_turns(model_kind, scenario, trace)
    return run_scenario(
        model_kind,
        scenario,
        trace,
        turn_cap,
        system_prompt=SYSTEM_PROMPT,
        execute_tool=execute_tool,
    )


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    typed_run_one: RunOne = run_one
    return run_scenarios("ch10_routing", model_kind, SCENARIOS, typed_run_one, turn_cap)
