# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Four cells of one matrix, and the question is who can change the outcome.

`ch04_tool_failures`, `ch07_tool_http` and `ch08_tool_program` collected the
specimens. This chapter classifies them, and the classification is not a list
of failures -- it is a two-column table, because there are exactly two parties
who can do anything about a failed call:

    failure                                harness         model
    ------------------------------------------------------------------
    rate limit, 5xx, timeout on a read     backoff, 2-3    pointless
    timeout on a call that writes          none            none
    arguments the tool rejects             none            once
    not found, unauthorised, no such tool  none            none

Three of the four cells are `none`, which is the finding. Retry is not a
default with exceptions; it is one cell, and everything else is either
reporting or stopping.

**The tool declares facts, the harness decides policy.** `ToolCallError` carries
`retryable`, `idempotent` and `caller_can_fix` -- three questions only the
tool's author can answer, because only they know whether an identical call
could ever work, whether calling twice is safe, and whether different
arguments would have helped. `attempt_tool` answers a different question: how
many, how long, and then what. That one is the harness's, because the same
tool deserves five attempts in a batch job overnight and none behind a person
waiting for an answer.

**`retryable and idempotent` is one condition, and both halves are load
bearing.** `order_may_have_landed` is the cell that catches people: the
timeout is transient, so the failure *looks* retryable, and the call may
already have placed the order. Nothing in the outcome distinguishes "the
request never arrived" from "the reply never came back". So the harness does
not retry, and the sentence handed to the model says the outcome is unknown
rather than that it failed -- which is the only honest thing to say.

**The harness's only lever on the model is the sentence it hands back.** It
cannot make the model retry and cannot stop it; `_sentence` writes one of
three, and each one is the harness telling the model which cell it is in.
`ch07_tool_http` showed what happens without that: told only that a 404 had
failed, the live model called the service again, because trying again is a
reasonable inference from a sentence that does not say otherwise.

**The live run found the cell the matrix does not have.** `arguments_are_wrong`
was scripted as a correction: rejected at 20, called again at 12, done. The
live model answered the rejection with *two* calls in one reply -- 12 and 8 --
and reported "Ordered 20 bottles of milk total (12 + 8), as the per-order
maximum is 12."

It did not correct the argument. It read the bound, kept the user's intent,
and routed around it. Every part of the harness behaved as designed: the
sentence named the bound, the tool enforced the bound, both calls were within
it, and 20 bottles were ordered.

`MAX_PER_ORDER` is a bound on one call and not on the outcome, and a tool
cannot tell the difference because a tool never sees the sequence it is part
of. Worse: `place_order` is the tool declared *not* idempotent, and the model
called it twice on purpose. Idempotence stops the harness from retrying. It
has nothing to say about the model choosing to call again -- which is the
argument `guards` is going to need, because a limit that means anything has to
be checked where the running total lives, and that is never inside the tool.

**And the counter lives outside the loop.** `attempt_tool` counts; the model
is never told which attempt this is, because from inside the context attempt
four looks exactly like attempt one -- the failed attempts of a retried call
never enter the message list at all. `service_is_busy` costs two service
calls and the model sees one tool result.

Two things here are staged and worth naming. `CALLS` scripts what the service
does on each attempt, because a real service recovering is not something a
chapter can arrange -- so the second attempt asks for a path that works. And
`arguments_are_wrong` is rejected by the tool before any request, which is
where most argument failures actually come from; a service answering `400` is
the same cell with a round trip in front of it.

This is also the whole of retry in LangGraph. `ToolNode` reports
`ToolInvocationError` back to the model and re-raises everything else -- the
one cell where the model retrying is right, and the one cell where the harness
retrying is useless. There is no backoff anywhere in it.
"""

from support.scenario import Scenario, run_scenario, run_scenarios
from support.service import SERVICE, SERVICE_KEY, Canned, build_http_client
from support.trace import ModelKind, Span, Trace

import time
from functools import partial
from typing import Any, Literal

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

TURN_CAP = 4

# Policy, and it is the harness's. Not the tool's, which knows nothing about
# who is waiting, and not the model's, which cannot count attempts it never
# sees. Two numbers, in one place, changeable without touching a tool.
MAX_ATTEMPTS = 3

BACKOFF_SECONDS = 0.05

# The house rule the tool enforces before it calls anything. A bound like this
# is where most bad arguments are caught in practice.
MAX_PER_ORDER = 12

# Status codes where an identical call could later succeed. Note what is not
# here: 404 and 401 are answers, not outages.
TRANSIENT = frozenset({429, 500, 502, 503, 504})

PRICE = httpx.Response(200, json={"args": {"item": "milk"}, "price": "1.20"})

ORDER = httpx.Response(200, json={"args": {"item": "milk"}, "reference": "ord-8812"})

# What the service does on each attempt of one situation. Attempt two of
# `service_is_busy` asks for a path that answers, because a real service
# agreeing to recover on cue is the one thing a chapter cannot arrange -- so
# the request is scripted instead of the service.
CALLS: dict[str, list[tuple[str, Canned]]] = {
    "service_is_busy": [
        ("/status/429", httpx.Response(429, headers={"retry-after": "1"}, text="")),
        ("/get", PRICE),
    ],
    # Not a response. A mocked transport answers instantly, so the only way to
    # be slow is to raise what being slow would have raised.
    "order_may_have_landed": [("/delay/10", httpx.TimeoutException("mocked: no answer"))],
    "arguments_are_wrong": [("/get", ORDER)],
    "service_says_no": [("/status/404", httpx.Response(404, text=""))],
}


class ToolCallError(RuntimeError):
    """A call that produced no result, and the three facts about why.

    `retryable` and `caller_can_fix` describe the failure. `idempotent`
    describes the tool -- it is the same on every failure that tool ever
    raises -- and rides along because the harness needs both facts in one
    place to decide anything at all.
    """

    def __init__(
        self, message: str, *, retryable: bool, idempotent: bool, caller_can_fix: bool
    ) -> None:
        super().__init__(message)
        self.retryable: bool = retryable
        self.idempotent: bool = idempotent
        self.caller_can_fix: bool = caller_can_fix

    def facts(self) -> dict[str, Any]:
        """What the span records, and what `attempt_tool` decides on."""
        return {
            "message": str(self),
            "retryable": self.retryable,
            "idempotent": self.idempotent,
            "caller_can_fix": self.caller_can_fix,
        }


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of one of an item, in pounds."""
    raise AssertionError(f"price_of ['{item}'] is declared for its schema; fetch_price works")


@tool
def place_order(item: GroceryItem, quantity: int) -> str:
    """Order a quantity of an item. Placing an order twice orders it twice."""
    raise AssertionError(
        f"place_order ['{quantity}' x '{item}'] is declared for its schema; submit_order works"
    )


def call_service(
    scenario: str,
    attempt: int,
    params: dict[str, Any],
    idempotent: bool,
    span: Span,
    kind: ModelKind,
) -> None:
    """One request, and every failure it can produce, classified on the way out.

    Returns nothing: the situations here differ in how they fail, not in what
    they answer, so a body neither tool would read is noise in the trace.
    """
    calls = CALLS[scenario]
    path, canned = calls[min(attempt, len(calls)) - 1]
    client = build_http_client(kind, span, canned)
    try:
        response = client.get(
            f"{SERVICE}{path}",
            params=params,
            headers={"Authorization": f"Bearer {SERVICE_KEY}"},
            timeout=2 if path.startswith("/delay") else 10,
        )
    except httpx.TimeoutException as timeout:
        # Transient by nature, and unsafe by consequence when the tool writes.
        # The one cell where `retryable` is true and retrying is still wrong.
        raise ToolCallError(
            f"['{SERVICE}'] did not answer in time, and the call may have been received",
            retryable=True,
            idempotent=idempotent,
            caller_can_fix=False,
        ) from timeout
    if not response.is_success:
        raise ToolCallError(
            f"['{SERVICE}'] answered ['{response.status_code}']",
            retryable=response.status_code in TRANSIENT,
            idempotent=idempotent,
            caller_can_fix=False,
        )


def fetch_price(item: str, scenario: str, attempt: int, span: Span, kind: ModelKind) -> str:
    """A read. Calling it twice costs a request and changes nothing."""
    call_service(scenario, attempt, {"item": item}, True, span, kind)
    return "1.20"


def submit_order(
    item: str, quantity: int, scenario: str, attempt: int, span: Span, kind: ModelKind
) -> str:
    """A write. Calling it twice orders twice, which is the whole difference."""
    if not 1 <= quantity <= MAX_PER_ORDER:
        # Rejected before anything leaves the process, and the message names
        # the bound -- because the model can only fix what it is told.
        raise ToolCallError(
            f"quantity ['{quantity}'] must be between 1 and {MAX_PER_ORDER}",
            retryable=False,
            idempotent=False,
            caller_can_fix=True,
        )
    call_service(scenario, attempt, {"item": item, "quantity": quantity}, False, span, kind)
    return f"ordered {quantity} x {item}"


TOOLS: dict[str, Any] = {price_of.name: fetch_price, place_order.name: submit_order}

DECLARED_TOOLS = [place_order, price_of]


def attempt_tool(call: ToolCall, scenario: str, span: Span, kind: ModelKind) -> Any:
    """The policy: how many, how long, and then what.

    One condition decides everything, and both halves are load bearing --
    `retryable` because an identical call must be able to work, `idempotent`
    because a call that may already have run must not be made twice.

    The attempts are recorded and never appended to the message list. The
    model sees one result whether this took one request or three.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = TOOLS[call["name"]](
                **call["args"], scenario=scenario, attempt=attempt, span=span, kind=kind
            )
        except ToolCallError as failure:
            span.add_note("attempt", number=attempt, **failure.facts())
            if not (failure.retryable and failure.idempotent) or attempt == MAX_ATTEMPTS:
                raise
            wait = BACKOFF_SECONDS * 2 ** (attempt - 1)
            span.add_note("backoff", seconds=wait, before_attempt=attempt + 1)
            time.sleep(wait)
        else:
            # The one that worked is recorded too, carrying only its number.
            # Otherwise the span shows a failure and a result with nothing
            # saying they were different requests.
            span.add_note("attempt", number=attempt)
            return result
    raise AssertionError("unreachable: the loop returns a result or raises the last failure")


def sentence(name: str, failure: ToolCallError) -> str:
    """What the model is told, which is the only lever the harness has on it.

    Three sentences for four cells: the two that no one can retry read the
    same to the model, because "stop and say so" is the same instruction
    whether the reason is a 404 or an order that may already exist. What
    separates them is that one of them says the outcome is unknown.
    """
    if failure.caller_can_fix:
        return f"['{name}'] rejected the arguments: {failure}. Correct them and call it again."
    if failure.retryable and not failure.idempotent:
        return (
            f"['{name}'] did not confirm: {failure}. It may have taken effect. "
            "Do not call it again; say what is uncertain."
        )
    return f"['{name}'] failed: {failure}. It will not succeed. Answer without it."


def execute_tool(
    call: ToolCall, scenario: Scenario, model_kind: ModelKind, trace: Trace
) -> ToolMessage:
    """One tool call, retried by policy, and reported once however it went."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        try:
            result = attempt_tool(call, scenario.name, span, model_kind)
        except ToolCallError as failure:
            span.add_note("failed", **failure.facts())
            return ToolMessage(
                content=sentence(call["name"], failure), tool_call_id=call["id"], status="error"
            )
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


ASKS_PRICE = AIMessage(
    "",
    tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}],
    response_metadata={"finish_reason": "tool_calls"},
)


def asks_to_order(quantity: int, call_id: str) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {"name": "place_order", "args": {"item": "milk", "quantity": quantity}, "id": call_id}
        ],
        response_metadata={"finish_reason": "tool_calls"},
    )


def says(text: str) -> AIMessage:
    return AIMessage(text, response_metadata={"finish_reason": "stop"})


SCENARIOS = [
    # Two service calls, one tool result, and the model none the wiser.
    Scenario(
        name="service_is_busy",
        question="How much is milk?",
        tools=DECLARED_TOOLS,
        mock_model_replies=[ASKS_PRICE, says("Milk is 1.20.")],
    ),
    # The cell with teeth. Transient, and still nobody may try again.
    Scenario(
        name="order_may_have_landed",
        question="Order 2 bottles of milk.",
        tools=DECLARED_TOOLS,
        mock_model_replies=[
            asks_to_order(2, "c1"),
            says("The order may or may not have been placed. Please check before ordering again."),
        ],
    ),
    # The only cell where the model retrying is right, and it costs a turn.
    Scenario(
        name="arguments_are_wrong",
        question="Order 20 bottles of milk.",
        tools=DECLARED_TOOLS,
        mock_model_replies=[
            asks_to_order(20, "c1"),
            asks_to_order(12, "c2"),
            says("I ordered 12, the most allowed in one order."),
        ],
    ),
    # An answer, not an outage. One attempt, and nothing to be gained by more.
    Scenario(
        name="service_says_no",
        question="How much is milk?",
        tools=DECLARED_TOOLS,
        mock_model_replies=[ASKS_PRICE, says("I could not get the price of milk.")],
    ),
]


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    return run_scenarios(
        "ch09_who_retries",
        model_kind,
        SCENARIOS,
        partial(run_scenario, system_prompt=SYSTEM_PROMPT, execute_tool=execute_tool),
        turn_cap,
    )
