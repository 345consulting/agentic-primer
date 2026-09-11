# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The first tool that leaves the process, and what comes back with it.

Every tool so far returned from a dict or raised an exception someone wrote,
which made `ch04_tool_failures` honest but invented. A real HTTP call brings
four things at once, and each of them is material the next chapters need.

**Latency lands inside the tool span.** A turn's cost stops splitting two ways
and starts splitting three: the model provider, the library, and now a service
that is nothing to do with either.

**Failures arrive as status codes, with meanings attached.** These are the
specimens `ch09_who_retries` classifies, and no two of them want the same
treatment:

    service_answers      200   nothing to decide
    service_rate_limits  429   transient, and Retry-After says how long
    service_is_broken    500   transient, and it tells you nothing
    service_says_no      404   permanent; retrying is pure cost
    service_is_slow      timeout, and the call may still have landed

The last is the one with teeth. A timeout is transient by nature and unsafe by
consequence: you do not know whether the work happened, so a tool that is not
idempotent must not be retried even though the failure looks retryable. No
status code tells you that. Only the tool's author knows.

**There is a second credential**, and it is the first in this primer that is
not the model provider's. A harness holds more than one secret, and the
recorder has to be right about all of them.

**And there is a second wire.** The tool's HTTP is recorded with the same
hooks as the model's, in the tool's own span -- so a tool call now carries a
request and a response body, syntax-highlighted like any other.

Which is how this chapter finds something. The request record refuses to store
`Authorization`, by the allowlist in `support/models.py`: a denylist has to be
right about every auth header that will ever exist, an allowlist that is wrong
records nothing. That rule holds. And then httpbin echoes the request headers
back in its *response body*, which is recorded in full, and the credential we
withheld at one boundary arrives at the other.

The key here is fabricated and worth nothing, which is the only reason this is
safe to demonstrate. With a real one, the same code would write it to
`out/*.json` in plaintext, and every check we have would still pass.

No retry -- by us. The first live run retried anyway, and it is worth reading
before the retry chapters make it a subject: the model called the service
twice on `service_says_no`, a 404, having been told the call failed and to
answer without it. Eleven turns live against ten in the mock column.

A 404 is the one failure in this chapter that can never succeed. The model
cannot know that: it received a sentence, and trying again is a reasonable
inference from a sentence. Which is the whole argument for classification
living with the tool and the counter living outside the loop -- neither is
something the model can be persuaded into.
"""

from support.scenario import Scenario, run_scenario, run_scenarios
from support.service import SERVICE, SERVICE_KEY, Canned, build_http_client
from support.trace import ModelKind, Span, Trace

from functools import partial
from typing import Any, Literal

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

USER_PROMPT = "How much is milk?"

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

TURN_CAP = 4

# The path each situation asks httpbin for. `/get` answers and echoes; the
# `/status/...` paths fail on demand; `/delay/...` outlasts the timeout.
PATHS = {
    "service_answers": "/get",
    "service_rate_limits": "/status/429",
    "service_is_broken": "/status/500",
    "service_says_no": "/status/404",
    "service_is_slow": "/delay/10",
}

# What the mocked service returns, so a mock run makes no network call and the
# recorded wire is the same shape as a live one.
CANNED: dict[str, Canned] = {
    "service_answers": httpx.Response(
        200,
        json={
            "args": {"item": "milk"},
            # httpbin echoes what it was sent. The credential comes home here.
            "headers": {"Authorization": f"Bearer {SERVICE_KEY}", "Host": "httpbin.org"},
            "url": f"{SERVICE}/get?item=milk",
        },
    ),
    "service_rate_limits": httpx.Response(429, headers={"retry-after": "3"}, text=""),
    "service_is_broken": httpx.Response(500, text=""),
    "service_says_no": httpx.Response(404, text=""),
    # Not a response. A mocked transport replies instantly, so the only way
    # to be slow is to raise what being slow would have raised.
    "service_is_slow": httpx.TimeoutException("mocked: the service did not answer"),
}


class PriceServiceError(RuntimeError):
    """The price service did not answer. Carries what the service said."""


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of one of an item, in pounds."""
    # Declared, never dispatched. `@tool` builds the schema that goes to the
    # model provider; the thing that runs is `fetch_price` below, because it
    # needs a span to record into and a situation to be in, and a tool only
    # ever receives what the model chose to send it.
    raise AssertionError(
        f"price_of ['{item}'] is declared for its schema; fetch_price does the work"
    )


def fetch_price(item: str, scenario: str, span: Span, model_kind: ModelKind) -> str:
    """The tool's body, with everything it needs to record itself."""
    client = build_http_client(model_kind, span, CANNED[scenario])
    try:
        response = client.get(
            f"{SERVICE}{PATHS[scenario]}",
            params={"item": item},
            # The second credential. Sent on every call, refused by the
            # request recorder, and echoed back into the response record.
            headers={"Authorization": f"Bearer {SERVICE_KEY}"},
            timeout=2 if scenario == "service_is_slow" else 10,
        )
    except httpx.TimeoutException as timeout:
        raise PriceServiceError(f"['{SERVICE}'] did not answer in time") from timeout
    if not response.is_success:
        raise PriceServiceError(f"['{SERVICE}'] answered ['{response.status_code}']")
    return "1.20"


# The dispatch table and the declaration are two different objects, which
# ch04_tool_failures made a lesson of. Here they differ on purpose: the model
# is told about `price_of`, and what runs is `fetch_price`.
TOOLS: dict[str, Any] = {price_of.name: fetch_price}

DECLARED_TOOLS = [price_of]

MOCK_MODEL_REPLIES = [
    AIMessage(
        "",
        tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}],
        response_metadata={"finish_reason": "tool_calls"},
    ),
    AIMessage("Milk is 1.20.", response_metadata={"finish_reason": "stop"}),
]

MOCK_MODEL_GIVES_UP = [
    MOCK_MODEL_REPLIES[0],
    AIMessage("I could not get the price of milk.", response_metadata={"finish_reason": "stop"}),
]

SCENARIOS = [
    Scenario(name=name, question=USER_PROMPT, mock_model_replies=replies, tools=DECLARED_TOOLS)
    for name, replies in (
        ("service_answers", MOCK_MODEL_REPLIES),
        ("service_rate_limits", MOCK_MODEL_GIVES_UP),
        ("service_is_broken", MOCK_MODEL_GIVES_UP),
        ("service_says_no", MOCK_MODEL_GIVES_UP),
        ("service_is_slow", MOCK_MODEL_GIVES_UP),
    )
]


def execute_tool(
    call: ToolCall, scenario: Scenario, model_kind: ModelKind, trace: Trace
) -> ToolMessage:
    """One tool call, with the tool's own wire recorded in its span."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        try:
            result = TOOLS[call["name"]](
                **call["args"], scenario=scenario.name, span=span, model_kind=model_kind
            )
        except PriceServiceError as failure:
            span.add_note("failed", exception=type(failure).__name__, message=str(failure))
            return ToolMessage(
                content=f"['{call['name']}'] failed: {failure}. Answer without it.",
                tool_call_id=call["id"],
                status="error",
            )
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    return run_scenarios(
        "ch07_tool_http",
        model_kind,
        SCENARIOS,
        partial(run_scenario, system_prompt=SYSTEM_PROMPT, execute_tool=execute_tool),
        turn_cap,
    )
