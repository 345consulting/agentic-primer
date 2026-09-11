# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""A dispatch table you did not write.

`support/mcp.py` stands in for what an MCP server hands a client: three
catalogs, each returned by a call instead of written as a literal, and the
three matching calls a chapter uses to act on what it found. This chapter is
what changes because of that -- not how the protocol's wire format works.

Five scenarios, one per fact worth seeing:

    tool_discovered_not_declared        the familiar round trip from `ch02`
                                         onward, except nothing in this
                                         chapter's source names the tool the
                                         model calls -- `tools/list` did
    resource_lands_without_a_round_trip a client reads a resource and
                                         splices it into context before the
                                         model ever runs -- no `tool_call`,
                                         and the model cannot tell the
                                         result apart from typed text
    prompt_seeds_the_conversation       a human picks a name, fills an
                                         argument, and `prompts/get`'s reply
                                         becomes the seed messages -- no
                                         model involved in the choice either
    prompt_and_tool_are_two_separate_decisions
                                         the same prompt, this time with the
                                         tool it needs actually declared --
                                         nothing in MCP links a prompt to the
                                         tools its questions require, and the
                                         first scenario's live model proved
                                         it by guessing a price instead of
                                         checking one
    dispatch_table_order_is_not_a_promise
                                         the same catalog, discovered twice
                                         with different seeds, agrees on
                                         content and disagrees on order --
                                         which is what a stable cache prefix
                                         cannot survive

**The first three are a comparison of actors, not of mechanism.** Every one
of them ends with more text in the model's context than a human typed this
turn. What differs is who decided it should be there: the model, for a
tool -- it is the only primitive with a request/response shape at all. The
client, for a resource -- discovered and read on the harness's own say-so,
confirmed against a live wire body in this chapter's own development, where
`messages[1]` held the resource's text with nothing marking it as such. A
human, for a prompt -- picked and filled before turn one, leaving no residue
in the request that a template was ever involved.

**The fourth is a live-confirmed gap, not a hypothetical one.** `prompts/get`
returns messages; that is its whole contract. Nothing about it also declares
which tools those messages presuppose. `prompt_seeds_the_conversation`
declares none, on purpose, to keep the scenario isolated -- and this
chapter's own live run of it answered *"I don't have real-time access to
grocery store pricing, but butter typically costs around $4-$6..."* rather
than admitting it could not check. Attaching the prompt and attaching the
tool it needs are two separate decisions, made by whoever assembles the
request, and MCP gives you no seam that catches it if the second one is
forgotten. The fifth scenario is the same prompt with that decision made.

**The fifth is `ch22`'s `design_time` row under a specific threat.** A tool
declared as a Python literal has one order, forever, because you wrote it.
A tool list returned by a call has whatever order that call produced, and
nothing promises the next call agrees -- confirmed here directly:
`ToolServer.list_tools(seed=0)` and `list_tools(seed=4)` return the same
three names and disagree on their order, which means the serialized `tools`
block at the front of the request is different bytes each time. A provider's
prompt cache keys off a stable prefix; MCP's discovery step is a second
process's decision sitting inside that prefix, and this chapter's own
`support/mcp.py` build only needed nine trial seeds to find two that
disagree.
"""

from support.agent import ask_model, execute_tool, run_turns
from support.mcp import Json, PromptServer, ResourceServer, ToolServer
from support.trace import ModelKind, Trace

from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import create_model

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."


@tool
def price_of(item: str) -> str:
    """The current shelf price of an item, in pounds."""
    return f"1.20 for {item}"


@tool
def hours_of(day: str) -> str:
    """The store's opening hours on a given day."""
    return f"9am-6pm on {day}"


@tool
def stock_of(item: str) -> str:
    """How many units of an item are on the shelf."""
    return f"14 units of {item}"


TOOLS = {"price_of": price_of, "hours_of": hours_of, "stock_of": stock_of}


def _declaration(t: BaseTool) -> Json:
    return {"name": t.name, "description": t.description, "schema": t.get_input_jsonschema()}


def _handler(t: BaseTool) -> Callable[..., Any]:
    def run(**kw: Any) -> Any:
        return t.invoke(kw)

    return run


TOOL_SERVER = ToolServer(
    catalog=[_declaration(t) for t in TOOLS.values()],
    handlers={name: _handler(t) for name, t in TOOLS.items()},
)

_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _proxy_from_schema(entry: Json) -> BaseTool:
    """A tool the model can call, built from nothing but what `tools/list`
    returned -- `entry["schema"]`, plain JSON. No local Python function
    backs this; a real MCP client is in exactly this position, because a
    remote tool's implementation lives on the server that declared it, not
    on the machine building the request. Calling the proxy sends `name` and
    `args` to `TOOL_SERVER.call_tool` -- the only two things `tools/call`
    ever carries -- the same way a client would send them over the wire.

    This is the piece `tool_discovered_not_declared` would be cheating
    without: reusing `TOOLS[name]` for the *declaration* would mean the
    request was built from a Python object that happened to already exist,
    not from the JSON discovery actually returned.
    """
    schema = entry["schema"]
    properties: Json = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {
        name: (
            _JSON_TYPES.get(prop.get("type", ""), str),
            ... if name in required else None,
        )
        for name, prop in properties.items()
    }
    args_model = create_model(f"{entry['name']}_args", **fields)

    def call(**kwargs: Any) -> Any:
        return TOOL_SERVER.call_tool(entry["name"], kwargs)

    return StructuredTool.from_function(
        func=call, name=entry["name"], description=entry["description"], args_schema=args_model
    )


RESOURCE_SERVER = ResourceServer(
    catalog=[
        {
            "uri": "file:///pricing/policy.md",
            "name": "pricing-policy",
            "description": "This quarter's pricing policy",
            "mimeType": "text/markdown",
        }
    ],
    content={"file:///pricing/policy.md": "Round every price to the nearest 5p."},
)


def _price_check_template(args: Json) -> list[Json]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "human", "content": f"What does {args['item']} cost right now?"},
    ]


PROMPT_SERVER = PromptServer(
    catalog=[
        {
            "name": "price-check",
            "description": "Ask the current price of a named item",
            "arguments": [{"name": "item", "description": "the item to price", "required": True}],
        }
    ],
    templates={"price-check": _price_check_template},
)

_ROLE_TO_MESSAGE = {"system": SystemMessage, "human": HumanMessage}


def _round_trip(
    trace: Trace,
    model_kind: ModelKind,
    messages: list[BaseMessage],
    declared: Sequence[BaseTool],
    mock_replies: list[AIMessage],
    turn_cap: int,
) -> tuple[int, int, str]:
    """The turn loop, given an already-open scenario span -- so an MCP call
    made before this point (`list_tools`, `read_resource`, `get_prompt`) has
    somewhere to record a note, instead of happening in bare Python with no
    trace of it at all.
    """

    def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(trace, model_kind, mock_replies[turn - 1 :], so_far, declared)

    def execute(call: ToolCall) -> ToolMessage:
        return execute_tool(call, trace, lambda n, a, _s: TOOL_SERVER.call_tool(n, a))

    return run_turns(messages, trace, turn_cap, ask, execute)


def tool_discovered_not_declared(trace: Trace, model_kind: ModelKind) -> None:
    """`tools/list`, then the model asks for one by the name it discovered --
    the round trip is `ch02`'s; the catalog is not this chapter's literal.
    """
    with trace.span("scenario", name="tool_discovered_not_declared") as span:
        discovered = TOOL_SERVER.list_tools(seed=0)
        span.add_note("tools/list", catalog=discovered)
        # Built from `discovered` alone -- not `TOOLS[name]`. Reaching into
        # the module's own real functions here would have declared a tool
        # the request never actually needed a server for.
        declared = [_proxy_from_schema(t) for t in discovered]
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("How much is milk?"),
        ]
        mock_replies = [
            AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}]),
            AIMessage("Milk is 1.20."),
        ]
        _round_trip(trace, model_kind, messages, declared, mock_replies, turn_cap=2)


def resource_lands_without_a_round_trip(trace: Trace, model_kind: ModelKind) -> None:
    """`resources/list`, then `resources/read` -- read by the harness, not
    the model. The content is a message before the model ever runs.
    """
    with trace.span("scenario", name="resource_lands_without_a_round_trip") as span:
        catalog = RESOURCE_SERVER.list_resources()
        span.add_note("resources/list", catalog=catalog)
        uri = "file:///pricing/policy.md"
        policy = RESOURCE_SERVER.read_resource(uri)
        span.add_note("resources/read", uri=uri, content=policy)
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(f"Reference document:\n{policy}"),
            HumanMessage("How much should milk cost, rounded per the policy?"),
        ]
        mock_replies = [AIMessage("Milk should be rounded to the nearest 5p.")]
        _round_trip(trace, model_kind, messages, [], mock_replies, turn_cap=1)


def prompt_seeds_the_conversation(trace: Trace, model_kind: ModelKind) -> None:
    """`prompts/list`, then `prompts/get` with a human-filled argument -- its
    reply becomes turn one's messages, not a hand-written system/human pair.
    """
    with trace.span("scenario", name="prompt_seeds_the_conversation") as span:
        catalog = PROMPT_SERVER.list_prompts()
        span.add_note("prompts/list", catalog=catalog)
        args = {"item": "butter"}
        picked = PROMPT_SERVER.get_prompt("price-check", args)
        span.add_note("prompts/get", name="price-check", arguments=args, messages=picked)
        messages: list[BaseMessage] = [_ROLE_TO_MESSAGE[m["role"]](m["content"]) for m in picked]
        mock_replies = [AIMessage("Butter is 1.20.")]
        _round_trip(trace, model_kind, messages, [], mock_replies, turn_cap=1)


def prompt_and_tool_are_two_separate_decisions(trace: Trace, model_kind: ModelKind) -> None:
    """The same prompt, this time with `price_of` discovered and declared
    alongside it -- proving the contrast against `prompt_seeds_the_conversation`
    rather than just asserting it.
    """
    name = "prompt_and_tool_are_two_separate_decisions"
    with trace.span("scenario", name=name) as span:
        picked = PROMPT_SERVER.get_prompt("price-check", {"item": "butter"})
        span.add_note("prompts/get", name="price-check", messages=picked)
        discovered = TOOL_SERVER.list_tools()
        span.add_note("tools/list", catalog=discovered)
        declared = [_proxy_from_schema(t) for t in discovered if t["name"] == "price_of"]
        messages: list[BaseMessage] = [_ROLE_TO_MESSAGE[m["role"]](m["content"]) for m in picked]
        mock_replies = [
            AIMessage(
                "", tool_calls=[{"name": "price_of", "args": {"item": "butter"}, "id": "c3"}]
            ),
            AIMessage("Butter is 1.20."),
        ]
        _round_trip(trace, model_kind, messages, declared, mock_replies, turn_cap=2)


def dispatch_table_order_is_not_a_promise(trace: Trace, model_kind: ModelKind) -> None:
    """The same catalog, two seeds -- same names, different order, different
    bytes at the front of every request that follows.
    """
    with trace.span("scenario", name="dispatch_table_order_is_not_a_promise") as span:
        first = TOOL_SERVER.list_tools(seed=0)
        second = TOOL_SERVER.list_tools(seed=4)
        same_content = {t["name"] for t in first} == {t["name"] for t in second}
        same_order = [t["name"] for t in first] == [t["name"] for t in second]
        fingerprint_first = sha256(str([t["name"] for t in first]).encode()).hexdigest()[:12]
        fingerprint_second = sha256(str([t["name"] for t in second]).encode()).hexdigest()[:12]
        span.add_note(
            "discovered_twice",
            first_order=[t["name"] for t in first],
            second_order=[t["name"] for t in second],
            same_content=same_content,
            same_order=same_order,
            fingerprint_first=fingerprint_first,
            fingerprint_second=fingerprint_second,
        )
        declared = [_proxy_from_schema(t) for t in first]
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("How much is milk?"),
        ]
        mock_replies = [
            AIMessage("", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c2"}]),
            AIMessage("Milk is 1.20."),
        ]
        _round_trip(trace, model_kind, messages, declared, mock_replies, turn_cap=2)


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch23_mcp", model_kind=model_kind)

    tool_discovered_not_declared(trace, model_kind)
    resource_lands_without_a_round_trip(trace, model_kind)
    prompt_seeds_the_conversation(trace, model_kind)
    prompt_and_tool_are_two_separate_decisions(trace, model_kind)
    dispatch_table_order_is_not_a_promise(trace, model_kind)

    # Five independent scenarios, not one loop -- `ended` has no single
    # value to report, the same reasoning `ch22` used for the same shape.
    trace.close(turns=0, messages=0, ended="n/a")
    return trace
