# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The named points in the loop, and the three powers.

`ch10_routing`'s three `if`s were ours, hand-written, one-off: `route()`
decided where a call went, and the decision lived inside `execute_tool`
because that was the only place it could. A hook is the same kind of
decision, generalized into something attachable -- a named point in the
loop, and a list of functions anything can register against it, each one
allowed exactly one of three things:

    observe   reads what is there, returns nothing        cannot change the run
    modify    reads what is there, returns a replacement   changes what happens next
    veto      reads what is there, can refuse it           stops it happening at all

**Four actors could have a `pre`/`post` pair -- tool, model, agent, step --
and this chapter builds only `tool`'s.** A supervisor needs none of its own:
`ch12`'s own finding is that it is an ordinary agent, nested, so hooking one
is hooking `agent` and `tool` together, not a third kind. A workflow does not
act at all -- `ch13`'s finding -- so `step` wraps whatever its body is rather
than being an actor in its own right. `model`, `agent` and `step` get the
same treatment only when a later chapter needs one, the same graduation rule
as everywhere else here.

**`pre` and `post` do not offer the same veto.** Before dispatch, nothing has
happened yet, and a veto refuses the action outright. After dispatch, the
tool already ran -- there is nothing left to prevent, only a consequence
left to stop: the result never reaches the message list, and the loop ends
here instead of continuing on it. Two different powers wearing one name.

**Six scenarios, one hook each, isolating one cell of that grid at a time:**

    tool_pre_observe    logs whether the call is a write; changes nothing
    tool_pre_modify     "  Milk " arrives; "milk" is what gets dispatched
    tool_pre_veto       an item outside the catalog is refused before it runs
    tool_post_observe   logs the result; changes nothing
    tool_post_modify    an internal reference number is stripped before the
                         model ever sees it -- `ch07`'s theme, the other way
    tool_post_veto      the call succeeds and returns an implausible price;
                         the hook withholds it rather than let the loop
                         answer from it. This is `ch02_tool_call`'s oldest
                         finding, with a mechanism now built to catch it: an
                         unconstrained value became a fluent wrong answer
                         because nothing was watching after the call
                         returned. A post-veto is that watch.

`tool_pre_veto`'s catalog check is deliberately the simple, single-call kind
-- the harder case, a bound that only means something across many calls, is
`ch11_state`'s `total_ordered` and `ch19_guards`' reason to exist. This
chapter proves the mechanism; `guards` is where a veto gets a real one.

**A seventh scenario runs three hooks on one call, in order.**
`pre_tool_hooks_compose` chains observe, then modify, then veto -- and the
veto hook never sees the call the model made. It sees what the modify hook
already normalized. `"  Chocolate Milk "` becomes `"chocolate milk"` before
the catalog check ever runs, and it is the normalized string that gets
refused. That is what "hooks compose" actually means: not that more than one
can attach, but that each one's input is the one before it's output, and a
scenario with only one hook per point can never show that.
"""

from support.trace import ModelKind, Trace

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall, tool_call

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

CATALOG = frozenset({"bread", "butter", "chips", "milk"})

WRITE_TOOLS = frozenset({"place_order"})


def price_of(item: str) -> str:
    """The current shelf price of an item, in pounds.

    A plain function, not `@tool` -- nothing here declares a schema to a
    model, because nothing here calls one. Carries an internal reference
    number, the way a real pricing service's response might, so
    `tool_post_modify` has something to redact.
    """
    return f"1.20 for {item} [internal_ref=8821]"


def place_order(item: str, quantity: int) -> str:
    """Order a quantity of an item. No `Literal` enum constrains `item` here
    -- the catalog check below is a hook, not a schema, and a hook needs
    something outside the enum's protection to have anything to refuse.
    """
    return f"ordered {quantity} x {item}"


TOOLS: dict[str, Callable[..., Any]] = {"price_of": price_of, "place_order": place_order}


@dataclass
class HookVerdict:
    """What one hook decided. Independent fields -- a hook may leave a note
    regardless of whether it also modifies or vetoes, and the fields being
    separate is what makes `observe` provably powerless: it can fill `note`
    and nothing else, and nothing else is what changes the run.
    """

    note: str | None = None
    veto: str | None = None
    replacement: dict[str, Any] | None = None


type BeforeHook = Callable[[ToolCall], HookVerdict]
type AfterHook = Callable[[ToolCall, Any], HookVerdict]


def note_if_a_write(call: ToolCall) -> HookVerdict:
    """observe, pre. Reads the call, records what it saw, changes nothing."""
    kind = "write" if call["name"] in WRITE_TOOLS else "read"
    return HookVerdict(note=f"{call['name']} is a {kind}")


def normalize_item(call: ToolCall) -> HookVerdict:
    """modify, pre. Whitespace and case are not the model's to get right."""
    item = call["args"].get("item")
    if not isinstance(item, str):
        return HookVerdict()
    normalized = item.strip().lower()
    if normalized == item:
        return HookVerdict()
    return HookVerdict(replacement={**call["args"], "item": normalized})


def refuse_unlisted_items(call: ToolCall) -> HookVerdict:
    """veto, pre. A single-call check -- ch19_guards' bound needs more calls
    than this one to mean anything; this one does not.
    """
    item = call["args"].get("item")
    if item in CATALOG:
        return HookVerdict()
    return HookVerdict(veto=f"['{item}'] is not in the catalog")


def note_the_result(_call: ToolCall, _result: Any) -> HookVerdict:
    """observe, post. Reads the result, changes nothing."""
    return HookVerdict(note="result observed")


def redact_internal_reference(_call: ToolCall, result: Any) -> HookVerdict:
    """modify, post. `ch07_tool_http` found a credential leak the recorder
    never guarded; this is the same shape run the other way -- a tool's own
    result carrying something that should not reach the model at all.
    """
    text = str(result)
    marker = " [internal_ref="
    if marker not in text:
        return HookVerdict()
    return HookVerdict(replacement={"value": text.split(marker)[0]})


def broken_price_of(item: str) -> str:
    """A pricing service having a bad day -- used only by `tool_post_veto`,
    which needs a price worth refusing rather than the honest one above.
    """
    return f"0.00 for {item}"


def refuse_implausible_price(_call: ToolCall, result: Any) -> HookVerdict:
    """veto, post. The call already happened; what this stops is the
    consequence -- the loop answering from a number that should not be
    trusted. ch02_tool_call's oldest finding, with a mechanism for it now.
    """
    if str(result).startswith("0.00"):
        return HookVerdict(veto=f"price ['{result}'] is implausible; withheld")
    return HookVerdict()


def dispatch_with_hooks(
    call: ToolCall,
    trace: Trace,
    before_hooks: Sequence[BeforeHook],
    after_hooks: Sequence[AfterHook],
    tools: dict[str, Callable[..., Any]] | None = None,
) -> ToolMessage:
    """One call, run through a named point's hooks before and after dispatch.

    Not `execute_tool` from `support/agent.py`: that function has no hook
    points at all, and adding them there would be teaching the mechanism by
    editing scaffolding a reader cannot see change. It graduates once a
    chapter after this one needs it built in rather than built by hand.
    """
    dispatch = tools if tools is not None else TOOLS
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        for hook in before_hooks:
            verdict = hook(call)
            span.add_note(
                "hook",
                when="pre",
                by=hook.__name__,
                note=verdict.note,
                veto=verdict.veto,
                replacement=verdict.replacement,
            )
            if verdict.veto is not None:
                return ToolMessage(
                    content=f"['{call['name']}'] blocked before it ran: {verdict.veto}",
                    tool_call_id=call["id"],
                    status="error",
                )
            if verdict.replacement is not None:
                call = tool_call(name=call["name"], args=verdict.replacement, id=call["id"])

        result: Any = dispatch[call["name"]](**call["args"])
        # Recorded before any post hook runs, so a post-veto's trace proves
        # the call actually happened -- otherwise it would read identically
        # to a pre-veto that stopped it from ever running at all.
        span.add_note("dispatched", value=result)

        for after_hook in after_hooks:
            verdict = after_hook(call, result)
            span.add_note(
                "hook",
                when="post",
                by=after_hook.__name__,
                note=verdict.note,
                veto=verdict.veto,
                replacement=verdict.replacement,
            )
            if verdict.veto is not None:
                return ToolMessage(
                    content=f"['{call['name']}'] ran, but the result was withheld: {verdict.veto}",
                    tool_call_id=call["id"],
                    status="error",
                )
            if verdict.replacement is not None:
                result = verdict.replacement["value"]

        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run(model_kind: ModelKind = "mock") -> Trace:
    """`model_kind` is accepted and never read, the same honest no-op
    `ch13_workflow` used -- nothing here calls a model either.
    """
    trace = Trace(chapter="ch18_hooks", model_kind=model_kind)
    endings: list[str] = []

    scenarios: list[tuple[str, ToolCall, Sequence[BeforeHook], Sequence[AfterHook]]] = [
        (
            "tool_pre_observe",
            tool_call(name="place_order", args={"item": "milk", "quantity": 2}, id="c1"),
            [note_if_a_write],
            [],
        ),
        (
            "tool_pre_modify",
            tool_call(name="place_order", args={"item": "  Milk ", "quantity": 2}, id="c2"),
            [normalize_item],
            [],
        ),
        (
            "tool_pre_veto",
            tool_call(name="place_order", args={"item": "saffron", "quantity": 1}, id="c3"),
            [refuse_unlisted_items],
            [],
        ),
        (
            "tool_post_observe",
            tool_call(name="price_of", args={"item": "milk"}, id="c4"),
            [],
            [note_the_result],
        ),
        (
            "tool_post_modify",
            tool_call(name="price_of", args={"item": "milk"}, id="c5"),
            [],
            [redact_internal_reference],
        ),
        (
            "tool_post_veto",
            tool_call(name="price_of", args={"item": "milk"}, id="c6"),
            [],
            [refuse_implausible_price],
        ),
        (
            "pre_tool_hooks_compose",
            tool_call(
                name="place_order", args={"item": "  Chocolate Milk ", "quantity": 1}, id="c7"
            ),
            [note_if_a_write, normalize_item, refuse_unlisted_items],
            [],
        ),
    ]

    for name, call, before, after in scenarios:
        with trace.span("scenario", name=name) as span:
            # tool_post_veto needs a tool that actually returns an implausible
            # price to refuse -- a distinct dispatch table, local to this one
            # scenario, rather than a special case inside `price_of` itself.
            tools = {"price_of": broken_price_of} if name == "tool_post_veto" else None
            reply = dispatch_with_hooks(call, trace, before, after, tools)
            ended = f"{reply.status}: {reply.content}"
            span.add_note("ended", reason=ended)
        endings.append(ended)

    trace.close(turns=len(scenarios), messages=0, ended="; ".join(endings))
    return trace
