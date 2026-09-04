# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Everything that enters the context is a prompt.

`ch02_tool_call`'s `Literal[...]` finding and `ch04`'s hand-written rejection
sentences were both instances of this, arriving one at a time with no name
for the pattern connecting them. The claim, named now: every piece of text
the model reads and reacts to is a prompt, whatever field carries it into the
request. `system` and `human` are not a special case -- they are the one row
of a three-row table that happens to get reviewed every time it changes.

    authored          example                              reviewed
    per conversation  system, user                          as it is written
    at design time    tool descriptions, argument enums      once, or never
    during the run    tool results, a sub-agent's answer     never, by anyone

`mcp`, `mcp_injection`, `rag_injection`, `skills` are all specific, sharper
instances of the bottom row -- a document, a remote tool list, a web page.
This chapter builds none of them; each needs its own mechanism to exist
first (a document store, a runtime dispatch table), and building a throwaway
version here would leave those chapters nothing new to show. What this
chapter builds is the classification itself, checked against three requests
that are strict supersets of each other.

**The middle row is the one this primer's own instrument missed.** A tool's
docstring becomes its schema `description`; a `Literal` becomes an `enum`
-- both sent to the model provider on every call, neither ever shown in a
`context` note before this chapter. `Span.add_context` now records `tools`
alongside `messages`, and the fix is not cosmetic: a `Literal` declared as a
reusable type alias serializes as a `$ref` into a `$defs` block, so reading
only `tool.args` would have dropped the enum silently -- recording something
that looked complete while missing the one field the chapter is about. Every
trace from `ch09` onward already carries this; it was simply never surfaced.

**Three scenarios, each a strict superset of the one before, because the
bottom row cannot be isolated from the other two.** A tool result only
exists in a real request because a design-time tool was declared and a
per-conversation question asked for it -- there is no way to show "only a
tool result" without the two rows underneath it.

    authored_per_conversation   no tools -- system and human, and nothing
                                 else ever enters the context
    authored_at_design_time     a tool declared, called once, `turn_cap=1` so
                                 the call dispatches but nothing asks again --
                                 the previous plus a docstring and an enum,
                                 sent whether or not the call ever happens
    authored_during_the_run     the same tool, with a second turn -- the
                                 previous plus the call's own result, feeding
                                 into the very next request

`classify` reads a request's own `messages` and `tools` back and labels each
piece by which row produced it -- from the role a message carries or the
field a declaration arrived in, never by guessing from its text. One edge
case worth naming: the model's own prior reply is during-run-authored too,
the same row a tool result sits in, but a stranger member of it -- the
"nobody" who never reviewed it includes the model itself, which has already
moved on to generating something new by the time that reply comes back as
plain text in its own context.
"""

from support.agent import ask_model, execute_tool, run_turns
from support.trace import CONTEXT, Json, ModelKind, Trace

from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

type GroceryItem = Literal["bread", "butter", "chips", "milk"]


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of an item, in pounds."""
    return f"1.20 for {item}"


TOOLS = {"price_of": price_of}

DECLARED_TOOLS = [price_of]


def classify(messages: list[Json], tools: list[Json]) -> list[dict[str, str]]:
    """Every piece of one request's context, labelled by which row of the
    table produced it. The role a message carries, or the field a
    declaration arrived in, already says which bucket it is -- classifying
    it is reading that back, not guessing from the text.
    """
    found: list[dict[str, str]] = []
    for t in tools:
        found.append(
            {
                "what": f"tool description: {t['name']}",
                "bucket": "design_time",
                "reviewed": (
                    "once, when the function was written -- or never, if a dependency wrote it"
                ),
            }
        )
    for message in messages:
        role = message["role"]
        if role in ("system", "human"):
            found.append(
                {
                    "what": f"{role} message",
                    "bucket": "per_conversation",
                    "reviewed": "as it was written",
                }
            )
        elif role == "tool":
            found.append(
                {"what": "tool result", "bucket": "during_run", "reviewed": "never, by anyone"}
            )
        elif role == "ai":
            found.append(
                {
                    "what": "the model's own prior reply",
                    "bucket": "during_run",
                    "reviewed": "never, by anyone -- the model itself has moved on",
                }
            )
    return found


def run_one_scenario(
    trace: Trace,
    model_kind: ModelKind,
    name: str,
    question: str,
    tools: Sequence[BaseTool],
    mock_replies: list[AIMessage],
    turn_cap: int,
) -> list[str]:
    with trace.span("scenario", name=name) as span:
        messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]

        def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
            return ask_model(trace, model_kind, mock_replies[turn - 1 :], so_far, tools)

        def execute(call: ToolCall) -> ToolMessage:
            return execute_tool(call, trace, lambda n, a, _s: TOOLS[n].invoke(a))

        run_turns(messages, trace, turn_cap, ask, execute)

        # The last model call's own context is the fullest one: everything
        # accumulated so far, which is what a reader would actually be
        # handed if they asked "what did the model provider just see".
        (last_model,) = [trace.find_spans("model")[-1]]
        ctx = last_model.require(CONTEXT).payload
        found = classify(ctx["messages"], ctx["tools"])
        # `types` is the chapter's actual claim as data -- which rows of the
        # table this request drew from -- kept separate from `classifications`'
        # per-piece detail so the growth across scenarios reads at a glance
        # instead of requiring a count. Not `ENDED`: that label means "why
        # `run_turns` stopped" everywhere else it appears, and this note is
        # not that -- reusing it made a classification look like a model
        # signal it never was.
        types = sorted({f["bucket"] for f in found})
        span.add_note("classifications", found=found)
        span.add_note("prompts", types=types)
    return types


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch22_prompt_types", model_kind=model_kind)

    scenarios = [
        run_one_scenario(
            trace,
            model_kind,
            "authored_per_conversation",
            "Say hello, briefly.",
            [],
            [AIMessage("Hello! How can I help you today?")],
            turn_cap=1,
        ),
        run_one_scenario(
            trace,
            model_kind,
            "authored_at_design_time",
            "How much is milk?",
            DECLARED_TOOLS,
            [
                AIMessage(
                    "", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}]
                )
            ],
            turn_cap=1,
        ),
        run_one_scenario(
            trace,
            model_kind,
            "authored_during_the_run",
            "How much is milk?",
            DECLARED_TOOLS,
            [
                AIMessage(
                    "", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c2"}]
                ),
                AIMessage("Milk is 1.20."),
            ],
            turn_cap=2,
        ),
    ]

    # `ended` keeps its usual meaning -- this run has no single loop to have
    # stopped, it is three independent scenarios, so there is nothing to
    # report there. What grew, scenario over scenario, goes in `extra`
    # instead of being folded into a field that means something else.
    trace.close(turns=3, messages=0, ended="n/a", prompt_types_by_scenario=scenarios)
    return trace
