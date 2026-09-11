# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""A hook that reads the reply, per turn and not per run.

`ch18`'s hooks and `ch19`'s guard both watch a *tool* -- a call about to
dispatch, a result about to be trusted. A judge watches the *model*: the
reply itself, before anything downstream decides what to do with it. That is
the actor `ch18` deferred on purpose, and it needs its own, narrower
mechanism rather than reusing `HookVerdict`.

**A judge only ever passes or fails. There is no `replacement`.** Guards and
the tool hooks in `ch18` can rewrite what happens next; a judge's `Verdict`
has nowhere to put a rewrite even if it wanted to -- `passed: bool` and
`reason: str | None`, nothing else. The restriction is structural, the same
way `observe`'s powerlessness was: not a rule a judge chooses to follow, a
field that does not exist to break it.

**A rejection retries the model, and it is `ch09_who_retries`'
"bad arguments" cell, not its "transient" one.** These are not failures a
blind retry could fix by luck -- the model chose 500, and asking the exact
same question again would produce 500 again. `ch09`'s own rule for that
cell is "the model retries, once -- it can fix them", and fixing requires
being told what was wrong. So a rejection appends the rejected reply and a
correction -- a `ToolMessage` naming the problem if the reply asked for a
tool, a plain message if it did not -- before the next attempt asks again.
`MAX_ATTEMPTS = 3`, the same number `ch09`'s own retry policy uses.

**Three scenarios:**

    judge_rejects_a_tool_call        "order a couple bottles" gets a first
                                      reply asking for 500 -- rejected, not
                                      because the schema disallows it (an
                                      int is an int) but because it does not
                                      match what was asked. The retry asks
                                      for 2, and passes.
    judge_rejects_a_final_answer     a reply that never mentions milk,
                                      answering a question about its price,
                                      rejected for the same reason a tool
                                      result post-veto is -- content that
                                      does not meet what was expected. The
                                      retry answers the actual question.
    judge_gives_up_after_max_attempts every attempt asks for 500. Three
                                      strikes, then the honest thing `ch09`'s
                                      matrix always did for a permanent
                                      failure: say so and stop, rather than
                                      accepting the fourth wrong answer
                                      because the budget for judging it ran
                                      out. Scripted to fail every time on
                                      purpose -- nothing about "a couple
                                      bottles" is actually unreasonable, so a
                                      live model corrected once may simply
                                      get it right and never reach this
                                      ending at all.

The first two are `ch18_hooks`' two content-validation shapes -- a call that
looks reasonable to the schema but is not, a result that looks complete but
is missing something -- run one layer up, against the model's own reply
instead of a tool's.

**The live run passed all three on the first attempt -- none of them
rejects anything live.** `judge_rejects_a_tool_call` asks for a sensible
quantity immediately. `judge_rejects_a_final_answer`'s first reply already
says "milk" several times over, discussing its price, so the groundedness
check passes before a second attempt is ever needed. `judge_gives_up_after_
max_attempts` never produces the 500 the mock scripts on purpose. Every
scenario's name describes what the *mock* is built to demonstrate, not a
guarantee about what a live model will do -- the same limit every
scripted-adversary scenario in this primer has had since `ch09`. The first
version of this chapter did not know that and asserted the mock's outcome
unconditionally; the live run crashed it immediately, which is exactly the
failure mode `ask_and_judge` not telling the model why it was rejected
should have predicted -- a model with no memory of a failure has no reason
to behave differently the second time, and every attempt here was, in
effect, a first attempt.
"""

from support.agent import ask_model
from support.trace import ENDED, ModelKind, Span, Trace

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

MAX_ATTEMPTS = 3

REASONABLE_MAX_QUANTITY = 20


@tool
def place_order(item: GroceryItem, quantity: int) -> str:
    """Order a quantity of an item."""
    raise AssertionError(
        f"place_order ['{quantity}' x '{item}'] is declared for its schema; "
        "submit_order below runs, once the judge accepts it"
    )


def submit_order(item: str, quantity: int) -> str:
    return f"ordered {quantity} x {item}"


TOOLS = {"place_order": submit_order}

DECLARED_TOOLS = [place_order]


@dataclass
class Verdict:
    """A judge's whole vocabulary. No `replacement` field -- a judge cannot
    rewrite a reply, only accept or reject it, and there is nowhere on this
    type to put a rewrite even by mistake.
    """

    passed: bool
    reason: str | None = None


type Judge = Callable[[AIMessage], Verdict]


def judge_tool_call_is_reasonable(reply: AIMessage) -> Verdict:
    """A schema cannot catch this -- `quantity` is a valid `int` either way.
    What is wrong is that 500 does not match "a couple", and only reading
    the call against the question it answers can tell.
    """
    if not reply.tool_calls:
        return Verdict(passed=True)
    call = reply.tool_calls[0]
    quantity = call["args"].get("quantity", 0)
    if call["name"] == "place_order" and quantity > REASONABLE_MAX_QUANTITY:
        return Verdict(passed=False, reason=f"{quantity} is not reasonable for 'a couple'")
    return Verdict(passed=True)


def make_groundedness_judge(must_mention: str) -> Judge:
    """post, model. A reply that never addresses what was asked -- the same
    shape as `ch18`'s post-veto on a tool result, one layer up: content that
    looks complete but is missing the one thing that made it an answer.
    """

    def judge_answer_is_grounded(reply: AIMessage) -> Verdict:
        if reply.tool_calls:
            return Verdict(passed=True)
        if must_mention not in str(reply.content).lower():
            return Verdict(passed=False, reason=f"answer does not mention '{must_mention}'")
        return Verdict(passed=True)

    return judge_answer_is_grounded


def ask_and_judge(
    span: Span,
    trace: Trace,
    model_kind: ModelKind,
    attempts: Sequence[AIMessage],
    messages: list[BaseMessage],
    tools: Sequence[BaseTool],
    judge: Judge,
) -> tuple[AIMessage, int, bool]:
    """Ask, judge the reply, and retry up to `MAX_ATTEMPTS` on a rejection --
    telling the model what was wrong before asking again, the "bad
    arguments" cell's rule, not the "transient" cell's. A tool-call reply
    gets a `ToolMessage` explaining the rejection rather than a dispatched
    result, the same shape a real failure gets, and required by the same
    constraint `ch04` found: an assistant message with `tool_calls` needs a
    result for each one before the next request, or the model provider
    refuses it.
    """
    reply = AIMessage("")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = ask_model(trace, model_kind, [attempts[attempt - 1]], messages, tools)
        verdict = judge(reply)
        span.add_note("judged", attempt=attempt, passed=verdict.passed, reason=verdict.reason)
        if verdict.passed:
            return reply, attempt, True
        if attempt < MAX_ATTEMPTS:
            messages.append(reply)
            if reply.tool_calls:
                for call in reply.tool_calls:
                    messages.append(
                        ToolMessage(
                            content=f"rejected: {verdict.reason}. Try different arguments.",
                            tool_call_id=call["id"],
                            status="error",
                        )
                    )
            else:
                messages.append(HumanMessage(f"That does not work: {verdict.reason}. Try again."))
    return reply, MAX_ATTEMPTS, False


TOOL_CALL_ATTEMPTS = [
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 500}, "id": "c1"}],
    ),
    AIMessage(
        "",
        tool_calls=[{"name": "place_order", "args": {"item": "milk", "quantity": 2}, "id": "c2"}],
    ),
]

FINAL_ANSWER_ATTEMPTS = [
    AIMessage("Sure, happy to help with grocery questions."),
    AIMessage("Milk is 1.20 per bottle."),
]

ALWAYS_ABSURD_ATTEMPTS = [
    AIMessage(
        "",
        tool_calls=[
            {"name": "place_order", "args": {"item": "milk", "quantity": 500}, "id": f"c{i}"}
        ],
    )
    for i in range(1, MAX_ATTEMPTS + 1)
]


def run_tool_call_scenario(trace: Trace, model_kind: ModelKind) -> str:
    with trace.span("scenario", name="judge_rejects_a_tool_call") as span:
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("Order a couple bottles of milk."),
        ]
        with trace.span("turn", number=1) as turn_span:
            reply, attempt, passed = ask_and_judge(
                turn_span,
                trace,
                model_kind,
                TOOL_CALL_ATTEMPTS,
                messages,
                DECLARED_TOOLS,
                judge_tool_call_is_reasonable,
            )
            messages.append(reply)
            if passed and reply.tool_calls:
                (call,) = reply.tool_calls
                with trace.span("tool", name=call["name"], id=call["id"]) as tool_span:
                    tool_span.add_note("args", **call["args"])
                    result = TOOLS[call["name"]](**call["args"])
                    tool_span.add_note("result", value=result)
                ended = f"accepted on attempt {attempt}: {result}"
            else:
                # A live model can pass without ever asking for a tool at
                # all -- "a couple" is answerable in prose. Reported, not
                # forced into the shape the mock was scripted to take.
                ended = f"attempt {attempt}, passed={passed}: {reply.content or 'no tool call'}"
        span.add_note(ENDED, reason=ended)
    return ended


def run_final_answer_scenario(trace: Trace, model_kind: ModelKind) -> str:
    with trace.span("scenario", name="judge_rejects_a_final_answer") as span:
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("How much is milk?"),
        ]
        judge = make_groundedness_judge("milk")
        with trace.span("turn", number=1) as turn_span:
            reply, attempt, passed = ask_and_judge(
                turn_span, trace, model_kind, FINAL_ANSWER_ATTEMPTS, messages, [], judge
            )
            messages.append(reply)
        ended = f"attempt {attempt}, passed={passed}: {reply.content}"
        span.add_note(ENDED, reason=ended)
    return ended


def run_give_up_scenario(trace: Trace, model_kind: ModelKind) -> str:
    with trace.span("scenario", name="judge_gives_up_after_max_attempts") as span:
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("Order a couple bottles of milk."),
        ]
        with trace.span("turn", number=1) as turn_span:
            reply, attempt, passed = ask_and_judge(
                turn_span,
                trace,
                model_kind,
                ALWAYS_ABSURD_ATTEMPTS,
                messages,
                DECLARED_TOOLS,
                judge_tool_call_is_reasonable,
            )
        if passed:
            # The mock is scripted to fail three times on purpose; nothing
            # about "order a couple bottles" is actually unreasonable, and a
            # live model corrected once may just get it right immediately.
            ended = f"accepted on attempt {attempt} instead of giving up"
        else:
            reason = judge_tool_call_is_reasonable(reply).reason
            ended = f"gave up after {attempt} attempts: {reason}"
        span.add_note(ENDED, reason=ended)
    return ended


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch21_judge", model_kind=model_kind)

    endings = [
        run_tool_call_scenario(trace, model_kind),
        run_final_answer_scenario(trace, model_kind),
        run_give_up_scenario(trace, model_kind),
    ]

    trace.close(turns=3, messages=0, ended="; ".join(endings))
    return trace
