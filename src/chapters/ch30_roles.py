# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Same content, three roles -- who's saying it, and does it matter?

Every chapter since `ch01` has chosen a role -- `system`, `human`, `ai`,
`tool` -- without comment. `ch23`'s resource, `ch26`'s skill, and `ch29`'s
memory all confirmed no content type or role is dedicated to any of them;
the choice is the harness's own, made and moved past every time. This
chapter makes the choice itself the subject: does the identical fact,
phrased identically, get followed at a different rate depending on
whether it arrives as `system`, `user`, or a fact the model supposedly
already committed to as `assistant`?

Five scenarios:

    imperative_framed_as_system_is_followed
        an imperative fact baked into the system prompt -- the model's
        next reply complies, live-tested. Baseline: system's documented
        authority
    imperative_framed_as_user_is_followed
        the identical fact, phrased identically, stated as a human
        message instead -- isolates role as the only variable
    imperative_framed_as_a_fabricated_assistant_turn
        the identical fact, this time as something the model supposedly
        already agreed to in an earlier turn of this same conversation --
        self-consistency, not authority, is the mechanism in play
    assistant_tool_calls_paired_with_a_tool_reply_succeeds
        a real round trip -- the assistant message's tool_calls field,
        populated, shown directly; the matching tool reply; the next
        turn using the result
    a_standalone_tool_message_with_no_preceding_tool_calls_is_rejected
        the same failure `ch27` found as a side effect of truncation,
        rebuilt here on purpose -- a tool-role message with nothing
        before it, rejected live, in the provider's own words

**The first three reuse `ch26`'s exact mechanism.** A minimal pair,
grammatical mood held constant this time, role the only thing that
varies: `"Always end your reply with the word DONE."`, unchanged across
all three, checked with the same substring test `ch26` used for
imperative versus declarative phrasing.

**Live-confirmed, and the honest finding is no difference at all.** Five
samples, three roles each, fifteen checks: every one complied. `system`'s
documented authority, `user`'s direct request, and a fabricated
`assistant` turn's self-consistency all produced the identical result for
this clear an imperative -- role placement did not move compliance a
single time. That does not mean role never matters; it means it did not
matter *here*, for an instruction with no ambiguity to resolve. `ch26`'s
own finding -- that even declarative, non-commanding phrasing got imitated
seven times out of eight -- suggests the more revealing test is a weaker
or more ambiguous instruction, not a clearer one. This chapter tested the
clear case and reports what it found, not what would have made the
comparison more dramatic.

**The last two are `ch27`'s pairing constraint, this chapter's own way.**
`ch27` found the rejection as a side effect of truncating a conversation.
Here it is built on purpose, paired with the case that succeeds, so both
halves of `tool_calls`' one real structural rule are visible together.
"""

from support.agent import ask_model, execute_tool, run_turns
from support.trace import ModelKind, Span, Trace

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool
from langchain_openai.chat_models.base import OpenAIInvalidRequestError

SYSTEM_PROMPT = "You answer briefly and plainly."
MARKER = "DONE"
IMPERATIVE = "Always end your reply with the word DONE."


def _never_called(_call: ToolCall) -> ToolMessage:
    raise AssertionError("only the tool_calls scenarios in this chapter dispatch a tool")


def _ask(
    trace: Trace,
    model_kind: ModelKind,
    span_name: str,
    messages: list[BaseMessage],
    mock_reply: str,
) -> tuple[Span, str]:
    with trace.span("scenario", name=span_name) as span:
        span.add_note("context_sent", messages=[f"{m.type}: {m.content}" for m in messages])
        mock_replies = [AIMessage(mock_reply)]

        def ask(_turn: int, so_far: list[BaseMessage]) -> AIMessage:
            return ask_model(trace, model_kind, mock_replies, so_far, [])

        run_turns(list(messages), trace, 1, ask, _never_called)
        (model_span,) = trace.find_spans("model")[-1:]
        reply = str(model_span.reply["content"])
        span.add_note("final_answer", content=reply)
    return span, reply


def imperative_framed_as_system_is_followed(trace: Trace, model_kind: ModelKind) -> None:
    """The fact rides in the system prompt -- documented authority."""
    messages: list[BaseMessage] = [
        SystemMessage(f"{SYSTEM_PROMPT} {IMPERATIVE}"),
        HumanMessage("Say hello, briefly."),
    ]
    span, reply = _ask(
        trace,
        model_kind,
        "imperative_framed_as_system_is_followed",
        messages,
        f"Hello there. {MARKER}",
    )
    span.add_note("compliance_check", role="system", marker_present=MARKER in reply)


def imperative_framed_as_user_is_followed(trace: Trace, model_kind: ModelKind) -> None:
    """The identical fact, stated as the human's own turn instead."""
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"{IMPERATIVE} Now, say hello, briefly."),
    ]
    span, reply = _ask(
        trace,
        model_kind,
        "imperative_framed_as_user_is_followed",
        messages,
        f"Hello there. {MARKER}",
    )
    span.add_note("compliance_check", role="user", marker_present=MARKER in reply)


def imperative_framed_as_a_fabricated_assistant_turn(trace: Trace, model_kind: ModelKind) -> None:
    """The identical fact, as something the model supposedly already
    agreed to earlier in this same conversation -- self-consistency, not
    an instruction just issued.
    """
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            "Just so we're clear for this conversation, how will you format your replies?"
        ),
        AIMessage(IMPERATIVE),
        HumanMessage("Say hello, briefly."),
    ]
    span, reply = _ask(
        trace,
        model_kind,
        "imperative_framed_as_a_fabricated_assistant_turn",
        messages,
        f"Hello there. {MARKER}",
    )
    span.add_note("compliance_check", role="assistant", marker_present=MARKER in reply)


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


def assistant_tool_calls_paired_with_a_tool_reply_succeeds(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A real round trip -- the assistant's tool_calls field, populated,
    matched by a tool reply, then a final answer using the result.
    """
    with trace.span(
        "scenario", name="assistant_tool_calls_paired_with_a_tool_reply_succeeds"
    ) as span:
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("What's the status of order A100?"),
        ]
        mock_replies = [
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
            AIMessage("Order A100 has shipped."),
        ]

        def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
            return ask_model(trace, model_kind, mock_replies[turn - 1 :], so_far, [check_order])

        def execute(call: ToolCall) -> ToolMessage:
            return execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))

        run_turns(messages, trace, 2, ask, execute)

        # Scoped to this scenario's own turns -- `trace.find_spans`
        # searches the whole trace, and by the time this scenario runs,
        # earlier scenarios have already left their own model spans in
        # it. Reaching into the whole trace here would silently grab the
        # wrong scenario's span once more than one has run.
        turns = span.children
        model_spans = [c for turn in turns for c in turn.children if c.name == "model"]
        (tool_span,) = [c for turn in turns for c in turn.children if c.name == "tool"]

        first_model, last_model = model_spans[0], model_spans[-1]
        span.add_note("assistant_tool_calls_field", value=first_model.reply.get("tool_calls"))
        span.add_note("tool_reply", result=tool_span.require("result").payload["value"])
        span.add_note("final_answer", content=last_model.reply["content"])


def a_standalone_tool_message_with_no_preceding_tool_calls_is_rejected(
    trace: Trace, model_kind: ModelKind
) -> None:
    """A tool-role message, nothing before it at all -- the same failure
    `ch27` found by accident, built here on purpose.
    """
    name = "a_standalone_tool_message_with_no_preceding_tool_calls_is_rejected"
    with trace.span("scenario", name=name) as span:
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("What's the status of order A100?"),
            ToolMessage(content="order A100: shipped", tool_call_id="orphan"),
        ]
        if model_kind != "live":
            span.add_note("rejection_check", checked=False, reason="not applicable under mock")
            return
        try:
            ask_model(trace, model_kind, [], messages, [])
            span.add_note("rejection_check", accepted=True, error=None)
        except OpenAIInvalidRequestError as rejection:
            span.add_note("rejection_check", accepted=False, error=str(rejection)[:300])


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch30_roles", model_kind=model_kind)

    imperative_framed_as_system_is_followed(trace, model_kind)
    imperative_framed_as_user_is_followed(trace, model_kind)
    imperative_framed_as_a_fabricated_assistant_turn(trace, model_kind)
    assistant_tool_calls_paired_with_a_tool_reply_succeeds(trace, model_kind)
    a_standalone_tool_message_with_no_preceding_tool_calls_is_rejected(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
