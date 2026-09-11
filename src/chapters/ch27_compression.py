# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The list is too long; what do you drop, and what does it cost?

Every chapter since `ch01` has sent the whole message list, every call,
because the model provider remembers nothing between requests. That list
only grows. Eventually something has to leave it -- this chapter is what,
when, and what it actually costs, not assumed but measured against a fixed
three-round seed conversation every scenario starts from.

Ten scenarios:

    truncating_by_position_breaks_the_pairing
        naive index-based truncation can isolate a ToolMessage from the
        AIMessage.tool_calls that requested it -- `ch04`'s own pairing
        constraint, violated by construction, rejected by the provider
    truncating_by_turn_preserves_pairing_but_still_loses_information
        structure-aware truncation -- drop whole rounds, pairing intact,
        request succeeds. But the dropped round's fact is gone: ask for it
        and the model cannot answer
    tool_results_are_compressed_the_system_prompt_never_is
        `ch22`'s classification, proven -- only during-run tool results
        shrink; the system prompt and every declared tool's description
        are measured byte-for-byte unchanged
    recent_turns_stay_verbatim_older_ones_shrink
        a second, independent axis -- age, not role. Old rounds shrink
        whole, including the model's own worded reply, not just their raw
        tool result
    summarization_costs_a_model_call_and_still_loses_detail
        replacing old rounds with a summary is a real extra model call,
        its own span in the trace -- and a specific fact is still absent
        from what comes back
    compression_too_aggressive_breaks_the_very_next_turn
        not eventual loss -- the most recent result is compressed, and the
        very next question needed exactly that detail
    reactive_compression_only_fires_when_actually_needed
        proactive (compress on a fixed schedule) vs. reactive (compress
        only near a real threshold) -- reactive does nothing to a short
        conversation that never needed it
    compression_invalidates_the_cache_prefix
        every trace since `ch01` has carried real prompt-cache usage
        numbers, unremarked. Two identical calls hit cache; compress once,
        and the next call's usage shows a real miss -- live only
    a_pinned_message_is_never_compressed_even_when_eligible
        a new named point for `ch18`'s hook taxonomy -- pre-eviction. A
        guard vetoes compressing anything marked pinned, same `HookVerdict`
        shape, one point later in the pipeline
    only_relevant_tools_are_declared_this_turn
        compression is not only about history -- `ch23`'s dispatch table
        can itself be the dominant cost. Filtering to the relevant few
        tools instead of declaring a whole catalog is compression applied
        to the design-time row instead of during-run

**The seed conversation is the same shape for every scenario, on purpose.**
Three rounds, each asking about one order by id, each producing a real
tool call and a real result with a fact only that round knows -- so "did
this scenario lose information" is a checkable question, not a matter of
reading a summary and taking its word for it.
"""

from support.agent import ask_model, execute_tool, run_turns
from support.hooks import HookVerdict
from support.trace import ModelKind, Trace

from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool
from langchain_openai.chat_models.base import OpenAIInvalidRequestError

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

ORDERS: dict[str, str] = {
    "A100": "shipped, tracking XYZ001, via ground freight, arriving within five to seven days",
    "B200": "processing, ETA 2 days, awaiting final inspection before leaving the warehouse",
    "C300": "delivered on Monday, signed for at the front desk, condition confirmed undamaged",
}
ORDER_IDS = list(ORDERS)

TOOL_CATALOG: dict[str, str] = {
    "check_order": "Check the status of an order by id.",
    "check_inventory": "Check how many units of an item are in stock.",
    "check_store_hours": "Check the store's opening hours on a given day.",
    "check_return_policy": "Check the return window for a purchase.",
    "check_shipping_rate": "Check the shipping cost to a given region.",
    "check_gift_card_balance": "Check the remaining balance on a gift card.",
    "check_loyalty_points": "Check a customer's loyalty point balance.",
}


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return ORDERS[order_id]


def _never_called(_call: ToolCall) -> ToolMessage:
    raise AssertionError("no scenario in this chapter calls a tool other than check_order")


def _seed_conversation(trace: Trace, model_kind: ModelKind) -> list[BaseMessage]:
    """Three rounds, each a real tool round trip about one order. A round
    is exactly four messages: human, ai-with-tool-calls, tool, ai-reply.
    """
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    for i, order_id in enumerate(ORDER_IDS):
        messages.append(HumanMessage(f"What's the status of order {order_id}?"))
        mock_replies = [
            AIMessage(
                "",
                tool_calls=[{"name": "check_order", "args": {"order_id": order_id}, "id": f"c{i}"}],
            ),
            AIMessage(f"Order {order_id}: {ORDERS[order_id]}."),
        ]

        def ask(
            turn: int, so_far: list[BaseMessage], mock_replies: list[AIMessage] = mock_replies
        ) -> AIMessage:
            return ask_model(trace, model_kind, mock_replies[turn - 1 :], so_far, [check_order])

        def execute(call: ToolCall) -> ToolMessage:
            return execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))

        run_turns(messages, trace, 2, ask, execute)
    return messages


def _ask_one_more(
    trace: Trace, model_kind: ModelKind, messages: list[BaseMessage], question: str, mock_reply: str
) -> str:
    """Append a new human question to a finished message list, ask once
    more, return the reply text. Used to check what survived compression.
    """
    messages.append(HumanMessage(question))

    def ask(_turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(trace, model_kind, [AIMessage(mock_reply)], so_far, [])

    run_turns(messages, trace, 1, ask, _never_called)
    (model_span,) = trace.find_spans("model")[-1:]
    return str(model_span.reply["content"])


def truncating_by_position_breaks_the_pairing(trace: Trace, model_kind: ModelKind) -> None:
    """Keep only the last two messages by raw index -- lands exactly
    between the final round's tool result and its own tool call, orphaning
    the result. The provider rejects the malformed request.
    """
    with trace.span("scenario", name="truncating_by_position_breaks_the_pairing") as span:
        messages = _seed_conversation(trace, model_kind)
        truncated = messages[-2:]
        span.add_note(
            "truncated",
            kept_roles=[m.type for m in truncated],
            orphaned_tool_message="tool" in [m.type for m in truncated],
        )
        if model_kind == "live":
            try:
                ask_model(trace, model_kind, [], truncated, [])
                span.add_note("rejected", accepted=True, error=None)
            except OpenAIInvalidRequestError as rejection:
                span.add_note("rejected", accepted=False, error=str(rejection)[:300])
        else:
            span.add_note("rejected", accepted=None, error="not applicable under mock")


def truncating_by_turn_preserves_pairing_but_still_loses_information(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Drop the oldest whole round -- pairing intact, request succeeds,
    but the dropped round's fact is genuinely gone.
    """
    name = "truncating_by_turn_preserves_pairing_but_still_loses_information"
    with trace.span("scenario", name=name) as span:
        messages = _seed_conversation(trace, model_kind)
        # System (1) + rounds of 4 -- drop the oldest round, keep the rest.
        dropped_order = ORDER_IDS[0]
        truncated = [messages[0], *messages[5:]]
        span.add_note("truncated", dropped_round_for=dropped_order, remaining=len(truncated))
        reply = _ask_one_more(
            trace,
            model_kind,
            truncated,
            f"Without looking anything up again, just from what we already discussed: "
            f"what was the status of order {dropped_order}?",
            "I don't have any record of that order.",
        )
        span.add_note(
            "recall_check",
            order=dropped_order,
            fact_present=ORDERS[dropped_order].split(",")[0] in reply,
        )


def tool_results_are_compressed_the_system_prompt_never_is(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Only during-run tool results shrink -- the system prompt and every
    declared tool's description are measured, not assumed, unchanged.
    """
    name = "tool_results_are_compressed_the_system_prompt_never_is"
    with trace.span("scenario", name=name) as span:
        messages = _seed_conversation(trace, model_kind)
        before_system = str(messages[0].content)
        before_tool_chars = sum(len(str(m.content)) for m in messages if m.type == "tool")

        compressed: list[BaseMessage] = []
        for m in messages:
            if isinstance(m, ToolMessage):
                compressed.append(
                    ToolMessage(content="[older result omitted]", tool_call_id=m.tool_call_id)
                )
            else:
                compressed.append(m)

        after_system = str(compressed[0].content)
        after_tool_chars = sum(len(str(m.content)) for m in compressed if m.type == "tool")
        span.add_note(
            "compressed",
            system_unchanged=before_system == after_system,
            tool_declaration_name=check_order.name,
            tool_declaration_description=check_order.description,
            tool_chars_before=before_tool_chars,
            tool_chars_after=after_tool_chars,
        )


def recent_turns_stay_verbatim_older_ones_shrink(trace: Trace, model_kind: ModelKind) -> None:
    """A second axis, independent of role -- age. The oldest round shrinks
    whole, including the model's own worded reply, not just its raw
    result. The newest round stays exactly as it was.
    """
    name = "recent_turns_stay_verbatim_older_ones_shrink"
    with trace.span("scenario", name=name) as span:
        messages = _seed_conversation(trace, model_kind)
        # System (1) + round 0 (messages[1:5]) + round 1 (messages[5:9]) +
        # round 2 (messages[9:13]). Shrink round 0 whole; keep the rest.
        shrunk: list[BaseMessage] = [
            messages[0],
            HumanMessage("[earlier round omitted for length]"),
        ]
        shrunk.extend(messages[5:])
        newest_reply = str(messages[-1].content)
        span.add_note(
            "shrunk",
            oldest_round_kept_verbatim=False,
            newest_reply_kept_verbatim=str(shrunk[-1].content) == newest_reply,
            message_count_before=len(messages),
            message_count_after=len(shrunk),
        )


def summarization_costs_a_model_call_and_still_loses_detail(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Replacing the oldest round with a summary is a real extra model
    call -- its own span -- and the exact tracking number still doesn't
    survive into the summary text.
    """
    name = "summarization_costs_a_model_call_and_still_loses_detail"
    with trace.span("scenario", name=name) as span:
        messages = _seed_conversation(trace, model_kind)
        models_before = len(trace.find_spans("model"))
        to_summarize = messages[1:5]
        summarize_prompt: list[BaseMessage] = [
            SystemMessage("Summarize this exchange in one short sentence."),
            *to_summarize,
        ]
        mock_summary = AIMessage("The customer asked about an order and got a status update.")

        def ask(_turn: int, so_far: list[BaseMessage]) -> AIMessage:
            return ask_model(trace, model_kind, [mock_summary], so_far, [])

        summary_reply = ask(1, summarize_prompt)
        models_after = len(trace.find_spans("model"))
        span.add_note("summarized", extra_model_calls=models_after - models_before)
        detail = ORDERS[ORDER_IDS[0]].split(",")[1].strip()  # e.g. "tracking XYZ001"
        span.add_note(
            "detail_check", detail=detail, detail_present=detail in str(summary_reply.content)
        )


def compression_too_aggressive_breaks_the_very_next_turn(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Not eventual loss -- the most recent result, compressed, and the
    very next question needed exactly that detail. Compressing only the
    raw tool result isn't enough on its own -- the model's own worded
    reply from that round repeats the fact and survives untouched unless
    it is compressed too, found live: the first version of this scenario
    only touched the ToolMessage, and the fact came right back from the
    AIMessage sitting next to it.
    """
    name = "compression_too_aggressive_breaks_the_very_next_turn"
    with trace.span("scenario", name=name) as span:
        messages = _seed_conversation(trace, model_kind)
        last_order = ORDER_IDS[-1]
        # The last round is [..., tool, ai-reply] -- compress both, since
        # the worded reply restates the same fact the tool result carried.
        compressed = list(messages)
        original = compressed[-2]
        assert isinstance(original, ToolMessage)
        compressed[-2] = ToolMessage(content="[result omitted]", tool_call_id=original.tool_call_id)
        compressed[-1] = AIMessage("[response omitted]")
        reply = _ask_one_more(
            trace,
            model_kind,
            compressed,
            f"Without looking anything up again, repeat back the exact status you just gave me "
            f"for order {last_order}.",
            "I no longer have that information.",
        )
        span.add_note(
            "immediate_recall_check",
            order=last_order,
            fact_present=ORDERS[last_order].split(",")[0] in reply,
        )


def reactive_compression_only_fires_when_actually_needed(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """A short conversation, well under any real threshold -- reactive
    compression (fires only near the limit) leaves it untouched; proactive
    (fires on a fixed schedule) compresses it anyway, unnecessarily.
    """
    name = "reactive_compression_only_fires_when_actually_needed"
    with trace.span("scenario", name=name) as span:
        short: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage("What's the status of order A100?"),
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c0"}]
            ),
            ToolMessage(content=ORDERS["A100"], tool_call_id="c0"),
            AIMessage(f"Order A100: {ORDERS['A100']}."),
        ]
        threshold_chars = 5000
        actual_chars = sum(len(str(m.content)) for m in short)

        def reactive(messages: list[BaseMessage]) -> list[BaseMessage]:
            if sum(len(str(m.content)) for m in messages) < threshold_chars:
                return messages
            return [messages[0], *messages[-2:]]

        def proactive_every_round(messages: list[BaseMessage]) -> list[BaseMessage]:
            return [messages[0], *messages[-2:]]

        reactive_result = reactive(short)
        proactive_result = proactive_every_round(short)
        span.add_note(
            "policy_comparison",
            conversation_chars=actual_chars,
            threshold_chars=threshold_chars,
            reactive_did_nothing=reactive_result == short,
            proactive_compressed_anyway=len(proactive_result) < len(short),
        )


def compression_invalidates_the_cache_prefix(trace: Trace, model_kind: ModelKind) -> None:
    """Two identical calls hit cache; compress once, and the next call's
    usage shows a real miss. Live only -- the mock model never touches a
    provider's cache at all.
    """
    name = "compression_invalidates_the_cache_prefix"
    with trace.span("scenario", name=name) as span:
        if model_kind != "live":
            span.add_note("cache_check", checked=False, reason="not applicable under mock")
            return
        messages = _seed_conversation(trace, model_kind)
        reply1 = ask_model(trace, model_kind, [], list(messages), [])
        (span1,) = trace.find_spans("model")[-1:]
        usage1 = dict(reply1.usage_metadata or {})
        # Identical messages again -- should be a warmer cache than the first.
        reply2 = ask_model(trace, model_kind, [], list(messages), [])
        (span2,) = trace.find_spans("model")[-1:]
        usage2 = dict(reply2.usage_metadata or {})
        compressed = [messages[0], *messages[5:]]
        reply3 = ask_model(trace, model_kind, [], compressed, [])
        usage3 = dict(reply3.usage_metadata or {})
        del span1, span2
        span.add_note("usage_call_1", usage=usage1)
        span.add_note("usage_call_2_same_messages", usage=usage2)
        span.add_note("usage_call_3_after_compression", usage=usage3)


def _protect_pinned(pinned_id: str) -> Callable[[ToolMessage], HookVerdict]:
    def check(message: ToolMessage) -> HookVerdict:
        if message.tool_call_id == pinned_id:
            return HookVerdict(veto=f"tool_call_id {pinned_id!r} is pinned")
        return HookVerdict()

    return check


def a_pinned_message_is_never_compressed_even_when_eligible(
    trace: Trace, model_kind: ModelKind
) -> None:
    """Pre-eviction -- a new named point for the hook taxonomy. A guard
    vetoes compressing anything pinned, same HookVerdict shape as every
    hook since ch18, one point later in the pipeline.
    """
    name = "a_pinned_message_is_never_compressed_even_when_eligible"
    with trace.span("scenario", name=name) as span:
        messages = _seed_conversation(trace, model_kind)
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        pinned_id = tool_messages[0].tool_call_id
        guard = _protect_pinned(pinned_id)

        compressed: list[BaseMessage] = []
        for m in messages:
            if not isinstance(m, ToolMessage):
                compressed.append(m)
                continue
            verdict = guard(m)
            span.add_note(
                "hook",
                when="pre_eviction",
                by="_protect_pinned",
                tool_call_id=m.tool_call_id,
                veto=verdict.veto,
            )
            if verdict.veto is not None:
                compressed.append(m)
            else:
                compressed.append(
                    ToolMessage(content="[older result omitted]", tool_call_id=m.tool_call_id)
                )

        pinned_survived = any(
            isinstance(m, ToolMessage)
            and m.tool_call_id == pinned_id
            and str(m.content) == ORDERS[ORDER_IDS[0]]
            for m in compressed
        )
        others_compressed = sum(
            1 for m in compressed if m.type == "tool" and str(m.content) == "[older result omitted]"
        )
        span.add_note(
            "result", pinned_survived=pinned_survived, others_compressed=others_compressed
        )


def _select_relevant_tools(question: str, catalog: dict[str, str]) -> list[str]:
    """Plain keyword overlap between the question and each tool's name --
    the point is that filtering happens at all, not how smart it is.
    """
    words = set(question.lower().replace("?", "").split())
    scored = [(name, len(words & set(name.split("_")))) for name in catalog]
    return [name for name, score in scored if score > 0] or [next(iter(catalog))]


def only_relevant_tools_are_declared_this_turn(trace: Trace, _model_kind: ModelKind) -> None:
    """A whole catalog declared every turn costs more the bigger the
    catalog gets -- filtering to the relevant few keeps the declared cost
    flat regardless of how large the real catalog is.
    """
    name = "only_relevant_tools_are_declared_this_turn"
    with trace.span("scenario", name=name) as span:
        question = "What's the status of my order?"
        full_chars = sum(len(n) + len(d) for n, d in TOOL_CATALOG.items())
        selected = _select_relevant_tools(question, TOOL_CATALOG)
        selected_chars = sum(len(n) + len(TOOL_CATALOG[n]) for n in selected)
        span.add_note(
            "filtered",
            catalog_size=len(TOOL_CATALOG),
            declared_full_chars=full_chars,
            selected=selected,
            declared_selected_chars=selected_chars,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch27_compression", model_kind=model_kind)

    truncating_by_position_breaks_the_pairing(trace, model_kind)
    truncating_by_turn_preserves_pairing_but_still_loses_information(trace, model_kind)
    tool_results_are_compressed_the_system_prompt_never_is(trace, model_kind)
    recent_turns_stay_verbatim_older_ones_shrink(trace, model_kind)
    summarization_costs_a_model_call_and_still_loses_detail(trace, model_kind)
    compression_too_aggressive_breaks_the_very_next_turn(trace, model_kind)
    reactive_compression_only_fires_when_actually_needed(trace, model_kind)
    compression_invalidates_the_cache_prefix(trace, model_kind)
    a_pinned_message_is_never_compressed_even_when_eligible(trace, model_kind)
    only_relevant_tools_are_declared_this_turn(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
