# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""What triggers a hit, what triggers a miss, and what a hit actually is.

`ch27`'s last scenario proved compression invalidates the cache prefix, in
passing, on the way to a different point. This chapter is caching itself,
tested directly: what is actually cached, what preserves a hit, and what
specifically breaks one -- against real `usage_metadata` numbers every
time, never assumed.

**What's cached is not "the context" as a blob -- it's the model's
internal attention state for a specific sequence of input tokens.** A
cache lookup finds the longest matching prefix between this request and a
previously-cached one, reuses the computation for that shared portion, and
only computes fresh for whatever diverges after it -- and, found live,
what's *reported* back as `cache_read` is that true match rounded down to
a complete 128-token block, not the exact byte count. A short conversation
can undershoot the first block and report zero even on an exact repeat; a
longer one can clear several blocks, and an edit early enough breaks all
of them, while one late enough breaks none.

Six scenarios, every one live-only -- the mock model never touches a
provider's cache, so mock runs record that the check does not apply and
nothing more:

    identical_prefix_is_a_hit
        two consecutive, byte-identical calls -- the second shows real
        cache_read tokens, confirmed not assumed
    appending_to_the_end_still_hits
        call two is call one's exact messages plus one new turn -- the
        original prefix still hits; only the new suffix costs anything.
        This is what ch27's compression chapter was protecting
    changing_the_system_prompt_misses
        identical everything except one word in the system prompt -- a
        full miss, even though nearly everything else is unchanged. The
        cache is byte-exact, not similarity-based
    reordering_declared_tools_misses
        ch23's foreshadowed finding, tested directly -- same tools, same
        content, different discovery order -- a miss, proving order is
        part of the prefix, not just content
    editing_before_the_cached_boundary_misses_entirely
        an edit whose tokens fall before the last complete cached block
        breaks the match back to zero; the same edit placed after that
        boundary changes nothing. Not the clean "partial hit" this
        scenario set out to show -- the honest finding is what it found
    does_the_cache_go_cold_over_time
        the same identical prefix, with a real delay between calls -- does
        this provider's cache have an observable TTL, or does it stay
        warm across the delay this chapter can afford to wait
"""

from support.agent import ask_model
from support.trace import Json, ModelKind, Span, Trace

import time
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

SYSTEM_PROMPT_TEMPLATE = (
    "You are a support assistant for an online electronics retailer, session {nonce}. "
    "Answer briefly and plainly. Use the tools you are given when a question needs "
    "current data; otherwise answer from what you already know about store "
    "policy. Never make up a policy you have not been told."
)


def _system_prompt(nonce: str) -> str:
    """Every scenario gets its own nonce baked into the system prompt --
    found live: the provider's cache persists across separate calls and
    even separate process runs, so a shared, literal SYSTEM_PROMPT let
    earlier scenarios (and earlier debugging runs) silently warm what a
    later scenario's "cold" first call assumed was untouched. A unique
    prefix per scenario is the only way "call 1" is actually cold.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(nonce=nonce)


def _seed_messages(nonce: str) -> list[BaseMessage]:
    # Long enough to clear more than one cache block -- found live, this
    # provider caches in fixed 128-token blocks, and only a *complete*
    # block is ever reported cached, even on a byte-identical repeat (a
    # 211-token conversation showed cache_read=128, never 211). A short
    # seed can never show a genuinely partial hit, because there is only
    # ever one block for it to land in.
    return [
        SystemMessage(_system_prompt(nonce)),
        HumanMessage("What's the return policy for electronics?"),
        AIMessage("Electronics can be returned within 30 days if unopened, in original packaging."),
        HumanMessage("What about the shipping cost to Canada?"),
        AIMessage(
            "Shipping to Canada is a flat $15 fee, arriving in 5-7 business days by courier."
        ),
        HumanMessage("Do gift cards expire?"),
        AIMessage("No, gift cards never expire and carry no maintenance fee of any kind."),
        HumanMessage("What's the warranty on laptops?"),
        AIMessage("Laptops carry a one-year manufacturer warranty covering parts and labor."),
        HumanMessage("Can I combine two gift cards on one order?"),
        AIMessage("Yes, up to three gift cards can be applied to a single order at checkout."),
        HumanMessage("Do you price-match other retailers?"),
        AIMessage(
            "We price-match major retailers within 14 days of purchase, with proof of price."
        ),
        HumanMessage("Is express shipping available for international orders?"),
        AIMessage("Express shipping is available to most countries for an additional $25 fee."),
    ]


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


@tool
def check_inventory(item: str) -> str:
    """Check how many units of an item are in stock."""
    return f"{item}: 12 in stock"


@tool
def check_hours(day: str) -> str:
    """Check the store's opening hours on a given day."""
    return f"{day}: 9am-6pm"


def _cache_read(reply: AIMessage) -> int:
    usage: Json = dict(reply.usage_metadata or {})
    details: Json = dict(usage.get("input_token_details") or {})
    return int(details.get("cache_read", 0))


def _call(
    trace: Trace,
    model_kind: ModelKind,
    turn: int,
    label: str,
    messages: list[BaseMessage],
    tools: list[BaseTool],
) -> AIMessage:
    """One model call, wrapped in its own turn span -- the same shape
    every other chapter's page renders, so a turn here reads exactly like
    a turn anywhere else: the agent calling the model, once. `label` and
    the resulting `cache_read` are recorded on the turn itself, not just
    summarized at the end, so the impact is visible turn by turn.
    """
    with trace.span("turn", number=turn) as turn_span:
        reply = ask_model(trace, model_kind, [AIMessage("ok")], messages, tools)
        turn_span.add_note("cache_impact", label=label, cache_read=_cache_read(reply))
    return reply


def _skip_under_mock(span: Span) -> None:
    span.add_note("cache_check", checked=False, reason="not applicable under mock")


def _nonce() -> str:
    return uuid.uuid4().hex[:8]


def identical_prefix_is_a_hit(trace: Trace, model_kind: ModelKind) -> None:
    """Two consecutive, byte-identical calls -- the second should show
    real cache_read tokens.
    """
    with trace.span("scenario", name="identical_prefix_is_a_hit") as span:
        if model_kind != "live":
            _skip_under_mock(span)
            return
        messages = _seed_messages(_nonce())
        reply1 = _call(trace, model_kind, 1, "first call, necessarily cold", list(messages), [])
        reply2 = _call(trace, model_kind, 2, "identical repeat", list(messages), [])
        span.add_note(
            "result",
            first_call_cold=_cache_read(reply1) == 0,
            second_call_hit=_cache_read(reply2) > 0,
        )


def appending_to_the_end_still_hits(trace: Trace, model_kind: ModelKind) -> None:
    """Call two is call one's exact messages plus one new turn -- the
    original prefix should still hit.
    """
    with trace.span("scenario", name="appending_to_the_end_still_hits") as span:
        if model_kind != "live":
            _skip_under_mock(span)
            return
        messages = _seed_messages(_nonce())
        _call(trace, model_kind, 1, "first call, necessarily cold", list(messages), [])
        grown = [*messages, HumanMessage("And what's your warranty policy?")]
        reply2 = _call(trace, model_kind, 2, "same prefix plus one new turn", grown, [])
        span.add_note("result", appended_call_hit=_cache_read(reply2) > 0)


def changing_the_system_prompt_misses(trace: Trace, model_kind: ModelKind) -> None:
    """Identical everything except one word in the system prompt -- should
    be a full miss.
    """
    with trace.span("scenario", name="changing_the_system_prompt_misses") as span:
        if model_kind != "live":
            _skip_under_mock(span)
            return
        nonce = _nonce()
        messages = _seed_messages(nonce)
        _call(trace, model_kind, 1, "first call, necessarily cold", list(messages), [])
        edited = list(messages)
        edited[0] = SystemMessage(_system_prompt(nonce).replace("briefly", "concisely"))
        reply2 = _call(trace, model_kind, 2, "system prompt edited by one word", edited, [])
        span.add_note("result", edited_call_missed=_cache_read(reply2) == 0)


def reordering_declared_tools_misses(trace: Trace, model_kind: ModelKind) -> None:
    """Same tools, same content, different discovery order -- proving
    order is part of the prefix, not just content.

    A minimal system+human list, not `_seed_messages()`'s full history --
    found live: once tools are bound, DeepSeek's thinking mode requires
    every prior AIMessage to carry `reasoning_content`, which a
    hand-authored fake reply never has. Tool order is the point here, not
    conversation depth, so there is no reason to carry that risk.
    """
    with trace.span("scenario", name="reordering_declared_tools_misses") as span:
        if model_kind != "live":
            _skip_under_mock(span)
            return
        messages: list[BaseMessage] = [
            SystemMessage(_system_prompt(_nonce())),
            HumanMessage("What's the status of order A100?"),
        ]
        order_a = [check_order, check_inventory, check_hours]
        order_b = [check_hours, check_inventory, check_order]
        _call(trace, model_kind, 1, "first call, necessarily cold", list(messages), order_a)
        reply_same_order = _call(
            trace, model_kind, 2, "same tools, same order", list(messages), order_a
        )
        reply_reordered = _call(
            trace, model_kind, 3, "same tools, reversed order", list(messages), order_b
        )
        span.add_note(
            "result",
            same_order_hit=_cache_read(reply_same_order) > 0,
            reordered_missed=_cache_read(reply_reordered) < _cache_read(reply_same_order),
        )


def editing_before_the_cached_boundary_misses_entirely(trace: Trace, model_kind: ModelKind) -> None:
    """Not a clean "partial hit" -- found live, `cache_read` behaves like
    the true longest matching prefix, rounded *down* to a complete
    128-token block, not a set of independently-cached blocks. Editing a
    message whose tokens fall before the last complete block boundary
    breaks the match entirely, all the way back to zero; editing one
    whose tokens fall after that boundary changes nothing at all. An edit
    near message index 9 landed right on that boundary and flipped
    between runs, sensitive to a token or two of jitter from the nonce
    itself -- found live, not assumed -- so this scenario uses the first
    message after the system prompt (unambiguously early) and the
    second-to-last message (unambiguously late) instead of a borderline
    position, for a result that holds regardless of that jitter.
    """
    name = "editing_before_the_cached_boundary_misses_entirely"
    with trace.span("scenario", name=name) as span:
        if model_kind != "live":
            _skip_under_mock(span)
            return
        messages = _seed_messages(_nonce())
        _call(trace, model_kind, 1, "first call, necessarily cold", list(messages), [])
        reply_full_hit = _call(trace, model_kind, 2, "identical repeat", list(messages), [])
        edited_early = list(messages)
        edited_early[1] = HumanMessage("What's the return policy for kitchen appliances?")
        reply_early_edit = _call(trace, model_kind, 3, "earliest message edited", edited_early, [])
        edited_late = list(messages)
        edited_late[13] = HumanMessage("Is express shipping available for domestic orders?")
        reply_late_edit = _call(
            trace, model_kind, 4, "second-to-last message edited", edited_late, []
        )
        span.add_note(
            "result",
            early_edit_missed_entirely=_cache_read(reply_early_edit) == 0,
            late_edit_still_hits_fully=_cache_read(reply_late_edit) == _cache_read(reply_full_hit),
        )


def does_the_cache_go_cold_over_time(trace: Trace, model_kind: ModelKind) -> None:
    """Same identical prefix, a real delay in between -- an honest report
    of what was actually observed within a delay this chapter can afford,
    not a claim about the provider's real TTL.
    """
    delay_seconds = 45
    with trace.span("scenario", name="does_the_cache_go_cold_over_time") as span:
        if model_kind != "live":
            _skip_under_mock(span)
            return
        messages = _seed_messages(_nonce())
        _call(trace, model_kind, 1, "first call, necessarily cold", list(messages), [])
        time.sleep(delay_seconds)
        reply2 = _call(
            trace, model_kind, 2, f"identical repeat after {delay_seconds}s", list(messages), []
        )
        span.add_note(
            "result",
            still_warm_after_delay=_cache_read(reply2) > 0,
            note=f"observed only across {delay_seconds}s -- not proof of the provider's real TTL",
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch28_caching", model_kind=model_kind)

    identical_prefix_is_a_hit(trace, model_kind)
    appending_to_the_end_still_hits(trace, model_kind)
    changing_the_system_prompt_misses(trace, model_kind)
    reordering_declared_tools_misses(trace, model_kind)
    editing_before_the_cached_boundary_misses_entirely(trace, model_kind)
    does_the_cache_go_cold_over_time(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
