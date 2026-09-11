# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 28 is live-only -- the mock model never touches a provider's
cache, so these tests check the mock scaffolding runs cleanly and skips
correctly, not the cache claims themselves (those are asserted by hand
against real usage_metadata in the chapter docstring).
"""

from chapters import ch28_caching as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


SCENARIOS = [
    "identical_prefix_is_a_hit",
    "appending_to_the_end_still_hits",
    "changing_the_system_prompt_misses",
    "reordering_declared_tools_misses",
    "editing_before_the_cached_boundary_misses_entirely",
    "does_the_cache_go_cold_over_time",
]


def test_every_scenario_skips_the_cache_check_under_mock() -> None:
    trace = chapter.run()
    names = {s.attributes["name"] for s in trace.find_spans("scenario")}
    assert names == set(SCENARIOS)
    for span in trace.find_spans("scenario"):
        note = span.require("cache_check")
        assert note.payload["checked"] is False


def test_seed_messages_start_with_a_system_message_carrying_the_nonce() -> None:
    messages = chapter._seed_messages("abc123")
    assert messages[0].type == "system"
    assert "abc123" in str(messages[0].content)


def test_two_nonces_are_different() -> None:
    assert chapter._nonce() != chapter._nonce()


def test_cache_read_defaults_to_zero_with_no_usage_metadata() -> None:
    from langchain_core.messages import AIMessage

    reply = AIMessage("hi")
    assert chapter._cache_read(reply) == 0
