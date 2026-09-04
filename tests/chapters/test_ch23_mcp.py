# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 23 asserts what changes when a dispatch table, a document, or a
conversation seed arrives from a call instead of a literal -- five
scenarios, one per fact.
"""

from chapters import ch23_mcp as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_the_tool_round_trip_dispatches_through_the_discovered_catalog() -> None:
    span = scenario_span("tool_discovered_not_declared")
    kinds = [c.name for turn in span.children for c in turn.children]
    assert kinds == ["model", "tool", "model"]


def test_the_resource_scenario_has_no_tool_call_at_all() -> None:
    span = scenario_span("resource_lands_without_a_round_trip")
    kinds = [c.name for turn in span.children for c in turn.children]
    assert kinds == ["model"]


def test_the_resource_content_reached_the_model_as_a_plain_message() -> None:
    span = scenario_span("resource_lands_without_a_round_trip")
    (model,) = span.children[0].children
    contents = [m["content"] for m in model.context]
    assert any("Round every price to the nearest 5p" in c for c in contents)


def test_the_prompt_scenario_has_no_tool_call_either() -> None:
    span = scenario_span("prompt_seeds_the_conversation")
    kinds = [c.name for turn in span.children for c in turn.children]
    assert kinds == ["model"]


def test_the_prompt_seeded_messages_match_what_get_prompt_returned() -> None:
    span = scenario_span("prompt_seeds_the_conversation")
    (model,) = span.children[0].children
    picked = chapter.PROMPT_SERVER.get_prompt("price-check", {"item": "butter"})
    assert [m["content"] for m in model.context] == [m["content"] for m in picked]


def test_the_uri_never_reaches_the_model_only_the_content_does() -> None:
    span = scenario_span("resource_lands_without_a_round_trip")
    (model,) = span.children[0].children
    contents = " ".join(m["content"] for m in model.context)
    assert "file:///pricing/policy.md" not in contents


def test_the_prompt_alone_has_no_tool_declared() -> None:
    span = scenario_span("prompt_seeds_the_conversation")
    (model,) = span.children[0].children
    assert model.require("context").payload["tools"] == []


def test_the_prompt_with_its_tool_attached_calls_it() -> None:
    span = scenario_span("prompt_and_tool_are_two_separate_decisions")
    kinds = [c.name for turn in span.children for c in turn.children]
    assert kinds == ["model", "tool", "model"]


def test_the_prompt_with_its_tool_declares_only_price_of() -> None:
    span = scenario_span("prompt_and_tool_are_two_separate_decisions")
    (model,) = [c for c in span.children[0].children if c.name == "model"]
    declared = model.require("context").payload["tools"]
    assert [t["name"] for t in declared] == ["price_of"]


def test_two_discoveries_agree_on_content_and_disagree_on_order() -> None:
    span = scenario_span("dispatch_table_order_is_not_a_promise")
    note = span.require("discovered_twice")
    assert set(note.payload["first_order"]) == set(note.payload["second_order"])
    assert note.payload["same_content"] is True
    assert note.payload["same_order"] is False


def test_disagreeing_order_produces_a_different_fingerprint() -> None:
    span = scenario_span("dispatch_table_order_is_not_a_promise")
    note = span.require("discovered_twice")
    assert note.payload["fingerprint_first"] != note.payload["fingerprint_second"]


def test_tool_server_never_exposes_its_handlers_through_list_tools() -> None:
    discovered = chapter.TOOL_SERVER.list_tools()
    for entry in discovered:
        assert "handler" not in entry
        assert set(entry) == {"name", "description", "schema"}


def test_resource_server_list_carries_no_content() -> None:
    discovered = chapter.RESOURCE_SERVER.list_resources()
    for entry in discovered:
        assert "content" not in entry


def test_prompt_server_list_carries_no_messages() -> None:
    discovered = chapter.PROMPT_SERVER.list_prompts()
    for entry in discovered:
        assert "messages" not in entry
