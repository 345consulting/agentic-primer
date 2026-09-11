# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 18 asserts the three powers, at both ends of one call, and that
hooks attached to the same point see each other's work.
"""

from chapters import ch18_hooks as chapter
from support.trace import Span

from langchain_core.messages.tool import tool_call


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def hook_notes(span: Span) -> list[dict[str, object]]:
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    return [n.payload for n in tool_span.notes if n.label == "hook"]


# --- the hooks themselves, as pure functions -------------------------------


def test_observe_can_leave_a_note_and_nothing_else() -> None:
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 2}, id="c1")
    verdict = chapter.note_if_a_write(call)
    assert verdict.note is not None
    assert verdict.veto is None
    assert verdict.replacement is None


def test_modify_leaves_an_already_clean_argument_alone() -> None:
    call = tool_call(name="place_order", args={"item": "milk", "quantity": 2}, id="c1")
    verdict = chapter.normalize_item(call)
    assert verdict.replacement is None


def test_modify_normalizes_whitespace_and_case() -> None:
    call = tool_call(name="place_order", args={"item": "  Milk ", "quantity": 2}, id="c1")
    verdict = chapter.normalize_item(call)
    assert verdict.replacement == {"item": "milk", "quantity": 2}


def test_pre_veto_refuses_only_what_is_outside_the_catalog() -> None:
    listed = tool_call(name="place_order", args={"item": "milk", "quantity": 1}, id="c1")
    unlisted = tool_call(name="place_order", args={"item": "saffron", "quantity": 1}, id="c2")
    assert chapter.refuse_unlisted_items(listed).veto is None
    assert chapter.refuse_unlisted_items(unlisted).veto is not None


def test_post_modify_strips_the_internal_reference() -> None:
    call = tool_call(name="price_of", args={"item": "milk"}, id="c1")
    verdict = chapter.redact_internal_reference(call, "1.20 for milk [internal_ref=8821]")
    assert verdict.replacement == {"value": "1.20 for milk"}


def test_post_veto_refuses_only_an_implausible_price() -> None:
    call = tool_call(name="price_of", args={"item": "milk"}, id="c1")
    assert chapter.refuse_implausible_price(call, "1.20 for milk").veto is None
    assert chapter.refuse_implausible_price(call, "0.00 for milk").veto is not None


# --- the mechanism, through a trace -----------------------------------------


def test_pre_observe_does_not_change_the_dispatched_call() -> None:
    span = scenario_span("tool_pre_observe")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    assert tool_span.require("result").payload["value"] == "ordered 2 x milk"


def test_pre_modify_is_what_actually_dispatches() -> None:
    span = scenario_span("tool_pre_modify")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    # The tool ran with the normalized item, not the raw " Milk " it was sent.
    assert tool_span.require("result").payload["value"] == "ordered 2 x milk"


def test_pre_veto_never_reaches_the_tool() -> None:
    span = scenario_span("tool_pre_veto")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    assert tool_span.find("result") is None


def test_post_observe_does_not_change_the_result() -> None:
    span = scenario_span("tool_post_observe")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    assert "[internal_ref=" in str(tool_span.require("result").payload["value"])


def test_post_modify_changes_what_the_result_note_holds() -> None:
    span = scenario_span("tool_post_modify")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    assert tool_span.require("result").payload["value"] == "1.20 for milk"


def test_post_veto_ran_the_tool_but_withheld_the_result() -> None:
    span = scenario_span("tool_post_veto")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    # The call happened -- there is no result note, because it was withheld
    # after the fact, not refused before it.
    assert tool_span.find("result") is None
    hooks = hook_notes(span)
    assert hooks[-1]["veto"] is not None


def test_pre_veto_never_dispatched_but_post_veto_did() -> None:
    # Both endings have no `result` note -- without a `dispatched` note too,
    # the trace could not prove the difference the chapter's whole claim
    # rests on: one call never ran, the other ran and was discarded after.
    pre_tool = next(c for c in scenario_span("tool_pre_veto").children if c.name == "tool")
    post_tool = next(c for c in scenario_span("tool_post_veto").children if c.name == "tool")
    assert pre_tool.find("dispatched") is None
    assert post_tool.find("dispatched") is not None


def test_hooks_compose_and_the_veto_sees_the_modified_value() -> None:
    span = scenario_span("pre_tool_hooks_compose")
    hooks = hook_notes(span)
    assert [h["by"] for h in hooks] == [
        "note_if_a_write",
        "normalize_item",
        "refuse_unlisted_items",
    ]
    # The veto fired on the normalized string, not the one the model sent.
    veto_reason = hooks[-1]["veto"]
    assert isinstance(veto_reason, str)
    assert "chocolate milk" in veto_reason
    assert "Chocolate Milk" not in veto_reason


def test_three_hooks_run_in_order_on_one_call() -> None:
    span = scenario_span("pre_tool_hooks_compose")
    (tool_span,) = [c for c in span.children if c.name == "tool"]
    assert tool_span.find("result") is None
