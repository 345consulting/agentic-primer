# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The page must say when a column describes a different version of a chapter.

A stored trace records a past run. Two columns can agree turn for turn and
still have come from different sources -- the page looks coherent while
quietly comparing two different things.
"""

from support.trace import Json, ModelKind, Trace
from support.view import (
    _digest,
    _render_column,
    _render_span,
    _render_wire_fields,
    _unit_label,
    render,
    write_book,
)

import json
from pathlib import Path

from langchain_core.messages import AIMessage


def _recorded(kind: ModelKind, source: str) -> Json:
    trace = Trace(chapter="ch01_single_call", model_kind=kind, source=source)
    with trace.span("turn", number=1):
        pass
    trace.close(turns=1, messages=0, ended="test")
    return trace.as_json()


def test_a_column_recorded_from_the_current_source_is_not_flagged() -> None:
    current = Trace(chapter="ch01_single_call").source
    page = render({"mock": _recorded("mock", current)})
    # A matching stamp says nothing, so it is not shown at all -- only a
    # disagreement between the columns is worth a reader's attention.
    assert "different source" not in page
    assert current not in page


def test_a_column_recorded_from_a_different_source_is_flagged() -> None:
    page = render({"mock": _recorded("mock", "deadbeef0000")})
    assert "recorded from different source" in page
    assert "deadbeef0000" in page


def test_an_older_trace_with_no_stamp_at_all_is_flagged() -> None:
    stale = _recorded("live", "x")
    del stale["source"]
    page = render({"live": stale})
    assert "recorded from different source" in page
    assert "unknown" in page


def test_a_header_with_no_recorded_value_says_so_rather_than_rendering_blank() -> None:
    note = {"label": "wire request", "headers": {"authorization": None, "host": "api.example"}}
    fields = _render_wire_fields(note)
    assert "not recorded" in fields
    assert "api.example" in fields
    assert "headers (2)" in fields


def test_headers_are_collapsed_and_the_defaulted_params_are_not() -> None:
    note = {
        "label": "wire request",
        "headers": {"host": "api.example"},
        "defaulted_by_model_provider": ["temperature", "seed"],
    }
    fields = _render_wire_fields(note)
    assert "<details>" in fields
    assert "temperature, seed" in fields
    assert fields.index("defaulted (2)") < fields.index("headers (1)")


def test_a_trace_written_before_values_were_recorded_still_renders() -> None:
    note = {"label": "wire request", "header_names": ["accept", "host"]}
    fields = _render_wire_fields(note)
    assert "headers (2)" in fields
    assert fields.count("not recorded") == 2


def test_the_book_reads_in_chapter_order_not_the_order_of_the_files(tmp_path: Path) -> None:
    for chapter in ("ch02_tool_call", "ch01_single_call"):
        trace = Trace(chapter=chapter, model_kind="mock")
        with trace.span("turn", number=1):
            pass
        trace.close(turns=1, messages=0, ended="test")
        (tmp_path / f"{chapter}.mock.json").write_text(json.dumps(trace.as_json()))
    page = write_book(tmp_path).read_text()
    assert page.index("ch01_single_call") < page.index("ch02_tool_call")


def test_a_chapter_that_has_never_been_run_says_so_in_place(tmp_path: Path) -> None:
    # Going missing would be worse: the book is the reading order, so a gap
    # in it should be visible rather than silently closed up.
    page = write_book(tmp_path).read_text()
    assert "ch01_single_call" in page
    assert "not run" in page


def test_nothing_is_expanded_by_default(tmp_path: Path) -> None:
    # A page that opens everything is a dump. At thirty-five chapters it is
    # unreadable, so every level must be closed and say enough to be skipped.
    (tmp_path / "ch01_single_call.mock.json").write_text(json.dumps(_recorded("mock", "abc")))
    page = write_book(tmp_path).read_text()
    assert "<details open>" not in page


def test_a_closed_span_still_says_what_happened() -> None:
    # The digest is read off notes the span already carries; nothing new is
    # computed, so a closed row cannot claim more than the trace recorded.
    span = {
        "name": "tool",
        "attributes": {"name": "stock_on_hand"},
        "seq": 3,
        "thread": "MainThread",
        "elapsed_ms": 1.0,
        "model_provider_ms": None,
        "library_ms": None,
        "notes": [{"label": "args", "item": "milk"}, {"label": "result", "value": 2}],
        "children": [],
    }
    # Every row ends with what came out, labelled: position alone cannot
    # say whether a value is what went in or what came back.
    assert _digest(span) == "given: item=milk \u00b7 returned: 2"


def test_a_turn_reports_what_its_children_did() -> None:
    child = {
        "name": "model",
        "attributes": {},
        "seq": 2,
        "thread": "MainThread",
        "elapsed_ms": 1.0,
        "model_provider_ms": None,
        "library_ms": None,
        "notes": [{"label": "reply", "message": {"content": "", "tool_calls": [{"name": "x"}]}}],
        "children": [],
    }
    turn = {**child, "name": "turn", "notes": [], "children": [child]}
    # A turn is one invocation plus its tools, so what came out of it is
    # what came out of the model. The tools are one row down.
    assert _digest(turn) == "asked for: x"


def _span_json(name: str, /, **attributes: object) -> Json:
    # Positional-only, for the third time in this codebase: a span's
    # attributes can be called `name`, and the recorder's own signature must
    # not be what stops them. See Trace.span and Span.add_note.
    return {
        "name": name,
        "attributes": attributes,
        "seq": 1,
        "thread": "MainThread",
        "elapsed_ms": 1.0,
        "model_provider_ms": None,
        "library_ms": None,
        "notes": [],
        "children": [],
    }


def test_the_comparison_unit_is_named_by_what_the_chapter_opened() -> None:
    """It used to be hardcoded as "turn N", true only by accident.

    Every chapter so far opens turns at the top level. A chapter that runs the
    same loop under several conditions opens `scenario` spans instead, and the
    page has to say so rather than call them turns.
    """
    turns = {"mock": {"spans": [_span_json("turn", number=1)]}}
    assert _unit_label(turns, 0) == "turn 1"

    scenarios = {"mock": {"spans": [_span_json("scenario", name="length")]}}
    assert _unit_label(scenarios, 0) == "scenario length"


def test_a_unit_present_in_one_kind_and_not_the_other_is_still_named() -> None:
    # The live column may have fewer units than the mock; the heading comes
    # from whichever kind has one, so the rows stay aligned.
    traces = {"live": {"spans": [_span_json("scenario", name="content_filter")]}}
    assert _unit_label(traces, 0) == "scenario content_filter"
    assert "not run" in _render_column("mock", None, 0)


def test_notes_and_children_render_in_the_order_they_were_recorded() -> None:
    """The bug ch21_judge found: a note added between two child spans used
    to render after both children, showing a verdict before the reply it
    judged. `body` fixes that; this asserts the fix, not just that it runs.
    """
    trace = Trace(chapter="test", model_kind="mock")
    with trace.span("turn", number=1) as span:
        with trace.span("model", label="first-model"):
            pass
        span.add_note("judged", marker="first-verdict")
        with trace.span("model", label="second-model"):
            pass
        span.add_note("judged", marker="second-verdict")
    trace.close(turns=1, messages=0, ended="test")

    turn = trace.spans[0].as_json()
    kinds = [entry["kind"] for entry in turn["body"]]
    assert kinds == ["span", "note", "span", "note"]

    rendered = _render_span(turn)
    positions = [
        rendered.index(marker)
        for marker in ("first-model", "first-verdict", "second-model", "second-verdict")
    ]
    assert positions == sorted(positions)


def test_a_trace_recorded_before_body_existed_still_renders() -> None:
    # No "body" key at all -- the shape every out/*.json on disk had before
    # this fix. Falls back to notes-then-children rather than crashing.
    span: Json = {
        "name": "turn",
        "attributes": {"number": 1},
        "seq": 1,
        "thread": "MainThread",
        "elapsed_ms": None,
        "model_provider_ms": None,
        "library_ms": None,
        "notes": [{"label": "ended", "reason": "test"}],
        "children": [],
    }
    assert "ended" in _render_span(span)


def test_the_turn_digest_summarizes_the_last_model_attempt_not_the_first() -> None:
    # ch21_judge's other bug: a turn with two `model` children (a rejected
    # attempt, then an accepted retry) summarized itself from the first --
    # the one that failed, not the one whose outcome the turn actually has.
    trace = Trace(chapter="test", model_kind="mock")
    with trace.span("turn", number=1):
        with trace.span("model") as first:
            first.add_reply(_ai_message_stub("rejected answer"))
        with trace.span("model") as second:
            second.add_reply(_ai_message_stub("accepted answer"))
    trace.close(turns=1, messages=0, ended="test")

    digest = _digest(trace.spans[0].as_json())
    assert "accepted answer" in digest
    assert "rejected answer" not in digest


def _ai_message_stub(content: str) -> AIMessage:
    return AIMessage(content)
