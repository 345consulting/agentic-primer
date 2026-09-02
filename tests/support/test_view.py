# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The page must say when a column describes a different version of a chapter.

A stored trace records a past run. Two columns can agree turn for turn and
still have come from different sources -- the page looks coherent while
quietly comparing two different things.
"""

from support.trace import Json, ModelKind, Trace
from support.view import _digest, _render_wire_fields, render, write_book

import json
from pathlib import Path


def _recorded(kind: ModelKind, source: str) -> Json:
    trace = Trace(chapter="ch01_single_call", model_kind=kind, source=source)
    with trace.span("turn", number=1):
        pass
    trace.close(turns=1, messages=0, ended="test")
    return trace.as_json()


def test_a_column_recorded_from_the_current_source_is_not_flagged() -> None:
    current = Trace(chapter="ch01_single_call").source
    page = render({"mock": _recorded("mock", current)})
    assert "different source" not in page
    assert current in page


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
    assert _digest(span) == "(item=milk) -> 2"


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
    assert _digest(turn) == "model x"
