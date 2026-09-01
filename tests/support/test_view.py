"""The page must say when a column describes a different version of a chapter.

A stored trace records a past run. Two columns can agree turn for turn and
still have come from different sources -- the page looks coherent while
quietly comparing two different things.
"""

from support.trace import Json, Trace
from support.view import _wire_fields, render


def _recorded(kind: str, source: str) -> Json:
    trace = Trace(chapter="ch01_single_call", kind=kind, source=source)
    with trace.span("turn", number=1):
        pass
    trace.close(turns=1)
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
    fields = _wire_fields(note)
    assert "not recorded" in fields
    assert "api.example" in fields
    assert "headers (2)" in fields


def test_headers_are_collapsed_and_the_defaulted_params_are_not() -> None:
    note = {
        "label": "wire request",
        "headers": {"host": "api.example"},
        "defaulted_by_provider": ["temperature", "seed"],
    }
    fields = _wire_fields(note)
    assert "<details>" in fields
    assert "temperature, seed" in fields
    assert fields.index("defaulted (2)") < fields.index("headers (1)")


def test_a_trace_written_before_values_were_recorded_still_renders() -> None:
    note = {"label": "wire request", "header_names": ["accept", "host"]}
    fields = _wire_fields(note)
    assert "headers (2)" in fields
    assert fields.count("not recorded") == 2
