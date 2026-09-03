# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 15 asserts that a half-arrived tool call still looks well-formed.

The negative test is the point: a fragment parses, and reads as a legitimate
call. Only the raw argument string can tell you it is not one yet.
"""

from chapters import ch15_stream_tools as chapter
from support.trace import Span

import json


def stream_spans() -> list[Span]:
    trace = chapter.run()
    return trace.find_spans("stream")


def chunk_notes(span: Span) -> list[dict[str, object]]:
    return [note.payload for note in span.notes if note.label == "chunk"]


def test_a_fragment_parses_into_a_well_formed_call_anyway() -> None:
    """The chapter. Every fragment reads as a legitimate `tool_calls` entry."""
    notes = chunk_notes(stream_spans()[0])
    incomplete = [n for n in notes if not n["arguments_complete"]]
    assert incomplete, "expected at least one fragment before the arguments finished"
    for note in incomplete:
        parsed = note["parsed_as"]
        assert isinstance(parsed, list)
        # Not empty, not an error, not flagged -- a call, with arguments.
        assert len(parsed) == 1


def test_the_arguments_of_an_early_fragment_are_wrong() -> None:
    notes = chunk_notes(stream_spans()[0])
    first = notes[0]
    assert first["arguments_complete"] is False
    assert first["parsed_as"] == [{}]  # price_of(), with no item at all


def test_the_raw_fragment_is_the_only_thing_that_says_it_is_incomplete() -> None:
    for note in chunk_notes(stream_spans()[0]):
        raw = note["raw_arguments"]
        assert isinstance(raw, str)
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            assert note["arguments_complete"] is False
        else:
            assert note["arguments_complete"] is True


def test_judging_every_chunk_judges_arguments_that_had_not_arrived() -> None:
    span = stream_spans()[0]
    verdicts = [note.payload for note in span.notes if note.label == "judged"]
    early = [v for v in verdicts if not v["knew_it_was_a_fragment"]]
    assert len(early) >= 1
    # And nothing raised -- that is what makes it dangerous rather than noisy.


def test_the_second_scenario_judges_nothing_and_dispatches_once() -> None:
    trace = chapter.run()
    (_, waited) = trace.find_spans("scenario")
    tools = [c for c in waited.children if c.name == "tool"]
    assert len(tools) == 1
    assert tools[0].require("args").payload == {"item": "milk"}
    assert tools[0].require("result").payload["value"] == "1.20 for milk"


def test_only_the_waiting_scenario_calls_the_tool() -> None:
    trace = chapter.run()
    (judged_every_chunk, _) = trace.find_spans("scenario")
    assert [c.name for c in judged_every_chunk.children] == ["model"]
