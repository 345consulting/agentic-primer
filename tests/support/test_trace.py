# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The recorder must carry everything the adapter carried.

These exist because of the finding dated 2026-09-01 in LEARNINGS.md: a field the
recorder drops is indistinguishable from a field the provider never sent.
"""

from support.chapters import UNKNOWN_SOURCE, source_hash
from support.trace import Json, Trace

from langchain_core.messages import AIMessage


def _reply(message: AIMessage) -> Json:
    trace = Trace(chapter="test")
    with trace.span("model") as span:
        span.add_reply(message)
    trace.close(turns=0, messages=0, ended="test")
    described: Json = trace.spans[0].notes[0].payload["message"]
    return described


def test_reasoning_is_recorded_because_it_is_billed() -> None:
    described = _reply(AIMessage("answer", additional_kwargs={"reasoning_content": "thinking"}))
    assert described["additional_kwargs"] == {"reasoning_content": "thinking"}


def test_the_providers_own_reason_for_stopping_is_recorded() -> None:
    described = _reply(AIMessage("answer", response_metadata={"finish_reason": "length"}))
    assert described["finish_reason"] == "length"


def test_usage_is_recorded_including_the_reasoning_split() -> None:
    usage = {
        "input_tokens": 102,
        "output_tokens": 55,
        "total_tokens": 157,
        "output_token_details": {"reasoning": 23},
    }
    described = _reply(AIMessage("answer", usage_metadata=usage))
    usage_recorded: Json = described["usage"]
    assert usage_recorded["output_token_details"] == {"reasoning": 23}


def test_closing_with_a_span_still_open_is_an_error() -> None:
    trace = Trace(chapter="test")
    manager = trace.span("leaked")
    manager.__enter__()
    try:
        raised = False
        try:
            trace.close(turns=0, messages=0, ended="test")
        except RuntimeError:
            raised = True
        assert raised
    finally:
        manager.__exit__(None, None, None)


def test_a_real_chapter_is_stamped_with_its_source() -> None:
    trace = Trace(chapter="ch01_single_call")
    assert trace.source == source_hash("ch01_single_call")
    assert trace.source != UNKNOWN_SOURCE
    assert trace.as_json()["source"] == trace.source


def test_a_chapter_with_no_file_stamps_unknown_rather_than_raising() -> None:
    assert Trace(chapter="test").source == UNKNOWN_SOURCE


def test_a_missing_note_names_what_was_actually_recorded() -> None:
    # Returning None would push the defect downstream to whoever unwraps it.
    trace = Trace(chapter="test")
    with trace.span("model") as span:
        span.add_note("context", messages=[])
    trace.close(turns=0, messages=0, ended="test")
    try:
        trace.find_spans("model")[0].require("reply")
    except KeyError as error:
        assert "context" in str(error)
    else:
        raise AssertionError("a missing note must raise")


def test_a_note_can_carry_a_payload_key_called_label() -> None:
    # A tool's argument names are chosen by the model, not by us. A tool with
    # a parameter called `label` must not collide with the recorder's own
    # signature -- the same collision `Trace.span` had with `name`.
    trace = Trace(chapter="test")
    with trace.span("tool") as span:
        span.add_note("args", label="a value", name="another")
    trace.close(turns=0, messages=0, ended="test")
    assert trace.find_spans("tool")[0].require("args").payload == {
        "label": "a value",
        "name": "another",
    }
