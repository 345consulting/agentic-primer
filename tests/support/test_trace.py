"""The recorder must carry everything the adapter carried.

These exist because of the finding dated 2026-09-01 in FINDINGS.md: a field the
recorder drops is indistinguishable from a field the provider never sent.
"""

from support.trace import UNKNOWN_SOURCE, Json, Trace, source_hash

from langchain_core.messages import AIMessage


def _reply(message: AIMessage) -> Json:
    trace = Trace(chapter="test")
    with trace.span("model") as span:
        span.reply(message)
    trace.close()
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
            trace.close()
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
