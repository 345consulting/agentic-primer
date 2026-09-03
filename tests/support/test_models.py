# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""A secret this code never holds is a secret it cannot leak into a trace.

The wire hooks write to a file on disk. These assert the one rule that makes
that safe: no credential-bearing header value is ever put in the record.
"""

from support.models import (
    RECORDED_REQUEST_HEADERS,
    SECRET_HEADERS,
    RecordingTransport,
    _request_headers,
    _response_headers,
    build_wire_hooks,
)
from support.trace import Trace

from unittest.mock import patch

import httpx


def test_an_allowed_request_header_is_recorded() -> None:
    recorded = _request_headers(httpx.Headers({"host": "api.example"}))
    assert recorded == {"host": "api.example"}
    assert "host" in RECORDED_REQUEST_HEADERS


def test_the_sdks_machine_fingerprint_is_recorded() -> None:
    recorded = _request_headers(httpx.Headers({"x-stainless-os": "MacOS"}))
    assert recorded == {"x-stainless-os": "MacOS"}


def test_a_request_header_this_code_has_never_heard_of_is_withheld() -> None:
    """The whole point of an allowlist: an unknown auth header is not a leak."""
    unknown = httpx.Headers({"x-api-key": "sk-live", "api-key": "sk-live"})
    assert _request_headers(unknown) == {"x-api-key": None, "api-key": None}


def test_the_authorization_header_is_withheld_from_a_request() -> None:
    headers = httpx.Headers({"authorization": "Bearer sk-live", "host": "api.example"})
    assert _request_headers(headers)["authorization"] is None


def test_response_header_values_are_recorded() -> None:
    recorded = _response_headers(httpx.Headers({"x-ds-trace-id": "abc123"}))
    assert recorded == {"x-ds-trace-id": "abc123"}


def test_a_credential_header_is_withheld_in_both_directions() -> None:
    for name in SECRET_HEADERS:
        recorded = _response_headers(httpx.Headers({name: "secret", "date": "now"}))
        assert recorded[name] is None, name
        assert recorded["date"] == "now"


def test_a_streaming_response_is_not_force_read_before_it_can_be_iterated() -> None:
    """`.read()` blocks until the whole body has arrived. Calling it inside
    the response hook would force an event stream to buffer completely
    before `ch14_stream`'s loop ever gets to iterate it, collapsing every
    chunk's real arrival time into the instant the hook ran.
    """
    trace = Trace(chapter="test", model_kind="live")
    with trace.span("model") as span:
        hooks = build_wire_hooks(span)
        (on_response,) = hooks["response"]
        response = httpx.Response(200, headers={"content-type": "text/event-stream"})
        with patch.object(response, "read") as read:
            on_response(response)
        read.assert_not_called()


def test_an_ordinary_response_is_still_read_and_recorded_whole() -> None:
    trace = Trace(chapter="test", model_kind="live")
    with trace.span("model") as span:
        hooks = build_wire_hooks(span)
        (on_response,) = hooks["response"]
        response = httpx.Response(
            200, headers={"content-type": "application/json"}, json={"ok": True}
        )
        with patch.object(response, "read", wraps=response.read) as read:
            on_response(response)
        read.assert_called_once()
    (note,) = [n for n in span.notes if n.label == "wire response"]
    assert note.payload["body"] == {"ok": True}


def test_a_streaming_response_records_every_frame_even_if_closed_early() -> None:
    """The bug this guards: recording after the `for` loop in `__iter__`
    depended on `StopIteration`, and a real SSE client never triggers one --
    it reads until it has seen `[DONE]` and calls `.close()` directly, never
    asking the generator for one more value. `close()` is where the fix
    records instead, so a caller that stops early is what this simulates.
    """

    def frames(_request: httpx.Request) -> httpx.Response:
        body = b'data: {"a": 1}\n\ndata: {"b": 2}\n\ndata: [DONE]\n\n'
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(body)
        )

    trace = Trace(chapter="test", model_kind="live")
    with trace.span("model") as span:
        transport = RecordingTransport(span, inner=httpx.MockTransport(frames))
        with (
            httpx.Client(transport=transport) as client,
            client.stream("GET", "https://example.test") as response,
        ):
            # One line only -- not exhausted -- then closed, the way an SSE
            # client stops as soon as it has what it needs.
            next(response.iter_lines())
    (note,) = [n for n in span.notes if n.label == "wire frame"]
    assert "a" in note.payload["raw"]
