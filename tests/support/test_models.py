# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""A secret this code never holds is a secret it cannot leak into a trace.

The wire hooks write to a file on disk. These assert the one rule that makes
that safe: no credential-bearing header value is ever put in the record.
"""

from support.models import (
    RECORDED_REQUEST_HEADERS,
    SECRET_HEADERS,
    _request_headers,
    _response_headers,
)

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
