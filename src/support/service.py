# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The seam between a mocked HTTP service and a real one.

Scaffolding, never a lesson, and the same shape as `support/models.py`: one
switch, so the only difference between a mock run and a live run is this
function. It exists because `ch07_tool_http` is the first chapter whose tool
leaves the process, and a tool that really calls the internet would make
`just gate` need the network -- which the tests have never needed.

Mock and live are about the *model* everywhere else in this primer. Here they
also decide whether a tool's HTTP is real, and the two axes come apart: a mock
run makes no network call at all, a live run makes two -- one to the model
provider and one to the service.

The recorded wire is the same shape either way, because `httpx` runs its event
hooks whether the transport is a socket or a stub. That is deliberate: the
mock column has to show what the live column would, or comparing them proves
nothing.
"""

from support.models import build_wire_hooks
from support.trace import ModelKind, Span

import httpx

# httpbin echoes the request back, headers included, which is exactly why it is
# useful here: it makes the credential's round trip visible.
SERVICE = "https://httpbin.org"

# Fabricated, and worth nothing. It is sent, echoed back by the service, and
# recorded in the response body -- which is the point of `ch07_tool_http`, and
# is only safe to demonstrate because there is no secret here to leak.
SERVICE_KEY = "primer-demo-key"


type Canned = httpx.Response | Exception


def build_http_client(model_kind: ModelKind, span: Span, canned: Canned | None) -> httpx.Client:
    """The one switch, mirroring `build_model`.

    `canned` is what the mocked service does and is ignored when live; `span`
    records the wire either way, using the same hooks as the model.

    A canned exception rather than a response is not a special case: it is the
    only way to mock a failure that is not an answer. A mocked transport
    replies instantly, so it cannot be slow -- it has to raise what being slow
    would have raised.
    """
    hooks = build_wire_hooks(span)
    if model_kind == "live":
        return httpx.Client(event_hooks=hooks, timeout=10)
    if canned is None:
        raise RuntimeError("a mock run needs a canned reply; none was given")

    def answer(_: httpx.Request) -> httpx.Response:
        if isinstance(canned, Exception):
            raise canned
        return canned

    return httpx.Client(event_hooks=hooks, transport=httpx.MockTransport(answer))
