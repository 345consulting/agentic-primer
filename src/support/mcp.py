# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""A stand-in for what an MCP server hands a client -- three catalogs, each
discovered by a call instead of written as a literal, and the three matching
calls a chapter uses to act on what it found.

Scaffolding, not the lesson -- the same bucket as `models.py`'s mock model
and `trace.py`, both imported starting in `ch01` and never taught inline
first. `ch23_mcp` and `ch24_mcp_injection` both need this, and neither is
*about* how a discovery stand-in is built; they are about what changes once
a dispatch table, a document, or a conversation seed arrives from a call
instead of a literal. That is `HookVerdict`'s opposite case: `ch18_hooks`
hand-built its mechanism first because the mechanism -- observe, modify,
veto -- was the chapter's entire content, and only graduated once `ch19`
needed it too. Nothing here is a chapter's content; it is what lets three
chapters show theirs. Not a protocol implementation -- no JSON-RPC framing,
no transport. What a real client sends on the wire is `ch23`'s prose, not
this file's job; this file's job is the one fact a Python literal cannot
produce -- a catalog is the return value of a call, and two calls need not
agree on order.

**Three primitives, three different actors deciding what happens next.** A
tool is discovered, declared to the model, and the *model* decides whether
to call it -- the familiar `tool_calls` round trip every chapter since `ch02`
already runs. A resource is discovered and read by the *client* -- its
content lands in context directly, no round trip through the model at all;
the model never sees `resources/list`, only whatever text a human or the
harness chose to attach. A prompt is discovered and picked by a *human*,
filled with arguments, and returned as a ready-made list of messages -- it
seeds a conversation before the model is ever invoked, and picking it is not
the model's decision either. Same shape once each primitive's content lands
in the message list -- text nobody in this run authored -- different actor
deciding whether it gets there.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

type Json = dict[str, Any]


def _discovered(catalog: Sequence[Json], seed: int | None) -> list[Json]:
    """The one behavior a literal cannot have: order that is not a promise.

    A real server's `list` response order depends on its own registration
    order, plugin load order, or nothing stable at all. `seed=None` returns
    the catalog as given, the same every time -- useful for a scenario that
    is not about ordering. Any other seed reshuffles it, so two "connections"
    to the same server can be shown disagreeing about order while agreeing
    on content.
    """
    items = list(catalog)
    if seed is not None:
        Random(seed).shuffle(items)
    return items


@dataclass
class ToolServer:
    """`tools/list` and `tools/call`. `catalog` is what crosses the wire on
    `list_tools` -- name, description, schema, all JSON. `handlers` never
    does; a real client cannot see a server's implementation, only call it
    by name and get a result back.
    """

    catalog: list[Json]
    handlers: dict[str, Callable[..., Any]] = field(repr=False)

    def list_tools(self, *, seed: int | None = None) -> list[Json]:
        return _discovered(self.catalog, seed)

    def call_tool(self, name: str, args: Json) -> Any:
        return self.handlers[name](**args)


@dataclass
class ResourceServer:
    """`resources/list` and `resources/read`. `catalog` carries uri, name,
    description, mime type -- enough to decide whether to read one without
    having read it yet. `content` is keyed by uri and, like a tool's
    handlers, never appears in `list_resources`.
    """

    catalog: list[Json]
    content: dict[str, str] = field(repr=False)

    def list_resources(self, *, seed: int | None = None) -> list[Json]:
        return _discovered(self.catalog, seed)

    def read_resource(self, uri: str) -> str:
        return self.content[uri]


@dataclass
class PromptServer:
    """`prompts/list` and `prompts/get`. `catalog` carries name, description,
    and the arguments a template accepts. `templates` maps a name to a
    function building the actual message list -- again never listed, only
    invoked by name once a human has picked it.
    """

    catalog: list[Json]
    templates: dict[str, Callable[[Json], list[Json]]] = field(repr=False)

    def list_prompts(self, *, seed: int | None = None) -> list[Json]:
        return _discovered(self.catalog, seed)

    def get_prompt(self, name: str, args: Json) -> list[Json]:
        return self.templates[name](args)
