# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""`support/mcp.py`'s own guarantees: discovery returns metadata only, and
order is not a promise unless asked to hold still.
"""

from support.mcp import PromptServer, ResourceServer, ToolServer


def make_tool_server() -> ToolServer:
    return ToolServer(
        catalog=[
            {"name": "a", "description": "d", "schema": {}},
            {"name": "b", "description": "d", "schema": {}},
            {"name": "c", "description": "d", "schema": {}},
        ],
        handlers={"a": lambda: "A", "b": lambda: "B", "c": lambda: "C"},
    )


def test_list_tools_with_no_seed_returns_the_catalog_as_given() -> None:
    server = make_tool_server()
    assert server.list_tools() == server.catalog


def test_list_tools_with_a_seed_can_return_a_different_order() -> None:
    server = make_tool_server()
    first = [t["name"] for t in server.list_tools(seed=0)]
    second = [t["name"] for t in server.list_tools(seed=4)]
    assert set(first) == set(second)
    assert first != second


def test_call_tool_dispatches_by_name_through_handlers() -> None:
    server = make_tool_server()
    assert server.call_tool("b", {}) == "B"


def test_resource_server_read_returns_content_not_listed() -> None:
    server = ResourceServer(
        catalog=[{"uri": "u1", "name": "n", "description": "d", "mimeType": "text/plain"}],
        content={"u1": "the actual bytes"},
    )
    listed = server.list_resources()
    assert "content" not in listed[0]
    assert server.read_resource("u1") == "the actual bytes"


def test_prompt_server_get_returns_the_template_result() -> None:
    server = PromptServer(
        catalog=[{"name": "greet", "description": "d", "arguments": []}],
        templates={"greet": lambda args: [{"role": "human", "content": f"hi {args['who']}"}]},
    )
    assert server.get_prompt("greet", {"who": "sanjeev"}) == [
        {"role": "human", "content": "hi sanjeev"}
    ]
