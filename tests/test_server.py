# tests/test_server.py
import os
import sys
import pytest

from typing import Union, List
import numpy as np

# 프로젝트 루트 기준으로 srcs 경로를 import 경로에 추가
ROOT = os.path.dirname(os.path.dirname(__file__))
SRCS = os.path.join(ROOT, "srcs")
if SRCS not in sys.path:
    sys.path.append(SRCS)

from fastmcp import Client
from server import MCPServerApp
from adapters.enVector_sdk import EnVectorSDKAdapter

@pytest.fixture
def mcp_server():
    """
    Create and return a FastMCP server instance for testing.
    Inject a fake adapter to avoid using the actual enVector SDK.
    """
    class FakeAdapter(EnVectorSDKAdapter):
        def __init__(self):
            pass  # Actual initialization not needed

        # ----------- Mocked method ----------- #
        def invoke_search(self, index_name: str, query: Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]], topk: int):
            # Return a fake response
            #   - Expected Return Type: List[Dict[str, Any]]
            return {"ok": True, "results": [{"id": 1, "score": 0.9, "metadata": {"fieldA": "valueA"}}]}

    app = MCPServerApp(adapter=FakeAdapter(), server_name="test-mcp")
    return app.mcp  # FastMCP Instance

# ----------- Search Tool Tests ----------- #
# Test cases for the 'search' tool in the MCP server
@pytest.mark.asyncio
async def test_tools_list_contains_search(mcp_server):
    # In-memory client: connects directly to the server instance without network/process
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert "search" in names  # Only 'search' tool is defined for now

# Happy Path Test
@pytest.mark.asyncio
async def test_call_tool_happy_path(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "search",
            {
                "index_name": "test_index",
                "query": [0.1, 0.2, 0.3],
                "topk": 5
            }
        )
        # FastMCP returns results as 'structured data + traditional content'.
        # Depending on implementation/version, accessors may differ, so we check both cases permissively
        data = getattr(result, "data", None) or getattr(result, "structured", None) \
               or getattr(result, "structured_content", None)

        assert data is not None, "No data returned from tool call"
        # Check the expected structure of the returned data from FakeAdapter
        # (key names may vary based on the actual adapter implementation)
        # Expected format:
        # {
        #     "ok": bool,
        #     "results": Any,          # Present if ok is True
        #     "error": str            # Present if ok is False
        # }
        assert data.get("ok") is True
        assert data.get("results", [{}])[0].get("metadata", {}).get("fieldA") == "valueA"

# Invalid Argument Type Test
@pytest.mark.asyncio
async def test_call_tool_invalid_args_type_error(mcp_server):
    async with Client(mcp_server) as client:
        # Invalid parameter value for query
        with pytest.raises(Exception):
            await client.call_tool(
                "search",
                {
                    "index_name": "test_index",
                    "query": "this_should_be_a_list_of_floats",  # Invalid type
                    "topk": 5
                }
            ) # Expected to raise an exception due to invalid argument type

# ----------- Search Tool Tests Finished ----------- #
