# tests/test_server.py
import os
import sys
import pytest

from typing import Union, List, Any, Dict, Optional

# Add srcs directory to import path relative to project root
ROOT = os.path.dirname(os.path.dirname(__file__))
SRCS = os.path.join(ROOT, "srcs")
if SRCS not in sys.path:
    sys.path.append(SRCS)

from fastmcp import Client
from server import MCPServerApp
from adapter.envector_sdk import EnVectorSDKAdapter

@pytest.fixture
def mcp_server():
    """
    Create and return a FastMCP server instance for testing.
    Inject a fake adapter to avoid using the actual enVector SDK.
    """
    class FakeAdapter(EnVectorSDKAdapter):
        def __init__(self):
            pass  # Actual initialization not needed

        # ----------- Mocked method: Insert ----------- #
        def invoke_insert(
                self,
                index_name: str,
                vectors: List[List[float]],
                metadata: Union[Any, List[Any]] = None
            ) -> Dict[str, Any]:
            return {"index_name": index_name, "vectors": vectors, "metadata": metadata}

        # ----------- Mocked method: Search ----------- #
        def invoke_search(self, index_name: str, query: Union[List[float], List[List[float]]], topk: int) -> List[Dict[str, Any]]:
            # Return a fake response
            #   - Expected Return Type: List[Dict[str, Any]]
            return [{"id": 1, "score": 0.9, "metadata": {"fieldA": "valueA"}}]

    app = MCPServerApp(adapter=FakeAdapter(), mcp_server_name="test-mcp")
    return app.mcp  # FastMCP Instance

# ----------- Insert Tool Tests ----------- #
# Test cases for the 'insert' tool in the MCP server
@pytest.mark.asyncio
async def test_tools_list_contains_insert(mcp_server):
    # In-memory client: connects directly to the server instance without network/process
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert "insert" in names  # Only 'insert' tool is defined for now

# Happy Path Test
@pytest.mark.asyncio
async def test_call_tool_insert_happy_path(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "insert",
            {
                "index_name": "test_index",
                "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "metadata": [{"field1": "value1"}, {"field2": "value2"}]
            }
        )
        # FastMCP returns results as 'structured data + traditional content'.
        # Depending on implementation/version, accessors may differ, so we check both cases permissively
        data = getattr(result, "data", None) or getattr(result, "structured", None) \
               or getattr(result, "structured_content", None)

        assert data is not None, "No data returned from tool call"
        assert data.get("ok") is True
        payload = data.get("results")
        assert isinstance(payload, dict)
        assert payload["index_name"] == "test_index"
        assert payload["vectors"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

# Invalid Argument Type Test
@pytest.mark.asyncio
async def test_call_tool_insert_invalid_args_type_error(mcp_server):
    async with Client(mcp_server) as client:
        # Invalid parameter value for vectors
        with pytest.raises(Exception):
            await client.call_tool(
                "insert",
                {
                    "index_name": "test_index",
                    "vectors": "this_should_be_a_list_of_floats_lists_or_else",  # Invalid type
                    "metadata": [{"field1": "value1"}, {"field2": "value2"}]
                }
            )  # Expected to raise an exception due to invalid argument type

# ----------- Insert Tool Tests Finished ----------- #

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
