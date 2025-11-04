# Summary of file: Main Server File (Runs with stdio)

"""
MCP Server Application using FastMCP and enVector SDK Adapter.
- Transport: Streamable HTTP
- Endpoint: http://<HOST>:<PORT>/mcp/ (default)
- Health Check: http://<HOST>:<PORT>/health/ (default)

Expected MCP Tool Return Format:
{
    "ok": bool,
    "results": Any,          # Present if ok is True
    "error": str            # Present if ok is False
}
"""

import argparse
from typing import Union, List, Dict, Any, Optional
import numpy as np
import os, sys, signal
import json

# Ensure current directory is in sys.path for module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from fastmcp import FastMCP  # pip install fastmcp
from adapter.envector_sdk import EnVectorSDKAdapter

# # For Health Check (Starlette Imports -> Included in FastMCP as dependency)
# from starlette.requests import Request
# from starlette.responses import PlainTextResponse

class MCPServerApp:
    """
    Main application class for the MCP server.
    """
    def __init__(
            self,
            adapter: EnVectorSDKAdapter,
            mcp_server_name: str = "envector_mcp_server"
        ) -> None:
        """
        Initializes the MCPServerApp with the given adapter and server name.
        Args:
            adapter (EnVectorSDKAdapter): The enVector SDK adapter instance.
            mcp_server_name (str): The name of the MCP server.
        """
        self.adapter = adapter
        self.mcp = FastMCP(name=mcp_server_name)

        # # ---------- Health Check Route ---------- #
        # @self.mcp.custom_route("/health/", methods=["GET"])
        # async def health_check(_: Request) -> PlainTextResponse:
        #     """
        #     Health check endpoint to verify server status.
        #     Returns:
        #         PlainTextResponse: A simple "OK" response indicating server health.
        #     """
        #     return PlainTextResponse("OK", status_code=200)

        # ---------- MCP Tools: Create Index ---------- #
        @self.mcp.tool(
            name="create_index",
            description=(
                "Create an index. There are 3 parameters to set. `index_name`, `dim`, and `index_params`. "
                "Set index_params accordingly: {'index_type': 'FLAT'} for a flat index or {'index_type': 'IVF_FLAT', 'nlist': <int>, 'default_nprobe': <int>} for IVF."
            )
        )
        async def tool_create_index(
                index_name: str,
                dim: int,
                index_params: Dict[str, Any]
            ) -> Dict[str, Any]:
            """
            MCP tool to create an index using the enVector SDK adapter.
            Call the adapter's call_create_index method.

            Args:
                index_name (str): The name of the index to create.
                dim (int): The dimensionality of the index.
                index_params (Dict[str, Any]): The parameters for the index.

            Returns:
                Dict[str, Any]: The create index results from the enVector SDK adapter.
            """
            return self.adapter.call_create_index(index_name=index_name, dim=dim, index_params=index_params)

        # ---------- MCP Tools: Get Index List ---------- #
        @self.mcp.tool(
            name="get_index_list",
            description=(
                "Get the list of indexes from the enVector SDK. "
                "No parameters are required. "
                "Returns the list of existing indexes."
            )
        )
        async def tool_get_index_list() -> Dict[str, Any]:
            """
            MCP tool to get the list of indexes using the enVector SDK adapter.
            Call the adapter's call_get_index_list method.

            Returns:
                Dict[str, Any]: The index list from the enVector SDK adapter.
            """
            return self.adapter.call_get_index_list()

        # ---------- MCP Tools: Get Index Info ---------- #
        @self.mcp.tool(
            name="get_index_info",
            description=(
                "Get information about a specific index from the enVector SDK. "
                "One parameter is required: `index_name`. "
                "Returns information about the specified index."
            )
        )
        async def tool_get_index_info(index_name: str) -> Dict[str, Any]:
            """
            MCP tool to get information about a specific index using the enVector SDK adapter.
            Call the adapter's call_get_index_info method.

            Args:
                index_name (str): The name of the index to retrieve information for.

            Returns:
                Dict[str, Any]: The index information from the enVector SDK adapter.
            """
            return self.adapter.call_get_index_info(index_name=index_name)

        # ---------- MCP Tools: Insert ---------- #
        @self.mcp.tool(
            name="insert",
            description=(
                "Insert vectors using enVector SDK. "
                "There are 3 parameters to set. `index_name`, `vectors`, and `metadata`. "
                "Field `metadata` is for attached information for each vector."
            )
        )
        async def tool_insert(
                index_name: str,
                vectors: Union[List[float], List[List[float]]],
                metadata: Union[Any, List[Any]] = None
            ) -> Dict[str, Any]:
            """
            MCP tool to perform insert using the enVector SDK adapter.
            Call the adapter's call_insert method.

            Args:
                index_name (str): The name of the index to insert into.
                vectors (Union[List[float], List[List[float]]]): The vector(s) to insert.
                metadata (Union[Any, List[Any]]): The list of metadata associated with the vectors.

            Returns:
                Dict[str, Any]: The insert results from the enVector SDK adapter.
            """
            # Instance normalization for vectors
            if isinstance(vectors, np.ndarray):
                vectors = [vectors.tolist()]
            elif isinstance(vectors, list) and all(isinstance(v, np.ndarray) for v in vectors):
                vectors = [v.tolist() for v in vectors]
            elif isinstance(vectors, list) and all(isinstance(v, float) for v in vectors):
                vectors = [vectors]
            elif isinstance(vectors, str):
                # If `vectors` is passed as a string, try to parse it as JSON
                try:
                    vectors = json.loads(vectors)
                except json.JSONDecodeError:
                    # If parsing fails, raise an error
                    raise ValueError("Invalid format has used or failed to parse JSON for `vectors` parameter. Caused by: " + vectors)

            # Instance normalization for metadata
            if metadata is not None and not isinstance(metadata, list):
                if isinstance(metadata, str):
                    # If `metadata` is passed as a string, try to parse it as JSON
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        # If parsing fails, wrap the string in a list
                        metadata = [metadata]
                else:
                    # If `metadata` is not a list or string, wrap it in a list
                    metadata = [metadata]
            return self.adapter.call_insert(index_name=index_name, vectors=vectors, metadata=metadata)

        # ---------- MCP Tools: Search ---------- #
        @self.mcp.tool(
            name="search",
            description=(
                "Search using enVector SDK. "
                "There are 3 parameters to set. `index_name`, `query`, and `topk`."
            )
        )
        async def tool_search(
                index_name: str,
                query: Union[List[float], List[List[float]]],
                topk: int
            ) -> Dict[str, Any]:
            """
            MCP tool to perform search using the enVector SDK adapter.
            Call the adapter's call_search method.

            Args:
                index_name (str): The name of the index to search.
                query (Union[List[float], List[List[float]]]): The search query.
                topk (int): The number of top results to return.

            Returns:
                Dict[str, Any]: The search results from the enVector SDK adapter.
            """
            if isinstance(query, np.ndarray):
                query = query.tolist()
            elif isinstance(query, list) and all(isinstance(q, np.ndarray) for q in query):
                query = [q.tolist() for q in query]
            return self.adapter.call_search(index_name=index_name, query=query, topk=topk)

    def run_http_service(self, host: str, port: int) -> None:
        """
        Runs the MCP server as an HTTP service.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
        """
        self.mcp.run(transport="http", host=host, port=port)

    def run_stdio_service(self) -> None:
        """
        Runs the MCP server using stdio transport (for local integrations).
        """
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the enVector MCP server.")
    parser.add_argument(
        "--mode",
        choices=("local", "remote"),
        default=os.getenv("MCP_SERVER_MODE", "remote"),
        help="Execution mode: 'local' uses stdio transport, 'remote' exposes HTTP transport.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_SERVER_HOST", "127.0.0.1"),
        help="HTTP bind host."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_SERVER_PORT", 8000)),
        help="HTTP bind port.",
    )
    parser.add_argument(
        "--address",
        default=os.getenv("MCP_SERVER_ADDRESS", None),
        help="HTTP bind address (host:port) of MCP Server. Overrides --host and --port if provided.",
    )
    parser.add_argument(
        "--server-name",
        default=os.getenv("MCP_SERVER_NAME", "envector_mcp_server"),
        help="Advertised MCP server name.",
    )
    parser.add_argument(
        "--envector-host",
        default=os.getenv("ENVECTOR_HOST", "127.0.0.1"),
        help="enVector endpoint hostname or IP.",
    )
    parser.add_argument(
        "--envector-port",
        type=int,
        default=int(os.getenv("ENVECTOR_PORT", 50050)),
        help="enVector endpoint port.",
    )
    parser.add_argument(
        "--envector-address",
        default=os.getenv("ENVECTOR_ADDRESS", None),
        help="enVector endpoint address (host:port). Overrides --envector-host and --envector-port if provided.",
    )
    parser.add_argument(
        "--envector-key-id",
        default=os.getenv("ENVECTOR_KEY_ID", "mcp_key"),
        help="enVector key identifier.",
    )
    parser.add_argument(
        "--envector-key-path",
        default=os.getenv("ENVECTOR_KEY_PATH", None),
        help="Path to the enVector key file.",
    )
    parser.add_argument(
        "--envector-eval-mode",
        default=os.getenv("ENVECTOR_EVAL_MODE", "rmp"),
        help="enVector evaluation mode (e.g., 'rmp', 'mm').",
    )
    args = parser.parse_args()
    run_mode = args.mode.lower()

    # Environment Variables for MCP Server Configuration
    """
    Environment Variables for MCP Server Configuration:
    - MCP_SERVER_HOST: The host address for the MCP server (default: "127.0.0.1")
    - MCP_SERVER_PORT: The port number for the MCP server (default: 8000)
    - MCP_SERVER_ADDRESS: The address (host:port) for the MCP server (overrides --host and --port if provided)
    - MCP_SERVER_NAME: The name of the MCP server (default: "envector_mcp_server")
    """
    if args.address:
        mcp_address = args.address.split(":")
        MCP_HOST = mcp_address[0]
        MCP_PORT = int(mcp_address[1]) if len(mcp_address) > 1 else 8000
    else:
        MCP_HOST = args.host
        MCP_PORT = args.port
    MCP_SERVER_NAME = args.server_name

    # Environment Variables for enVector SDK Configuration
    """
    Environment Variables for enVector SDK Configuration:
    - ENVECTOR_ENDPOINT: The endpoint URL of the `enVector` (default: "127.0.0.1")
    - ENVECTOR_PORT: The port number of the `enVector` (default: 50050)
    - ENVECTOR_KEY_ID: The key ID for the `enVector` SDK (default: "mcp_key")
    - ENVECTOR_EVAL_MODE: The evaluation mode of the `enVector` ["rmp", "mm"] (default: "rmp")
    """
    if args.envector_address:
        envector_address = args.envector_address.split(":")
        ENVECTOR_HOST = envector_address[0]
        ENVECTOR_PORT = int(envector_address[1]) if len(envector_address) > 1 else 50050
    else:
        ENVECTOR_HOST = args.envector_host
        ENVECTOR_PORT = args.envector_port
    ENVECTOR_KEY_ID = args.envector_key_id
    ENVECTOR_KEY_PATH = args.envector_key_path
    ENVECTOR_EVAL_MODE = args.envector_eval_mode

    adapter = EnVectorSDKAdapter(
        endpoint=ENVECTOR_HOST,
        port=ENVECTOR_PORT,
        key_id=ENVECTOR_KEY_ID,
        key_path=ENVECTOR_KEY_PATH,
        eval_mode=ENVECTOR_EVAL_MODE
    )
    app = MCPServerApp(adapter=adapter, mcp_server_name=MCP_SERVER_NAME)
    def _handle_shutdown(signum, frame):
        # parameter `frame` is not used, but required by signal handler signature
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        raise SystemExit(0)
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, _handle_shutdown)
    if run_mode == "local":
        app.run_stdio_service()
    else:
        app.run_http_service(host=MCP_HOST, port=MCP_PORT)
