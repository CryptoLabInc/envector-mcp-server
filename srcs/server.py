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

from typing import Union, List, Dict, Any
import numpy as np
import os, sys, signal

# Ensure current directory is in sys.path for module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from fastmcp import FastMCP  # pip install fastmcp
from adapter.enVector_sdk import EnVectorSDKAdapter

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

        # ---------- MCP Tools: Search ---------- #
        @self.mcp.tool(name="search", description="Search using enVector SDK")
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
                query (Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]]): The search query.
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


if __name__ == "__main__":
    # Environment Variables for MCP Server Configuration
    """
    Environment Variables for MCP Server Configuration:
    - MCP_SERVER_HOST: The host address for the MCP server (default: "127.0.0.1")
    - MCP_SERVER_PORT: The port number for the MCP server (default: 8000)
    - MCP_SERVER_NAME: The name of the MCP server (default: "envector_mcp_server")
    """
    MCP_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    MCP_PORT = int(os.getenv("MCP_SERVER_PORT", 8000))
    MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "envector_mcp_server")

    # Environment Variables for enVector SDK Configuration
    """
    Environment Variables for enVector SDK Configuration:
    - ENVECTOR_ENDPOINT: The endpoint URL of the `enVector` (default: "127.0.0.1")
    - ENVECTOR_PORT: The port number of the `enVector` (default: 50050)
    - ENVECTOR_KEY_ID: The key ID for the `enVector` SDK (default: "mcp_key")
    - ENVECTOR_EVAL_MODE: The evaluation mode of the `enVector` ["rmp", "mm"] (default: "mm")
    """
    ENVECTOR_ENDPOINT = os.getenv("ENVECTOR_ENDPOINT", "127.0.0.1")
    ENVECTOR_PORT = int(os.getenv("ENVECTOR_PORT", 50050))
    ENVECTOR_KEY_ID = os.getenv("ENVECTOR_KEY_ID", "mcp_key")
    ENVECTOR_EVAL_MODE = os.getenv("ENVECTOR_EVAL_MODE", "mm")

    adapter = EnVectorSDKAdapter(
        endpoint=ENVECTOR_ENDPOINT,
        port=ENVECTOR_PORT,
        key_id=ENVECTOR_KEY_ID,
        eval_mode=ENVECTOR_EVAL_MODE
    )
    app = MCPServerApp(adapter=adapter, mcp_server_name=MCP_SERVER_NAME)
    def _handle_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"\n[INFO] Received {sig_name}. Shutting down MCP server...", flush=True)
        raise SystemExit(0)
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, _handle_shutdown)
    try:
        app.run_http_service(host=MCP_HOST, port=MCP_PORT)
    finally:
        print("[INFO] MCP server stopped.", flush=True)
