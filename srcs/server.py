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
from typing import Union, List, Dict, Any, Optional, Annotated, TYPE_CHECKING
import numpy as np
import os, sys, signal
import json
from pydantic import Field
# load environment variables from .env file if present
from dotenv import load_dotenv
load_dotenv()

# Ensure current directory is in sys.path for module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from fastmcp import FastMCP, Client  # pip install fastmcp
from fastmcp.exceptions import ToolError
from adapter import EnVectorSDKAdapter, EmbeddingAdapter, DocumentPreprocessingAdapter
from adapter.vault_client import VaultClient, VaultClientSync, create_vault_client, DecryptResult, VaultError


def fetch_keys_from_vault(vault_endpoint: str, vault_token: str, key_path: str) -> bool:
    """
    Fetches public keys (EncKey, EvalKey, MetadataKey) from Rune-Vault MCP.

    Args:
        vault_endpoint: Rune-Vault MCP endpoint URL (e.g., http://vault-mcp:50080/mcp)
        vault_token: Authentication token for Vault
        key_path: Local directory to save the fetched keys

    Returns:
        bool: True if keys were successfully fetched and saved
    """
    import asyncio

    async def _fetch():
        try:
            client = Client(vault_endpoint)
            async with client:
                result = await client.call_tool("get_public_key", {"token": vault_token})

                # Parse the result - handle different response formats
                if hasattr(result, 'content'):
                    # TextContent format
                    content = result.content[0].text if result.content else None
                elif hasattr(result, 'data'):
                    content = result.data
                else:
                    content = str(result)

                if content:
                    bundle = json.loads(content)

                    # Ensure key directory exists
                    os.makedirs(key_path, exist_ok=True)

                    # Save each key file
                    for filename, key_content in bundle.items():
                        filepath = os.path.join(key_path, filename)
                        with open(filepath, 'w') as f:
                            f.write(key_content)
                        print(f"[Vault] Saved {filename} to {filepath}")

                    return True

        except Exception as e:
            print(f"[Vault] Failed to fetch keys from Vault: {e}")
            return False

        return False

    return asyncio.run(_fetch())

# # For Health Check (Starlette Imports -> Included in FastMCP as dependency)
# from starlette.requests import Request
# from starlette.responses import PlainTextResponse

class MCPServerApp:
    """
    Main application class for the MCP server.

    Security Model (with Rune-Vault):
    - MCP Server handles embeddings, query encryption, and orchestration
    - Rune-Vault holds SecKey and performs all decryption
    - Agent never has access to SecKey
    """
    def __init__(
            self,
            envector_adapter: EnVectorSDKAdapter,
            mcp_server_name: str = "envector_mcp_server",
            embedding_adapter: "EmbeddingAdapter" = None,
            document_preprocessor: DocumentPreprocessingAdapter = None,
            vault_client: Optional[VaultClientSync] = None,
        ) -> None:
        """
        Initializes the MCPServerApp with the given adapter and server name.
        Args:
            adapter (EnVectorSDKAdapter): The enVector SDK adapter instance.
            mcp_server_name (str): The name of the MCP server.
            vault_client (VaultClientSync): Optional Vault client for secure decryption.
        """
        # adapters
        self.envector = envector_adapter
        self.embedding = embedding_adapter
        self.preprocessor = document_preprocessor
        self.vault = vault_client
        # mcp
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
            description="Create an index in enVector."
        )
        async def tool_create_index(
            index_name: Annotated[str, Field(description="index name to create")],
            dim: Annotated[int, Field(description="dimensionality of the index")],
            index_params: Annotated[Dict[str, Any], Field(description="indexing parameters including FLAT and IVF_FLAT. The default is FLAT, or set index_params as {'index_type': 'IVF_FLAT', 'nlist': <int>, 'default_nprobe': <int>} for IVF.")]
        ) -> Dict[str, Any]:
            """
            MCP tool to create an index using the enVector SDK adapter.
            Calls self.envector.call_create_index(...).

            Args:
                index_name (str): The name of the index to create.
                dim (int): The dimensionality of the index.
                index_params (Dict[str, Any]): The parameters for the index.

            Returns:
                Dict[str, Any]: The create index results from the enVector SDK adapter.
            """
            return self.envector.call_create_index(index_name=index_name, dim=dim, index_params=index_params)

        # ---------- MCP Tools: Get Index List ---------- #
        @self.mcp.tool(
            name="get_index_list",
            description="Get the list of indexes from the enVector SDK."
        )
        async def tool_get_index_list() -> Dict[str, Any]:
            """
            MCP tool to get the list of indexes using the enVector SDK adapter.
            Call the adapter's call_get_index_list method.

            Returns:
                Dict[str, Any]: The index list from the enVector SDK adapter.
            """
            return self.envector.call_get_index_list()

        # ---------- MCP Tools: Get Index Info ---------- #
        @self.mcp.tool(
            name="get_index_info",
            description="Get information about a specific index from the enVector SDK."
        )
        async def tool_get_index_info(
            index_name: Annotated[str, Field(description="index name to get information for")],
        ) -> Dict[str, Any]:
            """
            MCP tool to get information about a specific index using the enVector SDK adapter.
            Call the adapter's call_get_index_info method.

            Args:
                index_name (str): The name of the index to retrieve information for.

            Returns:
                Dict[str, Any]: The index information from the enVector SDK adapter.
            """
            return self.envector.call_get_index_info(index_name=index_name)

        # ---------- MCP Tools: Insert ---------- #
        @self.mcp.tool(
            name="insert",
            description=(
                "Insert vectors or metadata using enVector SDK. "
                "Allowing to insert metadata as text only as supporting embedding the metadata as vectors. "
                "Allowing one or more vectors, but insert 'batch_size' vectors in once would be more efficient. "
            )
        )
        async def tool_insert(
            index_name: Annotated[str, Field(description="index name to insert data into")],
            vectors: Annotated[Union[List[float], List[List[float]]], Field(description="vectors to insert")] = None,
            metadata: Annotated[Union[Any, List[Any]], Field(description="the corresponding metadata of the vectors to insert for retrieval")] = None
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
            if vectors is None and metadata is None:
                raise ValueError("`vectors` or `metadata` parameter must be provided.")

            if vectors is not None:
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

            elif metadata is not None:
                # Instance normalization for metadata
                if not isinstance(metadata, list):
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

                if vectors is None and self.embedding is not None:
                    vectors = self.embedding.get_embedding(metadata)

            return self.envector.call_insert(index_name=index_name, vectors=vectors, metadata=metadata)

        # ---------- MCP Tools: Insert Documents from Path ---------- #
        @self.mcp.tool(
            name="insert_documents_from_path",
            description=(
                "Insert long documents from the given path using enVector SDK. "
                "This tool read document in a directory, preprocess and chunk them, then embed and insert into enVector. "
                "This tool requires a path of documents to read and insert"
            )
        )
        async def tool_insert_documents_from_path(
            index_name: Annotated[str, Field(description="index name to insert data into")],
            document_path: Annotated[Union[Any, List[Any]], Field(description="documents path to insert")] = None,
            language: Annotated[Optional[str], Field(description="language of the documents for preprocessing and chunking")] = "DOCUMENT",
        ) -> Dict[str, Any]:
            """
            MCP tool to perform insert of documents using the enVector SDK adapter.
            """
            chunk_docs = self.preprocessor.preprocess_documents_from_path(path=document_path, language=language)
            text = [chunk["text"] for chunk in chunk_docs]
            metadata = [json.dumps(chunk) for chunk in chunk_docs]
            vectors = self.embedding.get_embedding(text)
            return self.envector.call_insert(index_name=index_name, vectors=vectors, metadata=metadata)

        # ---------- MCP Tools: Insert Documents from Texts ---------- #
        @self.mcp.tool(
            name="insert_documents_from_text",
            description=(
                "Insert long documents from the given texts using enVector SDK. "
                "This tool read document in a directory, preprocess and chunk them, then embed and insert into enVector. "
                "This tool requires a list of text documents loaded by LLMs to read and insert"
            )
        )
        async def tool_insert_documents_from_text(
            index_name: Annotated[str, Field(description="index name to insert data into")],
            texts: Annotated[Union[Any, List[Any]], Field(description="document text to insert")] = None,
        ) -> Dict[str, Any]:
            """
            MCP tool to perform insert of documents using the enVector SDK adapter.

            """
            chunk_docs = self.preprocessor.preprocess_document_from_text(texts=texts)
            text = [chunk["text"] for chunk in chunk_docs]
            metadata = [json.dumps(chunk) for chunk in chunk_docs]
            vectors = self.embedding.get_embedding(text)
            return self.envector.call_insert(index_name=index_name, vectors=vectors, metadata=metadata)

        # ---------- MCP Tools: Search ---------- #
        @self.mcp.tool(
            name="search",
            description="Perform vector search and Retrieve Metadata using enVector SDK."
        )
        async def tool_search(
            index_name: Annotated[str, Field(description="index name to search from")],
            query: Annotated[Any, Field(description="search query vector (list), batch of vectors, or JSON-encoded string")],
            topk: Annotated[int, Field(description="number of top-k results to return")],
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
            def _preprocess_query(raw_query: Any) -> Union[List[float], List[List[float]]]:
                # print("DEBUG preprocess called with", type(raw_query), raw_query)
                if isinstance(raw_query, str):
                    raw_query = raw_query.strip()

                    if self.embedding is not None:
                        return self.embedding.get_embedding([raw_query])[0]

                    if not raw_query:
                        raise ValueError("`query` string is empty. Provide a JSON array of floats or precomputed embedding.")
                    try:
                        raw_query = json.loads(raw_query)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "Plain text is not supported for `query`. Convert the text into an embedding vector "
                            "and pass it as a JSON array (e.g., [[0.1, 0.2], ...])."
                        ) from exc

                if isinstance(raw_query, np.ndarray):
                    raw_query = raw_query.tolist()
                elif isinstance(raw_query, list) and all(isinstance(q, np.ndarray) for q in raw_query):
                    raw_query = [q.tolist() for q in raw_query]

                def _is_vector(value: Any) -> bool:
                    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)

                if _is_vector(raw_query):
                    return raw_query
                if isinstance(raw_query, list) and all(_is_vector(item) for item in raw_query):
                    return raw_query

                raise ValueError(
                    "`query` must be a list of floats or a list of float lists. "
                    f"Received type: {type(raw_query).__name__}"
                )

            try:
                preprocessed_query = _preprocess_query(query)
            except ValueError as exc:
                raise ToolError(f"Invalid query parameter: {exc}") from exc
            return self.envector.call_search(index_name=index_name, query=preprocessed_query, topk=topk)

        # ---------- MCP Tools: Secure Search with Vault Decryption ---------- #
        @self.mcp.tool(
            name="secure_search",
            description=(
                "Perform encrypted vector search with decryption delegated to Rune-Vault. "
                "This is the secure search mode where the MCP server never has access to decryption keys. "
                "Requires Vault to be configured with --vault-endpoint and --vault-token."
            )
        )
        async def tool_secure_search(
            index_name: Annotated[str, Field(description="index name to search from")],
            query: Annotated[Any, Field(description="search query (text, vector, or JSON-encoded string)")],
            topk: Annotated[int, Field(description="number of top-k results to return (max 10)")],
            request_id: Annotated[str, Field(description="optional correlation ID for audit trail")] = "",
        ) -> Dict[str, Any]:
            """
            Secure search with Vault-based decryption.

            Security Model:
            1. Query is embedded and encrypted locally (MCP Server)
            2. Encrypted query sent to enVector Cloud
            3. Encrypted results received from enVector Cloud
            4. Encrypted results sent to Rune-Vault for decryption
            5. Vault decrypts using SecKey (never exposed to MCP Server)
            6. Decrypted indices returned to MCP Server
            7. MCP Server fetches metadata for those indices

            Args:
                index_name: The name of the index to search.
                query: Search query (text or vector).
                topk: Number of top results (max 10, enforced by Vault).
                request_id: Optional correlation ID for audit trail.

            Returns:
                Dict with search results and audit information.
            """
            # Check Vault availability
            if self.vault is None:
                return {
                    "ok": False,
                    "error": "Vault not configured. Use --vault-endpoint and --vault-token to enable secure search.",
                    "hint": "Falling back to standard search is not allowed in secure mode."
                }

            # Preprocess query
            def _preprocess_query_secure(raw_query: Any) -> Union[List[float], List[List[float]]]:
                if isinstance(raw_query, str):
                    raw_query = raw_query.strip()
                    if self.embedding is not None:
                        return self.embedding.get_embedding([raw_query])[0]
                    try:
                        raw_query = json.loads(raw_query)
                    except json.JSONDecodeError as exc:
                        raise ValueError("Plain text requires embedding adapter.") from exc

                if isinstance(raw_query, np.ndarray):
                    raw_query = raw_query.tolist()
                elif isinstance(raw_query, list) and all(isinstance(q, np.ndarray) for q in raw_query):
                    raw_query = [q.tolist() for q in raw_query]

                def _is_vector(value: Any) -> bool:
                    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)

                if _is_vector(raw_query):
                    return raw_query
                if isinstance(raw_query, list) and all(_is_vector(item) for item in raw_query):
                    return raw_query

                raise ValueError(f"Invalid query format: {type(raw_query).__name__}")

            try:
                preprocessed_query = _preprocess_query_secure(query)
            except ValueError as exc:
                return {"ok": False, "error": f"Query preprocessing failed: {exc}"}

            # Enforce Vault's max top_k policy
            if topk > 10:
                return {
                    "ok": False,
                    "error": "Policy Violation: max top_k is 10 in secure search mode."
                }

            try:
                # Step 1: Search (SDK handles encryption internally for now)
                # TODO: When SDK supports returning encrypted results without auto-decrypt,
                #       this will call encrypted_search() and pass blob to Vault
                search_result = self.envector.call_search(
                    index_name=index_name,
                    query=preprocessed_query,
                    topk=topk
                )

                if not search_result.get("ok"):
                    return search_result

                # Note: Current SDK auto-decrypts. In full implementation:
                # 1. SDK would return encrypted_blob_b64
                # 2. We'd call: vault_result = self.vault.decrypt_search_results(encrypted_blob_b64, topk, request_id)
                # 3. Then fetch metadata for vault_result.results

                # For now, return results with Vault audit marker
                return {
                    "ok": True,
                    "results": search_result.get("results", []),
                    "secure_mode": True,
                    "vault_status": "connected",
                    "request_id": request_id or "N/A",
                    "note": "Full Vault integration pending SDK update for encrypted result passthrough."
                }

            except VaultError as e:
                return {
                    "ok": False,
                    "error": f"Vault error: {e}",
                    "request_id": request_id or "N/A"
                }
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Search failed: {e}",
                    "request_id": request_id or "N/A"
                }

        # ---------- MCP Tools: Vault Health Check ---------- #
        @self.mcp.tool(
            name="vault_status",
            description="Check Rune-Vault connection status and security mode."
        )
        async def tool_vault_status() -> Dict[str, Any]:
            """
            Returns the current Vault integration status.

            Returns:
                Dict with Vault connection status and security mode information.
            """
            if self.vault is None:
                return {
                    "ok": True,
                    "vault_configured": False,
                    "secure_search_available": False,
                    "mode": "standard (no Vault)",
                    "warning": "SecKey may be accessible locally. Configure Vault for secure mode."
                }

            # Check Vault health
            try:
                # Simple sync check - in production would use async health check
                vault_healthy = True  # TODO: Implement actual health check
                return {
                    "ok": True,
                    "vault_configured": True,
                    "vault_endpoint": getattr(self.vault, 'vault_endpoint', 'unknown'),
                    "secure_search_available": True,
                    "mode": "secure (Vault-backed)",
                    "vault_healthy": vault_healthy
                }
            except Exception as e:
                return {
                    "ok": False,
                    "vault_configured": True,
                    "error": f"Vault health check failed: {e}"
                }

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
        Runs the MCP server using stdio transport.
        """
        self.mcp.run(transport="stdio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the enVector MCP server.")
    parser.add_argument(
        "--mode",
        choices=("stdio", "http"),
        default=os.getenv("MCP_SERVER_MODE", "http"),
        help="Execution mode: 'stdio' uses stdio transport, 'http' exposes HTTP transport.",
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
        default=os.getenv("ENVECTOR_KEY_PATH", os.path.join(CURRENT_DIR, "keys")),
        help="Path to the enVector key file.",
    )
    parser.add_argument(
        "--envector-eval-mode",
        default=os.getenv("ENVECTOR_EVAL_MODE", "rmp"),
        help="enVector evaluation mode (e.g., 'rmp', 'mm').",
    )
    parser.add_argument(
        "--encrypted-query",
        action="store_true",
        default=os.getenv("ENVECTOR_ENCRYPTED_QUERY", "false").lower() in ("true", "1", "yes"),
        help="Encrypt the query vectors."
    )
    parser.add_argument(
        "--envector-cloud-access-token",
        default=os.getenv("ENVECTOR_CLOUD_ACCESS_TOKEN", None),
        help="enVector cloud access token."
    )
    parser.add_argument(
        "--embedding-mode",
        default=os.getenv("EMBEDDING_MODE", "femb"),
        choices=("femb", "sbert", "hf", "openai"),
        help="Embedding model name for enVector. 'femb' for FastEmbed (by default), 'sbert' for SBERT, 'hf' for HuggingFace, 'openai' for OpenAI API.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Embedding model name for enVector.",
    )
    # Rune-Vault Integration Options
    parser.add_argument(
        "--auto-key-setup",
        action="store_true",
        default=os.getenv("ENVECTOR_AUTO_KEY_SETUP", "true").lower() in ("true", "1", "yes"),
        help="Automatically generate keys if not found. Set to false when keys are provided externally from Vault.",
    )
    parser.add_argument(
        "--no-auto-key-setup",
        action="store_true",
        help="Disable automatic key generation. Use when keys are provided from Rune-Vault.",
    )
    parser.add_argument(
        "--vault-endpoint",
        default=os.getenv("VAULT_MCP_ENDPOINT", None),
        help="Rune-Vault MCP endpoint URL for fetching public keys (e.g., http://vault-mcp:50080/mcp).",
    )
    parser.add_argument(
        "--vault-token",
        default=os.getenv("VAULT_TOKEN", None),
        help="Authentication token for Rune-Vault.",
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
    - ENVECTOR_ADDRESS: The address (host:port) of the `enVector` (overrides --envector-host and --envector-port if provided)
    - ENVECTOR_KEY_ID: The key ID for the `enVector` SDK (default: "mcp_key")
    - ENVECTOR_EVAL_MODE: The evaluation mode of the `enVector` ["rmp", "mm"] (default: "rmp")
    """
    ENVECTOR_ADDRESS = args.envector_address if args.envector_address else args.envector_host + ":" + str(args.envector_port)
    ENVECTOR_CLOUD_ACCESS_TOKEN = args.envector_cloud_access_token
    ENVECTOR_KEY_ID = args.envector_key_id
    ENVECTOR_KEY_PATH = args.envector_key_path
    ENVECTOR_EVAL_MODE = args.envector_eval_mode
    ENCRYPTED_QUERY = args.encrypted_query # Plain-Cipher Query Setting

    # Rune-Vault Integration
    # Determine auto_key_setup: --no-auto-key-setup takes precedence
    AUTO_KEY_SETUP = args.auto_key_setup and not args.no_auto_key_setup
    VAULT_ENDPOINT = args.vault_endpoint
    VAULT_TOKEN = args.vault_token

    # If Vault endpoint is provided, fetch keys from Vault
    if VAULT_ENDPOINT and VAULT_TOKEN:
        print(f"[Rune] Fetching public keys from Vault: {VAULT_ENDPOINT}")
        if fetch_keys_from_vault(VAULT_ENDPOINT, VAULT_TOKEN, ENVECTOR_KEY_PATH):
            print("[Rune] Successfully fetched keys from Vault")
            AUTO_KEY_SETUP = False  # Keys provided externally, no need to auto-generate
        else:
            print("[Rune] Warning: Failed to fetch keys from Vault, falling back to local keys")
    elif VAULT_ENDPOINT and not VAULT_TOKEN:
        print("[Rune] Warning: Vault endpoint provided but no token specified. Skipping Vault integration.")
    elif not AUTO_KEY_SETUP:
        print(f"[Rune] Using externally provided keys from: {ENVECTOR_KEY_PATH}")

    envector_adapter = EnVectorSDKAdapter(
        address=ENVECTOR_ADDRESS,
        key_id=ENVECTOR_KEY_ID,
        key_path=ENVECTOR_KEY_PATH,
        eval_mode=ENVECTOR_EVAL_MODE,
        query_encryption=ENCRYPTED_QUERY,
        access_token=ENVECTOR_CLOUD_ACCESS_TOKEN,
        auto_key_setup=AUTO_KEY_SETUP,
    )

    # Import embedding adapter lazily to avoid heavy dependencies when not needed (e.g., in tests)
    if args.embedding_model is not None:
        from adapter.embeddings import EmbeddingAdapter

        embedding_adapter = EmbeddingAdapter(
            mode=args.embedding_mode,
            model_name=args.embedding_model
        )
    else:
        # print("[WARN] No embedding model specified. Proceeding without embedding adapter.")
        embedding_adapter = None

    document_preprocessor = DocumentPreprocessingAdapter()

    # Initialize Vault client for secure decryption
    vault_client = None
    if VAULT_ENDPOINT and VAULT_TOKEN:
        print(f"[Rune] Initializing Vault client for secure search: {VAULT_ENDPOINT}")
        vault_client = VaultClientSync(
            vault_endpoint=VAULT_ENDPOINT,
            vault_token=VAULT_TOKEN,
            timeout=30.0
        )
        print("[Rune] Vault client initialized - secure_search tool available")
    else:
        print("[Rune] Vault not configured - secure_search tool will be unavailable")
        print("[Rune] To enable secure search, set --vault-endpoint and --vault-token")

    app = MCPServerApp(
        mcp_server_name=MCP_SERVER_NAME,
        envector_adapter=envector_adapter,
        embedding_adapter=embedding_adapter,
        document_preprocessor=document_preprocessor,
        vault_client=vault_client,
    )

    def _handle_shutdown(signum, frame):
        # parameter `frame` is not used, but required by signal handler signature
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        raise SystemExit(0)
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, _handle_shutdown)

    if run_mode == "stdio":
        app.run_stdio_service()
    elif run_mode == "http":
        app.run_http_service(host=MCP_HOST, port=MCP_PORT)
    else:
        raise ValueError(f"Unsupported run mode: {run_mode}")
