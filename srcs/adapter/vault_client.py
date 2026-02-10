"""
Vault Client for Rune-Vault MCP Integration

This client handles communication between envector-mcp-server and Rune-Vault.
All decryption operations are delegated to Vault, which holds the SecKey.

Security Model:
- MCP server NEVER has access to SecKey
- All decryption requests go through Vault
- Audit trail maintained by Vault
"""

import os
import json
import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DecryptResult:
    """Result from Vault decryption of the result ciphertext."""
    ok: bool
    results: List[Dict[str, Any]]  # [{index: int, score: float}, ...] — similarity values
    request_id: str
    timestamp: float
    total_vectors: int = 0
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecryptResult":
        return cls(
            ok=data.get("ok", False),
            results=data.get("results", []),
            request_id=data.get("request_id", ""),
            timestamp=data.get("timestamp", 0),
            total_vectors=data.get("total_vectors", 0),
            error=data.get("error")
        )


class VaultError(Exception):
    """Error communicating with Vault."""
    pass


class VaultClient:
    """
    Async HTTP client for Rune-Vault MCP decryption service.

    The Vault holds the FHE SecKey and performs all decryption operations.
    This client sends result ciphertext (from encrypted similarity search) to Vault
    and receives top-k indices with similarity values.

    Usage:
        client = VaultClient(
            vault_endpoint="http://vault:50080",
            vault_token="your-token"
        )
        result = await client.decrypt_search_results(
            encrypted_blob_b64="base64...",
            top_k=5
        )
    """

    def __init__(
        self,
        vault_endpoint: str,
        vault_token: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize Vault client.

        Args:
            vault_endpoint: URL to Vault MCP (e.g., "http://vault-mcp:50080")
            vault_token: Authentication token for Vault
            timeout: Request timeout in seconds
            max_retries: Number of retries on transient failures
        """
        self.vault_endpoint = vault_endpoint.rstrip("/")
        self.vault_token = vault_token
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def decrypt_search_results(
        self,
        encrypted_blob_b64: str,
        top_k: int = 5,
        request_id: Optional[str] = None
    ) -> DecryptResult:
        """
        Call Vault MCP to decrypt result ciphertext from encrypted similarity
        search.

        The Cloud computes inner products between the encrypted query and
        stored encrypted embeddings, producing an LWE ciphertext packed
        into CKKS LRWE form. This method sends that result ciphertext to
        Vault for decryption with SecKey.

        Args:
            encrypted_blob_b64: Base64-encoded result ciphertext from
                encrypted similarity search on enVector Cloud
            top_k: Number of top results to return (max 10)
            request_id: Optional correlation ID for audit trail

        Returns:
            DecryptResult with top-k indices and similarity values

        Raises:
            VaultError: If Vault call fails after retries
        """
        if not request_id:
            request_id = f"mcp_{uuid.uuid4().hex[:12]}"

        # Build MCP tool call payload
        # FastMCP SSE endpoint expects tool calls in a specific format
        payload = {
            "method": "tools/call",
            "params": {
                "name": "decrypt_scores",
                "arguments": {
                    "token": self.vault_token,
                    "encrypted_blob_b64": encrypted_blob_b64,
                    "top_k": top_k,
                    "request_id": request_id
                }
            }
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                client = await self._get_client()

                # Call Vault MCP endpoint
                # FastMCP typically exposes /mcp/v1/tools/call or similar
                response = await client.post(
                    f"{self.vault_endpoint}/mcp/v1/tools/call",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    result_data = response.json()

                    # Extract tool result from MCP response
                    if "result" in result_data:
                        tool_result = result_data["result"]
                        if isinstance(tool_result, str):
                            tool_result = json.loads(tool_result)
                        return DecryptResult.from_dict(tool_result)
                    elif "content" in result_data:
                        # Alternative response format
                        content = result_data["content"]
                        if isinstance(content, list) and len(content) > 0:
                            text = content[0].get("text", "{}")
                            return DecryptResult.from_dict(json.loads(text))

                    # Unexpected format
                    raise VaultError(f"Unexpected response format: {result_data}")

                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("Retry-After", "5"))
                    logger.warning(f"Vault rate limited, waiting {retry_after}s")
                    await self._async_sleep(retry_after)
                    continue

                else:
                    raise VaultError(
                        f"Vault returned {response.status_code}: {response.text}"
                    )

            except httpx.TimeoutException as e:
                last_error = VaultError(f"Vault timeout: {e}")
                logger.warning(f"Vault timeout (attempt {attempt + 1}/{self.max_retries})")

            except httpx.HTTPError as e:
                last_error = VaultError(f"HTTP error: {e}")
                logger.warning(f"Vault HTTP error (attempt {attempt + 1}/{self.max_retries}): {e}")

            except json.JSONDecodeError as e:
                last_error = VaultError(f"Invalid JSON from Vault: {e}")
                break  # Don't retry JSON errors

            # Exponential backoff
            if attempt < self.max_retries - 1:
                await self._async_sleep(2 ** attempt)

        raise last_error or VaultError("Unknown error communicating with Vault")

    async def _async_sleep(self, seconds: float):
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)

    async def health_check(self) -> bool:
        """Check if Vault is reachable."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.vault_endpoint}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Vault health check failed: {e}")
            return False


class VaultClientSync:
    """
    Synchronous wrapper for VaultClient.

    Use this when async is not available (e.g., in synchronous code paths).
    """

    def __init__(
        self,
        vault_endpoint: str,
        vault_token: str,
        timeout: float = 30.0
    ):
        self.vault_endpoint = vault_endpoint.rstrip("/")
        self.vault_token = vault_token
        self.timeout = timeout

    def decrypt_search_results(
        self,
        encrypted_blob_b64: str,
        top_k: int = 5,
        request_id: Optional[str] = None
    ) -> DecryptResult:
        """
        Synchronously call Vault MCP to decrypt result ciphertext.

        Args:
            encrypted_blob_b64: Base64-encoded result ciphertext from
                encrypted similarity search on enVector Cloud
            top_k: Number of top results to return
            request_id: Optional correlation ID

        Returns:
            DecryptResult with top-k indices and similarity values
        """
        if not request_id:
            request_id = f"mcp_sync_{uuid.uuid4().hex[:12]}"

        payload = {
            "method": "tools/call",
            "params": {
                "name": "decrypt_scores",
                "arguments": {
                    "token": self.vault_token,
                    "encrypted_blob_b64": encrypted_blob_b64,
                    "top_k": top_k,
                    "request_id": request_id
                }
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.vault_endpoint}/mcp/v1/tools/call",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    result_data = response.json()
                    if "result" in result_data:
                        tool_result = result_data["result"]
                        if isinstance(tool_result, str):
                            tool_result = json.loads(tool_result)
                        return DecryptResult.from_dict(tool_result)

                raise VaultError(f"Vault returned {response.status_code}: {response.text}")

        except Exception as e:
            raise VaultError(f"Failed to call Vault: {e}")


def create_vault_client(
    vault_endpoint: Optional[str] = None,
    vault_token: Optional[str] = None,
    async_mode: bool = True
) -> Optional[VaultClient | VaultClientSync]:
    """
    Factory function to create Vault client from environment variables.

    Environment variables:
    - VAULT_ENDPOINT: URL to Vault MCP (e.g., "http://vault:50080")
    - VAULT_TOKEN: Authentication token for Vault

    Args:
        vault_endpoint: Override for VAULT_ENDPOINT
        vault_token: Override for VAULT_TOKEN
        async_mode: If True, return async client; else sync client

    Returns:
        VaultClient or VaultClientSync if configured, None otherwise
    """
    endpoint = vault_endpoint or os.getenv("VAULT_ENDPOINT")
    token = vault_token or os.getenv("VAULT_TOKEN")

    if not endpoint or not token:
        logger.info("Vault not configured (VAULT_ENDPOINT or VAULT_TOKEN missing)")
        return None

    if async_mode:
        return VaultClient(vault_endpoint=endpoint, vault_token=token)
    else:
        return VaultClientSync(vault_endpoint=endpoint, vault_token=token)
