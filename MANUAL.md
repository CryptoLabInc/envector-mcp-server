# enVector MCP Server User Manual
## Description
This document let users know how to use `enVector MCP Server`

## Repository Structure (Essentials Only)
```bash
srcs/
 ├─ server.py               # MCP Server entrypoint (HTTP/STDIO modes)
 └─ adapters/
     └─ enVector_sdk.py     # `enVector` SDK Adapter (Class)
examples/
 └─ ...                     # Example Codes
tests/
 └─ ...                     # Test Codes (pyTest)
requirements.txt            # List of Required Package
README.md                   # Introduction of `enVector MCP Server` GitHub repository
MANUAL.md                   # User Manual
```

## Basic requirements
- Python 3.10+ (3.12 recommended)
- Packages
    ```bash
    pip install -r requirements.txt
    ```
- Environment Variable Set-Up
    + Use `.env` file (Recommended)
        ```bash
        source .env
        ```
    + CLI overrides (optional)
        Every setting has default value, but, you can check option with `python srcs/server.py --help` and overwrite each value with CLI.

## Run (Server/Client)
### How to use (MCP Server)
```bash
# Remote HTTP mode (default)
python srcs/server.py --mode remote \
    --host 0.0.0.0 \
    --port 8000 \
    --server-name envector_mcp_server \
    --envector-endpoint 127.0.0.1 \
    --envector-port 50050 \
    --envector-key-id mcp_key \
    --envector-eval-mode mm

# Local STDIO mode (for MCP desktop integrations)
python srcs/server.py --mode local
```
- If omitted, all parameter follows
    1) `.env`
    2) Evironment Variable
    3) Default Values
- `STDIO` mode communicate with standard I/O only, so log might not be seen. Please connect to MCP Host.

### How to use (Client)
1. Attach to IDE or else

    ex. `VS Code`, `Claude`, etc.
     - Please follow description of each module
     - For example, (Claude Desktop)
        ```json
        {
            "mcpServers": {
                "enVectorMCP": {
                "command": "/path/to/python",
                "args": [
                    "/path/to/envector-mcp-server/srcs/server.py",
                    "--mode",
                    "local",
                    "--envector-endpoint",
                    "IP",
                    "--envector-port",
                    "50050"
                ]
                }
            }
        }
        ```

2. Use Python client (FastMCP Client)

    For example,
    ```python
    import asyncio
    from fastmcp import Client

    async def main():
        client = Client("http://localhost:8000/mcp")
        async with client:
            tools = await client.list_tools()
            print([t.name for t in tools])  # ['search', ...]

            result = await client.call_tool(
                "search", {"index_name": "test_index_name", ...}
                # and so on...
            )

            print(result)           # Instance
            # print(result.data)    # JSON (Different from version)
            # print(result.content) # Text Block (or else)

    asyncio.run(main())
    ```

3. Use `curl`

    Basic format is `JSON-RPC 2.0`
    1) Create Session
        ```bash
        curl -i -X POST http://localhost:8000/mcp \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d '{
            "jsonrpc":"2.0",
            "id":1,
            "method":"initialize",
            "params":{
            "protocolVersion":"2025-06-18",
            "capabilities":{"sampling":{}, "elicitation":{}},
            "clientInfo":{"name":"curl-test","version":"0.1.0"}
            }
        }'
        ```

    2) Notice Initialization Completed
        ```bash
        curl -i -X POST http://localhost:8000/mcp \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -H 'MCP-Protocol-Version: 2025-06-18' \
        -H 'Mcp-Session-Id: {RESPONSED SESSION ID}' \
        -d '{
            "jsonrpc":"2.0",
            "method":"notifications/initialized"
        }'

        ```

    3) List up tool-list: `tools/list`
        ```bash
        curl -sS -X POST http://localhost:8000/mcp \
            -H 'Content-Type: application/json' \
            -d '{
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }'
        ```

    4) Run tool: `tools/call`
        ```bash
        curl -sS -X POST http://localhost:8000/mcp \
            -H 'Content-Type: application/json' \
            -d '{
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                "name": "tool_name",
                "arguments": { "paramA": "valueA", "paramB": valueB }
                }
            }'
        ```

## Fast Trouble Shooting
### Error List
- 404/405:
    + Is URL `/mcp`?
    + Is HTTP method `Post`?
- Unknown tool:
    + Is tool name correct?
- Input type error:
    + Check TypeHint
