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
- Python Packages

    ```bash
    pip install -r requirements.txt
    ```

- Environment Variable Set-Up

    1. Use `.env` file

        ```bash
        source .env
        ```

    2. CLI overrides

        Every setting has default value, but, you can check option with `python srcs/server.py --help` and overwrite each value with CLI.

    If omitted this environment variable setup, all parameters in `server.py` follow:
        1) `.env`,
        2) Evironment Variable, and
        3) Default Values.

## Run MCP Server

### 1. How to run MCP Server in your service

Configurate your config files (e.g. `/path/to/Claude/claude_desktop_config.json`):

```json
{
    "mcpServers": {
        "enVectorMCP": {
            "command": "/path/to/python",
            "args": [
                "/path/to/envector-mcp-server/srcs/server.py",
                "--mode",
                "http",
                "--envector-address",
                "ENVECTORHOST:50050",
                "--envector-key-path",
                "/path/to/keys"
            ],
            "cwd": "/path/to/envector-mcp-server",
            "description": "enVector MCP server stores the user's vector data and their corresponding metadata for semantic search."
        },
        ...
    }
}
```

Note that,
- some AI service providers including Claude Desktop have an option that 1) run the MCP server in the service, and 2) connect the running MCP server.

### 2. How to run MCP Server directly

Run the following Python script in `/path/to/envector-mcp-server/`:

```bash
# Remote HTTP mode (default)
python srcs/server.py \
    --mode "http" \
    --host "localhost" \
    --port "8000" \
    --server-name "envector_mcp_server" \
    --envector-address "ENVECTORHOST:50050" \
    --envector-key-id "mcp_key" \
    --envector-key-path "/path/to/keys"

# Local STDIO mode (for MCP desktop integrations)
python srcs/server.py \
    --mode "stdio"
```

Note that,
- `stdio` mode communicate with standard I/O only, so log might not be seen. Please connect to MCP Host.


## Connect MCP Server (Client)

### 1. Attach to your AI service (Recommended)

Attach to your AI service (e.g. Claude, Gemini, VSCode, etc.).

For example, in Gemini CLI, configurate `.gemini/settings.json` to connect the running enVector MCP server:

```json
{
    "mcpServers": [
        {
            "name": "envector-mcp-server",
            "httpUrl": "http://localhost:8000/mcp",
            "description": "enVector MCP server stores the user's vector data and their corresponding metadata for semantic search."
        },
        ...
    ],
    ...
}
```

The configuration files in AI services:
- Claude Desktop: `claude_desktop_config.json`
- Gemini CLI: `.gemini/settings.json`
- Cursor: `.cursor/mcp.json`
- Codex: `.codes/config.toml`
- Cline: `cline_mcp_settings.json`

### 2. Use Python client

Python package `fastmcp` provices Client method.
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

### 3. Use `curl`

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
