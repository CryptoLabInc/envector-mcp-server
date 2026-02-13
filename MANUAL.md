# enVector MCP Server User Manual
## Description
This document let users know how to use `enVector MCP Server`

## Repository Structure (Essentials Only)
```bash
├── MANUAL.md                       # User Manual
├── README.md                       # Introduction of enVector MCP Server
├── requirements.txt                # Required Python Package
├── srcs
│   ├── adapter
│   │   ├── __init__.py
│   │   ├── document_preprocess.py  # Document Preprocessor for loading and chunking
│   │   ├── embeddings.py           # Embedding Model
│   │   └── envector_sdk.py         # `enVector` SDK Adapter (Class)
│   └── server.py                   # MCP Server entrypoint (HTTP/STDIO modes)
└── tests                           # Test Codes (pyTest)
    └── test_server.py
```

## Supporting Tools

- `get_index_list`: Get the list of indexes in enVector.
- `get_index_info`: Get information about a specific index in enVector.
- `create_index`: Create an index in enVector.
- `insert`: Insert vectors and the corresponding metadata into enVector index. Support to specify embedding model to get embedding vectors to insert.
- `search`: Perform homomorphic encrypted vector similarity search and retrieve metadata from enVector. Support to specify embedding model to get embedding vectors to search.
- `remember`: Vault-secured organizational memory recall. Orchestrates a 3-step pipeline: (1) homomorphic encrypted vector similarity search on encrypted index, (2) Vault decryption of result ciphertext + top-k selection, (3) metadata retrieval. Requires `RUNEVAULT_ENDPOINT` and `RUNEVAULT_TOKEN` environment variables.
- `vault_status`: Check Rune-Vault connection status and security mode.
- `insert_documents_from_path`: Insert documents from the given path. Support to read and chunk the document file, get embedding of texts and insert them into enVector.
- `insert_documents_from_text`: Insert documents from the given texts. Support to chunk the document file, get embedding of texts and insert them into enVector.

## Prerequisites
- Python 3.10+ (3.12 recommended)

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

- Python Packages

    ```bash
    pip install -r requirements.txt
    ```

- Environment Variable Set-Up

    1. Use `.env` file to set environmental variables

    2. CLI Options

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
                "--envector-key-path",
                "/path/to/keys"
            ],
            "env": {
                "ENVECTOR_ADDRESS": "cluster-xxx.clusters.envector.io",
                "ENVECTOR_API_KEY": "YOUR_API_KEY"
            },
            "cwd": "/path/to/envector-mcp-server",
            "description": "enVector MCP server stores the user's vector data and their corresponding metadata for semantic search."
        },
    }
}
```

Note that, some AI service providers including Claude Desktop have an option that 1) run the MCP server in the service, and 2) connect the running MCP server.

### 2. How to run MCP Server directly

Run the following Python script in `/path/to/envector-mcp-server/`:

```bash
# Remote HTTP mode (default) - enVector Cloud
export ENVECTOR_ADDRESS="cluster-xxx.clusters.envector.io"
export ENVECTOR_API_KEY="YOUR_API_KEY"

python srcs/server.py \
    --mode "http" \
    --host "localhost" \
    --port "8000" \
    --server-name "envector_mcp_server" \
    --envector-key-id "mcp_key" \
    --envector-key-path "/path/to/keys" \
    --embedding-mode "femb" \
    --embedding-model "sentence-transformers/all-MiniLM-L6-v2"

# Local STDIO mode (for MCP desktop integrations)
python srcs/server.py \
    --mode "stdio"
```

Note that,
- `stdio` mode communicate with standard I/O only, so log might not be seen. Please connect to MCP Host.

## MCP Server Options

### CLI Options

Arguments to run Python scripts:

- 💻 MCP execution
    - `--mode`: MCP execution mode, supporting `http` (default) and `stdio` transports.
    - `--host`: MCP HTTP bind host. The default is `127.0.0.1`.
    - `--port`: MCP HTTP bind port. The default is `8000`.
    - `--address`: MCP HTTP bind address. Overrides `--host` and `--port` if provided.
    - `--server-name`: MCP server name. The default is `envector_mcp_server`.

- 🔌 enVector connection
    - `--envector-address` or `ENVECTOR_ADDRESS`: enVector endpoint address (`{host}:{port}` or enVector Cloud endpoint ends with `.clusters.envector.io`). For Cloud, prefer environment variable.
    - `ENVECTOR_API_KEY` (env var only): access token of enVector Cloud.

- 🔑 enVector options
    - `--envector-key-id`: enVector key id (identifier).
    - `--envector-key-path`: path to enVector key files.
    - `--envector-eval-mode`: enVector FHE evaluation mode. Recommend to use `rmp` (default) mode for more flexible usage.
    - `--encrypted-query`: whether to encrypt the query vectors. The index is encrypted by default.
    - `--no-auto-key-setup`: disable automatic key generation (default: auto-generate enabled). Use when keys are provided externally (e.g., from Rune-Vault).

    > ⚠️ **Note**: MCP server holds the key for homomorphic encryption as MCP server is a enVector Client.

- 🔐 Rune-Vault Integration (Optional, env var only)
    - `RUNEVAULT_ENDPOINT`: Rune-Vault MCP endpoint URL for fetching public keys.
    - `RUNEVAULT_TOKEN`: Authentication token for Rune-Vault.

    > 💡 **Rune Integration**: When integrated with Rune, the Vault MCP manages cryptographic keys centrally. The envector-mcp-server fetches public keys (EncKey, EvalKey) from Vault at startup, while secret key remains securely in Vault for decryption operations. See [Rune Architecture](#rune-integration) for details.

    > ⚠️ **Security**: Credentials (`ENVECTOR_API_KEY`, `RUNEVAULT_TOKEN`) must be provided via environment variables only.

- ⚙️ Embedding options
    - `--embedding-mode`: Mode of the embedding model. Supports `femb` (FastEmb), `hf` (huggingface), `sbert` (SBERT; sentence-transformers), and `openai` (OpenAI API). For `openai`, required to set environmental variable `OPENAI_API_KEY`.
    - `--embedding-model`: Embedding model name to use enVector. The `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` is set as the default, whose embedding dimension is 384.

<details>
<summary>Supporting embedding models</summary>

- models supported by [`fastembed`](https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models)
- models supported by [`transformers`](https://huggingface.co/models?pipeline_tag=sentence-similarity&library=transformers&sort=trending)
- models supported by [`sentence-transformers`](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- models supported by [`openai`](https://platform.openai.com/docs/guides/embeddings)

</details>

### Use environment variables

Copy `.env.example` to `.env` and configure `.env` as you want.

```bash
# MCP execution
MCP_SERVER_MODE="http"
MCP_SERVER_ADDRESS="127.0.0.1:8000"
MCP_SERVER_NAME="envector_mcp_server"

# enVector Cloud
ENVECTOR_ADDRESS="cluster-xxx.clusters.envector.io"
ENVECTOR_API_KEY=""

# enVector options
ENVECTOR_KEY_ID="mcp_key"
ENVECTOR_KEY_PATH="./keys"
ENVECTOR_EVAL_MODE="rmp"
ENVECTOR_ENCRYPTED_QUERY="false"
ENVECTOR_AUTO_KEY_SETUP="true"

# Rune-Vault information
RUNEVAULT_ENDPOINT=""
RUNEVAULT_TOKEN=""

# Embedding mode
EMBEDDING_MODE="femb"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```


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
    ],
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


## Rune Integration

When used with [Rune](https://github.com/CryptoLabInc/rune), the envector-mcp-server operates in a distributed key management architecture:

### Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Rune Architecture                      │
└───────────────────────────────────────────────────────────┘

  Agent (Claude/Gemini/Custom)
    │
    │  MCP tool calls: insert, search, remember, vault_status
    ▼
  ┌──────────────────────────────────────────┐
  │         envector-mcp-server              │  ← Scalable Workers
  ├──────────────────────────────────────────┤
  │ Uses: EncKey, EvalKey (public keys)      │
  │                                          │
  │ Tools:                                   │
  │  insert / search    → direct pipeline    │
  │  remember           → Vault pipeline:    │
  │    1. scoring() → ciphertext             │
  │    2. Vault decrypt + top-k selection    │
  │    3. get_metadata_by_indices → results  │
  └─────────┬──────────────────┬─────────────┘
            │                  │
            ▼                  ▼
  ┌──────────────────┐  ┌──────────────────────┐
  │  enVector Cloud  │  │   Rune-Vault MCP     │
  │ (Encrypted Store)│  │  (secret key holder) │
  │                  │  │  - get_public_key()  │
  │ Encrypted vectors│  │  - decrypt_scores()  │
  │  & metadata      │  │  Admin-controlled    │
  └──────────────────┘  └──────────────────────┘
```

### Configuration for Rune

#### Option 1: Fetch keys from Rune-Vault at startup (Recommended)

```bash
export ENVECTOR_ADDRESS="cluster-xxx.clusters.envector.io"
export ENVECTOR_API_KEY="YOUR_API_KEY"
export RUNEVAULT_ENDPOINT="http://vault-mcp:50080/mcp"
export RUNEVAULT_TOKEN="envector-team-alpha"

python srcs/server.py \
    --mode "http" \
    --no-auto-key-setup
```

#### Option 2: Use pre-distributed keys

```bash
export ENVECTOR_ADDRESS="cluster-xxx.clusters.envector.io"
export ENVECTOR_API_KEY="YOUR_API_KEY"

# Keys are pre-distributed to /shared/keys by Vault or deployment pipeline
python srcs/server.py \
    --mode "http" \
    --envector-key-path "/shared/keys" \
    --no-auto-key-setup
```

### Environment Variables for Rune

```bash
# Disable auto key generation
ENVECTOR_AUTO_KEY_SETUP="false"

# Rune-Vault integration (Option 1)
RUNEVAULT_ENDPOINT="http://vault-mcp:50080/mcp"
RUNEVAULT_TOKEN="envector-team-alpha"

# Pre-distributed keys (Option 2)
ENVECTOR_KEY_PATH="/shared/keys"
```

### Docker Compose Example

```yaml
services:
  vault-mcp:
    image: rune/vault-mcp:latest
    volumes:
      - vault_keys:/secure/keys
    ports:
      - "127.0.0.1:50080:50080"

  envector-mcp:
    image: envector/mcp-server:latest
    environment:
      - RUNEVAULT_ENDPOINT=http://vault-mcp:50080/mcp
      - RUNEVAULT_TOKEN=${RUNEVAULT_TOKEN}
      - ENVECTOR_AUTO_KEY_SETUP=false
      - ENVECTOR_ADDRESS=cluster-xxx.clusters.envector.io
      - ENVECTOR_API_KEY=${ENVECTOR_API_KEY}
    depends_on:
      - vault-mcp
```

### Key Distribution Flow

1. **Startup**: envector-mcp-server calls Vault's `get_public_key` tool
2. **Key Fetch**: Vault returns EncKey.json, EvalKey.json
3. **Local Save**: Keys are saved to `--envector-key-path` directory
4. **SDK Init**: pyenvector SDK initializes with fetched keys (`auto_key_setup=False`)
5. **Operations**: Insert/Search operations use public keys for encryption

### Remember Pipeline (Vault-Secured Recall)

When an agent calls the `remember` tool, the MCP server orchestrates a 3-step pipeline:

1. **Search**: `index.scoring(query)` → result ciphertext (base64-serialized). The Cloud computes encrypted similarity scores and packs into a result ciphertext.
2. **Vault Decrypt**: Send result ciphertext to Vault's `decrypt_scores(token, blob, top_k)`. Vault decrypts with secret key to obtain similarity values, selects top-k → `[{index, score}, ...]`
3. **Retrieve**: `index.indexer.get_metadata(indices)` → plaintext metadata returned to agent

Secret key never leaves Vault. The MCP server and agent only see the result ciphertext and final metadata.
