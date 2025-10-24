# enVector-MCP-Server
We supply MCP Server of `enVector`, `CryptoLab Inc.`'s HE(Homomorphic Encryption)-Based vector search engine.

## What is MCP?
`MCP`, which stands for `Model Context Protocol`, is a protocol used by AI application for access to external data, tool, worflow, and so on. It is kind of pre-defined JSON format protocol.

There are 3 participant in `MCP` communication
- Host
    + AI application
    + ex. `VS Code`, `Claude`, and so on
- Client
    + Connection module of Host
    + Form of expansion or add-on module (1:1 for each server)
- Server
    + Supplier of Data/Tools
    + In our case, `enVector`

## How `enVector-MCP-Server` can be implemented to services?
As `enVector` is vector search engine based on HE, this `enVector MCP Server` can be used in some cases like below.
1. AI chat-bot emplaced in private network(IntraNet/Air-gapped Net) need to use protection-required dataset. To get data, AI send query to `enVector` via `enVector MCP Server`. `enVector` do vector search and returns encryptred scoreboard. Then AI decrypt scoreboard and require most appropriate vector's metadata to secured DB. After get response, AI decrypt dataset and show it to user.
2. SW Developer, who is affiliated at some SW Dev Team, is runnig new project. They wanna refer to their previous project to reuse some codes. In case they are using code assistant module and their previous project is under protection with private repository, code assistant generate new skeleton codes first and then, try to search similar codes in DB. As codes are protected as encrypted form, code assistant AI call `enVector MCP Server` to search code candidates via `enVector`. Then, `enVector` returns scoreboard of codes stored in secured DB. Code Assistant AI now can require top-k code blocks stored in specific index in DB. After decrypting returned code blocks, code assistant AI can improve skeleton codes with them.
In given two cases, `enVector` can run on anywhere and each terminal(devices) user just need to add `enVector MCP Server` on their AI assistant (or else). With pre-defined protocol that `enVector MCP Server` uses, all 'secured-data search' will be processed automatically.