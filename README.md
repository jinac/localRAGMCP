# local RAG MCP
This repo is an example of putting together a RAG system. It was built trying to create a vector store of local pdfs and expose them via MCP for search and context.

## To run MCP
'uv run main.py'

## To run MCPO/OpenAPI Tool Server
'uvx mcpo --port 8000 -- uv run main.py'