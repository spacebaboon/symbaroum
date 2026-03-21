# Symbaroum RAG

A local RAG (Retrieval-Augmented Generation) system for querying the Symbaroum Core Rulebook, using [LlamaIndex](https://www.llamaindex.ai/) and [Ollama](https://ollama.com/).

## Stack

- **LLM**: `qwen3:14b` via Ollama
- **Embeddings**: `nomic-embed-text` via Ollama
- **Indexing**: LlamaIndex with persistent vector store

## Usage

1. Install dependencies:
   ```
   pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama
   ```

2. Ensure Ollama is running with the required models:
   ```
   ollama pull qwen3:14b
   ollama pull nomic-embed-text
   ```

3. Run:
   ```
   python rag_symbaroum.py
   ```

On first run, the index is built from the markdown-converted rulebook and persisted to `index/`. Subsequent runs load from disk.
