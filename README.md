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

## Overall plan

Start with the simplest RAG approach, then keep refining with more sophisicated techniques, comparing the quality
of the results as we go. The project is more about learning within a known context than making something that I need.

1. Simple RAG approach

2. Add BG25 keyword tokenisation and combine results

3. Use reranker to order results for better scoring

4. Add GraphRAG to better use relationships between contexts

5. Convert to LazyGraphRAG approach
