# Symbaroum GM Assistant

A local, private RAG system for querying Symbaroum rulebooks and adventure content in natural language. Runs entirely on local hardware — no cloud dependencies for inference.

Built as a learning project for RAG/LLM techniques, using Symbaroum as the domain. This could be applied to any similar game system; some references and prompts are specific to Symbaroum.

## Stack

| Component       | Technology                                           |
| --------------- | ---------------------------------------------------- |
| PDF processing  | Docling (HybridChunker, Nomic tokenizer alignment)   |
| Embeddings      | `nomic-embed-text` via Ollama (port 11436)           |
| Vector store    | LlamaIndex with file-backed JSON                     |
| Keyword search  | LlamaIndex BM25Retriever + manual RRF fusion         |
| Knowledge graph | LightRAG (NetworkX + NanoVectorDB, file-backed)      |
| Reranker        | `BAAI/bge-reranker-base` cross-encoder               |
| Query router    | `qwen3:1.7b` classifies queries → hybrid or LightRAG |
| LLM (answers)   | `qwen3:14b` via Ollama (port 11434)                  |
| LLM (utility)   | `qwen3:1.7b` via Ollama (port 11435), think=False    |
| Web server      | FastAPI + Starlette StreamingResponse (SSE)          |
| Framework       | LlamaIndex + LightRAG + direct Ollama Python client  |
| Package manager | uv                                                   |

All three Ollama instances run concurrently to keep models resident in VRAM, avoiding load/unload overhead between pipeline steps.

## File Structure

```
symbaroum/
  api/
    __init__.py
    app.py          # FastAPI server, SSE streaming endpoint
    models.py       # Pydantic request/response models
    pipelines.py    # all retrieval logic (shared by web UI and CLI)
  static/
    index.html      # single-page web frontend
  rag_query.py      # CLI entry point (imports from api/pipelines.py)
  rag_symbaroum.py  # legacy v0.5 script (kept for reference)
  build_lightrag_index.py   # one-time LightRAG indexer
  test_lightrag.py  # LightRAG smoke tests
  start.sh / stop.sh
```

## Setup

### Prerequisites

- Ollama installed and running
- CUDA-capable GPU recommended (tested on RTX 4080 16GB)
- uv

### Install

```bash
git clone <repo>
cd symbaroum
uv sync
```

Pull the required Ollama models:

```bash
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
```

Place your Symbaroum PDFs in `data/`.

### Building the indexes

On first run, build the vector and BM25 indexes:

```bash
uv run rag_symbaroum.py
```

This converts PDFs via Docling and saves indexes to `index/vector/` and `index/bm25/`. Subsequent runs load from disk in seconds.

Build the LightRAG knowledge graph index (one-time, slow — expect several hours on a single book):

```bash
./start.sh
uv run build_lightrag_index.py
```

The indexer saves progress after each chunk and resumes automatically if interrupted.

### Running

```bash
./start.sh                                                        # start all three Ollama instances
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload   # web UI
uv run rag_query.py                                               # CLI alternative
```

```bash
./stop.sh    # shut down Ollama instances
```

Set `SYMBAROUM_DEBUG=1` for detailed retrieval output and timings.

## How It Works

Each query goes through a routing pipeline:

1. **Routing** — `qwen3:1.7b` classifies the query as `hybrid` or `lightrag`
2. **Hybrid path** (rules lookups, stat blocks, single-topic factual queries):
   - Query rewriting and keyword extraction run in parallel via `asyncio.gather`
   - Vector search + BM25 retrieval, fused with reciprocal rank fusion (BM25 weighted 2x)
   - Cross-encoder reranker (`bge-reranker-base`) scores and filters to top 8 chunks
   - `qwen3:14b` synthesises a grounded answer
3. **LightRAG path** (cross-referencing, relationships, adventure+rules connections):
   - Queries the knowledge graph in `mix` mode (graph traversal + vector search combined)
   - `qwen3:14b` synthesises from graph context with LLM response caching for repeat queries

Typical query times: hybrid ~10–40s, LightRAG ~60s first call / ~1s cached.

## Project Evolution

| Version | What changed                                                         |
| ------- | -------------------------------------------------------------------- |
| v0.1    | Basic vector RAG with LlamaIndex + Nomic                             |
| v0.2    | BM25 hybrid retrieval, manual RRF fusion                             |
| v0.3    | Cross-encoder reranker, query rewriting, few-shot keyword extraction |
| v0.4    | Docling PDF → HybridChunker (replaced Marker Markdown pipeline)      |
| v0.5    | Parallel utility LLM calls, dedicated small model — 5x query speedup |
| v0.6    | LightRAG knowledge graph, query router, three-instance Ollama setup  |
| v0.7    | FastAPI web UI, SSE streaming, pipeline refactor into shared module  |

## Roadmap

- **Qdrant** — replace in-memory JSON store for multi-document scale
- **Multi-document support** — index remaining Symbaroum books
- **Content generation** — NPCs, encounters, locations consistent with Symbaroum lore

## Legal

Symbaroum is © [Free League Publishing](https://freeleaguepublishing.com/). This project contains no copyrighted game content — it is a retrieval tool only. You must supply your own legally obtained copies of the Symbaroum PDFs to use it.
