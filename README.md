# Symbaroum GM Assistant

A local, private RAG system for querying Symbaroum rulebooks and adventure content in natural language. Runs entirely on local hardware — no cloud dependencies for inference.

Built as a learning project for RAG/LLM techniques, using Symbaroum as the domain. This could be applied to any similar game system, or other systems, some of the references and prompts are specific to this game system.

## Stack

| Component       | Technology                                         |
| --------------- | -------------------------------------------------- |
| PDF processing  | Docling (HybridChunker, Nomic tokenizer alignment) |
| Embeddings      | `nomic-embed-text` via Ollama                      |
| Vector store    | LlamaIndex with file-backed JSON                   |
| Keyword search  | LlamaIndex BM25Retriever + manual RRF fusion       |
| Reranker        | `BAAI/bge-reranker-base` cross-encoder             |
| LLM (answers)   | `qwen3:14b` via Ollama (port 11434)                |
| LLM (utility)   | `qwen3:1.7b` via Ollama (port 11435), think=False  |
| Framework       | LlamaIndex + direct Ollama Python client           |
| Package manager | uv                                                 |

## Setup

### Prerequisites

- Ollama installed and running
- CUDA-capable GPU recommended (tested on RTX 4080)
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

### Running

```bash
./start.sh          # starts both Ollama instances
uv run rag_symbaroum.py
```

```bash
./stop.sh           # shuts down Ollama instances
```

On first run, Docling converts the PDFs and builds the vector + BM25 indexes (saved to `index/`). Subsequent runs load from disk in seconds.

## How It Works

Queries go through a pipeline:

1. **Query rewriting** — LLM rewrites the query for better vector retrieval
2. **Keyword extraction** — few-shot prompted LLM extracts Symbaroum-specific named entities
3. **Hybrid retrieval** — vector search + BM25, fused with reciprocal rank fusion (BM25 weighted 2x)
4. **Reranking** — cross-encoder reranker scores and filters retrieved chunks
5. **Answer generation** — `qwen3:14b` synthesises a grounded answer from the top chunks

Steps 1 and 2 run in parallel via `ThreadPoolExecutor`. Total query time: ~12s.

## Project Evolution

| Version | What changed                                                         |
| ------- | -------------------------------------------------------------------- |
| v0.1    | Basic vector RAG with LlamaIndex + Nomic                             |
| v0.2    | BM25 hybrid retrieval, manual RRF fusion                             |
| v0.3    | Cross-encoder reranker, query rewriting, few-shot keyword extraction |
| v0.4    | Docling PDF → HybridChunker (replaced Marker Markdown pipeline)      |
| v0.5    | Parallel utility LLM calls, dedicated small model — 5x query speedup |

## Roadmap

- **LightRAG (GraphRAG)** — solve cross-referencing between adventure encounters and bestiary entries
- **Qdrant** — replace in-memory JSON store for multi-document scale
- **Multi-document support** — index remaining Symbaroum books
- **Web UI** — Gradio or FastAPI frontend
- **Content generation** — NPCs, encounters, locations consistent with Symbaroum lore

## Legal

Symbaroum is © [Free League Publishing](https://freeleaguepublishing.com/). This project contains no copyrighted game content — it is a retrieval tool only. You must supply your own legally obtained copies of the Symbaroum PDFs to use it.
