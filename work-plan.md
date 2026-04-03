# Symbaroum GM Assistant — RAG Project Work Plan

> **Living document** — update as priorities shift, new learnings emerge, or scope changes.  
> Primary purpose: learning RAG/LLM techniques through building something useful. Secondary: a working GM assistant for Symbaroum.  
> Commercial relevance: architecture decisions should be kept in mind as a potential SME RAG product template.

---

## Project Overview

A local, private RAG (Retrieval-Augmented Generation) system that allows a Game Master to query Symbaroum rulebooks and adventure content in natural language. Runs entirely on local hardware (RTX 4080, WSL Ubuntu), no cloud dependencies for inference.

### Current Stack

- **PDF processing**: Docling (HybridChunker with Nomic tokenizer alignment)
- **Embeddings**: Nomic-embed-text via Ollama (port 11436, dedicated instance)
- **Vector store**: LlamaIndex in-memory JSON (file-backed)
- **BM25**: LlamaIndex BM25Retriever with manual RRF fusion
- **Knowledge graph**: LightRAG (NetworkX + NanoVectorDB, file-backed)
- **Reranker**: BAAI/bge-reranker-base (cross-encoder)
- **Query router**: qwen3:1.7b classifies queries → hybrid or LightRAG path
- **LLM (answers)**: qwen3:14b via Ollama (port 11434)
- **LLM (utility)**: qwen3:1.7b via Ollama (port 11435), think=False
- **Web server**: FastAPI + Starlette StreamingResponse (SSE)
- **Framework**: LlamaIndex + LightRAG + direct Ollama Python client
- **Package manager**: uv

### Key Design Decisions Made

- Docling over Marker for chunking (context-aware, heading-prefixed embeddings)
- Manual RRF fusion over QueryFusionRetriever (more control, BM25 2x weighting)
- Direct async Ollama client for all utility calls (avoids event loop deadlocks in FastAPI)
- `asyncio.gather()` for parallel utility LLM calls (replaces ThreadPoolExecutor)
- Native Linux filesystem (~/) over /mnt/g/ (5x I/O performance improvement)
- Three concurrent Ollama instances to keep all models resident in VRAM simultaneously
- LightRAG over LazyGraphRAG (fully open source, active Ollama support)
- Router temperature=0 for deterministic path selection
- LightRAG `mix` mode for queries (graph traversal + vector combined)
- `build_lightrag_index.py` as a separate one-time indexing script (not part of inference)
- Pipeline logic in `api/pipelines.py` shared between FastAPI server and CLI
- Raw `StreamingResponse` with SSE wire format over `sse-starlette` (version compatibility issues with dict format in 2.x+)
- Newlines in SSE token data escaped as `\n` to avoid breaking SSE framing

---

## Completed Milestones

### v0.1 — Basic Vector RAG

- LlamaIndex + Ollama + Nomic embeddings
- SimpleDirectoryReader from Marker-converted Markdown
- Basic query loop with retrieved chunk debugging

### v0.2 — Hybrid Retrieval

- BM25 + vector search
- Manual reciprocal rank fusion (BM25 2x weighting)
- Fixed BM25 node persistence (JSON, not pickle)

### v0.3 — Reranking & Query Rewriting

- BAAI/bge-reranker-base cross-encoder reranker
- LLM-based query rewriting for better vector retrieval
- LLM-based keyword extraction with few-shot examples (Symbaroum-specific)
- Improved chunking thresholds (1500 char safety net)

### v0.4 — Docling Context-Aware Chunking

- Replaced Marker Markdown pipeline with Docling PDF → HybridChunker
- Nomic tokenizer alignment (512 token max)
- contextualize() prepends heading to every chunk
- Resolved Steadfast and adventure stat block retrieval
- Docling JSON cached for instant reload

### v0.5 — Performance Optimisation

- Parallel utility LLM calls (ThreadPoolExecutor)
- Dedicated small model for utility calls (qwen3:1.7b, port 11435)
- Direct Ollama Python client with think=False
- 5x query speedup: 60s → 12s
- Start/stop shell scripts for VRAM management

### v0.6 — LightRAG Knowledge Graph + Query Router ✓

- `build_lightrag_index.py`: standalone one-time indexer using Docling HybridChunker output
- LightRAG index: 881 chunks → 4215 entities, 8211 edges (core rulebook + tutorial adventure)
- Three Ollama instances (11434/11435/11436) kept resident simultaneously — no VRAM eviction
- `rag_query.py`: new inference script combining both pipelines behind a query router
- Router: qwen3:1.7b classifies queries → hybrid (rules/factual) or LightRAG (relational/cross-ref)
- LightRAG response caching: repeat queries ~1s (vs ~60s first call)
- Logging cleanup: LightRAG INFO suppressed in normal mode, DEBUG flag for full output
- Routing determinism: temperature=0 on router call

### v0.7 — FastAPI Web UI with SSE Streaming ✓

- Refactored pipeline logic into `api/pipelines.py` — shared by both web server and CLI
- `api/app.py`: FastAPI server with `POST /query` SSE streaming endpoint
- `api/models.py`: Pydantic request/response models
- `static/index.html`: single-page frontend — dark theme, markdown rendering, query history chips, question displayed above answer
- All utility LLM calls converted to fully async (`AsyncClient`) — eliminates event loop deadlocks
- `asyncio.gather()` replaces `ThreadPoolExecutor` for parallel keyword/rewrite calls
- SSE via raw `StreamingResponse` (sse-starlette 3.x incompatible with dict yield format)
- `rag_query.py` retained as thin CLI wrapper importing from `api/pipelines.py`
- Query input cleared after submission; question shown above answer in UI

---

## Current State

### What Works Well

- Rules queries: accurate, well-cited answers for abilities, conditions, combat rules
- Adventure queries: finding NPCs, scenes, tactics with correct details
- Cross-reference queries: LightRAG correctly handles adventure+rules relationships
- Named entity retrieval: few-shot keyword extraction resolves generic terms to named entities
- Routing: consistently sends factual queries to hybrid, relational queries to LightRAG
- Query timing: hybrid ~10–40s, LightRAG ~60s first call / ~1s cached
- Web UI: streaming responses, pipeline badge, query history chips, markdown rendering
- All three models stay resident in VRAM — no load/unload overhead

### Known Limitations

- LightRAG INFO logging cannot be suppressed via Python logging (uses print internally)
- Keyword extraction occasionally misfires on abstract queries (rewards, thematic questions)
- LightRAG first-call latency ~60s (mitigated by response cache for repeat queries)
- Single document only (core rulebook + tutorial adventure)
- Hybrid path delivers answer as a single chunk (no token-level streaming — LlamaIndex query engine is synchronous)

---

## Immediate Next Steps

### Task 3: Qdrant Vector Database

**Goal**: Replace in-memory JSON store with proper vector DB for multi-document scale  
**Priority**: Medium (needed before adding a second document)  
**Estimated effort**: 0.5 sessions

#### Subtasks

- Install Qdrant via Docker: `docker run -p 6333:6333 qdrant/qdrant`
- Add `llama-index-vector-stores-qdrant` dependency
- Replace `VectorStoreIndex` JSON store with Qdrant backend in `api/pipelines.py`
- Test incremental upsert (add new book without full rebuild)
- Add Qdrant startup to start.sh
- Update stop.sh to preserve Qdrant data between sessions
- Git tag: v0.8-qdrant

---

### Task 4: Multi-Document Support

**Goal**: Add remaining Symbaroum content (adventure modules, additional sourcebooks)  
**Priority**: High (unlocks the full GM assistant vision)  
**Dependencies**: Qdrant (Task 3)

#### Subtasks

- Inventory all owned Symbaroum PDFs
- Run Docling conversion on each (one-time, save JSON)
- Design metadata schema: `{source: "core_rulebook", type: "rules|adventure|bestiary"}`
- Add per-document metadata to nodes at index time
- Update query engine to support metadata filtering
- Update keyword extraction prompt with new content awareness
- Rebuild LightRAG index with all documents
- Test cross-document queries: "adventures that feature corruption themes"

---

## Medium-Term Roadmap

### Task 5: Homebrew Content Generation

**Goal**: Generate NPCs, encounters, locations consistent with Symbaroum lore  
**Priority**: Medium  
**Dependencies**: Good multi-document coverage, LightRAG

#### Subtasks

- Design generation prompts using retrieved lore as context
- NPC generator: name, background, stats appropriate to location/role
- Encounter generator: monsters appropriate to region + threat level
- Location generator: ruins/settlements consistent with Davokar lore
- Test output consistency with canonical content
- Consider structured output (JSON stat blocks vs narrative)

---

### Task 6: NPC/Encounter Generator (Structured)

**Goal**: Generate game-ready stat blocks and encounter descriptions  
**Priority**: Medium  
**Dependencies**: Task 5

#### Subtasks

- Define Symbaroum stat block schema (all attributes, abilities, traits, tactics)
- Fine-tune prompts to generate valid stat blocks
- Validate generated stats against game balance guidelines from rulebook
- Consider lightweight fine-tuning on extracted stat block data
- Integration with web UI as a separate generator tab

---

## Long-Term Considerations

### Potential Commercial Direction

The architecture being built here maps cleanly to an SME RAG product:

- Docling for document processing
- Hybrid retrieval (vector + BM25 + reranker)
- LightRAG for knowledge graph
- Qdrant for scalable vector storage
- FastAPI + SSE streaming web UI

Key differences for a commercial product:

- Cloud LLM API (Mistral/Claude) instead of local Ollama
- Multi-tenant document isolation
- Authentication and usage tracking
- EU data residency (Mistral AI or OVHcloud inference)
- GDPR compliance documentation

### Other RPG Systems

The pipeline is system-agnostic. Once working well for Symbaroum, the same approach applies to:

- Call of Cthulhu (already own PDFs)
- Alien RPG (already own PDFs)

A future refactor could make the system configurable per-game with different prompt templates.

---

## Technical Debt & Cleanup

- Add few-shot example for rewards/experience queries to keyword extraction prompt
- Handle Ollama connection errors gracefully (retry logic, friendly error in web UI)
- Add query history to a local SQLite DB for reviewing past sessions
- Consider abstracting the retrieval pipeline so LightRAG and hybrid RAG share a common interface
- True token streaming on hybrid path (requires bypassing LlamaIndex query engine for answer generation)
- Update start.sh to launch uvicorn alongside Ollama instances
- Update `.env.sample` with all current config vars

---

## Environment Reference

```
Hardware:     RTX 4080 16GB, Windows 11, WSL2 Ubuntu
Project:      ~/projects/symbaroum/
Data:         ~/projects/symbaroum/data/ (gitignored)
Indexes:      ~/projects/symbaroum/index/vector/, index/bm25/, index/lightrag/
Models:       ~/.ollama/models/
Primary LLM:  qwen3:14b on localhost:11434
Utility LLM:  qwen3:1.7b on localhost:11435 (think=False, temperature=0 for routing)
Embeddings:   nomic-embed-text on localhost:11436
```

### Session Startup

```bash
~/projects/symbaroum/start.sh

# Web UI:
cd ~/projects/symbaroum && uv run uvicorn api.app:app --host 0.0.0.0 --port 8000

# CLI:
cd ~/projects/symbaroum && uv run rag_query.py
```

### Session Shutdown

```bash
~/projects/symbaroum/stop.sh
```

### Git Tags

| Tag                            | Description                                                   |
| ------------------------------ | ------------------------------------------------------------- |
| v0.1-basic-rag                 | Basic vector RAG with LlamaIndex                              |
| v0.2-hybrid-retrieval          | BM25 + vector hybrid search                                   |
| v0.3-reranking-query-rewriting | Cross-encoder reranker, query rewriting                       |
| v0.4-docling-chunking          | Docling HybridChunker integration                             |
| v0.5-performance               | Parallel utility calls, 5x speedup                            |
| v0.6-lightrag                  | LightRAG knowledge graph, query router, three-instance Ollama |
| v0.7-webui                     | FastAPI web UI, SSE streaming, pipeline refactor              |

---

## Session Notes

### 2026-03-21 / 22 / 23 (Sessions 1-3)

- Established baseline: PDF → Marker → Markdown → LlamaIndex
- Migrated WSL to G: drive, fixed C: drive space crisis
- Moved to uv, native Linux filesystem
- Built hybrid RAG, reranker, query rewriting
- Switched to Docling for context-aware chunking
- Implemented parallel utility LLM calls
- Key insight: Docling's contextualize() (heading-prefixed text) dramatically improves retrieval
- Key insight: keyword extraction needs few-shot Symbaroum examples to resolve generic terms to named entities
- Key insight: utility LLM calls were 81% of total query time — small model + direct client → 5x speedup

### 2026-04-03 (Session 4)

- Built `build_lightrag_index.py` — standalone one-time LightRAG indexer using Docling HybridChunker output
- Used `test_lightrag.py` to prove LightRAG queries work with three-Ollama-instance setup before committing to full index build
- Full index build: 881 chunks → 4215 entities, 8211 edges
- Built `rag_query.py` — new inference script with query router + both pipelines
- Built FastAPI web UI with SSE streaming: `api/app.py`, `api/pipelines.py`, `api/models.py`, `static/index.html`
- Refactored pipeline logic into shared module consumed by both CLI and web server
- UI fixes via Claude Code: query cleared after submit, question shown above answer, env var deduplication
- Key insight: qwen3.5 models do not reliably obey think=False — stayed with qwen3 family
- Key insight: three concurrent Ollama instances (14b/1.7b/nomic) all fit in RTX 4080 VRAM simultaneously
- Key insight: LightRAG response cache makes repeat queries ~1s regardless of graph complexity
- Key insight: sync Ollama client deadlocks inside FastAPI event loop — must use AsyncClient throughout
- Key insight: sse-starlette 3.x dropped dict yield format — raw StreamingResponse more reliable

---

_Last updated: 2026-04-03_
