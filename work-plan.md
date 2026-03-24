# Symbaroum GM Assistant — RAG Project Work Plan

> **Living document** — update as priorities shift, new learnings emerge, or scope changes.  
> Primary purpose: learning RAG/LLM techniques through building something useful. Secondary: a working GM assistant for Symbaroum.  
> Commercial relevance: architecture decisions should be kept in mind as a potential SME RAG product template.

---

## Project Overview

A local, private RAG (Retrieval-Augmented Generation) system that allows a Game Master to query Symbaroum rulebooks and adventure content in natural language. Runs entirely on local hardware (RTX 4080, WSL Ubuntu), no cloud dependencies for inference.

### Current Stack

- **PDF processing**: Docling (HybridChunker with Nomic tokenizer alignment)
- **Embeddings**: Nomic-embed-text via Ollama
- **Vector store**: LlamaIndex in-memory JSON (file-backed)
- **BM25**: LlamaIndex BM25Retriever with manual RRF fusion
- **Reranker**: BAAI/bge-reranker-base (cross-encoder)
- **LLM (answers)**: qwen3:14b via Ollama
- **LLM (utility)**: qwen3:1.7b via second Ollama instance (port 11435), think=False
- **Framework**: LlamaIndex + direct Ollama Python client
- **Package manager**: uv

### Key Design Decisions Made

- Docling over Marker for chunking (context-aware, heading-prefixed embeddings)
- Manual RRF fusion over QueryFusionRetriever (more control, BM25 2x weighting)
- Direct Ollama client for utility calls (bypasses LlamaIndex for think=False support)
- Parallel utility LLM calls (ThreadPoolExecutor, ~0.5s vs ~48s sequential)
- Native Linux filesystem (~/) over /mnt/g/ (5x I/O performance improvement)

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

---

## Current State

### What Works Well

- Rules queries: accurate, well-cited answers for abilities, conditions, combat rules
- Adventure queries: finding NPCs, scenes, tactics with correct details
- Named entity retrieval: few-shot keyword extraction resolves "elves" → "Godrai Saran-Ri"
- Query timing: ~12s total (0.5s utility, 2s retrieval/reranking, 8-10s answer)

### Known Limitations

- Cross-referencing: adventure encounter descriptions + bestiary stat blocks in separate chunks
- qwen3.5:9b: excellent answer quality but think=False not working via LlamaIndex
- "What are the main events in The Promised Land" occasionally fails (retrieval variance)
- Rewards/experience queries miss adventure-specific content
- Single document only (core rulebook + tutorial adventure)

---

## Immediate Next Steps

### Task 1: LightRAG Integration (GraphRAG)

**Goal**: Solve the cross-referencing problem — connect adventure encounters to bestiary entries to rule descriptions  
**Priority**: High  
**Estimated effort**: 1-2 sessions

#### Why LightRAG over LazyGraphRAG

LazyGraphRAG is conceptually superior (NLP-only indexing, zero LLM calls at index time) but Microsoft's standalone open source implementation is not fully available for local use — it's primarily deployed via Azure/Microsoft Discovery. LightRAG is fully open source, actively maintained, has explicit Ollama support, and the concepts transfer directly. LazyGraphRAG remains worth revisiting if a clean local release lands.

#### Subtasks

- [ ] Install LightRAG (`uv add lightrag-hku`)
- [ ] Write Docling→LightRAG text converter (reuse contextualized HybridChunker output)
- [ ] Configure LightRAG with Ollama LLM and Nomic embeddings
- [ ] Benchmark entity extraction models (qwen3:8b vs qwen3:14b vs qwen3:1.7b)
- [ ] Run one-time indexing, save graph to `index/lightrag/`
- [ ] Build query router: rules queries → hybrid RAG, adventure/cross-ref queries → LightRAG
- [ ] Test cross-reference queries: "monsters in The Promised Land with abilities"
- [ ] Add LightRAG index path to .env and start/stop scripts
- [ ] Git tag: v0.6-lightrag

#### Research notes

- LightRAG input: plain text or Markdown (not Docling JSON natively)
- Solution: custom loader using contextualized HybridChunker output joined with separators
- LightRAG GitHub: `HKUDS/LightRAG`
- Key config: custom LLM function (async) pointing at Ollama, custom embedding function pointing at Nomic
- Entity extraction is a one-time cost — ~900 chunks on qwen3:14b, expect 30-60 mins
- Consider qwen3:8b for extraction: cheaper, still good at structured entity recognition

---

### Task 2: qwen3.5:9b Custom LLM Wrapper (Optional)

**Goal**: Enable qwen3.5:9b for answer generation with thinking disabled  
**Priority**: Low-Medium (investigate as separate task)  
**Estimated effort**: 0.5 sessions

#### Subtasks

- [ ] Create custom LlamaIndex LLM class wrapping direct Ollama client
- [ ] Pass think=False at the API level, not via additional_kwargs
- [ ] Test answer quality vs qwen3:14b on standard query set
- [ ] Benchmark: qwen3.5:9b no-think vs qwen3:14b with thinking
- [ ] Update .env and start.sh if adopted

#### Notes

- qwen3.5:9b without thinking: excellent quality, 35s (too slow)
- qwen3.5:9b with broken think=False: 71s + garbage output (unusable)
- qwen3:14b current baseline: ~12s total, good quality
- Target: qwen3.5:9b at ~12s with equal or better quality

---

## Near-Term Roadmap

### Task 3: Multi-Document Support

**Goal**: Add remaining Symbaroum content (adventure modules, additional sourcebooks)  
**Priority**: High (unlocks the full GM assistant vision)  
**Dependencies**: Qdrant (Task 4)

#### Subtasks

- [ ] Inventory all owned Symbaroum PDFs
- [ ] Run Docling conversion on each (one-time, save JSON)
- [ ] Design metadata schema: `{source: "core_rulebook", type: "rules|adventure|bestiary"}`
- [ ] Add per-document metadata to nodes at index time
- [ ] Update query engine to support metadata filtering
- [ ] Update keyword extraction prompt with new content awareness
- [ ] Test cross-document queries: "adventures that feature corruption themes"

---

### Task 4: Qdrant Vector Database

**Goal**: Replace in-memory JSON store with proper vector DB for multi-document scale  
**Priority**: Medium (needed before second document)  
**Estimated effort**: 0.5 sessions

#### Subtasks

- [ ] Install Qdrant via Docker: `docker run -p 6333:6333 qdrant/qdrant`
- [ ] Add `llama-index-vector-stores-qdrant` dependency
- [ ] Replace `VectorStoreIndex` JSON store with Qdrant backend
- [ ] Test incremental upsert (add new book without full rebuild)
- [ ] Add Qdrant startup to start.sh
- [ ] Update stop.sh to preserve Qdrant data between sessions
- [ ] Git tag: v0.7-qdrant

---

### Task 5: Simple Web UI

**Goal**: A basic web interface — nicer than CLI, good learning exercise in serving ML pipelines  
**Priority**: Medium (nice to have, not critical)  
**Estimated effort**: 1 session

#### Subtasks

- [ ] Choose framework: Gradio (quickest) or FastAPI + minimal HTML (more control)
- [ ] Implement REST endpoint: POST /query → {answer, timings, chunks}
- [ ] Basic web UI: query input, answer display, source chunks toggle
- [ ] Handle streaming responses for faster perceived performance
- [ ] Consider session history: last N queries visible
- [ ] Git tag: v0.8-webui

#### Notes

- Gradio is the fastest path to a working UI and good for learning
- FastAPI + minimal HTML is better if you want to understand the serving layer
- Not mobile-critical — desktop browser is fine

---

## Medium-Term Roadmap

### Task 6: Homebrew Content Generation

**Goal**: Generate NPCs, encounters, locations consistent with Symbaroum lore  
**Priority**: Medium  
**Dependencies**: Good multi-document coverage, LightRAG

#### Subtasks

- [ ] Design generation prompts using retrieved lore as context
- [ ] NPC generator: name, background, stats appropriate to location/role
- [ ] Encounter generator: monsters appropriate to region + threat level
- [ ] Location generator: ruins/settlements consistent with Davokar lore
- [ ] Test output consistency with canonical content
- [ ] Consider structured output (JSON stat blocks vs narrative)

---

### Task 7: NPC/Encounter Generator (Structured)

**Goal**: Generate game-ready stat blocks and encounter descriptions  
**Priority**: Medium  
**Dependencies**: Task 7

#### Subtasks

- [ ] Define Symbaroum stat block schema (all attributes, abilities, traits, tactics)
- [ ] Fine-tune prompts to generate valid stat blocks
- [ ] Validate generated stats against game balance guidelines from rulebook
- [ ] Consider lightweight fine-tuning on extracted stat block data
- [ ] Integration with web UI as a separate generator tab

---

## Long-Term Considerations

### Potential Commercial Direction

The architecture being built here maps cleanly to an SME RAG product:

- Docling for document processing
- Hybrid retrieval (vector + BM25 + reranker)
- LightRAG for knowledge graph
- Qdrant for scalable vector storage
- FastAPI + web UI

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

- [ ] Remove timing instrumentation from production query loop (or make DEBUG-only)
- [ ] Add `.env.sample` entries for all new config vars as they're added
- [ ] Write README.md with setup instructions
- [ ] Handle Ollama connection errors gracefully (retry logic)
- [ ] Add query history to a local SQLite DB for reviewing past sessions
- [ ] Consider abstracting the retrieval pipeline so LightRAG and hybrid RAG share a common interface
- [ ] Suppress XLMRobertaTokenizerFast warning properly

---

## Environment Reference

```
Hardware:     RTX 4080 16GB, Windows 11, WSL2 Ubuntu on G:\WSL\Ubuntu
Project:      ~/projects/symbaroum/
Data:         ~/projects/symbaroum/data/ (gitignored)
Indexes:      ~/projects/symbaroum/index/vector/ and index/bm25/
Models:       ~/.ollama/models/
Primary LLM:  qwen3:14b on localhost:11434
Utility LLM:  qwen3:1.7b on localhost:11435 (think=False)
Embeddings:   nomic-embed-text via Ollama
```

### Session Startup

```bash
~/projects/symbaroum/start.sh
cd ~/projects/symbaroum && uv run rag_symbaroum.py
```

### Session Shutdown

```bash
~/projects/symbaroum/stop.sh
```

### Git Tags

| Tag                            | Description                             |
| ------------------------------ | --------------------------------------- |
| v0.1-basic-rag                 | Basic vector RAG with LlamaIndex        |
| v0.2-hybrid-retrieval          | BM25 + vector hybrid search             |
| v0.3-reranking-query-rewriting | Cross-encoder reranker, query rewriting |
| v0.4-docling-chunking          | Docling HybridChunker integration       |
| v0.5-performance               | Parallel utility calls, 5x speedup      |

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

---

_Last updated: 2026-03-24_
