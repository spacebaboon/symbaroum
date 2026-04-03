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
- **Query router**: qwen3:1.7b classifies queries → hybrid or lightrag path
- **LLM (answers)**: qwen3:14b via Ollama (port 11434)
- **LLM (utility)**: qwen3:1.7b via Ollama (port 11435), think=False
- **Framework**: LlamaIndex + LightRAG + direct Ollama Python client
- **Package manager**: uv

### Key Design Decisions Made

- Docling over Marker for chunking (context-aware, heading-prefixed embeddings)
- Manual RRF fusion over QueryFusionRetriever (more control, BM25 2x weighting)
- Direct Ollama client for utility calls (bypasses LlamaIndex for think=False support)
- Parallel utility LLM calls (ThreadPoolExecutor, ~0.5s vs ~48s sequential)
- Native Linux filesystem (~/) over /mnt/g/ (5x I/O performance improvement)
- Three concurrent Ollama instances to keep all models resident in VRAM simultaneously
- LightRAG over LazyGraphRAG (fully open source, active Ollama support; LazyGraphRAG not cleanly available for local use)
- Router temperature=0 for deterministic path selection
- LightRAG `mix` mode for queries (graph traversal + vector combined)
- `build_lightrag_index.py` as a separate one-time indexing script (not part of inference)
- `rag_query.py` as the primary inference entry point (replaces `rag_symbaroum.py` for daily use)

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

---

## Current State

### What Works Well

- Rules queries: accurate, well-cited answers for abilities, conditions, combat rules
- Adventure queries: finding NPCs, scenes, tactics with correct details
- Named entity retrieval: few-shot keyword extraction resolves generic terms to named entities
- Cross-reference queries: LightRAG correctly handles adventure+rules relationships
- Routing: consistently sends factual queries to hybrid, relational queries to LightRAG
- Query timing: hybrid ~10–40s, LightRAG ~60s first call / ~1s cached
- All three models stay resident in VRAM — no load/unload overhead

### Known Limitations

- LightRAG INFO logging cannot be suppressed via Python logging (uses print internally)
- Keyword extraction occasionally misfires on abstract queries (e.g. extracted "Godrai Saran-Ri" for a rewards query) — BM25 path still recovers via reranker
- LightRAG first-call latency ~60s (mitigated by response cache for repeat queries)
- Single document only (core rulebook + tutorial adventure)
- `asyncio.get_event_loop()` deprecation warning during LightRAG init (cosmetic, not functional)

---

## Immediate Next Steps

### Task 2: Qdrant Vector Database

**Goal**: Replace in-memory JSON store with proper vector DB for multi-document scale  
**Priority**: Medium (needed before adding a second document)  
**Estimated effort**: 0.5 sessions

#### Subtasks

- Install Qdrant via Docker: `docker run -p 6333:6333 qdrant/qdrant`
- Add `llama-index-vector-stores-qdrant` dependency
- Replace `VectorStoreIndex` JSON store with Qdrant backend
- Test incremental upsert (add new book without full rebuild)
- Add Qdrant startup to start.sh
- Update stop.sh to preserve Qdrant data between sessions
- Git tag: v0.7-qdrant

---

### Task 3: Multi-Document Support

**Goal**: Add remaining Symbaroum content (adventure modules, additional sourcebooks)  
**Priority**: High (unlocks the full GM assistant vision)  
**Dependencies**: Qdrant (Task 2)

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

### Task 4: FastAPI Web UI with Streaming

**Goal**: Serve the GM assistant via a web interface with streaming responses, replacing the CLI REPL  
**Priority**: Medium  
**Estimated effort**: 1-2 sessions  
**Dependencies**: None (can run alongside existing `rag_query.py` CLI)

---

#### Overview

Refactor `rag_query.py` from a top-to-bottom script into a proper FastAPI application with:

- A streaming `POST /query` endpoint using server-sent events (SSE)
- A single-page HTML frontend served by FastAPI
- Startup/shutdown lifecycle hooks for index loading
- Full async pipeline (removing `asyncio.run()` workarounds)

The existing pipeline logic, prompts, and routing should be preserved exactly — this is a serving layer change, not a retrieval change.

---

#### File Structure

```
symbaroum/
  api/
    __init__.py
    app.py           # FastAPI app, lifespan, /query endpoint
    pipelines.py     # refactored hybrid + lightrag pipeline functions (async)
    models.py        # Pydantic request/response models
  static/
    index.html       # single-page frontend
  rag_query.py       # CLI entry point (kept, imports from api/)
```

---

#### Subtasks

**4.1 — Add dependencies**

```
uv add fastapi uvicorn[standard] sse-starlette
```

`sse-starlette` provides the `EventSourceResponse` class for server-sent events.

---

**4.2 — Create `api/models.py`**

Pydantic models for request/response:

```python
from pydantic import BaseModel
from typing import Literal

class QueryRequest(BaseModel):
    query: str

class QueryMetadata(BaseModel):
    pipeline: Literal["hybrid", "lightrag"]
    timings: dict[str, float]
    total: float
```

---

**4.3 — Refactor pipelines into `api/pipelines.py`**

Extract all pipeline logic from `rag_query.py` into importable async functions. Key changes from the current script:

- All index loading moved into an `initialise()` async function (called once at startup)
- `run_hybrid_pipeline()` becomes `async def hybrid_pipeline(query: str) -> AsyncGenerator[str, None]`
- `run_lightrag_pipeline()` becomes `async def lightrag_pipeline(query: str) -> AsyncGenerator[str, None]`
- Both yield token chunks as strings for SSE streaming
- `route_query()` and utility LLM calls use `asyncio.get_event_loop().run_in_executor()` to avoid blocking the event loop (they use the sync Ollama client)
- All state (indexes, rag instance, retrievers, reranker) held as module-level variables, initialised once

**Hybrid pipeline streaming approach:**
LlamaIndex's `query_engine.query()` is synchronous. Options:

- Run it in a thread executor and stream the complete response as a single SSE event (simplest)
- Switch to `astream_chat()` if using a chat engine — yields tokens natively (more complex)
- Recommended for v0.7: run in executor, emit a "thinking" SSE event while waiting, then emit the full answer. True token streaming can be added later.

**LightRAG streaming approach:**
LightRAG's `aquery()` supports streaming via `QueryParam(stream=True)` which returns an `AsyncGenerator`. Yield each chunk directly as an SSE event. This gives true token-level streaming for the LightRAG path.

---

**4.4 — Create `api/app.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from api.models import QueryRequest
from api import pipelines
import json, time

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipelines.initialise()   # load all indexes on startup
    yield
    # optional: graceful shutdown of Ollama clients

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/query")
async def query(request: QueryRequest):
    async def event_stream():
        total_start = time.time()

        # Route
        pipeline = await pipelines.route(request.query)
        yield {"event": "routing", "data": json.dumps({"pipeline": pipeline})}

        # Stream answer
        timings = {}
        t = time.time()
        if pipeline == "lightrag":
            async for chunk in pipelines.lightrag_pipeline(request.query):
                yield {"event": "token", "data": chunk}
            timings["lightrag_query"] = time.time() - t
        else:
            async for chunk in pipelines.hybrid_pipeline(request.query):
                yield {"event": "token", "data": chunk}
            timings["hybrid"] = time.time() - t

        # Done event with metadata
        yield {
            "event": "done",
            "data": json.dumps({
                "pipeline": pipeline,
                "timings": timings,
                "total": time.time() - total_start,
            })
        }

    return EventSourceResponse(event_stream())
```

---

**4.5 — Create `static/index.html`**

Single self-contained HTML file, no build step, no external JS frameworks. Requirements:

- Text input + submit button
- Answer display area that appends tokens as they arrive (SSE `token` events)
- Pipeline badge: shows `hybrid` or `lightrag` once routing event arrives
- Timing display: shown after `done` event
- Loading state: disable input + show spinner while query is in flight
- Query history: last 5 queries shown as clickable chips to re-run
- Error handling: display friendly message if SSE connection fails

Use the `EventSource` API with `fetch` + `ReadableStream` (since `EventSource` doesn't support POST):

```javascript
const response = await fetch("/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query }),
});
const reader = response.body.getReader();
// parse SSE lines manually: "event: token\ndata: ...\n\n"
```

Style: minimal, dark theme, monospace answer font. No CSS frameworks needed.

---

**4.6 — Keep `rag_query.py` as CLI entry point**

Update to import from `api/pipelines.py` rather than duplicating logic:

```python
# rag_query.py — CLI wrapper
import asyncio
from api import pipelines

async def main():
    await pipelines.initialise()
    # existing REPL loop, calling pipelines.hybrid_pipeline() / pipelines.lightrag_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
```

This ensures CLI and web UI stay in sync — one source of truth for pipeline logic.

---

**4.7 — Update `start.sh`**

Add uvicorn startup:

```bash
# In start.sh, after starting Ollama instances:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload &
echo "Web UI: http://localhost:8000"
```

`--reload` is fine for development; remove for stable use.

---

**4.8 — Update `.env.sample`**

Add:

```
SYMBAROUM_HOST=0.0.0.0
SYMBAROUM_PORT=8000
```

---

#### Testing Checklist

- [ ] `uv run uvicorn api.app:app` starts without errors, indexes load
- [ ] `GET /` returns the HTML page
- [ ] `POST /query` with a rules query streams tokens and ends with `done` event
- [ ] `POST /query` with a relational query routes to lightrag and streams
- [ ] Frontend displays tokens as they arrive (not all at once after completion)
- [ ] Pipeline badge appears immediately after routing (before answer streams)
- [ ] CLI (`uv run rag_query.py`) still works after refactor
- [ ] `start.sh` brings up all services including uvicorn

---

#### Notes & Decisions

- **True streaming on hybrid path**: LlamaIndex's synchronous query engine makes true token streaming non-trivial. Emitting a single SSE event with the complete answer after running in a thread executor is acceptable for v0.7. The LightRAG path will have real token streaming. Revisit if the UX difference is noticeable.
- **No authentication**: localhost only, not exposed externally. If WSL port forwarding is needed, add a note but no auth logic required.
- **CORS**: Not needed for localhost serving from the same FastAPI app. Add if frontend is ever served separately.
- **Git tag**: `v0.7-webui`

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

- Add few-shot example for rewards/experience queries to keyword extraction prompt
- Remove timing instrumentation from production query loop (or make DEBUG-only) ✓ (done in v0.6)
- Add `.env.sample` entries for all new config vars as they're added
- Write README.md with setup instructions ✓ (done in v0.6)
- Handle Ollama connection errors gracefully (retry logic)
- Add query history to a local SQLite DB for reviewing past sessions
- Consider abstracting the retrieval pipeline so LightRAG and hybrid RAG share a common interface
- Suppress XLMRobertaTokenizerFast warning properly ✓ (done in v0.6)
- Fix `asyncio.get_event_loop()` deprecation in LightRAG init (cosmetic)

---

## Environment Reference

```
Hardware:     RTX 4080 16GB, Windows 11, WSL2 Ubuntu on G:\WSL\Ubuntu
Project:      ~/projects/symbaroum/
Data:         ~/projects/symbaroum/data/ (gitignored)
Indexes:      ~/projects/symbaroum/index/vector/, index/bm25/, index/lightrag/
Models:       ~/.ollama/models/
Primary LLM:  qwen3:14b on localhost:11434
Utility LLM:  qwen3:1.7b on localhost:11435 (think=False, temperature=0 for routing)
Embeddings:   nomic-embed-text on localhost:11436
```

### Session Startup

```
~/projects/symbaroum/start.sh
cd ~/projects/symbaroum && uv run rag_query.py
```

### Session Shutdown

```
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
- Full index build: 881 chunks → 4215 entities, 8211 edges (completed in tmux)
- Built `rag_query.py` — new inference script with query router + both pipelines
- Tested against benchmark queries: routing consistently correct, answer quality good on both paths
- Key insight: qwen3.5 models do not reliably obey think=False — stayed with qwen3 family
- Key insight: three concurrent Ollama instances (14b/1.7b/nomic) all fit in RTX 4080 VRAM simultaneously
- Key insight: LightRAG response cache makes repeat queries ~1s regardless of graph complexity
- Known issue: keyword extraction misfires occasionally on abstract queries (rewards, thematic questions)

---

_Last updated: 2026-04-03_
