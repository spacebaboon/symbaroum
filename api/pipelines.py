"""
api/pipelines.py

All retrieval pipeline logic. Imported by both:
  - api/app.py  (FastAPI web server)
  - rag_query.py (CLI)

Call initialise() once at startup before using any pipeline functions.
"""

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator

import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
import ollama as ollama_client

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_DIR     = os.environ.get("SYMBAROUM_INDEX_DIR",    "./index/vector")
BM25_PATH     = os.environ.get("SYMBAROUM_BM25_DIR",     "./index/bm25")
LIGHTRAG_DIR  = os.environ.get("SYMBAROUM_LIGHTRAG_DIR", "./index/lightrag")
LLM_MODEL     = os.environ.get("SYMBAROUM_LLM_MODEL",    "qwen3:14b")
UTILITY_MODEL = os.environ.get("SYMBAROUM_UTILITY_MODEL","qwen3:1.7b")
EMBED_MODEL   = os.environ.get("SYMBAROUM_EMBED_MODEL",  "nomic-embed-text")
EMBED_HOST    = os.environ.get("SYMBAROUM_EMBED_HOST",   "http://127.0.0.1:11436")
DEBUG         = os.environ.get("SYMBAROUM_DEBUG", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

if not DEBUG:
    logging.getLogger("lightrag").setLevel(logging.WARNING)
    logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Module-level state — populated by initialise()
# ---------------------------------------------------------------------------

_rag: LightRAG | None = None
_bm25_retriever: BM25Retriever | None = None
_vector_retriever = None
_reranker: FlagEmbeddingReranker | None = None
_initialised = False

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QA_PROMPT = PromptTemplate(
    "You are an expert on the Symbaroum tabletop RPG rules and the tutorial adventure 'The Promised Land'. "
    "Using the context below, provide a thorough and complete answer. "
    "For rules queries: include all relevant mechanics, numbers, special cases, and cite specific ability or condition names. "
    "For adventure queries: provide information and summaries useful to the Game Master, "
    "including relevant NPCs, locations, challenges, and rewards. "
    "If the context is insufficient, say so clearly.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Detailed Answer:"
)

# ---------------------------------------------------------------------------
# LightRAG Ollama functions (async, used internally by LightRAG)
# ---------------------------------------------------------------------------

async def _nomic_embed(texts: list[str]) -> np.ndarray:
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11436")
    data = await client.embed(
        model=EMBED_MODEL,
        input=texts,
        keep_alive="1h",
    )
    return np.array(data["embeddings"])


async def _ollama_chat_nothink(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.pop("hashing_kv", None)
    kwargs.pop("max_tokens", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    client = ollama_client.AsyncClient(host="http://127.0.0.1:11434")
    response = await client.chat(
        model=LLM_MODEL,
        messages=messages,
        think=False,
        options={"num_ctx": 16384},
    )
    return response.message.content


# ---------------------------------------------------------------------------
# Utility LLM calls — all async, using AsyncClient on port 11435
# ---------------------------------------------------------------------------

async def _extract_keywords(query: str) -> str:
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11435")
    response = await client.chat(
        model=UTILITY_MODEL,
        messages=[{"role": "user", "content":
            "You are helping search a Symbaroum RPG rulebook and adventure. "
            "Extract the most specific searchable terms from this query. "
            "Prefer specific proper nouns and character names over generic terms. "
            "For example:\n"
            "  'elves in The Promised Land' → 'Godrai Saran-Ri'\n"
            "  'the thief who stole the Sun Stone' → 'Keler Sun Stone'\n"
            "  'undead villain in the adventure' → 'Mal-Rogan'\n"
            "Return only 2-4 specific terms, nothing else, no punctuation:\n"
            f"Query: {query}\n"
            "Specific terms:"
        }],
        think=False,
    )
    return response.message.content.strip()


async def _rewrite_query(query: str) -> str:
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11435")
    response = await client.chat(
        model=UTILITY_MODEL,
        messages=[{"role": "user", "content":
            "Rewrite this query to improve search retrieval against a Symbaroum RPG rulebook. "
            "Only use terms and names that are explicitly mentioned in the original query. "
            "Do not add names, factions, or terms not present in the original. "
            "Make the query more specific by expanding abbreviations and clarifying intent. "
            "Return only the rewritten query:\n"
            f"Original: {query}\n"
            "Rewritten:"
        }],
        think=False,
    )
    return response.message.content.strip().strip('/')


async def route(query: str) -> str:
    """Route query to 'hybrid' or 'lightrag' pipeline."""
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11435")
    response = await client.chat(
        model=UTILITY_MODEL,
        messages=[{"role": "user", "content":
            "You are routing queries for a Symbaroum RPG assistant. "
            "Decide which pipeline to use:\n"
            "- 'hybrid': simple factual lookups, rules definitions, stat blocks, "
            "single-topic queries (e.g. 'What is Steadfast?', 'How does initiative work?')\n"
            "- 'lightrag': cross-referencing multiple topics, relationship queries, "
            "adventure+rules connections (e.g. 'What monsters in The Promised Land and their abilities?', "
            "'What abilities relate to Resolute?', 'How do the elves connect to the adventure?')\n"
            "Reply with only the single word 'hybrid' or 'lightrag':\n"
            f"Query: {query}"
        }],
        think=False,
        options={"temperature": 0},
    )
    result = response.message.content.strip().lower()
    return "lightrag" if "lightrag" in result else "hybrid"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def initialise() -> None:
    """Load all indexes. Call once at startup."""
    global _rag, _bm25_retriever, _vector_retriever, _reranker, _initialised

    if _initialised:
        return

    # LlamaIndex settings
    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=600.0,
        temperature=0.1,
        context_window=8192,
        keepalive="24h",
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=EMBED_HOST,
    )

    # Hybrid index
    print("Loading hybrid index (vector + BM25)...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    with open(f"{BM25_PATH}/nodes.json", "r") as f:
        node_data = json.load(f)
    nodes = [
        TextNode(text=n["text"], id_=n["id"], metadata=n["metadata"])
        for n in node_data
    ]
    _bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)
    _vector_retriever = index.as_retriever(similarity_top_k=15)
    _reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=8)
    print("Hybrid index loaded.")

    # LightRAG index
    print("Loading LightRAG index...")
    _rag = LightRAG(
        working_dir=LIGHTRAG_DIR,
        llm_model_func=_ollama_chat_nothink,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=_nomic_embed,
        ),
        addon_params={
            "entity_types": [
                "Character", "Creature", "Ability", "Condition",
                "Attribute", "Faction", "Location", "Artifact",
                "Trait", "Adventure", "Rule",
            ]
        },
    )
    await _rag.initialize_storages()
    await initialize_pipeline_status()
    print("LightRAG index loaded.")

    _initialised = True
    print("All indexes ready.\n")


# ---------------------------------------------------------------------------
# StaticRetriever helper
# ---------------------------------------------------------------------------

class _StaticRetriever(BaseRetriever):
    def __init__(self, nodes):
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._nodes


# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------

async def hybrid_pipeline(query: str) -> AsyncGenerator[str, None]:
    """
    Runs hybrid retrieval (BM25 + vector + reranker).

    Keyword extraction and query rewriting run concurrently via asyncio.gather.
    LlamaIndex retrieval and generation are synchronous and run in a thread
    executor to avoid blocking the event loop.

    Yields the complete answer as a single chunk, followed by a timings sentinel.
    """
    loop = asyncio.get_running_loop()
    timings: dict[str, float] = {}

    # --- keyword extraction + query rewriting in parallel ---
    t = time.time()
    keywords, rewritten = await asyncio.gather(
        _extract_keywords(query),
        _rewrite_query(query),
    )
    timings["keywords+rewrite"] = time.time() - t

    if not keywords.strip():
        keywords = query
    if not rewritten.strip():
        rewritten = query

    if DEBUG:
        print(f"  BM25 keywords: {keywords}")
        print(f"  Rewritten query: {rewritten}")

    # --- retrieval (sync LlamaIndex, run in executor) ---
    t = time.time()
    vector_nodes, bm25_nodes = await asyncio.gather(
        loop.run_in_executor(None, _vector_retriever.retrieve, rewritten),
        loop.run_in_executor(None, _bm25_retriever.retrieve, keywords),
    )
    timings["retrieval"] = time.time() - t

    # --- RRF fusion ---
    t = time.time()
    seen_ids: dict[str, float] = {}
    for rank, node in enumerate(vector_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 1 / (rank + 1)
    for rank, node in enumerate(bm25_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 2 / (rank + 1)

    all_nodes: dict[str, TextNode] = {}
    for n in vector_nodes + bm25_nodes:
        all_nodes[n.node_id] = n.node

    fused = sorted(
        all_nodes.values(), key=lambda n: seen_ids[n.node_id], reverse=True
    )[:15]
    timings["fusion"] = time.time() - t

    # --- reranking (sync, run in executor) ---
    t = time.time()
    fused_with_scores = [NodeWithScore(node=n, score=seen_ids[n.node_id]) for n in fused]

    def _rerank():
        return _reranker.postprocess_nodes(
            fused_with_scores, query_bundle=QueryBundle(query)
        )

    reranked = await loop.run_in_executor(None, _rerank)
    timings["reranking"] = time.time() - t

    if DEBUG:
        print("\n  Reranked chunks:")
        for i, node in enumerate(reranked):
            print(f"  [{i+1}] Score: {node.score:.3f} | {node.text[:100]}")

    # --- answer generation (sync, run in executor) ---
    t = time.time()

    def _generate():
        static_retriever = _StaticRetriever(reranked)
        qe = RetrieverQueryEngine.from_args(
            static_retriever, response_mode="tree_summarize"
        )
        qe.update_prompts({"response_synthesizer:text_qa_template": QA_PROMPT})
        return str(qe.query(query))

    answer = await loop.run_in_executor(None, _generate)
    timings["llm_answer"] = time.time() - t

    yield answer
    yield f"__timings__{json.dumps(timings)}"


# ---------------------------------------------------------------------------
# LightRAG pipeline
# ---------------------------------------------------------------------------

async def lightrag_pipeline(query: str) -> AsyncGenerator[str, None]:
    """
    Queries LightRAG in mix mode.
    Yields the answer (as a single chunk on cache hit, or streamed if supported),
    followed by a timings sentinel.
    """
    t = time.time()

    response = await _rag.aquery(
        query,
        param=QueryParam(mode="mix", enable_rerank=False),
    )

    # aquery returns a plain string (cache hit or non-streaming)
    yield str(response)

    timings = {"lightrag_query": time.time() - t}
    yield f"__timings__{json.dumps(timings)}"