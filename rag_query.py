import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from llama_index.core import Settings, StorageContext, PromptTemplate, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
import ollama as ollama_client

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

load_dotenv()

# Config
INDEX_DIR    = os.environ.get("SYMBAROUM_INDEX_DIR",    "./index/vector")
BM25_PATH    = os.environ.get("SYMBAROUM_BM25_DIR",     "./index/bm25")
LIGHTRAG_DIR = os.environ.get("SYMBAROUM_LIGHTRAG_DIR", "./index/lightrag")
LLM_MODEL    = os.environ.get("SYMBAROUM_LLM_MODEL",    "qwen3:14b")
UTILITY_MODEL = os.environ.get("SYMBAROUM_UTILITY_MODEL", "qwen3:1.7b")
EMBED_MODEL  = os.environ.get("SYMBAROUM_EMBED_MODEL",  "nomic-embed-text")
EMBED_HOST   = os.environ.get("SYMBAROUM_EMBED_HOST",   "http://127.0.0.1:11436")
DEBUG        = os.environ.get("SYMBAROUM_DEBUG", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Logging — suppress LightRAG/nano-vectordb INFO unless DEBUG
# ---------------------------------------------------------------------------

if not DEBUG:
    logging.getLogger("lightrag").setLevel(logging.WARNING)
    logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

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

utility_client = ollama_client.Client(host="http://127.0.0.1:11435")

# ---------------------------------------------------------------------------
# Utility LLM functions
# ---------------------------------------------------------------------------

def extract_keywords(query: str) -> str:
    response = utility_client.chat(
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


def rewrite_query(query: str) -> str:
    response = utility_client.chat(
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


def route_query(query: str) -> str:
    """
    Decide which pipeline to use for this query.
    Returns 'hybrid' or 'lightrag'.

    - hybrid:   simple factual lookups, rules queries, stat blocks
    - lightrag: cross-referencing, relationships, thematic/global queries
    """
    response = utility_client.chat(
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
    # Fallback to hybrid if response is unexpected
    return "lightrag" if "lightrag" in result else "hybrid"

# ---------------------------------------------------------------------------
# LightRAG async helpers
# ---------------------------------------------------------------------------

async def nomic_embed(texts: list[str]) -> np.ndarray:
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11436")
    data = await client.embed(
        model=EMBED_MODEL,
        input=texts,
        keep_alive="1h",
    )
    return np.array(data["embeddings"])


async def ollama_chat_nothink(
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


async def lightrag_query(rag: LightRAG, query: str) -> str:
    return await rag.aquery(
        query,
        param=QueryParam(mode="mix", enable_rerank=False),
    )


# ---------------------------------------------------------------------------
# Load indexes
# ---------------------------------------------------------------------------

print("Loading hybrid index (vector + BM25)...")
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)

with open(f"{BM25_PATH}/nodes.json", "r") as f:
    node_data = json.load(f)
nodes = [TextNode(text=n["text"], id_=n["id"], metadata=n["metadata"]) for n in node_data]
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)
vector_retriever = index.as_retriever(similarity_top_k=15)
reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=8)
print("Hybrid index loaded.")

print("Loading LightRAG index...")
rag = LightRAG(
    working_dir=LIGHTRAG_DIR,
    llm_model_func=ollama_chat_nothink,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=nomic_embed,
    ),
    addon_params={
        "entity_types": [
            "Character", "Creature", "Ability", "Condition",
            "Attribute", "Faction", "Location", "Artifact",
            "Trait", "Adventure", "Rule",
        ]
    },
)
asyncio.run(rag.initialize_storages())
print("LightRAG index loaded.")

# ---------------------------------------------------------------------------
# Hybrid pipeline components
# ---------------------------------------------------------------------------

qa_prompt = PromptTemplate(
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


class StaticRetriever(BaseRetriever):
    def __init__(self, nodes):
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._nodes


def run_hybrid_pipeline(query: str) -> tuple[str, dict]:
    timings = {}

    t = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        kw_future = executor.submit(extract_keywords, query)
        rw_future = executor.submit(rewrite_query, query)
        keywords = kw_future.result()
        rewritten = rw_future.result()
    timings['keywords+rewrite'] = time.time() - t

    if not keywords.strip():
        keywords = query
    if not rewritten.strip():
        rewritten = query

    if DEBUG:
        print(f"  BM25 keywords: {keywords}")
        print(f"  Rewritten query: {rewritten}")

    t = time.time()
    vector_nodes = vector_retriever.retrieve(rewritten)
    timings['vector_retrieval'] = time.time() - t

    t = time.time()
    bm25_nodes = bm25_retriever.retrieve(keywords)
    timings['bm25_retrieval'] = time.time() - t

    t = time.time()
    seen_ids = {}
    for rank, node in enumerate(vector_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 1 / (rank + 1)
    for rank, node in enumerate(bm25_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 2 / (rank + 1)

    all_nodes = {}
    for n in vector_nodes + bm25_nodes:
        all_nodes[n.node_id] = n.node

    fused = sorted(all_nodes.values(), key=lambda n: seen_ids[n.node_id], reverse=True)[:15]
    timings['fusion'] = time.time() - t

    t = time.time()
    fused_with_scores = [NodeWithScore(node=n, score=seen_ids[n.node_id]) for n in fused]
    reranked = reranker.postprocess_nodes(fused_with_scores, query_bundle=QueryBundle(query))
    timings['reranking'] = time.time() - t

    if DEBUG:
        print("\n  Reranked chunks:")
        for i, node in enumerate(reranked):
            print(f"  [{i+1}] Score: {node.score:.3f} | {node.text[:100]}")

    t = time.time()
    static_retriever = StaticRetriever(reranked)
    qe = RetrieverQueryEngine.from_args(static_retriever, response_mode="tree_summarize")
    qe.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
    answer = str(qe.query(query))
    timings['llm_answer'] = time.time() - t

    return answer, timings


def run_lightrag_pipeline(query: str) -> tuple[str, dict]:
    timings = {}
    t = time.time()
    answer = asyncio.run(lightrag_query(rag, query))
    timings['lightrag_query'] = time.time() - t
    return answer, timings


# ---------------------------------------------------------------------------
# Main query loop
# ---------------------------------------------------------------------------

print("\nSymbaroum RAG ready. Type 'quit' to exit.\n")

while True:
    query = input("Query: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    total_start = time.time()

    # Route query
    t = time.time()
    pipeline = route_query(query)
    route_time = time.time() - t

    print(f"\nRouted to: {pipeline} ({route_time:.1f}s)")
    print("\nThinking...\n")

    if pipeline == "lightrag":
        answer, timings = run_lightrag_pipeline(query)
    else:
        answer, timings = run_hybrid_pipeline(query)

    total = time.time() - total_start

    print(f"Answer [{pipeline}]:\n{answer}\n")

    if DEBUG:
        print(f"Timings:")
        print(f"  {'route':20s}: {route_time:.1f}s")
        for stage, t in timings.items():
            print(f"  {stage:20s}: {t:.1f}s")
        print(f"  {'TOTAL':20s}: {total:.1f}s")
    else:
        print(f"[{pipeline} | {total:.1f}s]")

    print("-" * 60 + "\n")