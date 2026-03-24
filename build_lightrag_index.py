"""
Build LightRAG knowledge graph index from Symbaroum Docling JSON.
One-time operation — run this to build the graph, then use rag_symbaroum.py for queries.
Resumes automatically if interrupted (kv_store_doc_status.json tracks progress).
"""
import asyncio
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import ollama as ollama_client

from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

load_dotenv()

DOCLING_JSON = os.environ.get("SYMBAROUM_DOCLING_JSON", "./data/symbaroum_docling.json")
LIGHTRAG_DIR = os.environ.get("SYMBAROUM_LIGHTRAG_DIR", "./index/lightrag")
LLM_MODEL = os.environ.get("SYMBAROUM_LLM_MODEL", "qwen3:14b")

os.makedirs(LIGHTRAG_DIR, exist_ok=True)


async def nomic_embed(texts: list[str]) -> np.ndarray:
    """Embeddings via dedicated port 11436 to avoid evicting the main LLM."""
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11436")
    data = await client.embed(
        model="nomic-embed-text",
        input=texts,
        keep_alive="1h",
    )
    return np.array(data["embeddings"])


async def ollama_chat_nothink(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """LLM via chat API with think=False — avoids the generate API which ignores think=False."""
    kwargs.pop("hashing_kv", None)
    kwargs.pop("max_tokens", None)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    t = time.time()
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11434")
    response = await client.chat(
        model=LLM_MODEL,
        messages=messages,
        think=False,
        options={"num_ctx": 16384},
    )
    elapsed = time.time() - t
    # Log slow LLM calls (>30s may indicate thinking mode crept back in)
    if elapsed > 30:
        print(f"  ⚠ Slow LLM call: {elapsed:.1f}s")
    return response.message.content


def build_corpus() -> str:
    """
    Load Docling JSON and convert to contextualized text for LightRAG.
    Reuses the same HybridChunker pipeline as the main RAG system.
    """
    t = time.time()
    print(f"Loading Docling document from {DOCLING_JSON}...")
    doc = DoclingDocument.load_from_json(DOCLING_JSON)
    print(f"  Loaded in {time.time() - t:.1f}s")

    t = time.time()
    print("Chunking with HybridChunker...")
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1"),
        max_tokens=512,
    )
    chunker = HybridChunker(tokenizer=tokenizer)
    chunks = list(chunker.chunk(dl_doc=doc))
    print(f"  {len(chunks)} chunks created in {time.time() - t:.1f}s")

    t = time.time()
    corpus = "\n\n---\n\n".join(
        chunker.contextualize(chunk) for chunk in chunks
    )
    print(f"  Corpus: {len(corpus):,} chars, {len(corpus.split()):,} words ({time.time() - t:.1f}s)")
    return corpus


async def main():
    total_start = time.time()
    print("=== Symbaroum LightRAG Index Builder ===")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")

    # Build chunks (not joined corpus)
    t = time.time()
    print(f"Loading Docling document from {DOCLING_JSON}...")
    doc = DoclingDocument.load_from_json(DOCLING_JSON)
    print(f"  Loaded in {time.time() - t:.1f}s")

    t = time.time()
    print("Chunking with HybridChunker...")
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1"),
        max_tokens=512,
    )
    chunker = HybridChunker(tokenizer=tokenizer)
    chunks = list(chunker.chunk(dl_doc=doc))
    texts = [chunker.contextualize(chunk) for chunk in chunks]
    print(f"  {len(texts)} chunks created in {time.time() - t:.1f}s")

    # Check resume state
    status_file = os.path.join(LIGHTRAG_DIR, "kv_store_doc_status.json")
    if os.path.exists(status_file):
        with open(status_file) as f:
            status = json.load(f)
        completed = sum(1 for v in status.values() if v.get("status") == "processed")
        if completed > 0:
            print(f"\nResuming — {completed} chunks already processed.")

    print(f"\nInitialising LightRAG at {LIGHTRAG_DIR}...")
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

    await rag.initialize_storages()

    print(f"\nInserting {len(texts)} chunks individually...")
    print("Progress saved after each chunk — safe to interrupt and resume.\n")

    extraction_start = time.time()
    for i, text in enumerate(texts):
        chunk_start = time.time()
        await rag.ainsert(text)
        elapsed = time.time() - chunk_start
        total_so_far = time.time() - extraction_start
        avg = total_so_far / (i + 1)
        remaining = avg * (len(texts) - i - 1)
        print(f"Chunk {i+1}/{len(texts)} — {elapsed:.1f}s this chunk, "
              f"~{timedelta(seconds=int(remaining))} remaining")

    total_elapsed = time.time() - total_start
    print(f"\n=== Indexing complete ===")
    print(f"Extraction time: {timedelta(seconds=int(time.time() - extraction_start))}")
    print(f"Total time:      {timedelta(seconds=int(total_elapsed))}")
    print(f"Finished:        {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())