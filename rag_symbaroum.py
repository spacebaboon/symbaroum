import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from docling.chunking import HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from FlagEmbedding import FlagReranker
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Config
PDF_PATH = os.environ.get("SYMBAROUM_PDF_PATH", "./data/symbaroum_core_rulebook.pdf")
DOCLING_JSON = os.environ.get("SYMBAROUM_DOCLING_JSON", "./data/symbaroum_docling.json")
INDEX_DIR = os.environ.get("SYMBAROUM_INDEX_DIR", "./index/vector")
BM25_PATH = os.environ.get("SYMBAROUM_BM25_DIR", "./index/bm25")
LLM_MODEL = os.environ.get("SYMBAROUM_LLM_MODEL", "qwen3:14b")
EMBED_MODEL = os.environ.get("SYMBAROUM_EMBED_MODEL", "nomic-embed-text")
DEBUG = os.environ.get("SYMBAROUM_DEBUG", "").lower() in ("1", "true", "yes")

# Set up models
Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=240.0,
    temperature=0.1,
    context_window=8192,
)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)


def extract_keywords(query: str) -> str:
    """Extract key search terms from natural language query for BM25."""
    response = Settings.llm.complete(
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
    )
    return str(response).strip()


def rewrite_query(query: str) -> str:
    """Rewrite query to be more specific for better retrieval."""
    response = Settings.llm.complete(
        "Rewrite this query to improve search retrieval against a Symbaroum RPG rulebook. "
        "Only use terms and names that are explicitly mentioned in the original query. "
        "Do not add names, factions, or terms not present in the original. "
        "Make the query more specific by expanding abbreviations and clarifying intent. "
        "Return only the rewritten query:\n"
        f"Original: {query}\n"
        "Rewritten:"
    )
    return str(response).strip()


# Build or load index
if os.path.exists(INDEX_DIR) and os.path.exists(BM25_PATH):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    with open(f"{BM25_PATH}/nodes.json", "r") as f:
        node_data = json.load(f)
    nodes = [TextNode(text=n["text"], id_=n["id"], metadata=n["metadata"]) for n in node_data]
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)

else:
    print("Building index...")

    # Convert or load cached Docling document
    if Path(DOCLING_JSON).exists():
        print("Loading cached Docling document...")
        doc = DoclingDocument.load_from_json(DOCLING_JSON)
    else:
        print("Converting PDF with Docling (one-time)...")
        converter = DocumentConverter()
        result = converter.convert(PDF_PATH)
        doc = result.document
        doc.save_as_json(DOCLING_JSON)
        print(f"Saved to {DOCLING_JSON}")

    # Chunk with Nomic-aligned tokenizer
    print("Chunking with HybridChunker...")
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1"),
        max_tokens=512,
    )
    chunker = HybridChunker(tokenizer=tokenizer)
    chunks = list(chunker.chunk(dl_doc=doc))

    # Convert to LlamaIndex nodes using contextualized text
    nodes = [
        TextNode(
            text=chunker.contextualize(chunk),
            metadata={"source": "symbaroum_core_rulebook"}
        )
        for chunk in chunks
    ]
    print(f"Created {len(nodes)} chunks")

    index = VectorStoreIndex(nodes, show_progress=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    # Build and persist BM25 index
    os.makedirs(BM25_PATH, exist_ok=True)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)
    node_data = [{"text": n.text, "id": n.node_id, "metadata": n.metadata} for n in nodes]
    with open(f"{BM25_PATH}/nodes.json", "w") as f:
        json.dump(node_data, f)

    print(f"Index saved to {INDEX_DIR}")


# Custom prompt
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

# Retrievers
vector_retriever = index.as_retriever(similarity_top_k=15)
reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=8)


class StaticRetriever(BaseRetriever):
    def __init__(self, nodes):
        self._nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._nodes


print("\nSymbaroum RAG ready. Type 'quit' to exit.\n")
while True:
    query = input("Query: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    keywords = extract_keywords(query)
    rewritten = rewrite_query(query)

    if DEBUG:
        print(f"BM25 keywords: {keywords}")
        print(f"Rewritten query: {rewritten}")

    # Retrieve from both
    vector_nodes = vector_retriever.retrieve(rewritten)
    bm25_nodes = bm25_retriever.retrieve(keywords)

    # Reciprocal rank fusion with BM25 boost
    seen_ids = {}
    for rank, node in enumerate(vector_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 1 / (rank + 1)
    for rank, node in enumerate(bm25_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 2 / (rank + 1)

    # Deduplicate and sort
    all_nodes = {}
    for n in vector_nodes + bm25_nodes:
        all_nodes[n.node_id] = n.node

    fused = sorted(all_nodes.values(), key=lambda n: seen_ids[n.node_id], reverse=True)[:15]

    # Rerank
    fused_with_scores = [NodeWithScore(node=n, score=seen_ids[n.node_id]) for n in fused]
    reranked = reranker.postprocess_nodes(fused_with_scores, query_bundle=QueryBundle(query))

    if DEBUG:
        print("\nReranked chunks:")
        for i, node in enumerate(reranked):
            print(f"\n[{i + 1}] Score: {node.score:.3f}")
            print(node.text[:200])

    # Query
    static_retriever = StaticRetriever(reranked)
    qe = RetrieverQueryEngine.from_args(static_retriever, response_mode="tree_summarize")
    qe.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

    print("\nThinking...\n")
    start = time.time()
    response = qe.query(query)
    elapsed = time.time() - start
    print(f"\nAnswer:\n{response}\n")
    print(f"Time: {elapsed:.1f}s")
    print("-" * 60 + "\n")