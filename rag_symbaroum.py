import json
import os
import time
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Config
MARKDOWN_DIR = os.environ.get(
    "SYMBAROUM_MARKDOWN_DIR",
    "./marker_output/symbaroum_core_rulebook"
)
INDEX_DIR = os.environ.get(
    "SYMBAROUM_INDEX_DIR",
    "./index"
)
LLM_MODEL = os.environ.get("SYMBAROUM_LLM_MODEL", "qwen3:14b")
EMBED_MODEL = os.environ.get("SYMBAROUM_EMBED_MODEL", "nomic-embed-text")
BM25_PATH = os.environ.get("SYMBAROUM_BM25_DIR", "./bm25_index")
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
        "Extract only the specific proper nouns, names, and unique terms from this query. "
        "Ignore generic words like 'ability', 'rule', 'condition', 'what', 'how', 'the'. "
        "Return only 1-3 specific terms, nothing else, no punctuation:\n"
        f"Query: {query}\n"
        "Specific terms:"
    )
    return str(response).strip()

if os.path.exists(INDEX_DIR) and os.path.exists(BM25_PATH):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    # Reconstruct BM25 retriever from saved nodes
    with open(f"{BM25_PATH}/nodes.json", "r") as f:
        node_data = json.load(f)
    nodes = [TextNode(text=n["text"], id_=n["id"], metadata=n["metadata"]) for n in node_data]
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=15,
    )
else:
    print("Building index from markdown...")
    documents = SimpleDirectoryReader(
        MARKDOWN_DIR,
        required_exts=[".md"]
    ).load_data()

    md_parser = MarkdownNodeParser()
    md_nodes = md_parser.get_nodes_from_documents(documents)

    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=25)
    nodes = []
    for node in md_nodes:
        if len(node.text) > 800:
            sub_nodes = splitter.get_nodes_from_documents([node])
            nodes.extend(sub_nodes)
        else:
            nodes.append(node)

    sizes = [(len(n.text), i) for i, n in enumerate(nodes)]
    sizes.sort(reverse=True)
    print(f"Largest chunks: {sizes[:5]}")

    nodes = [n for n in nodes if len(n.text) < 1500]
    print(f"Created {len(nodes)} chunks")

    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    # Build and persist BM25 index by saving nodes
    os.makedirs(BM25_PATH, exist_ok=True)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10,
    )
    # Save node texts and ids for BM25 reconstruction
    node_data = [{"text": n.text, "id": n.node_id, "metadata": n.metadata} for n in nodes]
    with open(f"{BM25_PATH}/nodes.json", "w") as f:
        json.dump(node_data, f)

    print(f"Index saved to {INDEX_DIR}")

# Custom prompt for detailed answers
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

# Vector retriever
vector_retriever = index.as_retriever(similarity_top_k=15)

print("\nSymbaroum RAG ready. Type 'quit' to exit.\n")
while True:
    query = input("Query: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    # Extract keywords for BM25
    keywords = extract_keywords(query)
    if DEBUG:
        print(f"BM25 keywords: {keywords}")

    # Retrieve from both
    vector_nodes = vector_retriever.retrieve(query)
    bm25_nodes = bm25_retriever.retrieve(keywords)

    # Simple reciprocal rank fusion
    seen_ids = {}
    for rank, node in enumerate(vector_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 1/(rank + 1)
    for rank, node in enumerate(bm25_nodes):
        seen_ids[node.node_id] = seen_ids.get(node.node_id, 0) + 1/(rank + 1)

    # Combine all nodes, deduplicate, sort by fused score
    all_nodes = {}
    for n in vector_nodes + bm25_nodes:
        # n is NodeWithScore, we need the underlying node
        all_nodes[n.node_id] = n.node  # extract the actual node

    fused = sorted(all_nodes.values(), key=lambda n: seen_ids[n.node_id], reverse=True)[:10]
    
    if DEBUG:
        print("\nRetrieved chunks:")
        for i, node in enumerate(fused):
            print(f"\n[{i+1}] Score: {seen_ids[node.node_id]:.3f}")
            print(node.text[:200])

    # Rerank fused results
    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-base",
        top_n=5,
    )
    fused_with_scores = [NodeWithScore(node=n, score=seen_ids[n.node_id]) for n in fused]
    reranked = reranker.postprocess_nodes(fused_with_scores, query_bundle=QueryBundle(query))
    fused = [n.node for n in reranked]

    # Query using fused nodes directly
    class StaticRetriever(BaseRetriever):
        def __init__(self, nodes):
            self._nodes = nodes
            super().__init__()
        def _retrieve(self, query_bundle):
            return self._nodes

    static_retriever = StaticRetriever([NodeWithScore(node=n, score=seen_ids[n.node_id]) for n in fused])
    qe = RetrieverQueryEngine.from_args(static_retriever, response_mode="tree_summarize")
    qe.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

    print("\nThinking...\n")
    start = time.time()
    response = qe.query(query)
    elapsed = time.time() - start
    print(f"\nAnswer:\n{response}\n")
    print(f"Time: {elapsed:.1f}s")
    print("-" * 60 + "\n")