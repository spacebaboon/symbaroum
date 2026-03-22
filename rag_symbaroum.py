import os
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Config
MARKDOWN_DIR = "/mnt/g/llm/symbaroum/marker_output/symbaroum_core_rulebook"
INDEX_DIR = "/mnt/g/llm/symbaroum/index"
LLM_MODEL = "qwen3:14b"
EMBED_MODEL = "nomic-embed-text"

# Set up models
Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=240.0,  # Increase timeout for longer responses
    temperature=0.1, # Low temp for factual rules queries
    context_window=8192,
)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

# Build or load index
if os.path.exists(INDEX_DIR):
    print("Loading existing index...")
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Building index from markdown...")
    documents = SimpleDirectoryReader(MARKDOWN_DIR).load_data()
    
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=25)
    nodes = splitter.get_nodes_from_documents(documents)

    # Debug: find largest chunks
    sizes = [(len(n.text), i) for i, n in enumerate(nodes)]
    sizes.sort(reverse=True)
    print(f"Largest chunks: {sizes[:5]}")

    # Filter out anything over 1500 chars as a safety net
    nodes = [n for n in nodes if len(n.text) < 1500]
    print(f"After filtering: {len(nodes)} chunks")

    print(f"Created {len(nodes)} chunks")
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")

# Query loop
query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="tree_summarize",  # Better for longer, comprehensive answers
)

# For debugging, use retriever directly
retriever = index.as_retriever(similarity_top_k=10)

print("\nSymbaroum RAG ready. Type 'quit' to exit.\n")
while True:
    query = input("Query: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    retrieved = retriever.retrieve(query)
    print("\nRetrieved chunks:")
    for i, node in enumerate(retrieved):
        print(f"\n[{i+1}] Score: {node.score:.3f}")
        print(node.text[:200])
    print("\nThinking...\n")
    start = time.time()
    response = query_engine.query(query)
    elapsed = time.time() - start
    print(f"\nAnswer:\n{response}\n")
    print(f"Time: {elapsed:.1f}s")
    print("-" * 60 + "\n")