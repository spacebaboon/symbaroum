import os
import time
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

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

# Set up models
Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=240.0,
    temperature=0.1,
    context_window=8192,
)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

# Build or load index
if os.path.exists(INDEX_DIR):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Building index from markdown...")
    documents = SimpleDirectoryReader(
        MARKDOWN_DIR,
        required_exts=[".md"]
    ).load_data()

    # Two-pass chunking: respect markdown headings first,
    # then split any oversized nodes with SentenceSplitter
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

    # Debug: find largest chunks
    sizes = [(len(n.text), i) for i, n in enumerate(nodes)]
    sizes.sort(reverse=True)
    print(f"Largest chunks: {sizes[:5]}")

    # Safety net filter
    nodes = [n for n in nodes if len(n.text) < 1500]
    print(f"Created {len(nodes)} chunks")

    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")

# Custom prompt for detailed answers
qa_prompt = PromptTemplate(
    "You are an expert on the Symbaroum tabletop RPG rules. "
    "Using the context below, provide a thorough and complete answer. "
    "Include all relevant mechanics, numbers, and special cases. "
    "If the context is insufficient, say so clearly.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Detailed Answer:"
)

# Query engine and retriever
query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="tree_summarize",
)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

retriever = index.as_retriever(similarity_top_k=10)

print("\nSymbaroum RAG ready. Type 'quit' to exit.\n")
while True:
    query = input("Query: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

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