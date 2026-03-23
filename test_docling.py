"""
Test Docling chunking on the Symbaroum PDF directly.
Checks quality on known problem areas: Steadfast ability, Scene 8-2 elves.
"""
import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

PDF_PATH = "/home/ben/projects/symbaroum/symbaroum_core_rulebook.pdf"
DOC_JSON_PATH = "/home/ben/projects/symbaroum/symbaroum_docling.json"

# Convert or load from cache
if Path(DOC_JSON_PATH).exists():
    print("Loading cached Docling document...")
    from docling.datamodel.document import DoclingDocument
    doc = DoclingDocument.load_from_json(DOC_JSON_PATH)
    print("Loaded.")
else:
    print("Converting PDF with Docling (one-time, will be cached)...")
    converter = DocumentConverter()
    result = converter.convert(PDF_PATH)
    doc = result.document
    doc.save_as_json(DOC_JSON_PATH)
    print(f"Conversion done. Saved to {DOC_JSON_PATH}")

# Chunk with Nomic-aligned tokenizer
print("\nChunking with HybridChunker...")
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1"),
    max_tokens=512,
)
chunker = HybridChunker(tokenizer=tokenizer)
chunks = list(chunker.chunk(dl_doc=doc))
print(f"Created {len(chunks)} chunks")

# Search for known problem chunks
print("\n=== Searching for STEADFAST ===")
for i, chunk in enumerate(chunks):
    if "STEADFAST" in chunk.text.upper() and "Mental disciplines" in chunk.text:
        print(f"Chunk {i}:")
        print(f"Contextualized:\n{chunker.contextualize(chunk)[:500]}")
        print()

print("\n=== Searching for Scene 8-2 / Godrai stat block ===")
for i, chunk in enumerate(chunks):
    if "Acrobatics" in chunk.text and "Marksman" in chunk.text:
        print(f"Chunk {i}:")
        print(f"Contextualized:\n{chunker.contextualize(chunk)[:500]}")
        print()

print("\n=== Searching for Godrai dialogue chunks ===")
for i, chunk in enumerate(chunks):
    if "Godrai" in chunk.text or "SCENE 8-2" in chunk.text:
        print(f"Chunk {i}:")
        print(f"Contextualized:\n{chunker.contextualize(chunk)[:300]}")
        print()
        if i > 10:
            print("... (limiting output)")
            break