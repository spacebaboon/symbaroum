#!/bin/bash
echo "Starting Symbaroum RAG services..."

OLLAMA_MODELS=/mnt/g/llm/ollama/models

# Primary Ollama is managed by systemd
sudo systemctl start ollama
sleep 2
curl -s http://localhost:11434/api/version > /dev/null && echo "Primary Ollama: OK" || echo "Primary Ollama: FAILED"

# Secondary instance (utility LLM - qwen3:1.7b)
if ! curl -s http://localhost:11435/api/version > /dev/null 2>&1; then
    echo "Starting secondary Ollama (port 11435)..."
    nohup env OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=$OLLAMA_MODELS ollama serve > /tmp/ollama2.log 2>&1 &
    sleep 3
fi
curl -s http://localhost:11435/api/version > /dev/null && echo "Secondary Ollama: OK" || echo "Secondary Ollama: FAILED"

# Tertiary instance (embeddings - nomic-embed-text)
if ! curl -s http://localhost:11436/api/version > /dev/null 2>&1; then
    echo "Starting tertiary Ollama (port 11436)..."
    nohup env OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODELS=$OLLAMA_MODELS ollama serve > /tmp/ollama3.log 2>&1 &
    sleep 3
fi
curl -s http://localhost:11436/api/version > /dev/null && echo "Tertiary Ollama: OK" || echo "Tertiary Ollama: FAILED"

# Pre-load main LLM in background (takes a while)
echo "Loading qwen3:14b..."
curl -s --max-time 120 http://localhost:11434/api/generate \
    -d '{"model":"qwen3:14b","prompt":"hi","stream":false,"keep_alive":"1h"}' > /dev/null &

# Pre-load utility LLM synchronously (fast)
echo "Loading qwen3:1.7b..."
curl -s --max-time 30 http://localhost:11435/api/generate \
    -d '{"model":"qwen3:1.7b","prompt":"hi","stream":false,"keep_alive":"1h"}' > /dev/null

# Pre-load embedding model synchronously using embed endpoint
echo "Loading nomic-embed-text..."
curl -s --max-time 30 http://localhost:11436/api/embed \
    -d '{"model":"nomic-embed-text","input":["hi"]}' > /dev/null

echo "All models loaded."
echo "Ready! Run: uv run rag_symbaroum.py"
