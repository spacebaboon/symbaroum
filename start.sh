#!/bin/bash
echo "Starting Symbaroum RAG services..."

# Start primary Ollama if not already running
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "Starting primary Ollama..."
    ollama serve > /tmp/ollama1.log 2>&1 &
    sleep 2
fi

# Start secondary Ollama instance for utility model
if ! curl -s http://localhost:11435/api/version > /dev/null 2>&1; then
    echo "Starting secondary Ollama..."
    nohup env OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=~/.ollama/models ollama serve > /tmp/ollama2.log 2>&1 &
    sleep 2
fi

# Pre-load models
echo "Loading models into VRAM..."
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3:14b","prompt":"hi","stream":false}' > /dev/null
curl -s http://localhost:11435/api/generate -d '{"model":"qwen3:1.7b","prompt":"hi","stream":false}' > /dev/null

echo "Ready! Run: cd ~/projects/symbaroum && uv run rag_symbaroum.py"
    