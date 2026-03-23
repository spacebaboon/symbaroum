#!/bin/bash
echo "Stopping Symbaroum RAG services..."

# Unload models from VRAM first
curl -s http://localhost:11434/api/generate -d '{"model":"qwen3:14b","keep_alive":0}' > /dev/null 2>&1
curl -s http://localhost:11435/api/generate -d '{"model":"qwen3:1.7b","keep_alive":0}' > /dev/null 2>&1

# Stop secondary Ollama instance
pkill -f "OLLAMA_HOST=127.0.0.1:11435" 2>/dev/null

echo "Models unloaded. Primary Ollama still running for other uses."
echo "To fully stop Ollama: pkill ollama"
