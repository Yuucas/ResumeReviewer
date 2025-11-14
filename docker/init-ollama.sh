#!/bin/bash
# docker/init-ollama.sh

set -e

echo "��� Initializing Ollama models..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
max_retries=30
count=0

while [ $count -lt $max_retries ]; do
    if curl -f http://localhost:11434/api/tags &>/dev/null; then
        echo "✅ Ollama is ready!"
        break
    fi
    count=$((count + 1))
    echo "Waiting... ($count/$max_retries)"
    sleep 2
done

if [ $count -eq $max_retries ]; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull embedding model
echo "��� Pulling embedding model: nomic-embed-text..."
ollama pull nomic-embed-text

# Pull LLM model  
echo "��� Pulling LLM model: qwen3:latest..."
ollama pull qwen3:latest

echo "✅ All models ready!"
ollama list
