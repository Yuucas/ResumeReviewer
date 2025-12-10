#!/bin/bash
# docker/init-ollama.sh

set -e

echo "Initialising Ollama models..."

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
max_retries=30
count=0

# Check OLLAMA_HOST env var
OLLAMA_HOST=${OLLAMA_HOST:-localhost:11434}
echo "Targeting Ollama at: $OLLAMA_HOST"

while [ $count -lt $max_retries ]; do
    echo "Attempting to list models..."
    if ollama list; then
        echo "✅ Ollama is ready!"
        break
    fi
    count=$((count + 1))
    echo "Waiting for Ollama... ($count/$max_retries)"
    sleep 2
done

if [ $count -eq $max_retries ]; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Helper function to pull with retry
pull_model() {
    local model=$1
    local retries=5
    local count=0
    
    echo "⬇️ Pulling model: $model..."
    
    while [ $count -lt $retries ]; do
        if ollama pull "$model"; then
            echo "✅ Successfully downloaded $model"
            return 0
        fi
        
        count=$((count + 1))
        echo "⚠️ Pull failed, retrying... ($count/$retries)"
        sleep 5
    done
    
    echo "❌ Failed to download $model after $retries attempts"
    return 1
}

# Pull models
pull_model "nomic-embed-text"
pull_model "qwen2.5:latest"

echo "✅ All models ready!"
ollama list
