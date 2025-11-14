#!/bin/bash
# scripts/logs.sh

SERVICE=${1:-all}

if [ "$SERVICE" = "all" ]; then
    echo "í³‹ Showing all logs..."
    docker compose logs -f
elif [ "$SERVICE" = "ollama" ]; then
    echo "í³‹ Showing Ollama logs..."
    docker compose logs -f ollama
elif [ "$SERVICE" = "app" ]; then
    echo "í³‹ Showing app logs..."
    docker compose logs -f resume-rag
else
    echo "Usage: ./scripts/logs.sh [all|ollama|app]"
    exit 1
fi
