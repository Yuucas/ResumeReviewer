#!/bin/bash
# scripts/setup.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}��� Setting up Resume RAG System with Docker...${NC}\n"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not available.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}��� Creating directories...${NC}"
mkdir -p output .cache chroma_db dataset/{data_scientist,fullstack_engineer,it}

# Build Docker images
echo -e "${BLUE}��� Building Docker images...${NC}"
docker compose build

# Start Ollama service
echo -e "${BLUE}��� Starting Ollama service...${NC}"
docker compose up -d ollama

# Wait for Ollama to be ready
echo -e "${YELLOW}⏳ Waiting for Ollama to start (30-60 seconds)...${NC}"
sleep 15

# Check Ollama health
retries=0
max_retries=30

while [ $retries -lt $max_retries ]; do
    if curl -f http://localhost:11434/api/tags &>/dev/null; then
        echo -e "\n${GREEN}✅ Ollama is ready!${NC}\n"
        break
    fi
    echo -n "."
    sleep 2
    retries=$((retries + 1))
done

if [ $retries -eq $max_retries ]; then
    echo -e "\n${RED}❌ Ollama failed to start. Check logs: docker compose logs ollama${NC}"
    exit 1
fi

# Pull models
echo -e "${BLUE}��� Pulling Ollama models...${NC}"
echo -e "${YELLOW}(This may take several minutes - models are large)${NC}\n"

echo -e "${GREEN}Pulling embedding model: nomic-embed-text${NC}"
docker compose exec ollama ollama pull nomic-embed-text

echo -e "\n${GREEN}Pulling LLM model: qwen3:latest${NC}"
docker compose exec ollama ollama pull qwen3:latest

# Verify models
echo -e "\n${BLUE}��� Installed models:${NC}"
docker compose exec ollama ollama list

echo -e "\n${GREEN}✅ Setup complete!${NC}\n"
echo "Next steps:"
echo "  1. Add your resume PDFs to the 'dataset' folder"
echo "  2. Run: ./scripts/init.sh"
echo "  3. Run: ./scripts/find-candidates.sh --job-file job.txt --top-k 5"
