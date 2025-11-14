#!/bin/bash
# scripts/init.sh

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Ì¥ß Initializing Resume RAG System...${NC}\n"

# Check if Ollama is running
if ! curl -f http://localhost:11434/api/tags &>/dev/null; then
    echo -e "${RED}‚ùå Ollama is not running. Run ./scripts/setup.sh first.${NC}"
    exit 1
fi

# Run initialization
docker compose run --rm resume-rag python -m src.main init "$@"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}‚úÖ Initialization complete!${NC}"
else
    echo -e "\n${RED}‚ùå Initialization failed.${NC}"
    exit 1
fi
