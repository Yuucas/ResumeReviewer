#!/bin/bash
# scripts/find-candidates.sh

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Ì¥ç Finding best candidates...${NC}\n"

# Check if system is initialized
if [ ! -d "chroma_db" ]; then
    echo -e "${RED}‚ùå System not initialized. Run ./scripts/init.sh first.${NC}"
    exit 1
fi

# Run candidate search
docker compose run --rm resume-rag python -m src.main find-best-k "$@"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}‚úÖ Search complete!${NC}"
else
    echo -e "\n${RED}‚ùå Search failed.${NC}"
    exit 1
fi
