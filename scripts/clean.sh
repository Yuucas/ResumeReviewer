#!/bin/bash
# scripts/clean.sh

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${YELLOW}‚ö†Ô∏è  WARNING: This will remove all containers, volumes, and data!${NC}"
read -p "Are you sure? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo -e "${RED}Ì∑ëÔ∏è  Cleaning up...${NC}"

# Stop and remove containers
docker compose down

# Remove volumes (this deletes the database!)
echo "Removing volumes..."
docker compose down -v

# Remove images
echo "Removing images..."
docker compose down --rmi all

# Clean local data
read -p "Also delete local data (chroma_db, .cache, output)? (yes/no): " delete_local

if [ "$delete_local" = "yes" ]; then
    rm -rf chroma_db .cache output
    echo "Local data removed"
fi

echo -e "${GREEN}‚úÖ Cleanup complete!${NC}"
