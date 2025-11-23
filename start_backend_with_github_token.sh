#!/bin/bash
# Linux/Mac script to start backend with GitHub token

echo "========================================"
echo "Resume Reviewer Backend Starter"
echo "========================================"
echo ""

# Check if GITHUB_TOKEN is provided as argument
if [ -z "$1" ]; then
    echo "Usage: ./start_backend_with_github_token.sh YOUR_GITHUB_TOKEN"
    echo ""
    echo "Or set it first:"
    echo "  export GITHUB_TOKEN=ghp_your_token_here"
    echo "  ./start_backend_with_github_token.sh"
    echo ""

    # Check if already set in environment
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "[ERROR] GITHUB_TOKEN not set!"
        echo ""
        echo "Please either:"
        echo "  1. Run: ./start_backend_with_github_token.sh ghp_your_token"
        echo "  2. Set environment variable first: export GITHUB_TOKEN=ghp_your_token"
        echo ""
        exit 1
    else
        echo "[OK] Using GITHUB_TOKEN from environment"
        echo "Token: ${GITHUB_TOKEN:0:10}..."
    fi
else
    # Set token from argument
    export GITHUB_TOKEN=$1
    echo "[OK] GITHUB_TOKEN set from argument"
    echo "Token: ${GITHUB_TOKEN:0:10}..."
fi

echo ""
echo "Starting backend server..."
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Start uvicorn
uvicorn backend.app.main:app --reload
