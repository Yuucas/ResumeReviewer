#!/bin/bash
# scripts/run.sh

# Run any command in the resume-rag container
docker compose run --rm resume-rag "$@"
