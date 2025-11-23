#!/bin/bash
# Quick start script for Resume Reviewer Docker setup

set -e

echo "======================================================================="
echo "Resume Reviewer - Docker Quick Start"
echo "======================================================================="
echo ""

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed!"
    echo "Please install docker-compose"
    exit 1
fi

echo "✓ Docker is installed"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "📝 To enable GitHub analysis, edit .env and add your GITHUB_TOKEN"
    echo ""
else
    echo "✓ .env file exists"
    echo ""
fi

# Stop any existing containers
echo "Stopping existing containers..."
docker-compose down 2>/dev/null || true
echo ""

# Pull latest images
echo "Pulling Docker images..."
docker-compose pull
echo ""

# Build services
echo "Building application containers..."
docker-compose build
echo ""

# Start services
echo "Starting all services..."
docker-compose up -d
echo ""

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 5

# Check service status
echo ""
echo "======================================================================="
echo "Service Status"
echo "======================================================================="
docker-compose ps
echo ""

# Check if backend is healthy
echo "Checking backend health..."
for i in {1..30}; do
    if curl -f http://localhost:8000/api/health &>/dev/null; then
        echo "✓ Backend is healthy!"
        break
    fi
    echo "Waiting for backend... ($i/30)"
    sleep 2
done

echo ""
echo "======================================================================="
echo "Resume Reviewer is Ready!"
echo "======================================================================="
echo ""
echo "Access the application:"
echo "  • Frontend:  http://localhost:3000"
echo "  • Backend:   http://localhost:8000"
echo "  • API Docs:  http://localhost:8000/docs"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
echo "======================================================================="
