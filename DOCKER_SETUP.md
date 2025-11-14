# Docker Setup Guide

Complete guide for running the Resume RAG System using Docker.

## Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** v2.0+
- Minimum 8GB RAM (16GB recommended)
- 20GB free disk space

## Quick Start

### 1. Clone and Navigate

```bash
cd resume-reviewer-rag
```

### 2. Build and Start All Services

```bash
docker-compose up --build
```

This will start:
- **Ollama** (LLM service) on port 11434
- **Backend** (FastAPI) on port 8000
- **Frontend** (React/Nginx) on port 3000

### 3. Pull Required Models

In a new terminal, pull the Ollama models:

```bash
# Pull embedding model
docker exec -it resume-rag-ollama ollama pull nomic-embed-text

# Pull LLM model
docker exec -it resume-rag-ollama ollama pull qwen3:latest
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Services Overview

### Ollama Service
- **Container**: `resume-rag-ollama`
- **Port**: 11434
- **Purpose**: Provides LLM and embedding models
- **Volume**: `ollama_data` (persists downloaded models)

### Backend Service
- **Container**: `resume-rag-backend`
- **Port**: 8000
- **Technology**: FastAPI + Python 3.11
- **Volumes**:
  - `./dataset` - Resume files
  - `./src` - Source code (mounted for development)
  - `./chroma_db` - Vector database
  - `./.cache` - Embeddings cache
  - `./output` - Output files

### Frontend Service
- **Container**: `resume-rag-frontend`
- **Port**: 3000
- **Technology**: React + Vite + Nginx
- **Build**: Multi-stage (optimized production build)

## Docker Commands

### Start Services

```bash
# Start all services
docker-compose up

# Start in detached mode (background)
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f ollama
```

### Service Management

```bash
# Restart a service
docker-compose restart backend

# Rebuild a specific service
docker-compose up --build backend

# Scale a service (if supported)
docker-compose up --scale backend=2
```

## Development Mode

For development with hot reload:

### Backend Development

```bash
# The backend volume mounts allow live code changes
# Edit files in backend/app/ and they'll update automatically
docker-compose restart backend  # Only if needed
```

### Frontend Development

For frontend development, it's easier to run locally:

```bash
cd frontend
npm install
npm run dev  # Starts on http://localhost:5173
```

Update `VITE_API_URL` in `.env.local`:
```
VITE_API_URL=http://localhost:8000
```

## Data Persistence

### Volumes

- **ollama_data**: Stores downloaded Ollama models
- **chroma_db**: Vector database (mapped to `./chroma_db`)
- **.cache**: Embeddings cache (mapped to `./.cache`)

### Backup Data

```bash
# Backup vector database
docker cp resume-rag-backend:/app/chroma_db ./backup/chroma_db

# Backup Ollama models
docker run --rm -v resume-reviewer-rag_ollama_data:/data -v $(pwd):/backup alpine tar czf /backup/ollama_backup.tar.gz -C /data .
```

### Restore Data

```bash
# Restore vector database
docker cp ./backup/chroma_db resume-rag-backend:/app/

# Restore Ollama models
docker run --rm -v resume-reviewer-rag_ollama_data:/data -v $(pwd):/backup alpine tar xzf /backup/ollama_backup.tar.gz -C /data
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs backend

# Check if ports are in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Ollama Models Not Found

```bash
# List available models
docker exec -it resume-rag-ollama ollama list

# Pull missing models
docker exec -it resume-rag-ollama ollama pull nomic-embed-text
docker exec -it resume-rag-ollama ollama pull qwen3:latest
```

### Out of Memory

Increase Docker memory:
- Docker Desktop: Settings → Resources → Memory (set to 8GB+)
- Linux: Edit `/etc/docker/daemon.json`

### Backend Can't Connect to Ollama

```bash
# Check if Ollama is running
docker-compose ps ollama

# Test connection from backend
docker exec -it resume-rag-backend curl http://ollama:11434/api/tags
```

### Frontend Can't Connect to Backend

Check `VITE_API_URL` environment variable:
```bash
# Should be http://localhost:8000 for Docker setup
docker exec resume-rag-frontend env | grep VITE_API_URL
```

## Production Deployment

### Environment Variables

Create `.env` file:

```bash
cp .env.example .env
# Edit .env with production values
```

### Security Considerations

1. **Change Default Ports** (if exposed to internet)
2. **Add Authentication** to backend endpoints
3. **Use HTTPS** (add reverse proxy like Traefik/Nginx)
4. **Limit CORS Origins** in `backend/app/main.py`
5. **Use Secrets Management** for sensitive data

### Reverse Proxy Example (Nginx)

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_read_timeout 600s;
    }
}
```

## GPU Support (Optional)

For faster LLM inference with NVIDIA GPU:

1. Install **NVIDIA Container Toolkit**
2. Uncomment GPU section in `docker-compose.yml`:

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. Restart services:

```bash
docker-compose down
docker-compose up -d
```

## Performance Optimization

### Reduce Memory Usage

- Use smaller LLM models (e.g., `qwen3:4b`)
- Reduce `TOP_K_CANDIDATES` to 2-3
- Lower `CHUNK_SIZE` to 2000

### Speed Up Initialization

- Pre-download models before first run
- Use SSD for Docker volumes
- Allocate more CPU cores to Docker

## Monitoring

### Health Checks

```bash
# Check all service health
docker-compose ps

# Backend health
curl http://localhost:8000/api/health

# Frontend health
curl http://localhost:3000
```

### Resource Usage

```bash
# Monitor resource usage
docker stats

# Specific container
docker stats resume-rag-backend
```

## Cleanup

### Remove Everything

```bash
# Stop and remove containers, networks
docker-compose down

# Also remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker rmi $(docker images 'resume-reviewer-rag*' -q)
```

### Clean Docker System

```bash
# Remove unused data
docker system prune

# Remove everything (use with caution)
docker system prune -a --volumes
```

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Review troubleshooting section
3. Ensure all prerequisites are met
4. Verify port availability

## Architecture Diagram

```
┌─────────────────┐
│   Frontend      │
│ (React + Nginx) │
│   Port: 3000    │
└────────┬────────┘
         │
         │ HTTP
         ▼
┌─────────────────┐
│    Backend      │
│   (FastAPI)     │
│   Port: 8000    │
└────────┬────────┘
         │
         │ HTTP
         ▼
┌─────────────────┐
│     Ollama      │
│  (LLM Service)  │
│  Port: 11434    │
└─────────────────┘
```

## Next Steps

1. **Initialize System**: Go to http://localhost:3000/search
2. **Click "Initialize System"**: Indexes all resumes
3. **Upload Resumes**: Use "Upload Resume" button
4. **Search Candidates**: Enter job description and search!
