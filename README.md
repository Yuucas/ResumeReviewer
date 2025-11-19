# Resume Reviewer RAG

AI-powered resume screening system with a modern web interface, using Retrieval Augmented Generation (RAG) with local LLMs.

## ✨ Features

- **Modern Web Interface** - React-based responsive UI with real-time search
- **Local AI Processing** - Uses Ollama (no external API calls, privacy-first)
- **Semantic Search** - Understanding meaning and context, not just keywords
- **Intelligent Ranking** - LLM-powered candidate analysis with detailed insights
- **Resume Upload** - Easy PDF upload with automatic categorization
- **Docker-based** - Complete containerization for easy deployment
- **REST API** - FastAPI backend with comprehensive endpoints
- **Real-time Statistics** - Track your resume database metrics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Environment                        │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Frontend   │    │   Backend    │    │    Ollama    │ │
│  │   (React)    │◄──►│  (FastAPI)   │◄──►│  (Local LLM) │ │
│  │   Port 3000  │    │   Port 8000  │    │  Port 11434  │ │
│  │              │    │              │    │              │ │
│  │ - Search UI  │    │ - REST API   │    │ - qwen3      │ │
│  │ - Upload     │    │ - RAG Logic  │    │ - embeddings │ │
│  │ - Analytics  │    │ - ChromaDB   │    │              │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** (20.10+)
- **Docker Compose** (1.29+)
- **8GB+ RAM** recommended
- **20GB+ free disk space** (for models and data)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Yuucas/ResumeReviewer.git
cd ResumeReviewer
```

2. **Create environment file** (optional):
```bash
cp .env.example .env
# Edit .env if you need to customize settings
```

3. **Start the application**:
```bash
docker-compose up --build -d
```

4. **Pull Ollama models**:
```bash
# Embedding model (for semantic search)
docker exec -it resume-rag-ollama ollama pull nomic-embed-text

# LLM model (for candidate analysis)
docker exec -it resume-rag-ollama ollama pull qwen3:latest
```

5. **Access the application**:
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

### Adding Resume Data

You can add resumes in two ways:

**Option 1: Upload via Web Interface** (Recommended)
1. Navigate to http://localhost:3000
2. Click "Upload Resume" button
3. Select PDF file and role category
4. Click "Upload"

**Option 2: Direct File Copy**
```bash
# Copy PDFs to appropriate role folders
cp /path/to/resumes/*.pdf dataset/data_scientist/
cp /path/to/resumes/*.pdf dataset/fullstack_engineer/
cp /path/to/resumes/*.pdf dataset/it/

# Restart backend to re-index
docker-compose restart backend
```

## 💻 Usage

### Web Interface

1. **Search for Candidates**:
   - Go to the "Search" page
   - Enter job description
   - Select role category (optional)
   - Set minimum years of experience (optional)
   - Choose number of candidates (1-10)
   - Click "Find Best Candidates"
   - View detailed analysis results

2. **Upload New Resumes**:
   - Click "Upload Resume" on any page
   - Select PDF file
   - Choose appropriate role category
   - Submit and re-index database

3. **View Statistics**:
   - Navigate to "Statistics" page
   - See total resumes, role distribution
   - Track database health

### API Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# Search candidates
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Python developer with ML experience",
    "top_k": 5,
    "role_category": "data_scientist",
    "min_experience": 3
  }'

# Get statistics
curl http://localhost:8000/api/stats

# Upload resume
curl -X POST http://localhost:8000/api/upload \
  -F "file=@resume.pdf" \
  -F "role_category=data_scientist"
```

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up -d

# Build and start (after code changes)
docker-compose up --build -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f ollama

# Restart a service
docker-compose restart backend

# Check service status
docker-compose ps

# Clean everything (including volumes)
docker-compose down -v
```

## 📁 Project Structure

```
resume-reviewer-rag/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   │   ├── health.py   # Health check
│   │   │   ├── search.py   # Candidate search
│   │   │   ├── stats.py    # Statistics
│   │   │   └── upload.py   # Resume upload
│   │   ├── core/           # Core business logic
│   │   │   └── rag_service.py
│   │   └── main.py         # FastAPI app
│   ├── Dockerfile          # Backend container
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Page components
│   │   │   ├── Home.jsx    # Landing page
│   │   │   ├── Search.jsx  # Search interface
│   │   │   ├── Analysis.jsx # Results display
│   │   │   └── Statistics.jsx
│   │   └── services/       # API client
│   ├── Dockerfile          # Frontend container
│   ├── nginx.conf          # Nginx configuration
│   └── package.json        # Node dependencies
│
├── src/                    # Core RAG System
│   ├── ingestion/          # Document processing
│   ├── vectorstore/        # ChromaDB integration
│   ├── retrieval/          # Search logic
│   ├── agents/             # LLM integration
│   └── utils/              # Utilities
│
├── dataset/                # Resume PDFs (not tracked)
│   ├── data_scientist/
│   ├── fullstack_engineer/
│   └── it/
│
├── docker-compose.yml      # Container orchestration
├── DOCKER_SETUP.md         # Detailed Docker guide
└── TROUBLESHOOTING.md      # Common issues & fixes
```

## ⚙️ Configuration

The application uses environment variables for configuration. See `.env.example`:

```bash
# Ollama Settings
OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=qwen3:latest
EMBEDDING_MODEL=nomic-embed-text
TEMPERATURE=0.3

# Search Settings
TOP_K_CANDIDATES=10
MIN_SIMILARITY_SCORE=0.3

# Chunking Settings
CHUNK_SIZE=3000
CHUNK_OVERLAP=200

# ChromaDB Settings
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=resume_embeddings
```

## 🔧 Development

### Backend Development

```bash
# Install dependencies locally
cd backend
pip install -r requirements.txt

# Run backend locally (requires Ollama running)
cd backend
uvicorn app.main:app --reload --port 8000
```

### Frontend Development

```bash
# Install dependencies
cd frontend
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Docling](https://github.com/DS4SD/docling) - Document parsing
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI library
- [TailwindCSS](https://tailwindcss.com/) - Styling framework

## 📧 Contact

For questions or issues, please open a [GitHub issue](https://github.com/Yuucas/ResumeReviewer/issues).

---
