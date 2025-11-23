# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Use relative imports
from .api import search, stats, health, upload, debug, history
from .db.database import init_db

# Create FastAPI app
app = FastAPI(
    title="Resume RAG API",
    description="AI-powered resume screening system using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables."""
    init_db()

# CORS middleware (allow frontend to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(stats.router, prefix="/api", tags=["statistics"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(debug.router, prefix="/api", tags=["debug"])
app.include_router(history.router, prefix="/api", tags=["history"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Resume RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
