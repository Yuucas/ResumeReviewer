# backend/app/api/health.py

from fastapi import APIRouter, status
from ..models.response import HealthResponse
from ..core.rag_service import get_rag_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Check API and system health."""
    rag_service = get_rag_service()
    
    return HealthResponse(
        status="healthy",
        ollama_available=rag_service.check_ollama_health(),
        database_initialized=rag_service.is_database_initialized()
    )
