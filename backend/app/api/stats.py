# backend/app/api/stats.py

from fastapi import APIRouter, HTTPException, status
from ..models.response import StatsResponse
from ..core.rag_service import get_rag_service

router = APIRouter()


@router.get("/stats", response_model=StatsResponse, status_code=status.HTTP_200_OK)
async def get_statistics():
    """
    Get system statistics.
    
    Returns information about:
    - Total resumes in database
    - Total chunks indexed
    - Distribution by role
    - Average experience
    - Database size
    """
    rag_service = get_rag_service()
    
    # Check if database is initialized
    if not rag_service.is_database_initialized():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database not initialized. Please run /api/initialize first."
        )
    
    stats = rag_service.get_statistics()
    
    return StatsResponse(**stats)
