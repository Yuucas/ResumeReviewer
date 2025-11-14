# backend/app/api/search.py

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from ..models.request import SearchRequest, InitializeRequest
from ..models.response import SearchResponse, InitializeResponse
from ..core.rag_service import get_rag_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_candidates(request: SearchRequest):
    """
    Search for best matching candidates.

    - **job_description**: Job description or requirements
    - **top_k**: Number of top candidates to return (1-20)
    - **role_category**: Filter by role (optional)
    - **min_experience**: Minimum years of experience (optional)
    """
    try:
        rag_service = get_rag_service()

        # Check if database is initialized
        if not rag_service.is_database_initialized():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Database not initialized. Please run /api/initialize first."
            )

        # Search
        result = rag_service.search_candidates(
            job_description=request.job_description,
            top_k=request.top_k,
            role_category=request.role_category,
            min_experience=request.min_experience
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Search failed")
            )

        # Return as plain JSON to avoid Pydantic serialization issues
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/initialize", response_model=InitializeResponse, status_code=status.HTTP_200_OK)
async def initialize_system(request: InitializeRequest):
    """
    Initialize the RAG system (load and index documents).
    
    - **force**: Force reindex all documents (default: false)
    """
    rag_service = get_rag_service()
    
    result = rag_service.initialize_system(force=request.force)
    
    return InitializeResponse(**result)
