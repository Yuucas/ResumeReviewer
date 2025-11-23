# backend/app/api/search.py

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from ..models.request import SearchRequest, InitializeRequest
from ..models.response import SearchResponse, InitializeResponse
from ..core.rag_service import get_rag_service
from ..db.database import get_db
from ..db.models import JobSearch, SearchResult
from ..utils.text_utils import extract_job_title
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_candidates(request: SearchRequest, db: Session = Depends(get_db)):
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

        # Save search to PostgreSQL database
        try:
            # Extract job title from description
            job_title = extract_job_title(request.job_description)

            # Create job search record
            job_search = JobSearch(
                job_title=job_title,
                job_description=request.job_description,
                top_k=request.top_k,
                total_candidates=result.get("total_candidates", 0),
                processing_time=result.get("processing_time")
            )
            db.add(job_search)
            db.flush()  # Get the ID without committing

            # Save each result
            for candidate in result.get("results", []):
                search_result = SearchResult(
                    job_search_id=job_search.id,
                    filename=candidate.get("filename"),
                    match_score=candidate.get("match_score"),
                    recommendation=candidate.get("recommendation"),
                    strengths=candidate.get("strengths"),
                    weaknesses=candidate.get("weaknesses"),
                    overall_assessment=candidate.get("overall_assessment"),
                    experience_years=candidate.get("experience_years"),
                    role_category=candidate.get("role_category"),
                    email=candidate.get("email")
                )

                # Add GitHub data if available
                github_data = candidate.get("github")
                if github_data:
                    search_result.github_username = github_data.get("username")
                    search_result.github_profile_url = github_data.get("profile_url")
                    search_result.github_relevance_score = github_data.get("relevance_score")
                    search_result.github_top_languages = github_data.get("top_languages")
                    search_result.github_relevant_projects = github_data.get("relevant_projects")
                    search_result.github_summary = github_data.get("summary")

                db.add(search_result)

            db.commit()
            logger.info(f"Saved search history: ID={job_search.id}, results={len(result.get('results', []))}")

            # Add search_id to result for frontend reference
            result["search_id"] = job_search.id

        except Exception as db_error:
            db.rollback()
            logger.error(f"Failed to save search to database: {db_error}")
            # Don't fail the search if database save fails
            pass

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
