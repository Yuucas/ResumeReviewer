"""
API endpoints for search history.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime

from ..db.database import get_db
from ..db.models import JobSearch, SearchResult

router = APIRouter()


@router.get("/history")
async def get_search_history(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get list of previous job searches.

    Args:
        limit: Maximum number of searches to return
        offset: Number of searches to skip (for pagination)

    Returns:
        List of job searches with summary information
    """
    try:
        # Query searches ordered by most recent first
        searches = db.query(JobSearch).order_by(desc(JobSearch.created_at)).offset(offset).limit(limit).all()

        # Format response
        history = []
        for search in searches:
            history.append({
                "id": search.id,
                "job_title": search.job_title,
                "job_description": search.job_description,
                "created_at": search.created_at.isoformat(),
                "total_candidates": search.total_candidates,
                "processing_time": search.processing_time,
                "result_count": len(search.results)
            })

        # Get total count for pagination
        total_count = db.query(JobSearch).count()

        return {
            "success": True,
            "history": history,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@router.get("/history/{search_id}")
async def get_search_detail(
    search_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed results for a specific search.

    Args:
        search_id: ID of the job search

    Returns:
        Complete search information with all candidate results
    """
    try:
        # Query the search
        search = db.query(JobSearch).filter(JobSearch.id == search_id).first()

        if not search:
            raise HTTPException(status_code=404, detail=f"Search with ID {search_id} not found")

        # Format results
        results = []
        for result in search.results:
            result_data = {
                "filename": result.filename,
                "match_score": result.match_score,
                "recommendation": result.recommendation,
                "strengths": result.strengths,
                "weaknesses": result.weaknesses,
                "overall_assessment": result.overall_assessment,
                "experience_years": result.experience_years,
                "role_category": result.role_category,
                "email": result.email
            }

            # Add GitHub data if available
            if result.github_username:
                result_data["github"] = {
                    "username": result.github_username,
                    "profile_url": result.github_profile_url,
                    "relevance_score": result.github_relevance_score,
                    "top_languages": result.github_top_languages,
                    "relevant_projects": result.github_relevant_projects,
                    "summary": result.github_summary
                }

            results.append(result_data)

        return {
            "success": True,
            "search": {
                "id": search.id,
                "job_title": search.job_title,
                "job_description": search.job_description,
                "created_at": search.created_at.isoformat(),
                "total_candidates": search.total_candidates,
                "processing_time": search.processing_time,
                "top_k": search.top_k
            },
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve search details: {str(e)}")


@router.delete("/history/{search_id}")
async def delete_search(
    search_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a search from history.

    Args:
        search_id: ID of the job search to delete

    Returns:
        Success confirmation
    """
    try:
        # Query the search
        search = db.query(JobSearch).filter(JobSearch.id == search_id).first()

        if not search:
            raise HTTPException(status_code=404, detail=f"Search with ID {search_id} not found")

        # Delete (cascade will delete results too)
        db.delete(search)
        db.commit()

        return {
            "success": True,
            "message": f"Search {search_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete search: {str(e)}")


@router.delete("/history")
async def clear_all_history(
    confirm: bool = False,
    db: Session = Depends(get_db)
):
    """
    Clear all search history.

    Args:
        confirm: Must be True to confirm deletion

    Returns:
        Success confirmation with count of deleted searches
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Must set confirm=true to delete all history")

    try:
        # Count searches
        count = db.query(JobSearch).count()

        # Delete all
        db.query(JobSearch).delete()
        db.commit()

        return {
            "success": True,
            "message": f"Deleted {count} searches from history"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")
