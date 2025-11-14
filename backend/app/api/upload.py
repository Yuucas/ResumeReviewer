# backend/app/api/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import logging
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)

# Get project root (4 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

# Valid role categories
VALID_ROLES = ["data_scientist", "fullstack_engineer", "it"]


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_resume(
    file: UploadFile = File(...),
    role_category: str = Form(...),
):
    """
    Upload a new resume PDF to the dataset.

    - **file**: PDF file to upload
    - **role_category**: Category for the resume (data_scientist, fullstack_engineer, it)
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed"
            )

        # Validate role category
        if role_category not in VALID_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role category. Must be one of: {', '.join(VALID_ROLES)}"
            )

        # Create role directory if it doesn't exist
        role_dir = DATASET_ROOT / role_category
        role_dir.mkdir(parents=True, exist_ok=True)

        # Generate safe filename
        safe_filename = file.filename.replace(" ", "_")
        file_path = role_dir / safe_filename

        # Check if file already exists
        if file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A resume with filename '{safe_filename}' already exists in {role_category}"
            )

        # Save the file
        logger.info(f"Saving resume to: {file_path}")
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Successfully uploaded: {safe_filename} to {role_category}")

        return JSONResponse(
            content={
                "success": True,
                "message": "Resume uploaded successfully",
                "filename": safe_filename,
                "role_category": role_category,
                "file_path": str(file_path)
            },
            status_code=status.HTTP_201_CREATED
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload resume: {str(e)}"
        )


@router.get("/roles")
async def get_available_roles():
    """Get list of available role categories."""
    return {
        "roles": [
            {"value": "data_scientist", "label": "Data Scientist"},
            {"value": "fullstack_engineer", "label": "Full Stack Engineer"},
            {"value": "it", "label": "IT Professional"}
        ]
    }
