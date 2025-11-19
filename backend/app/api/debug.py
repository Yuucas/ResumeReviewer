"""Debug endpoint to check GitHub data in database"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/debug/github-metadata")
async def check_github_metadata():
    """Check if GitHub URL exists in database metadata"""
    try:
        from ..core.rag_service import get_rag_service

        rag_service = get_rag_service()

        # Get ChromaDB collection
        vectordb = rag_service.pipeline.vectordb

        # Get all chunks for Yukselcan_Sevil_Resume.pdf
        results = vectordb.collection.get(
            where={'filename': 'Yukselcan_Sevil_Resume.pdf'},
            include=['metadatas', 'documents'],
            limit=10
        )

        response_data = {
            "filename": "Yukselcan_Sevil_Resume.pdf",
            "total_chunks": len(results['ids']),
            "github_found": False,
            "github_url": None,
            "sample_metadata": []
        }

        # Check each chunk for GitHub URL
        for i, (chunk_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            sample = {
                "chunk_number": i + 1,
                "chunk_id": chunk_id,
                "has_extracted_info": 'extracted_info' in metadata,
                "github_in_metadata": False,
                "github_url": None
            }

            if 'extracted_info' in metadata:
                extracted = metadata['extracted_info']

                # Handle both dict and JSON string
                if isinstance(extracted, str):
                    import json
                    try:
                        extracted = json.loads(extracted)
                    except:
                        pass

                if isinstance(extracted, dict) and 'github' in extracted:
                    sample["github_in_metadata"] = True
                    sample["github_url"] = extracted['github']
                    response_data["github_found"] = True
                    response_data["github_url"] = extracted['github']

            response_data["sample_metadata"].append(sample)

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Debug endpoint error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
