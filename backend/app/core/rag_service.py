# backend/app/core/rag_service.py

import sys
import os
from pathlib import Path
import time
from typing import Optional

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.main import ResumeRAGPipeline
from src.utils import get_config, setup_logger

logger = setup_logger(level="INFO")


class RAGService:
    """Service wrapper for Resume RAG system."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.config = get_config()
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize RAG pipeline."""
        try:
            self.pipeline = ResumeRAGPipeline(self.config)
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def is_database_initialized(self) -> bool:
        """Check if vector database is initialized."""
        try:
            # Access vectordb through pipeline's lazy loading
            stats = self.pipeline.vectordb.get_statistics()
            return stats.get('total_documents', 0) > 0
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return False
    
    def initialize_system(self, force: bool = False) -> dict:
        """Initialize the RAG system using pipeline's ingest_and_index method."""
        try:
            start_time = time.time()
            
            logger.info(f"Starting system initialization (force={force})...")
            
            # Use the pipeline's ingest_and_index method
            self.pipeline.ingest_and_index(force_reindex=force)
            
            # Get statistics after initialization
            try:
                stats = self.pipeline.vectordb.get_statistics()
                documents_processed = stats.get('total_documents', 0)
                
                # Try to get more accurate chunk count
                if hasattr(self.pipeline.vectordb, 'collection'):
                    chunks_created = self.pipeline.vectordb.collection.count()
                else:
                    chunks_created = documents_processed  # Fallback
                    
            except Exception as e:
                logger.warning(f"Could not get exact statistics: {e}")
                documents_processed = 0
                chunks_created = 0
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": "System initialized successfully",
                "documents_processed": documents_processed,
                "chunks_created": chunks_created,
                "processing_time": round(processing_time, 2)
            }
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Initialization failed: {str(e)}",
                "documents_processed": 0,
                "chunks_created": 0,
                "processing_time": 0
            }
    
    def search_candidates(
        self,
        job_description: str,
        top_k: int = 5,
        role_category: Optional[str] = None,
        min_experience: float = 0.0
    ) -> dict:
        """Search for best matching candidates using pipeline's find_best_k method."""
        try:
            start_time = time.time()
            
            logger.info(f"Searching for top {top_k} candidates...")
            
            # Use pipeline's find_best_k method which does the complete pipeline
            candidates = self.pipeline.find_best_k(
                job_description=job_description,
                top_k=top_k,
                role_category=role_category if role_category else None,
                min_experience=min_experience
            )
            
            processing_time = time.time() - start_time
            
            # Format results for API response
            results = []
            for candidate in candidates:
                # Use to_dict() method to include all fields including GitHub data
                result = candidate.to_dict()

                # Debug: Log if GitHub data exists
                if 'github' in result:
                    logger.info(f"✓ GitHub data found for {candidate.filename}: @{result['github'].get('username')}")
                else:
                    logger.warning(f"✗ No GitHub data for {candidate.filename}")

                results.append(result)
            
            return {
                "success": True,
                "query": job_description,
                "total_candidates": len(results),
                "results": results,
                "processing_time": round(processing_time, 2)
            }
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return {
                "success": False,
                "query": job_description,
                "total_candidates": 0,
                "results": [],
                "processing_time": 0,
                "error": str(e)
            }
    
    def get_statistics(self) -> dict:
        """Get system statistics from vector database."""
        try:
            # Get stats from vectordb
            stats = self.pipeline.vectordb.get_statistics()
            
            # Get total count
            total_docs = stats.get('total_documents', 0)
            
            # Try to get actual count from collection
            try:
                if hasattr(self.pipeline.vectordb, 'collection'):
                    total_chunks = self.pipeline.vectordb.collection.count()
                else:
                    total_chunks = total_docs
            except:
                total_chunks = total_docs
            
            # Get role distribution if available
            roles = stats.get('sample_role_distribution', {})
            
            # Calculate database size
            try:
                db_path = Path(self.config.chroma_db_path)
                if db_path.exists():
                    db_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file()) / (1024 * 1024)
                else:
                    db_size = 0.0
            except:
                db_size = 0.0
            
            # Try to get average experience from metadata
            avg_experience = 0.0
            try:
                if hasattr(self.pipeline.vectordb, 'collection'):
                    results = self.pipeline.vectordb.collection.get(
                        limit=100,
                        include=['metadatas']
                    )
                    metadatas = results.get('metadatas', [])
                    experiences = [
                        m.get('years_of_experience', 0) 
                        for m in metadatas 
                        if isinstance(m, dict) and m.get('years_of_experience')
                    ]
                    if experiences:
                        avg_experience = sum(experiences) / len(experiences)
            except Exception as e:
                logger.debug(f"Could not calculate average experience: {e}")
            
            return {
                "total_resumes": total_docs,
                "total_chunks": total_chunks,
                "roles": roles,
                "average_experience": round(avg_experience, 2),
                "database_size_mb": round(db_size, 2)
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {
                "total_resumes": 0,
                "total_chunks": 0,
                "roles": {},
                "average_experience": 0.0,
                "database_size_mb": 0.0
            }
    
    def check_ollama_health(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get(
                f"{self.config.ollama_base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False


# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
