"""
Embeddings Module
Handles vector embeddings generation using Ollama.
Converts text chunks into dense vector representations for semantic search.
"""

import logging
import requests
import numpy as np
from typing import List, Dict, Optional, Union
import time
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """
    Generate embeddings using Ollama's embedding models.
    Supports models like nomic-embed-text, mxbai-embed-large, etc.
    """
    
    def __init__(self,
                 model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434",
                 batch_size: int = 32,
                 normalize: bool = True,
                 cache_embeddings: bool = True,
                 cache_dir: str = ".embedding_cache"):
        """
        Initialize Ollama embeddings client.
        
        Args:
            model: Ollama embedding model name
                   Options: 
                   - 'nomic-embed-text' (768 dims, recommended for English text)
                   - 'mxbai-embed-large' (1024 dims, higher quality)
                   - 'all-minilm' (384 dims, faster/smaller)
            base_url: Ollama API base URL
            batch_size: Number of texts to embed in one request
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            cache_embeddings: Whether to cache embeddings to disk
            cache_dir: Directory for caching embeddings
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_embeddings = cache_embeddings
        self.cache_dir = Path(cache_dir)
        
        if self.cache_embeddings:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify Ollama connection and model availability
        self._verify_connection()
        
        logger.info(f"OllamaEmbeddings initialized (model: {model}, normalize: {normalize})")
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.warning(f"Model '{self.model}' not found. Available models: {model_names}")
                logger.warning(f"Pulling model '{self.model}'... This may take a few minutes.")
                self._pull_model()
            else:
                logger.info(f"Model '{self.model}' is available")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running with: 'ollama serve'"
            )
        except Exception as e:
            logger.error(f"Error verifying Ollama connection: {str(e)}")
            raise
    
    def _pull_model(self):
        """Pull the embedding model if not available."""
        try:
            logger.info(f"Pulling model: {self.model}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300  # 5 minutes timeout for model download
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to pull model '{self.model}': {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of embedding vector
        """
        # Check cache first
        if self.cache_embeddings:
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            embedding = np.array(response.json()['embedding'], dtype=np.float32)
            
            # Normalize if requested
            if self.normalize:
                embedding = self._normalize_embedding(embedding)
            
            # Cache the embedding
            if self.cache_embeddings:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout embedding text: {text[:100]}...")
            raise
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress logging
        
        Returns:
            List of numpy arrays (embeddings)
        """
        if not texts:
            logger.warning("Empty texts list provided")
            return []
        
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Embedding {len(texts)} texts in {total_batches} batches")
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            start_time = time.time()
            
            # Embed each text in the batch
            for text in batch:
                try:
                    embedding = self.embed_text(text)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to embed text in batch {batch_num}: {str(e)}")
                    # Add zero vector as fallback
                    embeddings.append(np.zeros(768, dtype=np.float32))
            
            elapsed = time.time() - start_time
            
            if show_progress:
                logger.info(f"Batch {batch_num} completed in {elapsed:.2f}s "
                           f"({len(batch)/elapsed:.1f} texts/sec)")
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
        
        Returns:
            Numpy array of embedding vector
        """
        logger.debug(f"Embedding query: {query[:100]}...")
        return self.embed_text(query)
    
    def embed_chunks(self, chunks: List, show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for ResumeChunk objects.
        
        Args:
            chunks: List of ResumeChunk objects (from indexing.py)
            show_progress: Whether to show progress
        
        Returns:
            List of numpy arrays (embeddings)
        """
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        logger.info(f"Embedding {len(chunks)} resume chunks")
        
        embeddings = self.embed_documents(texts, show_progress=show_progress)
        
        return embeddings
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length (for cosine similarity).
        
        Args:
            embedding: Embedding vector
        
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use hash of text + model name as cache key
        content = f"{self.model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available."""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {str(e)}")
                return None
        
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding to disk."""
        try:
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            np.save(cache_file, embedding)
            logger.debug(f"Cached embedding for text: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {str(e)}")
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache cleared")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model.
        
        Returns:
            Embedding dimension (e.g., 768 for nomic-embed-text)
        """
        # Generate a test embedding to get dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # If embeddings are normalized, dot product = cosine similarity
        if self.normalize:
            return float(np.dot(embedding1, embedding2))
        else:
            # Calculate cosine similarity manually
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about embedding cache."""
        if not self.cache_dir.exists():
            return {'cached_embeddings': 0, 'cache_size_mb': 0}
        
        cached_files = list(self.cache_dir.glob('*.npy'))
        total_size = sum(f.stat().st_size for f in cached_files)
        
        return {
            'cached_embeddings': len(cached_files),
            'cache_size_mb': round(total_size / (1024 * 1024), 2)
        }


class EmbeddingBenchmark:
    """Benchmark different embedding models for resume tasks."""
    
    @staticmethod
    def benchmark_models(test_texts: List[str], 
                        models: List[str] = None) -> Dict[str, Dict]:
        """
        Benchmark multiple embedding models.
        
        Args:
            test_texts: Sample texts to embed
            models: List of model names to test
        
        Returns:
            Dictionary with benchmark results
        """
        if models is None:
            models = ['nomic-embed-text', 'mxbai-embed-large', 'all-minilm']
        
        results = {}
        
        for model in models:
            logger.info(f"Benchmarking model: {model}")
            
            try:
                embedder = OllamaEmbeddings(model=model, cache_embeddings=False)
                
                # Measure embedding time
                start_time = time.time()
                embeddings = embedder.embed_documents(test_texts, show_progress=False)
                elapsed_time = time.time() - start_time
                
                # Get dimension
                dimension = len(embeddings[0]) if embeddings else 0
                
                results[model] = {
                    'dimension': dimension,
                    'total_time': round(elapsed_time, 2),
                    'avg_time_per_text': round(elapsed_time / len(test_texts), 3),
                    'texts_per_second': round(len(test_texts) / elapsed_time, 2),
                    'status': 'success'
                }
                
            except Exception as e:
                results[model] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"Failed to benchmark {model}: {str(e)}")
        
        return results


# Utility functions
def embed_resume_chunks(chunks: List, 
                       model: str = "nomic-embed-text",
                       show_progress: bool = True) -> List[np.ndarray]:
    """
    Convenience function to quickly embed resume chunks.
    
    Args:
        chunks: List of ResumeChunk objects
        model: Embedding model to use
        show_progress: Show progress logging
    
    Returns:
        List of embeddings
    """
    embedder = OllamaEmbeddings(model=model)
    return embedder.embed_chunks(chunks, show_progress=show_progress)


def compute_similarity(text1: str, text2: str, model: str = "nomic-embed-text") -> float:
    """
    Compute similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        model: Embedding model to use
    
    Returns:
        Similarity score (0-1)
    """
    embedder = OllamaEmbeddings(model=model)
    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)
    return embedder.similarity(emb1, emb2)