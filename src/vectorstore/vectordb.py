"""
Vector Database Module
Handles storage and retrieval of embeddings using ChromaDB.
Provides semantic search capabilities for resume chunks.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector database for storing and retrieving resume embeddings.
    Uses ChromaDB for persistent storage and fast similarity search.
    """
    
    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "resumes",
                 embedding_dimension: int = 768,
                 distance_metric: str = "cosine"):
        """
        Initialize ChromaDB vector database.
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection (like a table in SQL)
            embedding_dimension: Dimension of embeddings (768 for nomic-embed-text)
            distance_metric: 'cosine', 'l2', or 'ip' (inner product)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"VectorDatabase initialized (collection: {collection_name}, "
                   f"metric: {distance_metric})")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We provide embeddings ourselves
            )
            
            count = collection.count()
            logger.info(f"Loaded existing collection '{self.collection_name}' "
                       f"with {count} documents")
            
        except Exception:
            # Collection doesn't exist, create it
            logger.info(f"Creating new collection: {self.collection_name}")
            
            # Map distance metric to ChromaDB format
            metric_map = {
                'cosine': 'cosine',
                'l2': 'l2',
                'ip': 'ip'  # inner product
            }
            
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": metric_map.get(self.distance_metric, 'cosine'),
                    "embedding_dimension": self.embedding_dimension,
                    "created_at": datetime.now().isoformat()
                },
                embedding_function=None
            )
            
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(self,
                     chunks: List,
                     embeddings: List[np.ndarray],
                     batch_size: int = 100) -> Dict[str, int]:
        """
        Add resume chunks and their embeddings to the database.
        
        Args:
            chunks: List of ResumeChunk objects
            embeddings: List of embedding vectors (must match chunks order)
            batch_size: Number of documents to add per batch
        
        Returns:
            Dictionary with statistics about the operation
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                           f"must have the same length")
        
        if not chunks:
            logger.warning("No chunks provided to add")
            return {
                'added': 0,
                'failed': 0,
                'total_in_collection': self.collection.count()
            }
        
        logger.info(f"Adding {len(chunks)} documents to collection '{self.collection_name}'")
        
        added_count = 0
        failed_count = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.info(f"Adding batch {batch_num}/{total_batches} "
                       f"({len(batch_chunks)} documents)")
            
            try:
                # Prepare data for ChromaDB
                ids = [chunk.chunk_id for chunk in batch_chunks]
                documents = [chunk.content for chunk in batch_chunks]
                embeddings_list = [emb.tolist() for emb in batch_embeddings]
                metadatas = [self._prepare_metadata(chunk) for chunk in batch_chunks]
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    documents=documents,
                    metadatas=metadatas
                )
                
                added_count += len(batch_chunks)
                logger.info(f"Batch {batch_num} added successfully")
                
            except Exception as e:
                logger.error(f"Failed to add batch {batch_num}: {str(e)}")
                failed_count += len(batch_chunks)
        
        logger.info(f"Added {added_count} documents, {failed_count} failed")
        
        return {
            'added': added_count,
            'failed': failed_count,
            'total_in_collection': self.collection.count()
        }
    
    def _prepare_metadata(self, chunk) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB storage.
        ChromaDB only supports: str, int, float, bool (no nested dicts or lists)
        """
        metadata = {
            # Basic info
            'source_file': chunk.source_file,
            'chunk_index': chunk.chunk_index,
            'section_type': chunk.section_type,
            'char_count': chunk.char_count,
            'word_count': chunk.word_count,
            
            # From chunk metadata
            'filename': chunk.metadata.get('filename', ''),
            'role_category': chunk.metadata.get('role_category', ''),
            'email': chunk.metadata.get('email', ''),
            'has_skills': chunk.metadata.get('has_skills', False),
            'has_education': chunk.metadata.get('has_education', False),
        }
        
        # Add years of experience (handle None)
        years_exp = chunk.metadata.get('years_of_experience')
        if years_exp is not None:
            metadata['years_of_experience'] = float(years_exp)
        else:
            metadata['years_of_experience'] = 0.0
        
        # Convert keywords list to JSON string (ChromaDB doesn't support lists)
        keywords = chunk.metadata.get('keywords', [])
        if keywords:
            metadata['keywords_json'] = json.dumps(keywords)
        else:
            metadata['keywords_json'] = '[]'
        
        return metadata
    
    def search(self,
              query_embedding: np.ndarray,
              top_k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {'role_category': 'data_scientist'})
        
        Returns:
            List of search results with documents, metadata, and distances
        """
        try:
            # Prepare query embedding
            query_embedding_list = query_embedding.tolist()
            
            # Build where clause for filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = self._format_results(results)
            
            logger.info(f"Search returned {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from filters.

        ChromaDB requires multiple conditions to be wrapped in $and/$or operators.

        Example:
            filters = {'role_category': 'data_scientist', 'years_of_experience': {'$gte': 5}}
            → {"$and": [{"role_category": "data_scientist"}, {"years_of_experience": {"$gte": 5}}]}
        """
        if not filters:
            return {}

        # If only one filter, return it directly
        if len(filters) == 1:
            key, value = list(filters.items())[0]
            return {key: value}

        # Multiple filters: wrap in $and
        conditions = []
        for key, value in filters.items():
            conditions.append({key: value})

        return {"$and": conditions}
    
    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """
        Format ChromaDB results into a cleaner structure.
        
        ChromaDB returns:
        {
            'ids': [['id1', 'id2', ...]],
            'documents': [['doc1', 'doc2', ...]],
            'metadatas': [[{...}, {...}, ...]],
            'distances': [[0.1, 0.2, ...]]
        }
        
        We convert to:
        [
            {'id': 'id1', 'document': 'doc1', 'metadata': {...}, 'distance': 0.1},
            {'id': 'id2', 'document': 'doc2', 'metadata': {...}, 'distance': 0.2},
            ...
        ]
        """
        formatted = []
        
        # ChromaDB wraps results in nested lists
        ids = results['ids'][0] if results['ids'] else []
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        for i in range(len(ids)):
            # Convert distance to similarity score (for cosine: 1 - distance)
            distance = distances[i] if i < len(distances) else 0.0
            similarity = 1.0 - distance  # Assumes cosine distance
            
            # Parse keywords back from JSON
            metadata = metadatas[i] if i < len(metadatas) else {}
            if 'keywords_json' in metadata:
                try:
                    metadata['keywords'] = json.loads(metadata['keywords_json'])
                    del metadata['keywords_json']  # Remove JSON string
                except:
                    metadata['keywords'] = []
            
            formatted.append({
                'id': ids[i],
                'document': documents[i] if i < len(documents) else '',
                'metadata': metadata,
                'distance': distance,
                'similarity': similarity
            })
        
        return formatted
    
    def search_by_text(self,
                      query_text: str,
                      embedder,
                      top_k: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using text query (convenience method).
        
        Args:
            query_text: Text query (e.g., job description)
            embedder: OllamaEmbeddings instance to embed the query
            top_k: Number of results
            filters: Metadata filters
        
        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query_text[:100]}...")
        
        # Embed the query
        query_embedding = embedder.embed_query(query_text)
        
        # Search
        results = self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        return results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            chunk_id: Chunk ID to retrieve
        
        Returns:
            Document data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if not results['ids']:
                return None
            
            # Parse metadata
            metadata = results['metadatas'][0]
            if 'keywords_json' in metadata:
                try:
                    metadata['keywords'] = json.loads(metadata['keywords_json'])
                    del metadata['keywords_json']
                except:
                    metadata['keywords'] = []
            
            return {
                'id': results['ids'][0],
                'document': results['documents'][0],
                'metadata': metadata,
                'embedding': np.array(results['embeddings'][0])
            }
            
        except Exception as e:
            logger.error(f"Failed to get document {chunk_id}: {str(e)}")
            return None
    
    def get_documents_by_filter(self,
                               filters: Dict[str, Any],
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get documents matching metadata filters (no embedding search).
        
        Args:
            filters: Metadata filters (e.g., {'role_category': 'data_scientist'})
            limit: Maximum number of results
        
        Returns:
            List of matching documents
        """
        try:
            where_clause = self._build_where_clause(filters)
            
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            formatted = []
            for i in range(len(results['ids'])):
                metadata = results['metadatas'][i]
                
                # Parse keywords
                if 'keywords_json' in metadata:
                    try:
                        metadata['keywords'] = json.loads(metadata['keywords_json'])
                        del metadata['keywords_json']
                    except:
                        metadata['keywords'] = []
                
                formatted.append({
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': metadata
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to get documents by filter: {str(e)}")
            return []
    
    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Delete documents matching filters.
        
        Args:
            filters: Metadata filters
        
        Returns:
            Number of documents deleted
        """
        try:
            # Get documents to delete
            docs = self.get_documents_by_filter(filters)
            ids_to_delete = [doc['id'] for doc in docs]
            
            if not ids_to_delete:
                logger.info("No documents found matching filters")
                return 0
            
            # Delete
            self.collection.delete(ids=ids_to_delete)
            
            logger.info(f"Deleted {len(ids_to_delete)} documents")
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return 0
    
    def update_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific document.
        
        Args:
            chunk_id: Document ID
            metadata: New metadata (will be merged with existing)
        
        Returns:
            True if successful
        """
        try:
            # ChromaDB doesn't have direct update, so we need to delete and re-add
            doc = self.get_by_id(chunk_id)
            
            if not doc:
                logger.warning(f"Document {chunk_id} not found")
                return False
            
            # Merge metadata
            updated_metadata = {**doc['metadata'], **metadata}
            
            # Delete old
            self.collection.delete(ids=[chunk_id])
            
            # Re-add with updated metadata
            self.collection.add(
                ids=[chunk_id],
                embeddings=[doc['embedding'].tolist()],
                documents=[doc['document']],
                metadatas=[self._prepare_metadata_dict(updated_metadata)]
            )
            
            logger.info(f"Updated metadata for {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            return False
    
    def _prepare_metadata_dict(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a raw metadata dict for ChromaDB."""
        prepared = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, list):
                prepared[f'{key}_json'] = json.dumps(value)
            elif value is None:
                prepared[key] = ''
        
        return prepared
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            total_count = self.collection.count()
            
            # Get sample to analyze
            sample = self.collection.get(limit=min(100, total_count))
            
            # Count by role category
            role_counts = {}
            section_counts = {}
            
            for metadata in sample['metadatas']:
                role = metadata.get('role_category', 'unknown')
                section = metadata.get('section_type', 'unknown')
                
                role_counts[role] = role_counts.get(role, 0) + 1
                section_counts[section] = section_counts.get(section, 0) + 1
            
            return {
                'total_documents': total_count,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_dimension,
                'distance_metric': self.distance_metric,
                'sample_role_distribution': role_counts,
                'sample_section_distribution': section_counts,
                'persist_directory': str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {'error': str(e)}
    
    def clear_collection(self):
        """Delete all documents from the collection."""
        try:
            # Delete collection
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
            # Recreate empty collection
            self.collection = self._get_or_create_collection()
            logger.info(f"Recreated empty collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise
    
    def reset_database(self):
        """Reset the entire database (delete all collections)."""
        try:
            self.client.reset()
            logger.info("Database reset successfully")
            
            # Recreate collection
            self.collection = self._get_or_create_collection()
            
        except Exception as e:
            logger.error(f"Failed to reset database: {str(e)}")
            raise


# Utility functions
def create_vector_database(persist_directory: str = "./chroma_db",
                          collection_name: str = "resumes") -> VectorDatabase:
    """
    Convenience function to create a vector database.
    
    Args:
        persist_directory: Where to store the database
        collection_name: Name of the collection
    
    Returns:
        VectorDatabase instance
    """
    return VectorDatabase(
        persist_directory=persist_directory,
        collection_name=collection_name
    )