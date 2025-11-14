from src.vectorstore.indexing import ResumeChunker, ResumeChunk, chunk_resume
from src.vectorstore.embeddings import OllamaEmbeddings, embed_resume_chunks, compute_similarity
from src.vectorstore.vectordb import VectorDatabase, create_vector_database

__all__ = [
    # Indexing/Chunking
    'ResumeChunker',
    'ResumeChunk',
    'chunk_resume',
    
    # Embeddings
    'OllamaEmbeddings',
    'embed_resume_chunks',
    'compute_similarity',
    
    # Vector Database
    'VectorDatabase',
    'create_vector_database',
]

__version__ = '1.0.0'