"""
Configuration Module
Handles application configuration and settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Application configuration."""

    # Paths (using absolute paths to avoid working directory issues)
    dataset_root: str = str(Path(__file__).parent.parent.parent / "dataset")
    chroma_db_path: str = str(Path(__file__).parent.parent.parent / "chroma_db")
    cache_dir: str = str(Path(__file__).parent.parent.parent / ".cache")
    output_dir: str = str(Path(__file__).parent.parent.parent / "output")
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen3:latest"
    
    # Embedding settings
    embedding_dimension: int = 768
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    
    # Chunking settings
    chunk_size: int = 3000
    chunk_overlap: int = 200
    min_chunk_size: int = 400
    max_chunk_size: int = 3200
    chunking_strategy: str = "semantic"  # or "fixed"
    
    # Retrieval settings
    top_k_chunks: int = 50
    top_k_candidates: int = 5
    min_similarity: float = 0.3
    aggregation_method: str = "weighted"  # "average", "max", or "weighted"
    
    # Reranking settings
    rerank_method: str = "hybrid"  # "keyword", "experience", "section", "hybrid", "llm"
    rerank_top_k: int = 5
    
    # LLM settings
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 120
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Create necessary directories."""
        Path(self.dataset_root).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'dataset_root': self.dataset_root,
            'chroma_db_path': self.chroma_db_path,
            'cache_dir': self.cache_dir,
            'output_dir': self.output_dir,
            'ollama_base_url': self.ollama_base_url,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'normalize_embeddings': self.normalize_embeddings,
            'cache_embeddings': self.cache_embeddings,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'chunking_strategy': self.chunking_strategy,
            'top_k_chunks': self.top_k_chunks,
            'top_k_candidates': self.top_k_candidates,
            'min_similarity': self.min_similarity,
            'aggregation_method': self.aggregation_method,
            'rerank_method': self.rerank_method,
            'rerank_top_k': self.rerank_top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout,
            'log_level': self.log_level,
            'log_file': self.log_file,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_json(self, json_path: str):
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {json_path}")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load config from environment variables."""
        # Get project root (3 levels up from this file: config.py -> utils -> src -> project_root)
        project_root = Path(__file__).parent.parent.parent

        return cls(
        dataset_root=os.getenv('DATASET_ROOT', str(project_root / 'dataset')),
        chroma_db_path=os.getenv('CHROMA_DB_PATH', str(project_root / 'chroma_db')),
        cache_dir=os.getenv('CACHE_DIR', str(project_root / '.cache')),
        output_dir=os.getenv('OUTPUT_DIR', str(project_root / 'output')),
        ollama_base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        embedding_model=os.getenv('EMBEDDING_MODEL', 'nomic-embed-text'),
        llm_model=os.getenv('LLM_MODEL', 'qwen3:latest'),
        chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
        top_k_chunks=int(os.getenv('TOP_K_CHUNKS', '50')),
        top_k_candidates=int(os.getenv('TOP_K_CANDIDATES', '10')),
        min_similarity=float(os.getenv('MIN_SIMILARITY', '0.3')),
        temperature=float(os.getenv('TEMPERATURE', '0.3')),
        max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
    )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config instance
    """
    global _config
    
    if _config is None:
        # Try to load from config file first
        config_path = Path('config.json')
        if config_path.exists():
            _config = Config.from_json(str(config_path))
            logger.info("Loaded config from config.json")
        else:
            # Fall back to defaults
            _config = Config()
            logger.info("Using default configuration")
    
    return _config


def set_config(config: Config):
    """
    Set the global configuration instance.
    
    Args:
        config: Config instance
    """
    global _config
    _config = config
    logger.info("Configuration updated")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Path to config JSON file (optional)
    
    Returns:
        Config instance
    """
    if config_path and Path(config_path).exists():
        config = Config.from_json(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        # Try environment variables
        config = Config.from_env()
        logger.info("Loaded config from environment variables")
    
    set_config(config)
    return config


def create_default_config(output_path: str = "config.json"):
    """
    Create a default configuration file.
    
    Args:
        output_path: Where to save the config file
    """
    config = Config()
    config.save_to_json(output_path)
    print(f"✓ Default configuration saved to {output_path}")
    print("  You can edit this file to customize settings")