"""
Ingestion Package
Handles document loading, parsing, and preprocessing for the resume RAG system.
"""

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.docling_parser import DoclingParser, parse_resume_to_markdown
from src.ingestion.preprocessor import ResumePreprocessor, preprocess_resume

import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Classes
    'DocumentLoader',
    'DoclingParser',
    'ResumePreprocessor',
    
    # Utility functions
    'parse_resume_to_markdown',
    'preprocess_resume',
    'ingest_resumes',  # New convenience function
]

__version__ = '1.0.0'


def ingest_resumes(dataset_path: str, 
                   use_ocr: bool = False,
                   extract_tables: bool = True,
                   save_to_disk: bool = False,
                   output_dir: str = None):
    """
    Complete ingestion pipeline: Load → Parse → Preprocess.
    
    Args:
        dataset_path: Path to dataset root directory
        use_ocr: Whether to use OCR for scanned PDFs
        extract_tables: Whether to extract table structures
        save_to_disk: Whether to save preprocessed markdown to disk
        output_dir: Directory to save preprocessed files (if save_to_disk=True)
    
    Returns:
        List of processed resumes with structure:
        [
            {
                'markdown': str,
                'metadata': dict,
                'preprocessed': dict,
                'file_path': str,
                'success': bool
            },
            ...
        ]
    
    Example:
        >>> from src.ingestion import ingest_resumes
        >>> resumes = ingest_resumes('dataset')
        >>> print(f"Processed {len(resumes)} resumes")
    """
    from pathlib import Path
    
    logger.info(f"Starting complete ingestion pipeline for: {dataset_path}")
    
    # Initialize components
    loader = DocumentLoader(dataset_path)
    parser = DoclingParser(use_ocr=use_ocr, extract_tables=extract_tables)
    preprocessor = ResumePreprocessor()
    
    # Step 1: Load resumes
    logger.info("Step 1: Loading resumes...")
    resumes = loader.get_all_resumes()
    logger.info(f"Found {len(resumes)} resume files")
    
    # Step 2: Parse to markdown
    logger.info("Step 2: Parsing resumes to markdown...")
    parsed_resumes = parser.parse_batch(resumes)
    successful_parses = sum(1 for r in parsed_resumes if r['success'])
    logger.info(f"Successfully parsed {successful_parses}/{len(parsed_resumes)} resumes")
    
    # Step 3: Preprocess markdown
    logger.info("Step 3: Preprocessing markdown content...")
    preprocessed_resumes = preprocessor.preprocess_batch(parsed_resumes)
    successful_preprocessing = sum(1 for r in preprocessed_resumes if r.get('success') and 'preprocessed' in r)
    logger.info(f"Successfully preprocessed {successful_preprocessing}/{len(preprocessed_resumes)} resumes")
    
    # Step 4: Optionally save to disk
    if save_to_disk and output_dir:
        logger.info(f"Step 4: Saving preprocessed files to {output_dir}...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for resume in preprocessed_resumes:
            if resume.get('success') and 'preprocessed' in resume:
                filename = resume['metadata']['document_info']['filename']
                role = resume['metadata']['role_info']['role_category']
                
                output_filename = f"{role}_{Path(filename).stem}_cleaned.md"
                file_path = output_path / output_filename
                
                if preprocessor.save_preprocessed(resume['preprocessed'], str(file_path)):
                    saved_count += 1
        
        logger.info(f"Saved {saved_count} preprocessed files to {output_dir}")
    
    logger.info("Ingestion pipeline complete!")
    
    return preprocessed_resumes