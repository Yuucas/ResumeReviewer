"""
Document Indexing Module
Handles chunking of resume documents and indexing into vector database.
Uses semantic chunking based on resume structure.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class ResumeChunk:
    """
    Represents a chunk of resume content with metadata.
    """
    content: str                    # The actual text content
    chunk_id: str                   # Unique identifier for this chunk
    source_file: str                # Original file path
    chunk_index: int                # Position in the document (0-indexed)
    section_type: str               # e.g., 'experience', 'education', 'skills'
    metadata: Dict                  # Additional metadata from parser
    char_count: int                 # Number of characters
    word_count: int                 # Number of words
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'content': self.content,
            'chunk_id': self.chunk_id,
            'source_file': self.source_file,
            'chunk_index': self.chunk_index,
            'section_type': self.section_type,
            'metadata': self.metadata,
            'char_count': self.char_count,
            'word_count': self.word_count
        }


class ResumeChunker:
    """
    Chunks resume documents using semantic, structure-aware strategies.
    Optimized for resume content (preserves sections, experience entries, etc.)
    """
    
    def __init__(self,
                 chunk_size: int = 1500,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 3000,
                 chunking_strategy: str = 'semantic'):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
            max_chunk_size: Maximum chunk size (larger chunks are split)
            chunking_strategy: 'semantic' (section-based) or 'fixed' (fixed-size)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunking_strategy = chunking_strategy
        
        logger.info(f"ResumeChunker initialized (strategy: {chunking_strategy}, "
                   f"size: {chunk_size}, overlap: {chunk_overlap})")
    
    def chunk_resume(self, 
                     preprocessed_resume: Dict[str, any],
                     add_context_prefix: bool = True) -> List[ResumeChunk]:
        """
        Chunk a single preprocessed resume.
        
        Args:
            preprocessed_resume: Result from ResumePreprocessor (must have 'preprocessed' key)
            add_context_prefix: Whether to add candidate context to each chunk
        
        Returns:
            List of ResumeChunk objects
        """
        if not preprocessed_resume.get('success'):
            logger.warning(f"Cannot chunk failed resume: {preprocessed_resume.get('file_path')}")
            return []
        
        if 'preprocessed' not in preprocessed_resume:
            logger.error(f"Resume missing 'preprocessed' data: {preprocessed_resume.get('file_path')}")
            return []
        
        markdown_content = preprocessed_resume['preprocessed']['cleaned_markdown']
        metadata = preprocessed_resume['metadata']
        file_path = preprocessed_resume['file_path']
        
        logger.info(f"Chunking resume: {metadata['document_info']['filename']}")
        
        # Choose chunking strategy
        if self.chunking_strategy == 'semantic':
            chunks = self._semantic_chunking(markdown_content, metadata, file_path, add_context_prefix)
        else:  # 'fixed'
            chunks = self._fixed_size_chunking(markdown_content, metadata, file_path, add_context_prefix)
        
        logger.info(f"Created {len(chunks)} chunks from {metadata['document_info']['filename']}")
        
        return chunks
    
    def _semantic_chunking(self, 
                          markdown_content: str,
                          metadata: Dict,
                          file_path: str,
                          add_context_prefix: bool) -> List[ResumeChunk]:
        """
        Semantic chunking: Split by resume sections (Experience, Education, Skills, etc.)
        This preserves semantic meaning and context.
        """
        chunks = []
        
        # Split into sections
        sections = self._split_into_sections(markdown_content)
        
        # Create context prefix (candidate summary)
        context_prefix = self._create_context_prefix(metadata) if add_context_prefix else ""
        
        chunk_index = 0
        
        for section_type, section_content in sections.items():
            if not section_content or not section_content.strip():
                continue
            
            # For long sections (like Experience), split into sub-chunks
            if len(section_content) > self.max_chunk_size:
                sub_chunks = self._split_large_section(section_content, section_type)
                
                for sub_chunk_content in sub_chunks:
                    chunk = self._create_chunk(
                        content=context_prefix + sub_chunk_content if context_prefix else sub_chunk_content,
                        chunk_index=chunk_index,
                        section_type=section_type,
                        metadata=metadata,
                        file_path=file_path
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            # For small sections, merge with context
            elif len(section_content) < self.min_chunk_size:
                # Try to merge with previous chunk
                if chunks and len(chunks[-1].content) + len(section_content) < self.max_chunk_size:
                    # Merge with previous
                    chunks[-1].content += f"\n\n{section_content}"
                    chunks[-1].char_count = len(chunks[-1].content)
                    chunks[-1].word_count = len(chunks[-1].content.split())
                else:
                    # Create new chunk
                    chunk = self._create_chunk(
                        content=context_prefix + section_content if context_prefix else section_content,
                        chunk_index=chunk_index,
                        section_type=section_type,
                        metadata=metadata,
                        file_path=file_path
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            # Normal-sized sections
            else:
                chunk = self._create_chunk(
                    content=context_prefix + section_content if context_prefix else section_content,
                    chunk_index=chunk_index,
                    section_type=section_type,
                    metadata=metadata,
                    file_path=file_path
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _fixed_size_chunking(self,
                            markdown_content: str,
                            metadata: Dict,
                            file_path: str,
                            add_context_prefix: bool) -> List[ResumeChunk]:
        """
        Fixed-size chunking: Split by character count with overlap.
        Less semantic but more consistent chunk sizes.
        """
        chunks = []
        
        # Create context prefix
        context_prefix = self._create_context_prefix(metadata) if add_context_prefix else ""
        
        # Split into paragraphs (preserve paragraph boundaries)
        paragraphs = [p.strip() for p in markdown_content.split('\n\n') if p.strip()]
        
        current_chunk = context_prefix
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds max size, save current chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and len(current_chunk) > self.min_chunk_size:
                chunk = self._create_chunk(
                    content=current_chunk,
                    chunk_index=chunk_index,
                    section_type='mixed',  # Fixed chunking doesn't preserve sections
                    metadata=metadata,
                    file_path=file_path
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                # Take last N characters from previous chunk as overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk != context_prefix else paragraph
        
        # Add final chunk
        if len(current_chunk.strip()) > self.min_chunk_size:
            chunk = self._create_chunk(
                content=current_chunk,
                chunk_index=chunk_index,
                section_type='mixed',
                metadata=metadata,
                file_path=file_path
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sections(self, markdown_content: str) -> Dict[str, str]:
        """
        Split markdown into resume sections.
        Returns dict: {section_type: content}
        """
        sections = {
            'header': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'projects': '',
            'certifications': '',
            'other': ''
        }
        
        lines = markdown_content.split('\n')
        current_section = 'header'  # Start with header (name, contact info)
        
        # Section keywords
        section_keywords = {
            'summary': ['summary', 'profile', 'objective', 'about'],
            'experience': ['experience', 'work history', 'employment', 'professional experience', 'work experience'],
            'education': ['education', 'academic', 'qualifications'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'licenses'],
            'other': ['other', 'additional', 'volunteer', 'activities', 'interests']
        }
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Check if this is a section header
            is_header = False
            
            if line_stripped.startswith('#'):
                header_text = line_stripped.strip('#').strip().lower()
                is_header = True
                
                # Identify section
                for section_key, keywords in section_keywords.items():
                    if any(keyword in header_text for keyword in keywords):
                        current_section = section_key
                        break
            
            elif len(line_stripped) < 50:
                for section_key, keywords in section_keywords.items():
                    if any(keyword.upper() in line_stripped.upper() for keyword in keywords):
                        is_header = True
                        current_section = section_key
                        break
            
            # Add content to current section
            if not is_header:
                sections[current_section] += line + '\n'
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def _split_large_section(self, section_content: str, section_type: str) -> List[str]:
        """
        Split a large section into smaller chunks while preserving semantic boundaries.
        For EXPERIENCE section, tries to keep individual jobs together.
        """
        chunks = []
        
        # For experience section, try to split by job entries
        if section_type == 'experience':
            # Look for job entries (company names or job titles followed by dates)
            # Pattern: Company name + optional location, then job title, then dates
            job_pattern = r'(?:^|\n)([^\n]+(?:Inc|LLC|Ltd|Corporation|Company|University)[^\n]*\n[^\n]+\n[A-Za-z]+\s+\d{4}\s*[-–—]\s*(?:Present|Current|[A-Za-z]+\s+\d{4}))'
            
            jobs = re.split(job_pattern, section_content, flags=re.MULTILINE)
            
            current_chunk = ""
            for job in jobs:
                if not job.strip():
                    continue
                
                # If adding this job exceeds max size, save current chunk
                if len(current_chunk) + len(job) > self.chunk_size and len(current_chunk) > self.min_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = job
                else:
                    current_chunk += job
            
            if current_chunk.strip():
                chunks.append(current_chunk)
        
        else:
            # For other sections, split by paragraphs
            paragraphs = [p for p in section_content.split('\n\n') if p.strip()]
            
            current_chunk = ""
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) > self.chunk_size and len(current_chunk) > self.min_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            
            if current_chunk.strip():
                chunks.append(current_chunk)
        
        return chunks if chunks else [section_content]
    
    def _create_context_prefix(self, metadata: Dict) -> str:
        """
        Create a context prefix with candidate summary for each chunk.
        This helps the LLM understand who the candidate is.
        """
        extracted = metadata.get('extracted_info', {})
        role_info = metadata.get('role_info', {})
        
        context_parts = []
        
        # Candidate name (from filename if available)
        filename = metadata.get('document_info', {}).get('filename', '')
        if filename:
            context_parts.append(f"Candidate: {filename.replace('.pdf', '').replace('_', ' ')}")
        
        # Role category
        if role_info.get('role_category'):
            context_parts.append(f"Applied for: {role_info['role_category'].replace('_', ' ').title()}")
        
        # Years of experience
        if extracted.get('years_of_experience'):
            context_parts.append(f"Total Experience: {extracted['years_of_experience']} years")
        
        # Email
        if extracted.get('email'):
            context_parts.append(f"Contact: {extracted['email']}")
        
        # Location
        if extracted.get('location'):
            context_parts.append(f"Location: {extracted['location']}")
        
        if context_parts:
            return "[Candidate Context: " + " | ".join(context_parts) + "]\n\n"
        
        return ""
    
    def _create_chunk(self,
                     content: str,
                     chunk_index: int,
                     section_type: str,
                     metadata: Dict,
                     file_path: str) -> ResumeChunk:
        """Create a ResumeChunk object."""
        
        # Generate unique chunk ID
        filename = metadata.get('document_info', {}).get('filename', 'unknown')
        chunk_id = f"{filename}_chunk_{chunk_index}"
        
        # Create metadata for this chunk
        chunk_metadata = {
            'filename': metadata.get('document_info', {}).get('filename'),
            'role_category': metadata.get('role_info', {}).get('role_category'),
            'years_of_experience': metadata.get('extracted_info', {}).get('years_of_experience'),
            'email': metadata.get('extracted_info', {}).get('email'),
            'has_skills': metadata.get('search_metadata', {}).get('has_skills'),
            'has_education': metadata.get('search_metadata', {}).get('has_education'),
            'keywords': metadata.get('search_metadata', {}).get('keywords', []),
        }
        
        return ResumeChunk(
            content=content.strip(),
            chunk_id=chunk_id,
            source_file=file_path,
            chunk_index=chunk_index,
            section_type=section_type,
            metadata=chunk_metadata,
            char_count=len(content),
            word_count=len(content.split())
        )
    
    def chunk_batch(self, 
                   preprocessed_resumes: List[Dict[str, any]],
                   add_context_prefix: bool = True) -> List[List[ResumeChunk]]:
        """
        Chunk a batch of preprocessed resumes.
        
        Args:
            preprocessed_resumes: List of preprocessed resumes
            add_context_prefix: Whether to add candidate context to chunks
        
        Returns:
            List of lists, where each inner list contains chunks for one resume
        """
        all_chunks = []
        
        logger.info(f"Chunking batch of {len(preprocessed_resumes)} resumes")
        
        for idx, resume in enumerate(preprocessed_resumes, 1):
            if not resume.get('success'):
                logger.warning(f"Skipping failed resume {idx}/{len(preprocessed_resumes)}")
                all_chunks.append([])
                continue
            
            logger.info(f"Chunking {idx}/{len(preprocessed_resumes)}: "
                       f"{resume['metadata']['document_info']['filename']}")
            
            chunks = self.chunk_resume(resume, add_context_prefix)
            all_chunks.append(chunks)
        
        total_chunks = sum(len(chunks) for chunks in all_chunks)
        logger.info(f"Batch chunking complete: {total_chunks} total chunks from {len(preprocessed_resumes)} resumes")
        
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[ResumeChunk]) -> Dict[str, any]:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of ResumeChunk objects
        
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'chunks_by_section': {}
            }
        
        char_counts = [chunk.char_count for chunk in chunks]
        
        # Count chunks by section
        chunks_by_section = {}
        for chunk in chunks:
            section = chunk.section_type
            chunks_by_section[section] = chunks_by_section.get(section, 0) + 1
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': round(sum(char_counts) / len(char_counts), 2),
            'min_chunk_size': min(char_counts),
            'max_chunk_size': max(char_counts),
            'total_characters': sum(char_counts),
            'chunks_by_section': chunks_by_section
        }
        
        return stats


# Utility function for quick chunking
def chunk_resume(preprocessed_resume: Dict[str, any], 
                chunk_size: int = 1500,
                chunk_overlap: int = 200) -> List[ResumeChunk]:
    """
    Convenience function to quickly chunk a resume.
    
    Args:
        preprocessed_resume: Preprocessed resume dict
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of ResumeChunk objects
    """
    chunker = ResumeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_resume(preprocessed_resume)