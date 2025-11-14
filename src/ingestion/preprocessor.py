"""
Preprocessor Module
Cleans and normalizes markdown content from parsed resumes for optimal RAG performance.
"""

import re
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ResumePreprocessor:
    """
    Preprocesses resume markdown content for RAG pipeline.
    Handles cleaning, normalization, and structure preservation.
    """
    
    def __init__(self,
                 remove_extra_whitespace: bool = True,
                 normalize_headers: bool = True,
                 remove_special_chars: bool = False,
                 preserve_bullet_points: bool = True,
                 preserve_tables: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            remove_extra_whitespace: Remove excessive blank lines and spaces
            normalize_headers: Standardize markdown header levels
            remove_special_chars: Remove special characters (be careful with this)
            preserve_bullet_points: Keep bullet point structure
            preserve_tables: Keep table formatting
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_headers = normalize_headers
        self.remove_special_chars = remove_special_chars
        self.preserve_bullet_points = preserve_bullet_points
        self.preserve_tables = preserve_tables
        
        logger.info(f"ResumePreprocessor initialized")
    
    def preprocess(self, markdown_content: str, metadata: Optional[Dict] = None) -> Dict[str, any]:
        """
        Preprocess resume markdown content.
        
        Args:
            markdown_content: Raw markdown from Docling parser
            metadata: Optional metadata from parser (for context-aware preprocessing)
        
        Returns:
            Dictionary containing:
            {
                'cleaned_markdown': str,      # Preprocessed markdown
                'original_length': int,       # Original character count
                'cleaned_length': int,        # Cleaned character count
                'reduction_percentage': float, # How much was reduced
                'preprocessing_steps': list   # Steps applied
            }
        """
        if not markdown_content or not markdown_content.strip():
            logger.warning("Empty markdown content provided")
            return {
                'cleaned_markdown': '',
                'original_length': 0,
                'cleaned_length': 0,
                'reduction_percentage': 0.0,
                'preprocessing_steps': []
            }
        
        original_length = len(markdown_content)
        cleaned = markdown_content
        steps_applied = []
        
        # Step 1: Remove page breaks and artifacts
        cleaned = self._remove_page_artifacts(cleaned)
        steps_applied.append('remove_page_artifacts')
        
        # Step 2: Normalize whitespace
        if self.remove_extra_whitespace:
            cleaned = self._normalize_whitespace(cleaned)
            steps_applied.append('normalize_whitespace')
        
        # Step 3: Normalize headers
        if self.normalize_headers:
            cleaned = self._normalize_headers(cleaned)
            steps_applied.append('normalize_headers')
        
        # Step 4: Clean bullet points
        if self.preserve_bullet_points:
            cleaned = self._normalize_bullet_points(cleaned)
            steps_applied.append('normalize_bullet_points')
        
        # Step 5: Fix common OCR/parsing errors
        cleaned = self._fix_common_errors(cleaned)
        steps_applied.append('fix_common_errors')
        
        # Step 6: Remove special characters (optional)
        if self.remove_special_chars:
            cleaned = self._remove_special_characters(cleaned)
            steps_applied.append('remove_special_chars')
        
        # Step 7: Normalize contact information
        cleaned = self._normalize_contact_info(cleaned)
        steps_applied.append('normalize_contact_info')
        
        # Step 8: Final cleanup
        cleaned = self._final_cleanup(cleaned)
        steps_applied.append('final_cleanup')
        
        cleaned_length = len(cleaned)
        reduction_percentage = round(((original_length - cleaned_length) / original_length) * 100, 2) if original_length > 0 else 0
        
        logger.info(f"Preprocessing complete: {original_length} → {cleaned_length} chars ({reduction_percentage}% reduction)")
        
        return {
            'cleaned_markdown': cleaned,
            'original_length': original_length,
            'cleaned_length': cleaned_length,
            'reduction_percentage': reduction_percentage,
            'preprocessing_steps': steps_applied
        }
    
    def _remove_page_artifacts(self, text: str) -> str:
        """Remove page numbers, headers, footers, and other artifacts."""
        # Remove page numbers (e.g., "Page 1 of 2", "1/2", etc.)
        text = re.sub(r'(?i)page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'^\d+\s*/\s*\d+$', '', text, flags=re.MULTILINE)
        
        # Remove common footer/header patterns
        text = re.sub(r'(?i)^confidential.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(?i)^resume.*\d{4}$', '', text, flags=re.MULTILINE)
        
        # Remove excessive dashes or underscores (decorative lines)
        text = re.sub(r'^[-_=]{3,}$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace tabs with spaces
        text = text.replace('\t', '    ')
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove more than 2 consecutive blank lines
        normalized_lines = []
        blank_count = 0
        
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    normalized_lines.append(line)
            else:
                blank_count = 0
                normalized_lines.append(line)
        
        text = '\n'.join(normalized_lines)
        
        # Remove excessive spaces (more than 2)
        text = re.sub(r' {3,}', '  ', text)
        
        return text
    
    def _normalize_headers(self, text: str) -> str:
        """Normalize markdown headers to consistent format."""
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Ensure space after # in headers
            if line.strip().startswith('#'):
                # Count the number of #
                hash_count = len(re.match(r'^#+', line.strip()).group())
                # Extract header text
                header_text = line.strip()[hash_count:].strip()
                # Reconstruct with proper spacing
                normalized_line = '#' * hash_count + ' ' + header_text
                normalized_lines.append(normalized_line)
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _normalize_bullet_points(self, text: str) -> str:
        """Normalize bullet points to consistent format."""
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Convert various bullet styles to consistent format
            # Handle: ●, •, *, -, ○, ▪, ▫, →
            if stripped and stripped[0] in ['●', '•', '○', '▪', '▫', '→']:
                # Replace with standard markdown bullet
                content = stripped[1:].strip()
                # Preserve indentation level
                indent = len(line) - len(line.lstrip())
                normalized_line = ' ' * indent + '- ' + content
                normalized_lines.append(normalized_line)
            
            # Ensure space after - or * in markdown lists
            elif re.match(r'^(\s*)[-*]\S', line):
                match = re.match(r'^(\s*)([-*])(.+)$', line)
                if match:
                    indent, bullet, content = match.groups()
                    normalized_line = indent + bullet + ' ' + content.strip()
                    normalized_lines.append(normalized_line)
                else:
                    normalized_lines.append(line)
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _fix_common_errors(self, text: str) -> str:
        """Fix common OCR and parsing errors."""
        # Fix common OCR mistakes
        replacements = {
            r'\bl\b': 'I',  # lowercase L misread as I (context-dependent)
            r'\b0\b(?=\s+[A-Z])': 'O',  # zero misread as O
            r'(?<=\w)l(?=\d)': '1',  # lowercase L before digit
            r'(?<=\d)O(?=\d)': '0',  # O between digits
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix broken words (words split by newlines with hyphens)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix multiple punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])(?=[A-Za-z])', r'\1 ', text)
        
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """
        Remove special characters (use with caution).
        Keeps: letters, numbers, basic punctuation, whitespace, markdown syntax
        """
        # Keep: a-z, A-Z, 0-9, spaces, newlines, basic punctuation, and markdown symbols
        text = re.sub(r'[^a-zA-Z0-9\s\n.,;:!?()\-\'"#*_\[\]@/\\+=]', '', text)
        return text
    
    def _normalize_contact_info(self, text: str) -> str:
        """Normalize contact information format."""
        # Normalize email (ensure proper format)
        text = re.sub(r'\s+@\s+', '@', text)
        text = re.sub(r'@\s+', '@', text)
        
        # Normalize phone numbers (remove excessive formatting)
        # Keep in format: +1 234 567-890 or (123) 456-7890
        text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
        
        # Normalize URLs (remove extra spaces)
        text = re.sub(r'(https?://)\s+', r'\1', text)
        text = re.sub(r'linkedin\.com\s*/\s*in\s*/\s*', 'linkedin.com/in/', text)
        text = re.sub(r'github\.com\s*/\s*', 'github.com/', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        # Remove leading/trailing whitespace from entire text
        text = text.strip()
        
        # Ensure single newline at end
        text = text.rstrip('\n') + '\n'
        
        # Remove any remaining control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def preprocess_batch(self, parsed_resumes: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Preprocess a batch of parsed resumes.
        
        Args:
            parsed_resumes: List of results from DoclingParser.parse_batch()
        
        Returns:
            List of resumes with added 'preprocessed' field
        """
        preprocessed_resumes = []
        
        logger.info(f"Preprocessing batch of {len(parsed_resumes)} resumes")
        
        for idx, resume in enumerate(parsed_resumes, 1):
            if not resume['success']:
                logger.warning(f"Skipping failed parse: {resume.get('file_path', 'unknown')}")
                preprocessed_resumes.append(resume)
                continue
            
            logger.info(f"Preprocessing {idx}/{len(parsed_resumes)}: {resume['metadata']['document_info']['filename']}")
            
            # Preprocess the markdown
            preprocessed = self.preprocess(
                markdown_content=resume['markdown'],
                metadata=resume['metadata']
            )
            
            # Add preprocessed data to resume
            resume['preprocessed'] = preprocessed
            preprocessed_resumes.append(resume)
        
        successful = sum(1 for r in preprocessed_resumes if r.get('success') and 'preprocessed' in r)
        logger.info(f"Batch preprocessing complete: {successful}/{len(parsed_resumes)} processed")
        
        return preprocessed_resumes
    
    def save_preprocessed(self, preprocessed_data: Dict[str, any], output_path: str) -> bool:
        """
        Save preprocessed markdown to a file.
        
        Args:
            preprocessed_data: Result from preprocess()
            output_path: Where to save the file
        
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(preprocessed_data['cleaned_markdown'])
            
            logger.info(f"Saved preprocessed markdown to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save preprocessed markdown: {str(e)}")
            return False


# Utility function for quick preprocessing
def preprocess_resume(markdown_content: str) -> str:
    """
    Convenience function to quickly preprocess resume markdown.
    
    Args:
        markdown_content: Raw markdown from parser
    
    Returns:
        Cleaned markdown string
    """
    preprocessor = ResumePreprocessor()
    result = preprocessor.preprocess(markdown_content)
    return result['cleaned_markdown']