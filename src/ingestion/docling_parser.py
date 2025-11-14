"""
Docling Parser Module
Handles parsing of PDF/DOCX resumes using Docling and converts to Markdown.
Extracts resume-specific metadata including calculated years of experience.
"""

from pathlib import Path
from typing import Dict, Optional, List
import logging
import re
from datetime import datetime
from dateutil import parser as date_parser

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Parser for resume documents using Docling.
    Converts PDF/DOCX to structured Markdown format with rich metadata.
    """
    
    def __init__(self, 
                 use_ocr: bool = False,
                 extract_tables: bool = True,
                 extract_images: bool = False):
        """
        Initialize Docling parser.
        
        Args:
            use_ocr: Whether to use OCR for scanned documents
            extract_tables: Whether to extract and preserve tables
            extract_images: Whether to extract images
        """
        self.use_ocr = use_ocr
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        
        # Configure pipeline options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = use_ocr
        self.pipeline_options.do_table_structure = extract_tables
        
        # Initialize document converter
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.MD
            ],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)}
        )
        
        logger.info(f"DoclingParser initialized (OCR: {use_ocr}, Tables: {extract_tables})")
    
    def parse_to_markdown(self, 
                          file_path: str, 
                          role_category: Optional[str] = None) -> Optional[Dict[str, any]]:
        """
        Parse a resume file and convert to Markdown with comprehensive metadata.
        
        Args:
            file_path: Path to the resume file (PDF/DOCX/DOC)
            role_category: Role category from dataset structure (e.g., 'data_scientist')
        
        Returns:
            Dictionary containing:
            {
                'markdown': str,           # Full markdown content
                'metadata': dict,          # Comprehensive resume metadata
                'file_path': str,          # Original file path
                'success': bool,           # Parse success status
                'error': str or None       # Error message if failed
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {
                'markdown': None,
                'metadata': {},
                'file_path': str(file_path),
                'success': False,
                'error': f"File not found: {file_path}"
            }
        
        try:
            logger.info(f"Parsing document: {file_path.name}")
            
            # Convert document
            result = self.converter.convert(str(file_path))
            
            # Extract markdown
            markdown_content = result.document.export_to_markdown()
            
            # Extract comprehensive metadata
            metadata = self._extract_resume_metadata(
                markdown_content=markdown_content,
                document=result.document,
                file_path=file_path,
                role_category=role_category
            )
            
            logger.info(f"Successfully parsed: {file_path.name} ({metadata['document_info']['num_pages']} pages)")
            
            return {
                'markdown': markdown_content,
                'metadata': metadata,
                'file_path': str(file_path.absolute()),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {str(e)}")
            return {
                'markdown': None,
                'metadata': {'document_info': {'filename': file_path.name}},
                'file_path': str(file_path.absolute()),
                'success': False,
                'error': str(e)
            }
    
    def _extract_resume_metadata(self, 
                                  markdown_content: str,
                                  document,
                                  file_path: Path,
                                  role_category: Optional[str] = None) -> Dict[str, any]:
        """
        Extract comprehensive metadata from resume.
        
        Returns:
            Comprehensive metadata dictionary organized by category
        """
        # 1. DOCUMENT INFO (from file/parsing)
        document_info = {
            'filename': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size_kb': round(file_path.stat().st_size / 1024, 2),
            'num_pages': len(document.pages) if hasattr(document, 'pages') else None,
            'has_tables': self._has_tables(document),
            'num_tables': len(document.tables) if hasattr(document, 'tables') else 0,
            'parsed_at': datetime.now().isoformat(),
            'parser_version': 'docling_v1',
        }
        
        # 2. ROLE CLASSIFICATION (from dataset structure)
        role_info = {
            'role_category': role_category,
            'source_folder': role_category,
        }
        
        # 3. CONTENT STATISTICS (for chunking & retrieval)
        content_stats = {
            'total_characters': len(markdown_content),
            'total_words': len(markdown_content.split()),
            'total_lines': len(markdown_content.split('\n')),
            'estimated_reading_time_minutes': round(len(markdown_content.split()) / 200, 1),
        }
        
        # 4. EXTRACTED INFO (heuristic extraction from markdown)
        extracted_info = self._extract_heuristic_info(markdown_content)
        
        # 5. SECTION DETECTION (for better chunking)
        sections = self._detect_sections(markdown_content)
        
        # 6. SEARCH/FILTER METADATA (for RAG retrieval)
        search_metadata = {
            'keywords': self._extract_keywords(markdown_content, role_category),
            'has_education': sections.get('has_education', False),
            'has_experience': sections.get('has_experience', False),
            'has_skills': sections.get('has_skills', False),
            'has_projects': sections.get('has_projects', False),
            'has_certifications': sections.get('has_certifications', False),
        }
        
        # Combine all metadata
        return {
            'document_info': document_info,
            'role_info': role_info,
            'content_stats': content_stats,
            'extracted_info': extracted_info,
            'sections': sections,
            'search_metadata': search_metadata,
        }
    
    def _extract_heuristic_info(self, markdown_content: str) -> Dict[str, any]:
        """
        Extract candidate information using heuristics (regex patterns).
        Calculates years of experience from work history date ranges.
        """
        extracted = {
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None,
            'location': None,
            'years_of_experience': None,
            'total_months_experience': None,
            'work_history': [],
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, markdown_content)
        if email_match:
            extracted['email'] = email_match.group(0)
        
        # Phone pattern
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        phone_match = re.search(phone_pattern, markdown_content)
        if phone_match:
            extracted['phone'] = phone_match.group(0)
        
        # LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, markdown_content.lower())
        if linkedin_match:
            extracted['linkedin'] = linkedin_match.group(0)
        
        # GitHub
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, markdown_content.lower())
        if github_match:
            extracted['github'] = github_match.group(0)
        
        # Location pattern (City, State ZIP or City, Country)
        location_pattern = r'([A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s*\d{5})'
        location_match = re.search(location_pattern, markdown_content)
        if location_match:
            extracted['location'] = location_match.group(0)
        
        # Extract years of experience from work history
        experience_data = self._calculate_years_of_experience(markdown_content)
        extracted['years_of_experience'] = experience_data['years']
        extracted['total_months_experience'] = experience_data['months']
        extracted['work_history'] = experience_data['work_history']
        
        return extracted
    
    def _calculate_years_of_experience(self, markdown_content: str) -> Dict[str, any]:
        """
        Calculate total years of professional experience from EXPERIENCE section.
        Excludes EDUCATION section dates.
        
        Returns:
            {
                'years': float,           # Total years (rounded to 1 decimal)
                'months': int,            # Total months
                'work_history': [...]     # List of job periods
            }
        """
        # Split content into sections
        sections = self._split_into_sections(markdown_content)
        
        print("SECTION: \n", sections)
        
        # Get only the EXPERIENCE section
        experience_section = sections.get('experience', '')
        
        if not experience_section:
            logger.warning("No EXPERIENCE section found in resume")
            return {'years': None, 'months': None, 'work_history': []}
        
        # Date range patterns
        # Matches: "January 2018 - Present", "Jan 2018 - Jan 2020", "01/2018 - 12/2020", etc.
        date_range_patterns = [
            # Month Year - Month Year (e.g., "January 2018 - Present")
            r'([A-Za-z]+\s+\d{4})\s*[-–—]\s*(Present|Current|[A-Za-z]+\s+\d{4})',
            # MM/YYYY - MM/YYYY (e.g., "01/2018 - 12/2020")
            r'(\d{1,2}/\d{4})\s*[-–—]\s*(Present|Current|\d{1,2}/\d{4})',
            # YYYY-MM - YYYY-MM (e.g., "2018-01 - 2020-12")
            r'(\d{4}-\d{1,2})\s*[-–—]\s*(Present|Current|\d{4}-\d{1,2})',
        ]
        
        work_history = []
        
        for pattern in date_range_patterns:
            matches = re.findall(pattern, experience_section, re.IGNORECASE)
            
            for match in matches:
                start_date_str = match[0]
                end_date_str = match[1]
                
                try:
                    # Parse start date
                    start_date = date_parser.parse(start_date_str, fuzzy=True)
                    
                    # Parse end date (handle "Present" or "Current")
                    if end_date_str.lower() in ['present', 'current']:
                        end_date = datetime.now()
                        end_date_display = 'Present'
                    else:
                        end_date = date_parser.parse(end_date_str, fuzzy=True)
                        end_date_display = end_date.strftime('%Y-%m')
                    
                    # Calculate duration in months
                    months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                    
                    # Only add if duration is positive
                    if months_diff > 0:
                        work_history.append({
                            'start_date': start_date.strftime('%Y-%m'),
                            'end_date': end_date_display,
                            'duration_months': months_diff,
                            'raw_text': f"{start_date_str} - {end_date_str}"
                        })
                    
                except Exception as e:
                    logger.debug(f"Could not parse date range: {start_date_str} - {end_date_str}. Error: {e}")
                    continue
        
        # Calculate total months (sum all experiences)
        total_months = sum(job['duration_months'] for job in work_history)
        total_years = round(total_months / 12, 1) if total_months > 0 else None
        
        if work_history:
            logger.info(f"Calculated experience: {total_years} years ({total_months} months) from {len(work_history)} jobs")
        else:
            logger.warning("No valid work history dates found")
        
        return {
            'years': total_years,
            'months': total_months,
            'work_history': work_history
        }
    
    def _split_into_sections(self, markdown_content: str) -> Dict[str, str]:
        """
        Split resume into major sections (EXPERIENCE, EDUCATION, SKILLS, etc.)
        Handles both single-column and multi-column layouts.
        
        Returns:
            Dictionary with section names as keys and section content as values
        """
        sections = {
            'experience': '',
            'education': '',
            'skills': '',
            'projects': '',
            'certifications': '',
            'summary': '',
            'other': ''
        }
        
        lines = markdown_content.split('\n')
        
        # Common section keywords
        section_keywords = {
            'experience': ['experience', 'work history', 'employment', 'professional experience', 'work experience', 'career history'],
            'education': ['education', 'academic', 'qualifications', 'academic background'],
            'skills': ['skills', 'technical skills', 'competencies', 'core competencies', 'technical competencies'],
            'projects': ['projects', 'portfolio', 'key projects'],
            'certifications': ['certifications', 'certificates', 'licenses', 'professional certifications'],
            'summary': ['summary', 'profile', 'objective', 'about', 'professional summary'],
            'other': ['other', 'additional', 'volunteer', 'activities', 'interests']
        }
        
        current_section = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Skip empty lines
            if not line_stripped:
                if current_section:
                    sections[current_section] += '\n'
                continue
            
            # Check if this line is a section header
            is_header = False
            detected_section = None
            
            # Method 1: Markdown headers (# or ##)
            if line_stripped.startswith('#'):
                header_text = line_stripped.strip('#').strip().lower()
                is_header = True
                
                for section_key, keywords in section_keywords.items():
                    if any(keyword in header_text for keyword in keywords):
                        detected_section = section_key
                        break
            
            # Method 2: Check for standalone keywords (case-insensitive, exact match or starts with)
            elif len(line_stripped) < 50:  # Section headers are usually short
                for section_key, keywords in section_keywords.items():
                    for keyword in keywords:
                        # Check if line is exactly the keyword or starts with it
                        if (line_lower == keyword or 
                            line_lower.startswith(keyword) or
                            line_stripped.upper() == keyword.upper() or
                            line_stripped.upper().startswith(keyword.upper())):
                            is_header = True
                            detected_section = section_key
                            break
                    if is_header:
                        break
            
            # Method 3: Bold text patterns (look for **EXPERIENCE** style)
            elif '**' in line_stripped:
                text_between_stars = re.findall(r'\*\*(.+?)\*\*', line_stripped)
                for text in text_between_stars:
                    text_lower = text.lower()
                    for section_key, keywords in section_keywords.items():
                        if any(keyword in text_lower for keyword in keywords):
                            is_header = True
                            detected_section = section_key
                            break
                    if is_header:
                        break
            
            if is_header and detected_section:
                # Switch to new section
                current_section = detected_section
            else:
                # Add content to current section
                if current_section:
                    sections[current_section] += line + '\n'
        
        # Clean up sections (remove leading/trailing whitespace)
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def _detect_sections(self, markdown_content: str) -> Dict[str, any]:
        """
        Detect common resume sections.
        Returns section presence and their detected headers.
        """
        content_lower = markdown_content.lower()
        
        sections = {
            'has_education': any(keyword in content_lower for keyword in 
                                ['education', 'academic', 'degree', 'university', 'college']),
            'has_experience': any(keyword in content_lower for keyword in 
                                 ['experience', 'employment', 'work history', 'career']),
            'has_skills': any(keyword in content_lower for keyword in 
                            ['skills', 'technical skills', 'competencies', 'proficiencies']),
            'has_projects': any(keyword in content_lower for keyword in 
                              ['projects', 'portfolio', 'work samples']),
            'has_certifications': any(keyword in content_lower for keyword in 
                                    ['certification', 'certificate', 'license']),
            'has_summary': any(keyword in content_lower for keyword in 
                             ['summary', 'profile', 'objective', 'about']),
        }
        
        # Detect section headers (lines starting with # or ##)
        section_headers = []
        for line in markdown_content.split('\n'):
            if line.strip().startswith('#'):
                section_headers.append(line.strip())
        
        sections['detected_headers'] = section_headers
        sections['num_sections'] = len(section_headers)
        
        return sections
    
    def _extract_keywords(self, markdown_content: str, role_category: Optional[str]) -> List[str]:
        """
        Extract relevant keywords for search/filtering.
        Combines role-specific and general technical terms.
        """
        keywords = []
        content_lower = markdown_content.lower()
        
        # Role-specific keyword sets
        role_keywords = {
            'data_scientist': [
                'python', 'c++', 'r', 'machine learning', 'deep learning', 'tensorflow', 
                'pytorch', 'sklearn', 'pandas', 'numpy', 'statistics', 'sql',
                'data analysis', 'visualization', 'nlp', 'computer vision', 'ai',
                'data science', 'analytics', 'modeling', 'forecasting', 'big data',
                'hadoop', 'spark', 'etl', 'data engineering', 'matplotlib', 'seaborn',
                'scikit-learn', 'feature engineering', 'scipy'
            ],
            'fullstack_engineer': [
                'javascript', 'typescript', 'react', 'angular', 'vue', 'node.js',
                'python', 'java', 'c#', 'sql', 'nosql', 'api', 'rest', 'graphql',
                'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'frontend',
                'backend', 'full stack', 'web development'
            ],
            'it': [
                'windows', 'linux', 'networking', 'security', 'cloud', 'aws',
                'azure', 'virtualization', 'vmware', 'active directory', 'cisco',
                'itil', 'devops', 'monitoring', 'troubleshooting', 'infrastructure',
                'system administration', 'helpdesk', 'support'
            ]
        }
        
        # Add role-specific keywords if they appear in content
        if role_category and role_category in role_keywords:
            for keyword in role_keywords[role_category]:
                if keyword in content_lower:
                    keywords.append(keyword)
        
        # Add general technical terms
        general_keywords = [
            'agile', 'scrum', 'git', 'github', 'jira', 'leadership',
            'team', 'project management', 'communication', 'problem solving'
        ]
        
        for keyword in general_keywords:
            if keyword in content_lower:
                keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
    
    def _has_tables(self, document) -> bool:
        """Check if document contains tables."""
        try:
            if hasattr(document, 'tables'):
                return len(document.tables) > 0
            return False
        except:
            return False
    
    def parse_batch(self, 
                    resume_infos: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Parse multiple resume files in batch with their metadata.
        
        Args:
            resume_infos: List of resume info dicts from DocumentLoader
                         (must have 'file_path' and 'role_category')
        
        Returns:
            List of parse results
        """
        results = []
        
        logger.info(f"Starting batch parsing of {len(resume_infos)} documents")
        
        for idx, resume_info in enumerate(resume_infos, 1):
            logger.info(f"Processing {idx}/{len(resume_infos)}: {resume_info['filename']}")
            
            result = self.parse_to_markdown(
                file_path=resume_info['file_path'],
                role_category=resume_info.get('role_category')
            )
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info(f"Batch parsing complete: {successful} successful, {failed} failed")
        
        return results
    
    def save_markdown(self, markdown_content: str, output_path: str) -> bool:
        """
        Save markdown content to a file.
        
        Args:
            markdown_content: Markdown string to save
            output_path: Path where to save the markdown file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Saved markdown to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save markdown: {str(e)}")
            return False
    
    def get_document_stats(self, file_path: str) -> Optional[Dict[str, any]]:
        """
        Get statistics about a document without full parsing.
        
        Args:
            file_path: Path to the document
        
        Returns:
            Dictionary with document statistics
        """
        try:
            result = self.converter.convert(str(file_path))
            
            stats = {
                'filename': Path(file_path).name,
                'num_pages': len(result.document.pages) if hasattr(result.document, 'pages') else None,
                'num_tables': len(result.document.tables) if hasattr(result.document, 'tables') else 0,
                'text_length': len(result.document.export_to_markdown()),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for {file_path}: {str(e)}")
            return None


# Utility function for quick parsing
def parse_resume_to_markdown(file_path: str, use_ocr: bool = False) -> Optional[str]:
    """
    Convenience function to quickly parse a resume to markdown.
    
    Args:
        file_path: Path to resume file
        use_ocr: Whether to use OCR
    
    Returns:
        Markdown string or None if failed
    """
    parser = DoclingParser(use_ocr=use_ocr)
    result = parser.parse_to_markdown(file_path)
    
    return result['markdown'] if result['success'] else None