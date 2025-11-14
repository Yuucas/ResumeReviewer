"""
Retrieval Module
Advanced retrieval strategies for finding the best candidate matches.
Combines semantic search, metadata filtering, and candidate aggregation.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CandidateMatch:
    """
    Represents a candidate match with aggregated information.
    """
    filename: str                           # Resume filename
    role_category: str                      # Applied role
    email: str                              # Contact email
    years_of_experience: float              # Total experience
    average_similarity: float               # Average chunk similarity
    max_similarity: float                   # Best chunk similarity
    matched_chunks: List[Dict[str, Any]]    # All matched chunks
    chunk_count: int                        # Number of matched chunks
    section_coverage: Dict[str, int]        # Sections found
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filename': self.filename,
            'role_category': self.role_category,
            'email': self.email,
            'years_of_experience': self.years_of_experience,
            'average_similarity': self.average_similarity,
            'max_similarity': self.max_similarity,
            'chunk_count': self.chunk_count,
            'section_coverage': self.section_coverage,
            'matched_chunks': self.matched_chunks
        }


class ResumeRetriever:
    """
    Advanced retrieval system for resume search.
    Provides multiple search strategies and candidate ranking.
    """
    
    def __init__(self, vectordb, embedder):
        """
        Initialize the retriever.
        
        Args:
            vectordb: VectorDatabase instance
            embedder: OllamaEmbeddings instance
        """
        self.vectordb = vectordb
        self.embedder = embedder
        
        logger.info("ResumeRetriever initialized")
    
    def search(self,
              query: str,
              top_k: int = 20,
              filters: Optional[Dict[str, Any]] = None,
              min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Basic semantic search for resume chunks.
        
        Args:
            query: Search query (job description, requirements, etc.)
            top_k: Number of chunks to retrieve
            filters: Metadata filters (e.g., {'role_category': 'data_scientist'})
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of matching chunks with similarity scores
        """
        logger.info(f"Searching for: '{query[:100]}...'")
        
        # Search vector database
        results = self.vectordb.search_by_text(
            query_text=query,
            embedder=self.embedder,
            top_k=top_k,
            filters=filters
        )
        
        # Filter by minimum similarity
        if min_similarity > 0:
            results = [r for r in results if r['similarity'] >= min_similarity]
            logger.info(f"Filtered to {len(results)} results above {min_similarity} similarity")
        
        return results
    
    def search_candidates(self,
                         query: str,
                         top_k_chunks: int = 50,
                         top_k_candidates: int = 10,
                         filters: Optional[Dict[str, Any]] = None,
                         min_similarity: float = 0.3,
                         aggregation_method: str = 'average') -> List[CandidateMatch]:
        """
        Search for candidates (aggregates chunks per resume).
        
        Args:
            query: Search query (job description)
            top_k_chunks: Number of chunks to retrieve from vector DB
            top_k_candidates: Number of top candidates to return
            filters: Metadata filters
            min_similarity: Minimum similarity threshold
            aggregation_method: 'average', 'max', or 'weighted'
                - average: Average similarity across all chunks
                - max: Use highest chunk similarity
                - weighted: Weight by section importance
        
        Returns:
            List of CandidateMatch objects, sorted by relevance
        """
        logger.info(f"Searching for top {top_k_candidates} candidates")
        
        # Step 1: Get relevant chunks
        chunks = self.search(
            query=query,
            top_k=top_k_chunks,
            filters=filters,
            min_similarity=min_similarity
        )
        
        if not chunks:
            logger.warning("No chunks found matching criteria")
            return []
        
        logger.info(f"Found {len(chunks)} relevant chunks from vector search")
        
        # Step 2: Group chunks by candidate (filename)
        candidates_data = self._group_chunks_by_candidate(chunks)
        
        logger.info(f"Grouped into {len(candidates_data)} unique candidates")
        
        # Step 3: Create CandidateMatch objects
        candidate_matches = []
        
        for filename, data in candidates_data.items():
            # Calculate aggregated similarity score
            if aggregation_method == 'average':
                score = np.mean([c['similarity'] for c in data['chunks']])
            elif aggregation_method == 'max':
                score = max([c['similarity'] for c in data['chunks']])
            elif aggregation_method == 'weighted':
                score = self._weighted_similarity(data['chunks'])
            else:
                score = np.mean([c['similarity'] for c in data['chunks']])
            
            # Count section coverage
            section_coverage = defaultdict(int)
            for chunk in data['chunks']:
                section = chunk['metadata'].get('section_type', 'unknown')
                section_coverage[section] += 1
            
            # Create candidate match
            candidate = CandidateMatch(
                filename=filename,
                role_category=data['role_category'],
                email=data['email'],
                years_of_experience=data['years_of_experience'],
                average_similarity=float(np.mean([c['similarity'] for c in data['chunks']])),
                max_similarity=float(max([c['similarity'] for c in data['chunks']])),
                matched_chunks=data['chunks'],
                chunk_count=len(data['chunks']),
                section_coverage=dict(section_coverage)
            )
            
            candidate_matches.append(candidate)
        
        # Step 4: Sort by score and return top_k
        candidate_matches.sort(key=lambda x: x.average_similarity, reverse=True)
        
        top_candidates = candidate_matches[:top_k_candidates]
        
        logger.info(f"Returning top {len(top_candidates)} candidates")
        
        return top_candidates
    
    def _group_chunks_by_candidate(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Group chunks by candidate filename.
        
        Args:
            chunks: List of chunk results from vector search
        
        Returns:
            Dictionary mapping filename to candidate data
        """
        candidates = defaultdict(lambda: {
            'chunks': [],
            'role_category': '',
            'email': '',
            'years_of_experience': 0.0
        })
        
        for chunk in chunks:
            filename = chunk['metadata'].get('filename', 'unknown')
            
            # Add chunk to candidate
            candidates[filename]['chunks'].append(chunk)
            
            # Store metadata (same for all chunks from this candidate)
            if not candidates[filename]['role_category']:
                candidates[filename]['role_category'] = chunk['metadata'].get('role_category', '')
                candidates[filename]['email'] = chunk['metadata'].get('email', '')
                candidates[filename]['years_of_experience'] = chunk['metadata'].get('years_of_experience', 0.0)
        
        return dict(candidates)
    
    def _weighted_similarity(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate weighted similarity based on section importance.
        
        Section weights:
        - experience: 0.4 (most important)
        - skills: 0.3
        - projects: 0.15
        - education: 0.1
        - summary: 0.05
        - other: 0.0
        """
        section_weights = {
            'experience': 0.4,
            'skills': 0.3,
            'projects': 0.15,
            'education': 0.1,
            'summary': 0.05,
            'certifications': 0.1,
            'header': 0.0,
            'other': 0.0
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for chunk in chunks:
            section = chunk['metadata'].get('section_type', 'other')
            weight = section_weights.get(section, 0.0)
            similarity = chunk['similarity']
            
            weighted_sum += similarity * weight
            total_weight += weight
        
        if total_weight == 0:
            return np.mean([c['similarity'] for c in chunks])
        
        return weighted_sum / total_weight
    
    def find_best_candidates_for_role(self,
                                     job_description: str,
                                     role_category: str,
                                     min_years_experience: float = 0.0,
                                     top_k: int = 5) -> List[CandidateMatch]:
        """
        Find best candidates for a specific role.
        
        Args:
            job_description: Job description or requirements
            role_category: Target role (e.g., 'data_scientist')
            min_years_experience: Minimum years of experience required
            top_k: Number of top candidates to return
        
        Returns:
            List of top candidates
        """
        logger.info(f"Finding candidates for role: {role_category}")
        logger.info(f"Minimum experience: {min_years_experience} years")
        
        # Build filters
        filters = {
            'role_category': role_category
        }
        
        if min_years_experience > 0:
            filters['years_of_experience'] = {'$gte': min_years_experience}
        
        # Search candidates
        candidates = self.search_candidates(
            query=job_description,
            top_k_chunks=100,  # Get more chunks for better coverage
            top_k_candidates=top_k,
            filters=filters,
            min_similarity=0.3,
            aggregation_method='weighted'
        )
        
        return candidates
    
    def compare_candidates(self,
                          candidate1: CandidateMatch,
                          candidate2: CandidateMatch,
                          criteria: List[str] = None) -> Dict[str, Any]:
        """
        Compare two candidates side-by-side.
        
        Args:
            candidate1: First candidate
            candidate2: Second candidate
            criteria: List of criteria to compare (default: all)
        
        Returns:
            Comparison dictionary
        """
        if criteria is None:
            criteria = [
                'average_similarity',
                'years_of_experience',
                'chunk_count',
                'section_coverage'
            ]
        
        comparison = {
            'candidate1': {
                'filename': candidate1.filename,
                'email': candidate1.email
            },
            'candidate2': {
                'filename': candidate2.filename,
                'email': candidate2.email
            },
            'comparison': {}
        }
        
        for criterion in criteria:
            val1 = getattr(candidate1, criterion, None)
            val2 = getattr(candidate2, criterion, None)
            
            if val1 is not None and val2 is not None:
                if isinstance(val1, (int, float)):
                    winner = 'candidate1' if val1 > val2 else 'candidate2' if val2 > val1 else 'tie'
                    comparison['comparison'][criterion] = {
                        'candidate1': val1,
                        'candidate2': val2,
                        'winner': winner
                    }
                else:
                    comparison['comparison'][criterion] = {
                        'candidate1': val1,
                        'candidate2': val2
                    }
        
        return comparison
    
    def get_candidate_details(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get all chunks and details for a specific candidate.
        
        Args:
            filename: Candidate's resume filename
        
        Returns:
            Dictionary with all candidate information
        """
        logger.info(f"Getting details for candidate: {filename}")
        
        # Get all chunks for this candidate
        chunks = self.vectordb.get_documents_by_filter(
            filters={'filename': filename},
            limit=1000
        )
        
        if not chunks:
            logger.warning(f"No chunks found for {filename}")
            return None
        
        # Group by section
        sections = defaultdict(list)
        metadata = chunks[0]['metadata']  # Same metadata for all chunks
        
        for chunk in chunks:
            section = chunk['metadata'].get('section_type', 'other')
            sections[section].append(chunk['document'])
        
        return {
            'filename': filename,
            'role_category': metadata.get('role_category', ''),
            'email': metadata.get('email', ''),
            'years_of_experience': metadata.get('years_of_experience', 0.0),
            'total_chunks': len(chunks),
            'sections': {
                section: '\n\n'.join(content_list)
                for section, content_list in sections.items()
            }
        }
    
    def search_by_skills(self,
                        required_skills: List[str],
                        top_k: int = 10,
                        match_threshold: float = 0.5) -> List[CandidateMatch]:
        """
        Search candidates by required skills.
        
        Args:
            required_skills: List of required skills (e.g., ['Python', 'TensorFlow', 'SQL'])
            top_k: Number of candidates to return
            match_threshold: Minimum similarity threshold
        
        Returns:
            List of matching candidates
        """
        logger.info(f"Searching for candidates with skills: {required_skills}")
        
        # Create query from skills
        skills_query = f"Skills and experience in: {', '.join(required_skills)}"
        
        # Search for candidates
        candidates = self.search_candidates(
            query=skills_query,
            top_k_chunks=100,
            top_k_candidates=top_k,
            min_similarity=match_threshold,
            aggregation_method='weighted'
        )
        
        # Filter candidates who have skills section
        candidates_with_skills = [
            c for c in candidates
            if 'skills' in c.section_coverage or 'experience' in c.section_coverage
        ]
        
        logger.info(f"Found {len(candidates_with_skills)} candidates with relevant skills")
        
        return candidates_with_skills
    
    def search_by_experience_level(self,
                                  job_description: str,
                                  experience_range: Tuple[float, float],
                                  top_k: int = 10) -> List[CandidateMatch]:
        """
        Search candidates within a specific experience range.
        
        Args:
            job_description: Job description
            experience_range: (min_years, max_years)
            top_k: Number of candidates to return
        
        Returns:
            List of matching candidates
        """
        min_years, max_years = experience_range
        
        logger.info(f"Searching for candidates with {min_years}-{max_years} years experience")
        
        # Get candidates with minimum experience
        candidates = self.search_candidates(
            query=job_description,
            top_k_chunks=100,
            top_k_candidates=top_k * 2,  # Get more, then filter
            filters={'years_of_experience': {'$gte': min_years}},
            min_similarity=0.3
        )
        
        # Filter by maximum experience
        filtered_candidates = [
            c for c in candidates
            if c.years_of_experience <= max_years
        ]
        
        logger.info(f"Found {len(filtered_candidates)} candidates in experience range")
        
        return filtered_candidates[:top_k]
    
    def multi_criteria_search(self,
                             job_description: str,
                             criteria: Dict[str, Any],
                             top_k: int = 10) -> List[CandidateMatch]:
        """
        Search with multiple criteria.
        
        Args:
            job_description: Job description
            criteria: Dictionary of criteria, e.g.:
                {
                    'role_category': 'data_scientist',
                    'min_experience': 3,
                    'max_experience': 8,
                    'required_skills': ['Python', 'Machine Learning'],
                    'min_similarity': 0.4
                }
            top_k: Number of candidates to return
        
        Returns:
            List of matching candidates
        """
        logger.info(f"Multi-criteria search with {len(criteria)} criteria")
        
        # Build filters
        filters = {}
        
        if 'role_category' in criteria:
            filters['role_category'] = criteria['role_category']
        
        if 'min_experience' in criteria:
            filters['years_of_experience'] = {'$gte': criteria['min_experience']}
        
        # Build enhanced query
        query = job_description
        if 'required_skills' in criteria:
            skills_text = ', '.join(criteria['required_skills'])
            query = f"{job_description}\n\nRequired skills: {skills_text}"
        
        # Get minimum similarity
        min_similarity = criteria.get('min_similarity', 0.3)
        
        # Search
        candidates = self.search_candidates(
            query=query,
            top_k_chunks=100,
            top_k_candidates=top_k * 2,  # Get more for filtering
            filters=filters,
            min_similarity=min_similarity,
            aggregation_method='weighted'
        )
        
        # Filter by max experience if specified
        if 'max_experience' in criteria:
            max_exp = criteria['max_experience']
            candidates = [c for c in candidates if c.years_of_experience <= max_exp]
        
        logger.info(f"Found {len(candidates)} candidates matching all criteria")
        
        return candidates[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.vectordb.get_statistics()


# Utility functions
def create_retriever(vectordb, embedder) -> ResumeRetriever:
    """
    Convenience function to create a retriever.
    
    Args:
        vectordb: VectorDatabase instance
        embedder: OllamaEmbeddings instance
    
    Returns:
        ResumeRetriever instance
    """
    return ResumeRetriever(vectordb=vectordb, embedder=embedder)