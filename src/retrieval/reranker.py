"""
Reranker Module
Advanced reranking strategies to improve candidate ranking quality.
Uses cross-encoder models and LLM-based reranking for better precision.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """
    Represents a reranked result with updated score.
    """
    original_rank: int
    new_rank: int
    item: Dict[str, Any]
    original_score: float
    rerank_score: float
    score_change: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_rank': self.original_rank,
            'new_rank': self.new_rank,
            'item': self.item,
            'original_score': self.original_score,
            'rerank_score': self.rerank_score,
            'score_change': self.score_change
        }


class ResumeReranker:
    """
    Reranks candidate matches using advanced scoring methods.
    Provides multiple reranking strategies.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the reranker.
        
        Args:
            llm_client: Optional OllamaClient for LLM-based reranking
        """
        self.llm_client = llm_client
        
        logger.info("ResumeReranker initialized")
    
    def rerank_candidates(self,
                         query: str,
                         candidates: List[Any],
                         method: str = 'hybrid',
                         top_k: Optional[int] = None) -> List[Any]:
        """
        Rerank candidates using specified method.
        
        Args:
            query: Job description or search query
            candidates: List of CandidateMatch objects
            method: Reranking method:
                - 'keyword': Keyword matching boost
                - 'experience': Experience-weighted reranking
                - 'section': Section coverage reranking
                - 'hybrid': Combines multiple signals
                - 'llm': LLM-based reranking (requires llm_client)
            top_k: Return only top K results (None = return all)
        
        Returns:
            Reranked list of candidates
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates using '{method}' method")
        
        # Choose reranking method
        if method == 'keyword':
            reranked = self._keyword_rerank(query, candidates)
        elif method == 'experience':
            reranked = self._experience_rerank(query, candidates)
        elif method == 'section':
            reranked = self._section_coverage_rerank(candidates)
        elif method == 'hybrid':
            reranked = self._hybrid_rerank(query, candidates)
        elif method == 'llm' and self.llm_client:
            reranked = self._llm_rerank(query, candidates)
        else:
            logger.warning(f"Unknown method '{method}', returning original order")
            reranked = candidates
        
        # Limit to top_k if specified
        if top_k and top_k < len(reranked):
            reranked = reranked[:top_k]
        
        logger.info(f"Reranking complete, returning {len(reranked)} candidates")
        
        return reranked
    
    def _keyword_rerank(self, query: str, candidates: List[Any]) -> List[Any]:
        """
        Rerank based on keyword matching.
        Boosts candidates whose content contains important query keywords.
        """
        logger.info("Applying keyword-based reranking")
        
        # Extract important keywords from query
        keywords = self._extract_keywords(query)
        
        logger.info(f"Extracted {len(keywords)} keywords: {keywords[:10]}")
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            # Get all text from matched chunks
            all_text = ' '.join([
                chunk['document'].lower()
                for chunk in candidate.matched_chunks
            ])
            
            # Count keyword matches
            keyword_score = sum(
                1 for keyword in keywords
                if keyword.lower() in all_text
            )
            
            # Normalize by number of keywords
            keyword_score = keyword_score / len(keywords) if keywords else 0
            
            # Combine with original similarity (70% original, 30% keyword)
            combined_score = (candidate.average_similarity * 0.7) + (keyword_score * 0.3)
            
            scored_candidates.append((combined_score, candidate))
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [candidate for _, candidate in scored_candidates]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        Focuses on technical terms, skills, and key requirements.
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'can', 'may', 'might', 'must', 'we', 'you',
            'they', 'them', 'their', 'this', 'that', 'these', 'those', 'looking'
        }
        
        # Split into words and clean
        words = re.findall(r'\b[a-zA-Z][a-zA-Z+#\.]*\b', text.lower())
        
        # Filter stop words and short words
        keywords = [
            word for word in words
            if word not in stop_words and len(word) > 2
        ]
        
        # Keep unique keywords
        unique_keywords = list(dict.fromkeys(keywords))
        
        # Prioritize technical terms (contains digits, +, #, .)
        tech_keywords = [
            word for word in unique_keywords
            if any(char in word for char in ['+', '#', '.']) or any(char.isdigit() for char in word)
        ]
        
        # Return tech keywords first, then others
        return tech_keywords + [k for k in unique_keywords if k not in tech_keywords]
    
    def _experience_rerank(self, query: str, candidates: List[Any]) -> List[Any]:
        """
        Rerank based on experience level matching.
        Extracts required experience from query and boosts matching candidates.
        """
        logger.info("Applying experience-based reranking")
        
        # Try to extract required experience from query
        required_years = self._extract_required_experience(query)
        
        if required_years is None:
            logger.info("No experience requirement found in query, using original ranking")
            return candidates
        
        logger.info(f"Required experience: {required_years} years")
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            candidate_years = candidate.years_of_experience
            
            # Calculate experience match score
            if candidate_years < required_years:
                # Penalize insufficient experience
                exp_score = candidate_years / required_years
            elif candidate_years <= required_years * 1.5:
                # Perfect match or slightly over
                exp_score = 1.0
            else:
                # Over-qualified, slight penalty
                exp_score = 0.9
            
            # Combine with original similarity (60% original, 40% experience)
            combined_score = (candidate.average_similarity * 0.6) + (exp_score * 0.4)
            
            scored_candidates.append((combined_score, candidate))
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [candidate for _, candidate in scored_candidates]
    
    def _extract_required_experience(self, query: str) -> Optional[float]:
        """
        Extract required years of experience from job description.
        
        Examples:
        - "5+ years" → 5.0
        - "3-5 years" → 4.0 (average)
        - "minimum 7 years" → 7.0
        """
        query_lower = query.lower()
        
        # Pattern 1: "X+ years"
        match = re.search(r'(\d+)\+?\s*years', query_lower)
        if match:
            return float(match.group(1))
        
        # Pattern 2: "X-Y years" (take average)
        match = re.search(r'(\d+)\s*-\s*(\d+)\s*years', query_lower)
        if match:
            min_years = float(match.group(1))
            max_years = float(match.group(2))
            return (min_years + max_years) / 2
        
        # Pattern 3: "minimum X years" or "at least X years"
        match = re.search(r'(?:minimum|at least)\s+(\d+)\s*years', query_lower)
        if match:
            return float(match.group(1))
        
        return None
    
    def _section_coverage_rerank(self, candidates: List[Any]) -> List[Any]:
        """
        Rerank based on section coverage.
        Candidates with more complete profiles rank higher.
        """
        logger.info("Applying section coverage reranking")
        
        # Ideal sections for a complete profile
        ideal_sections = {'experience', 'education', 'skills', 'projects', 'summary'}
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            candidate_sections = set(candidate.section_coverage.keys())
            
            # Calculate coverage score
            coverage = len(candidate_sections & ideal_sections) / len(ideal_sections)
            
            # Bonus for having experience and skills
            critical_sections = {'experience', 'skills'}
            has_critical = len(candidate_sections & critical_sections) == len(critical_sections)
            critical_bonus = 0.1 if has_critical else 0.0
            
            # Combine with original similarity (70% original, 30% coverage)
            combined_score = (candidate.average_similarity * 0.7) + (coverage * 0.3) + critical_bonus
            
            scored_candidates.append((combined_score, candidate))
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [candidate for _, candidate in scored_candidates]
    
    def _hybrid_rerank(self, query: str, candidates: List[Any]) -> List[Any]:
        """
        Hybrid reranking combining multiple signals.
        
        Weights:
        - Original similarity: 40%
        - Keyword matching: 25%
        - Experience matching: 20%
        - Section coverage: 15%
        """
        logger.info("Applying hybrid reranking")
        
        # Extract keywords and required experience
        keywords = self._extract_keywords(query)
        required_years = self._extract_required_experience(query)
        
        ideal_sections = {'experience', 'education', 'skills', 'projects', 'summary'}
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            # 1. Original similarity score (40%)
            similarity_score = candidate.average_similarity * 0.4
            
            # 2. Keyword matching score (25%)
            all_text = ' '.join([
                chunk['document'].lower()
                for chunk in candidate.matched_chunks
            ])
            keyword_matches = sum(1 for kw in keywords if kw.lower() in all_text)
            keyword_score = (keyword_matches / len(keywords) if keywords else 0.5) * 0.25
            
            # 3. Experience matching score (20%)
            if required_years:
                candidate_years = candidate.years_of_experience
                if candidate_years < required_years:
                    exp_score = (candidate_years / required_years) * 0.2
                elif candidate_years <= required_years * 1.5:
                    exp_score = 0.2
                else:
                    exp_score = 0.18
            else:
                exp_score = 0.1  # Neutral score if no requirement
            
            # 4. Section coverage score (15%)
            candidate_sections = set(candidate.section_coverage.keys())
            coverage = len(candidate_sections & ideal_sections) / len(ideal_sections)
            
            # Bonus for critical sections
            critical_sections = {'experience', 'skills'}
            has_critical = len(candidate_sections & critical_sections) == len(critical_sections)
            
            section_score = (coverage * 0.15) + (0.03 if has_critical else 0)
            
            # Combine all scores
            total_score = similarity_score + keyword_score + exp_score + section_score
            
            scored_candidates.append((total_score, candidate))
        
        # Sort by total score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [candidate for _, candidate in scored_candidates]
    
    def _llm_rerank(self, query: str, candidates: List[Any]) -> List[Any]:
        """
        LLM-based reranking using Ollama.
        Uses LLM to assess candidate-job fit more intelligently.
        
        Note: This is slower but potentially more accurate.
        """
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to hybrid reranking")
            return self._hybrid_rerank(query, candidates)
        
        logger.info("Applying LLM-based reranking (this may take a while)")
        
        scored_candidates = []
        
        for i, candidate in enumerate(candidates, 1):
            logger.info(f"LLM reranking candidate {i}/{len(candidates)}: {candidate.filename}")
            
            # Prepare candidate summary
            candidate_summary = self._prepare_candidate_summary(candidate)
            
            # Ask LLM to score candidate fit (0-100)
            score = self._llm_score_candidate(query, candidate_summary)
            
            # Normalize to 0-1
            normalized_score = score / 100.0
            
            scored_candidates.append((normalized_score, candidate))
        
        # Sort by LLM score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [candidate for _, candidate in scored_candidates]
    
    def _prepare_candidate_summary(self, candidate: Any) -> str:
        """
        Prepare a concise summary of candidate for LLM evaluation.
        """
        # Get top 3 most relevant chunks
        sorted_chunks = sorted(
            candidate.matched_chunks,
            key=lambda x: x['similarity'],
            reverse=True
        )[:3]
        
        chunk_texts = '\n\n'.join([
            f"[{chunk['metadata']['section_type']}]\n{chunk['document']}"
            for chunk in sorted_chunks
        ])
        
        summary = f"""
Candidate: {candidate.filename}
Role: {candidate.role_category}
Experience: {candidate.years_of_experience} years
Email: {candidate.email}

Key Information:
{chunk_texts}

Sections Covered: {', '.join(candidate.section_coverage.keys())}
"""
        return summary.strip()
    
    def _llm_score_candidate(self, job_description: str, candidate_summary: str) -> float:
        """
        Use LLM to score how well candidate matches job description.
        
        Returns:
            Score from 0-100
        """
        prompt = f"""You are an expert technical recruiter. Rate how well this candidate matches the job requirements.

Job Description:
{job_description}

Candidate Profile:
{candidate_summary}

Provide a match score from 0-100 where:
- 0-30: Poor match, missing critical requirements
- 31-50: Partial match, has some relevant skills
- 51-70: Good match, meets most requirements
- 71-85: Strong match, exceeds requirements
- 86-100: Exceptional match, ideal candidate

Respond with ONLY a number between 0 and 100, nothing else.
"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            
            # Extract number from response
            score_match = re.search(r'\b(\d+)\b', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0), 100)  # Clamp to 0-100
            else:
                logger.warning("Could not extract score from LLM response")
                return 50.0  # Default neutral score
                
        except Exception as e:
            logger.error(f"LLM scoring failed: {str(e)}")
            return 50.0  # Default neutral score
    
    def rerank_with_comparison(self,
                              query: str,
                              candidates: List[Any],
                              method: str = 'hybrid',
                              top_k: Optional[int] = None) -> List[RerankResult]:
        """
        Rerank and return detailed comparison of before/after rankings.
        
        Args:
            query: Job description
            candidates: List of candidates
            method: Reranking method
            top_k: Return top K results
        
        Returns:
            List of RerankResult objects showing ranking changes
        """
        # Store original rankings and scores
        original_rankings = {
            id(candidate): (i, candidate.average_similarity)
            for i, candidate in enumerate(candidates)
        }
        
        # Rerank
        reranked_candidates = self.rerank_candidates(
            query=query,
            candidates=candidates,
            method=method,
            top_k=top_k
        )
        
        # Build comparison results
        results = []
        
        for new_rank, candidate in enumerate(reranked_candidates):
            original_rank, original_score = original_rankings[id(candidate)]
            
            # Calculate new score (approximation based on position)
            new_score = candidate.average_similarity  # This is still the base score
            
            result = RerankResult(
                original_rank=original_rank + 1,  # 1-indexed for display
                new_rank=new_rank + 1,
                item=candidate.to_dict(),
                original_score=original_score,
                rerank_score=new_score,
                score_change=new_rank - original_rank  # Negative = improved
            )
            
            results.append(result)
        
        return results
    
    def explain_reranking(self, rerank_results: List[RerankResult]) -> str:
        """
        Generate human-readable explanation of reranking changes.
        
        Args:
            rerank_results: Results from rerank_with_comparison()
        
        Returns:
            Explanation text
        """
        explanation = ["Reranking Results:", ""]
        
        # Show significant changes
        improved = [r for r in rerank_results if r.score_change < 0]
        declined = [r for r in rerank_results if r.score_change > 0]
        unchanged = [r for r in rerank_results if r.score_change == 0]
        
        if improved:
            explanation.append(f"Improved Rankings ({len(improved)} candidates):")
            for result in sorted(improved, key=lambda x: x.score_change)[:5]:
                filename = result.item['filename']
                explanation.append(
                    f"  • {filename}: "
                    f"#{result.original_rank} → #{result.new_rank} "
                    f"(+{abs(result.score_change)} positions)"
                )
            explanation.append("")
        
        if declined:
            explanation.append(f"Declined Rankings ({len(declined)} candidates):")
            for result in sorted(declined, key=lambda x: x.score_change, reverse=True)[:5]:
                filename = result.item['filename']
                explanation.append(
                    f"  • {filename}: "
                    f"#{result.original_rank} → #{result.new_rank} "
                    f"(-{abs(result.score_change)} positions)"
                )
            explanation.append("")
        
        if unchanged:
            explanation.append(f"Unchanged: {len(unchanged)} candidates maintained their ranking")
        
        return '\n'.join(explanation)


# Utility function
def create_reranker(llm_client=None) -> ResumeReranker:
    """
    Convenience function to create a reranker.
    
    Args:
        llm_client: Optional LLM client for advanced reranking
    
    Returns:
        ResumeReranker instance
    """
    return ResumeReranker(llm_client=llm_client)