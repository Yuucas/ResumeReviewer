"""
Resume Analyzer Module
Uses LLM to analyze candidates and make hiring recommendations.
Provides intelligent candidate evaluation and ranking.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import json
from dataclasses import dataclass

from src.agents.prompt_templates import PromptBuilder, get_system_prompt

logger = logging.getLogger(__name__)


@dataclass
class CandidateAnalysis:
    """
    Represents a detailed analysis of a candidate.
    """
    filename: str
    role_category: str
    email: str
    years_of_experience: float
    
    # LLM-generated analysis
    strengths: List[str]
    weaknesses: List[str]
    match_score: float  # 0-100
    overall_assessment: str
    key_qualifications: List[str]
    recommendation: str  # 'strongly_recommend', 'recommend', 'maybe', 'not_recommended'
    
    # Supporting data
    matched_chunks: List[Dict[str, Any]]
    similarity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filename': self.filename,
            'role_category': self.role_category,
            'email': self.email,
            'years_of_experience': self.years_of_experience,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'match_score': self.match_score,
            'overall_assessment': self.overall_assessment,
            'key_qualifications': self.key_qualifications,
            'recommendation': self.recommendation,
            'similarity_score': self.similarity_score
        }


class ResumeAnalyzer:
    """
    Analyzes candidates using LLM intelligence.
    Provides detailed evaluations and hiring recommendations.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the analyzer.
        
        Args:
            llm_client: OllamaClient instance
        """
        self.llm_client = llm_client
        
        logger.info("ResumeAnalyzer initialized")
    
    def analyze_candidate(self,
                         candidate,
                         job_description: str,
                         include_chunks: bool = True) -> CandidateAnalysis:
        """
        Perform detailed analysis of a single candidate.
        
        Args:
            candidate: CandidateMatch object
            job_description: Job description/requirements
            include_chunks: Whether to include chunk data in analysis
        
        Returns:
            CandidateAnalysis object
        """
        logger.info(f"Analyzing candidate: {candidate.filename}")
        
        # Prepare candidate information
        candidate_info = self._prepare_candidate_info(candidate)
        
        # Generate analysis using LLM
        analysis = self._llm_analyze(job_description, candidate_info)
        
        # Create CandidateAnalysis object
        result = CandidateAnalysis(
            filename=candidate.filename,
            role_category=candidate.role_category,
            email=candidate.email,
            years_of_experience=candidate.years_of_experience,
            strengths=analysis.get('strengths', []),
            weaknesses=analysis.get('weaknesses', []),
            match_score=analysis.get('match_score', 0.0),
            overall_assessment=analysis.get('overall_assessment', ''),
            key_qualifications=analysis.get('key_qualifications', []),
            recommendation=analysis.get('recommendation', 'maybe'),
            matched_chunks=candidate.matched_chunks if include_chunks else [],
            similarity_score=candidate.average_similarity
        )
        
        logger.info(f"Analysis complete: {candidate.filename} - "
                   f"Score: {result.match_score}/100")
        
        return result
    
    def _prepare_candidate_info(self, candidate) -> str:
        """
        Prepare candidate information for LLM analysis.
        
        Args:
            candidate: CandidateMatch object
        
        Returns:
            Formatted candidate information string
        """
        # Get top 5 most relevant chunks
        sorted_chunks = sorted(
            candidate.matched_chunks,
            key=lambda x: x['similarity'],
            reverse=True
        )[:5]
        
        # Group chunks by section
        sections = {}
        for chunk in sorted_chunks:
            section = chunk['metadata']['section_type']
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk['document'])
        
        # Format information
        info_parts = [
            f"Candidate: {candidate.filename}",
            f"Role Applied For: {candidate.role_category.replace('_', ' ').title()}",
            f"Total Experience: {candidate.years_of_experience} years",
            f"Email: {candidate.email}",
            f"Semantic Match Score: {candidate.average_similarity:.2f}",
            "",
            "Resume Content:"
        ]
        
        for section, content_list in sections.items():
            info_parts.append(f"\n[{section.upper()}]")
            info_parts.append('\n'.join(content_list))
        
        return '\n'.join(info_parts)
    
    def _llm_analyze(self, job_description: str, candidate_info: str) -> Dict[str, Any]:
        """
        Use LLM to analyze candidate fit.
        Uses centralized prompt templates.
        """
        # Build prompt using template
        prompt = PromptBuilder.build_analysis_prompt(
            job_description=job_description,
            candidate_info=candidate_info
        )
        
        # Get system prompt
        system_prompt = get_system_prompt('recruiter')
        
        try:
            # Generate analysis with system prompt
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500,
                system_prompt=system_prompt
            )
            
            # Extract JSON from response
            analysis = self.llm_client.extract_json(response)
            
            if analysis is None:
                logger.warning("Failed to parse LLM analysis, using defaults")
                return self._default_analysis()
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return self._default_analysis()
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when LLM fails."""
        return {
            "match_score": 50.0,
            "strengths": ["Unable to analyze - LLM error"],
            "weaknesses": ["Analysis incomplete"],
            "key_qualifications": [],
            "overall_assessment": "Analysis could not be completed due to technical error.",
            "recommendation": "maybe"
        }
    
    def analyze_candidates_batch(self,
                                 candidates: List,
                                 job_description: str,
                                 max_candidates: Optional[int] = None) -> List[CandidateAnalysis]:
        """
        Analyze multiple candidates in batch.
        
        Args:
            candidates: List of CandidateMatch objects
            job_description: Job description
            max_candidates: Maximum number of candidates to analyze (None = all)
        
        Returns:
            List of CandidateAnalysis objects
        """
        if max_candidates and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        
        logger.info(f"Analyzing {len(candidates)} candidates")
        
        analyses = []
        
        for i, candidate in enumerate(candidates, 1):
            logger.info(f"Analyzing candidate {i}/{len(candidates)}")
            
            try:
                analysis = self.analyze_candidate(
                    candidate=candidate,
                    job_description=job_description,
                    include_chunks=False  # Don't include chunks in batch to save memory
                )
                analyses.append(analysis)
                
            except Exception as e:
                logger.error(f"Failed to analyze {candidate.filename}: {str(e)}")
                continue
        
        logger.info(f"Batch analysis complete: {len(analyses)} successful")
        
        return analyses
    
    def rank_candidates(self,
                       candidates: List,
                       job_description: str,
                       top_k: int = 5) -> List[CandidateAnalysis]:
        """
        Analyze and rank candidates by LLM match score.
        
        Args:
            candidates: List of CandidateMatch objects
            job_description: Job description
            top_k: Number of top candidates to return
        
        Returns:
            Ranked list of CandidateAnalysis objects
        """
        logger.info(f"Ranking top {top_k} candidates from {len(candidates)}")
        
        # Analyze all candidates
        analyses = self.analyze_candidates_batch(candidates, job_description)
        
        # Sort by match score
        analyses.sort(key=lambda x: x.match_score, reverse=True)
        
        # Return top k
        top_analyses = analyses[:top_k]
        
        logger.info(f"Top {len(top_analyses)} candidates ranked")
        
        return top_analyses
    
    def find_best_2_candidates(self,
                              candidates: List,
                              job_description: str) -> Tuple[CandidateAnalysis, CandidateAnalysis]:
        """
        Find the best 2 candidates for the role.
        
        Args:
            candidates: List of CandidateMatch objects
            job_description: Job description
        
        Returns:
            Tuple of (best_candidate, second_best_candidate)
        """
        logger.info("Finding best 2 candidates")
        
        # Get top 2 from ranking
        top_2 = self.rank_candidates(
            candidates=candidates,
            job_description=job_description,
            top_k=2
        )
        
        if len(top_2) < 2:
            raise ValueError(f"Not enough candidates (found {len(top_2)}, need 2)")
        
        return top_2[0], top_2[1]
    
    def compare_candidates(self,
                        candidate1: CandidateAnalysis,
                        candidate2: CandidateAnalysis,
                        job_description: str) -> str:
        """Generate detailed comparison using prompt template."""
        
        # Prepare candidate data
        cand1_data = {
            'name': candidate1.filename,
            'experience': candidate1.years_of_experience,
            'score': candidate1.match_score,
            'strengths': candidate1.strengths,
            'weaknesses': candidate1.weaknesses,
            'assessment': candidate1.overall_assessment
        }
        
        cand2_data = {
            'name': candidate2.filename,
            'experience': candidate2.years_of_experience,
            'score': candidate2.match_score,
            'strengths': candidate2.strengths,
            'weaknesses': candidate2.weaknesses,
            'assessment': candidate2.overall_assessment
        }
        
        # Build prompt
        prompt = PromptBuilder.build_comparison_prompt(
            job_description=job_description,
            candidate1=cand1_data,
            candidate2=cand2_data
        )
        
        # Get system prompt
        system_prompt = get_system_prompt('comparator')
        
        try:
            comparison = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000,
                system_prompt=system_prompt
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return "Comparison could not be completed due to technical error."
    
    def generate_hiring_report(self,
                              top_candidates: List[CandidateAnalysis],
                              job_description: str) -> str:
        """
        Generate comprehensive hiring report.
        
        Args:
            top_candidates: List of analyzed candidates
            job_description: Job description
        
        Returns:
            Formatted hiring report
        """
        logger.info(f"Generating hiring report for {len(top_candidates)} candidates")
        
        report_parts = [
            "="*70,
            "HIRING RECOMMENDATION REPORT",
            "="*70,
            "",
            "JOB DESCRIPTION:",
            job_description,
            "",
            "="*70,
            f"TOP {len(top_candidates)} CANDIDATES",
            "="*70,
            ""
        ]
        
        for i, candidate in enumerate(top_candidates, 1):
            report_parts.extend([
                f"\n{'#'*70}",
                f"RANK #{i}: {candidate.filename}",
                f"{'#'*70}",
                "",
                f"Contact: {candidate.email}",
                f"Experience: {candidate.years_of_experience} years",
                f"Match Score: {candidate.match_score}/100",
                f"Recommendation: {candidate.recommendation.replace('_', ' ').upper()}",
                "",
                "STRENGTHS:",
                *[f"  ✓ {strength}" for strength in candidate.strengths],
                "",
                "CONCERNS:",
                *[f"  • {weakness}" for weakness in candidate.weaknesses],
                "",
                "KEY QUALIFICATIONS:",
                *[f"  → {qual}" for qual in candidate.key_qualifications],
                "",
                "OVERALL ASSESSMENT:",
                candidate.overall_assessment,
                ""
            ])
        
        # Add final recommendation if we have exactly 2 candidates
        if len(top_candidates) == 2:
            comparison = self.compare_candidates(
                top_candidates[0],
                top_candidates[1],
                job_description
            )
            
            report_parts.extend([
                "="*70,
                "FINAL RECOMMENDATION",
                "="*70,
                "",
                comparison,
                ""
            ])
        
        report_parts.extend([
            "="*70,
            "END OF REPORT",
            "="*70
        ])
        
        return '\n'.join(report_parts)


# Utility function
def create_analyzer(llm_client) -> ResumeAnalyzer:
    """
    Convenience function to create an analyzer.
    
    Args:
        llm_client: OllamaClient instance
    
    Returns:
        ResumeAnalyzer instance
    """
    return ResumeAnalyzer(llm_client=llm_client)