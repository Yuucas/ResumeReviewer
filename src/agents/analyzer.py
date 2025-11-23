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
from src.agents.github_analyzer import GitHubAnalyzer

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

    # GitHub analysis (optional)
    github_username: Optional[str] = None
    github_profile_url: Optional[str] = None
    github_relevance_score: Optional[float] = None
    github_top_languages: Optional[List[str]] = None
    github_relevant_projects: Optional[List[Dict[str, Any]]] = None
    github_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
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

        # Add GitHub data if available
        if self.github_username:
            result['github'] = {
                'username': self.github_username,
                'profile_url': self.github_profile_url,
                'relevance_score': self.github_relevance_score,
                'top_languages': self.github_top_languages,
                'relevant_projects': self.github_relevant_projects,
                'summary': self.github_summary
            }

        return result


class ResumeAnalyzer:
    """
    Analyzes candidates using LLM intelligence.
    Provides detailed evaluations and hiring recommendations.
    """

    def __init__(self, llm_client, enable_github_analysis: bool = True):
        """
        Initialize the analyzer.

        Args:
            llm_client: OllamaClient instance
            enable_github_analysis: Whether to analyze GitHub profiles (default: True)
        """
        self.llm_client = llm_client
        self.enable_github_analysis = enable_github_analysis

        # Initialize GitHub analyzer if enabled
        if self.enable_github_analysis:
            self.github_analyzer = GitHubAnalyzer(llm_client=llm_client)
        else:
            self.github_analyzer = None

        logger.info(f"ResumeAnalyzer initialized (GitHub analysis: {enable_github_analysis})")
    
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

        # Analyze GitHub profile if available and enabled
        github_data = self._analyze_github_profile(candidate, job_description)

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
            similarity_score=candidate.average_similarity,
            # GitHub data
            github_username=github_data.get('username'),
            github_profile_url=github_data.get('profile_url'),
            github_relevance_score=github_data.get('relevance_score'),
            github_top_languages=github_data.get('top_languages'),
            github_relevant_projects=github_data.get('relevant_projects'),
            github_summary=github_data.get('summary')
        )

        logger.info(f"Analysis complete: {candidate.filename} - "
                   f"Score: {result.match_score}/100"
                   f"{' (with GitHub data)' if github_data.get('username') else ''}")

        return result

    def _analyze_github_profile(self, candidate, job_description: str) -> Dict[str, Any]:
        """
        Analyze candidate's GitHub profile if available.

        Args:
            candidate: CandidateMatch object
            job_description: Job description

        Returns:
            Dictionary with GitHub analysis data or empty dict if not available
        """
        if not self.enable_github_analysis or not self.github_analyzer:
            return {}

        # Try to get GitHub URL from candidate metadata
        github_url = None

        # Check if candidate has github field in metadata
        if hasattr(candidate, 'github') and candidate.github:
            github_url = candidate.github
        # Check in matched_chunks metadata
        elif hasattr(candidate, 'matched_chunks') and candidate.matched_chunks:
            for chunk in candidate.matched_chunks:
                if 'metadata' in chunk and 'extracted_info' in chunk['metadata']:
                    extracted_info = chunk['metadata']['extracted_info']

                    # Handle both dict and JSON string
                    if isinstance(extracted_info, str):
                        import json
                        try:
                            extracted_info = json.loads(extracted_info)
                        except:
                            continue

                    if isinstance(extracted_info, dict):
                        github = extracted_info.get('github')
                        if github:
                            github_url = f"https://{github}" if not github.startswith('http') else github
                            break

        if not github_url:
            logger.debug(f"No GitHub URL found for candidate: {candidate.filename}")
            return {}

        # Analyze GitHub profile
        try:
            logger.info(f"Analyzing GitHub profile: {github_url}")
            github_analysis = self.github_analyzer.analyze_github_profile(
                github_url=github_url,
                job_description=job_description,
                max_repos=30
            )

            if github_analysis:
                return {
                    'username': github_analysis.username,
                    'profile_url': github_analysis.profile_url,
                    'relevance_score': github_analysis.relevance_score,
                    'top_languages': github_analysis.top_languages,
                    'relevant_projects': github_analysis.relevant_projects,
                    'summary': github_analysis.summary
                }
            else:
                logger.warning(f"GitHub analysis failed for: {github_url}")
                return {}

        except Exception as e:
            logger.error(f"Error analyzing GitHub profile {github_url}: {str(e)}")
            return {}
    
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
            # OPTIMIZATION: Set to 1500 tokens for complete JSON responses
            # (was 800, increased to 1200, now 1500 to prevent truncation)
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500,
                system_prompt=system_prompt,
                json_mode=True
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
                                 max_candidates: Optional[int] = None,
                                 parallel: bool = True,
                                 max_workers: int = 3) -> List[CandidateAnalysis]:
        """
        Analyze multiple candidates in batch.

        Args:
            candidates: List of CandidateMatch objects
            job_description: Job description
            max_candidates: Maximum number of candidates to analyze (None = all)
            parallel: Use parallel processing (default: True for speed)
            max_workers: Number of parallel workers (default: 3)

        Returns:
            List of CandidateAnalysis objects
        """
        if max_candidates and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        logger.info(f"Analyzing {len(candidates)} candidates {'in parallel' if parallel else 'sequentially'}")

        if not parallel or len(candidates) == 1:
            # Sequential processing (original method)
            analyses = []
            for i, candidate in enumerate(candidates, 1):
                logger.info(f"Analyzing candidate {i}/{len(candidates)}")
                try:
                    analysis = self.analyze_candidate(
                        candidate=candidate,
                        job_description=job_description,
                        include_chunks=False
                    )
                    analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Failed to analyze {candidate.filename}: {str(e)}")
                    continue
            return analyses

        # OPTIMIZATION: Parallel processing (3x faster!)
        import concurrent.futures

        def analyze_one(candidate_tuple):
            """Analyze one candidate."""
            i, candidate = candidate_tuple
            try:
                logger.info(f"Analyzing candidate {i+1}/{len(candidates)}: {candidate.filename}")
                return self.analyze_candidate(
                    candidate=candidate,
                    job_description=job_description,
                    include_chunks=False
                )
            except Exception as e:
                logger.error(f"Failed to analyze {candidate.filename}: {str(e)}")
                return None

        # Process candidates in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            indexed_candidates = list(enumerate(candidates))
            results = executor.map(analyze_one, indexed_candidates)

        # Filter out None results
        analyses = [r for r in results if r is not None]

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