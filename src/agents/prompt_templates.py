"""
Prompt Templates Module
Centralized prompt templates for LLM interactions.
Provides reusable, well-crafted prompts for resume analysis tasks.
"""

from typing import Dict, List, Optional
from string import Template


class PromptTemplates:
    """
    Collection of prompt templates for resume analysis.
    Uses Python's Template class for safe string substitution.
    """
    
    # System prompts (set the LLM's role and behavior)
    SYSTEM_RECRUITER = """You are an expert technical recruiter with 15+ years of experience in hiring for technology roles. You specialize in evaluating candidates for data science, engineering, and IT positions. You are thorough, fair, and provide honest assessments based on evidence from candidates' resumes."""
    
    SYSTEM_ANALYST = """You are a senior talent analyst who excels at identifying candidate strengths and potential concerns. You provide balanced, evidence-based evaluations and clear recommendations."""
    
    SYSTEM_COMPARATOR = """You are an expert at comparing candidates side-by-side. You identify key differentiators and make clear recommendations based on job requirements."""
    
    # Main analysis prompt
    CANDIDATE_ANALYSIS = Template("""Analyze how well this candidate matches the job requirements.

JOB DESCRIPTION:
$job_description

CANDIDATE PROFILE:
$candidate_info

Provide a detailed analysis in the following JSON format:
{
    "match_score": <number 0-100>,
    "strengths": [<list of 3-5 key strengths with specific evidence>],
    "weaknesses": [<list of 2-4 potential concerns or gaps>],
    "key_qualifications": [<list of 3-5 most relevant qualifications>],
    "overall_assessment": "<2-3 sentence summary>",
    "recommendation": "<one of: strongly_recommend, recommend, maybe, not_recommended>"
}

SCORING GUIDE:
- 0-30: Poor match - Missing critical requirements, significant skill gaps
- 31-50: Partial match - Has some relevant skills but major gaps exist
- 51-70: Good match - Meets most requirements, minor gaps acceptable
- 71-85: Strong match - Exceeds requirements, well-qualified
- 86-100: Exceptional match - Ideal candidate, rare combination of skills

EVALUATION CRITERIA:
1. Technical Skills Match: Does the candidate have the required technical skills?
2. Experience Level: Is their experience appropriate for the role?
3. Relevant Projects: Have they worked on similar projects or problems?
4. Education Alignment: Does their education match requirements?
5. Career Trajectory: Is this role a logical next step for them?

Be honest and critical. Use specific evidence from the resume. If information is missing, note it as a concern.

Respond with ONLY valid JSON, nothing else.""")
    
    # Quick scoring prompt (faster, less detailed)
    QUICK_SCORE = Template("""Rate this candidate for the job from 0-100.

Job Requirements:
$job_description

Candidate Summary:
$candidate_summary

Provide ONLY a number between 0 and 100, nothing else.

Scoring:
- 0-30: Poor match
- 31-50: Partial match  
- 51-70: Good match
- 71-85: Strong match
- 86-100: Exceptional match""")
    
    # Candidate comparison prompt
    COMPARE_CANDIDATES = Template("""Compare these two candidates and determine who is better suited for the role.

JOB DESCRIPTION:
$job_description

CANDIDATE 1: $candidate1_name
- Experience: $candidate1_experience years
- Match Score: $candidate1_score/100
- Strengths: $candidate1_strengths
- Weaknesses: $candidate1_weaknesses
- Assessment: $candidate1_assessment

CANDIDATE 2: $candidate2_name
- Experience: $candidate2_experience years
- Match Score: $candidate2_score/100
- Strengths: $candidate2_strengths
- Weaknesses: $candidate2_weaknesses
- Assessment: $candidate2_assessment

Provide a detailed comparison covering:

1. OVERALL WINNER: State clearly who is the stronger candidate and why

2. KEY DIFFERENTIATORS: What are the main differences that matter for this role?

3. SPECIFIC ADVANTAGES: 
   - Why Candidate 1 might be better
   - Why Candidate 2 might be better

4. HIRING RECOMMENDATION: Who would you hire and why? Include any concerns.

5. TEAM FIT: Which candidate would integrate better with the team?

Write 4-5 paragraphs. Be specific and use evidence from their profiles. Make a clear recommendation.""")
    
    # Strengths extraction prompt
    EXTRACT_STRENGTHS = Template("""Based on this resume content, identify the candidate's TOP 5 STRENGTHS for the role.

Role Requirements:
$job_description

Resume Content:
$resume_content

For each strength, provide:
1. The strength (concise)
2. Evidence from the resume

Format as a JSON array:
[
    {"strength": "...", "evidence": "..."},
    {"strength": "...", "evidence": "..."},
    ...
]

Respond with ONLY valid JSON.""")
    
    # Weaknesses/concerns extraction prompt
    EXTRACT_WEAKNESSES = Template("""Based on this resume and job requirements, identify 3-4 CONCERNS or POTENTIAL WEAKNESSES.

Job Requirements:
$job_description

Resume Content:
$resume_content

Consider:
- Missing skills or qualifications
- Experience gaps
- Lack of relevant projects
- Career trajectory concerns
- Overqualification risks
- Any red flags

Format as a JSON array of concerns:
[
    {"concern": "...", "explanation": "..."},
    {"concern": "...", "explanation": "..."},
    ...
]

Be constructive but honest. If there are no major concerns, mention minor points.

Respond with ONLY valid JSON.""")
    
    # Skills extraction prompt
    EXTRACT_SKILLS = Template("""Extract all technical skills mentioned in this resume.

Resume Content:
$resume_content

Categorize skills into:
- Programming Languages
- Frameworks/Libraries
- Tools/Technologies
- Databases
- Cloud Platforms
- Methodologies
- Other

Format as JSON:
{
    "programming_languages": [...],
    "frameworks": [...],
    "tools": [...],
    "databases": [...],
    "cloud_platforms": [...],
    "methodologies": [...],
    "other": [...]
}

Respond with ONLY valid JSON.""")
    
    # Experience summary prompt
    SUMMARIZE_EXPERIENCE = Template("""Summarize this candidate's professional experience concisely.

Experience Section:
$experience_content

Provide:
1. Total years of experience
2. Key roles held
3. Main achievements (top 3)
4. Career progression pattern

Write 3-4 sentences. Focus on progression and impact.

Respond in plain text, no JSON.""")
    
    # Red flags detection prompt
    DETECT_RED_FLAGS = Template("""Analyze this resume for potential red flags or concerns.

Resume Content:
$resume_content

Check for:
- Employment gaps (>6 months)
- Job hopping (many short-term roles)
- Declining career trajectory
- Lack of career progression
- Inconsistencies or vague descriptions
- Overqualification concerns
- Missing critical information

Format as JSON:
{
    "red_flags_found": true/false,
    "flags": [
        {"flag": "...", "severity": "low/medium/high", "explanation": "..."},
        ...
    ],
    "overall_risk_level": "low/medium/high"
}

Be fair but vigilant. If no red flags, set red_flags_found to false and flags to empty array.

Respond with ONLY valid JSON.""")
    
    # Culture fit assessment prompt
    ASSESS_CULTURE_FIT = Template("""Assess this candidate's potential culture fit based on their resume.

Company Culture:
$company_culture

Candidate Resume:
$resume_content

Evaluate:
- Communication style (from how resume is written)
- Values alignment (inferred from career choices)
- Team collaboration (from experience descriptions)
- Learning orientation (from skill development)
- Leadership potential (from achievements)

Provide assessment in JSON:
{
    "culture_fit_score": <0-100>,
    "positive_indicators": [...],
    "concerns": [...],
    "assessment": "<2-3 sentence summary>"
}

Respond with ONLY valid JSON.""")
    
    # Salary expectation estimation
    ESTIMATE_SALARY = Template("""Based on this candidate's profile, estimate a reasonable salary range.

Role: $role
Location: $location
Candidate Experience: $years_experience years
Skills: $skills
Industry: $industry

Consider:
- Years of experience
- Skill level and breadth
- Previous roles and companies
- Market rates for this role
- Location

Provide estimation in JSON:
{
    "minimum": <number>,
    "maximum": <number>,
    "currency": "USD",
    "confidence": "low/medium/high",
    "justification": "<explanation>"
}

Respond with ONLY valid JSON.""")
    
    # Interview questions generator
    GENERATE_INTERVIEW_QUESTIONS = Template("""Generate 5-7 targeted interview questions for this candidate.

Job Description:
$job_description

Candidate Profile:
$candidate_info

Generate questions that:
1. Probe their claimed expertise
2. Explore experience gaps
3. Assess problem-solving ability
4. Evaluate cultural fit
5. Test technical knowledge depth

Format as JSON:
{
    "technical_questions": [...],
    "behavioral_questions": [...],
    "situation_questions": [...]
}

Each question should be specific to THIS candidate, not generic.

Respond with ONLY valid JSON.""")
    
    # Batch ranking prompt (for multiple candidates)
    BATCH_RANKING = Template("""Rank these candidates for the role from best to worst.

Job Description:
$job_description

Candidates:
$candidates_summary

Provide ranking in JSON:
{
    "ranked_candidates": [
        {
            "rank": 1,
            "filename": "...",
            "score": <0-100>,
            "rationale": "<why this rank>"
        },
        ...
    ],
    "top_recommendation": "<name of #1 candidate>",
    "reasoning": "<why they're the best choice>"
}

Respond with ONLY valid JSON.""")
    
    # Hiring decision prompt (final recommendation)
    HIRING_DECISION = Template("""Make a final hiring recommendation for these top candidates.

Job Description:
$job_description

Hiring Context:
- Budget: $budget
- Start Date: $start_date
- Team Size: $team_size
- Urgency: $urgency

Top Candidates:
$candidates_summary

Provide recommendation in JSON:
{
    "recommended_candidate": "<name>",
    "hire_confidence": "low/medium/high",
    "decision_rationale": "<detailed explanation>",
    "backup_candidate": "<name>",
    "concerns": [...],
    "next_steps": [...]
}

Consider:
- Skills match
- Cultural fit
- Budget alignment
- Team needs
- Risk factors

Respond with ONLY valid JSON.""")


class PromptBuilder:
    """
    Helper class to build prompts with validation.
    """
    
    @staticmethod
    def build_analysis_prompt(job_description: str, candidate_info: str) -> str:
        """
        Build candidate analysis prompt.
        
        Args:
            job_description: Job description/requirements
            candidate_info: Formatted candidate information
        
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.CANDIDATE_ANALYSIS.substitute(
            job_description=job_description.strip(),
            candidate_info=candidate_info.strip()
        )
    
    @staticmethod
    def build_comparison_prompt(
        job_description: str,
        candidate1: Dict,
        candidate2: Dict
    ) -> str:
        """
        Build candidate comparison prompt.
        
        Args:
            job_description: Job description
            candidate1: Dict with keys: name, experience, score, strengths, weaknesses, assessment
            candidate2: Dict with keys: name, experience, score, strengths, weaknesses, assessment
        
        Returns:
            Formatted prompt string
        """
        return PromptTemplates.COMPARE_CANDIDATES.substitute(
            job_description=job_description.strip(),
            candidate1_name=candidate1['name'],
            candidate1_experience=candidate1['experience'],
            candidate1_score=candidate1['score'],
            candidate1_strengths=', '.join(candidate1['strengths']),
            candidate1_weaknesses=', '.join(candidate1['weaknesses']),
            candidate1_assessment=candidate1['assessment'],
            candidate2_name=candidate2['name'],
            candidate2_experience=candidate2['experience'],
            candidate2_score=candidate2['score'],
            candidate2_strengths=', '.join(candidate2['strengths']),
            candidate2_weaknesses=', '.join(candidate2['weaknesses']),
            candidate2_assessment=candidate2['assessment']
        )
    
    @staticmethod
    def build_quick_score_prompt(job_description: str, candidate_summary: str) -> str:
        """Build quick scoring prompt."""
        return PromptTemplates.QUICK_SCORE.substitute(
            job_description=job_description.strip(),
            candidate_summary=candidate_summary.strip()
        )
    
    @staticmethod
    def build_strengths_prompt(job_description: str, resume_content: str) -> str:
        """Build strengths extraction prompt."""
        return PromptTemplates.EXTRACT_STRENGTHS.substitute(
            job_description=job_description.strip(),
            resume_content=resume_content.strip()
        )
    
    @staticmethod
    def build_weaknesses_prompt(job_description: str, resume_content: str) -> str:
        """Build weaknesses extraction prompt."""
        return PromptTemplates.EXTRACT_WEAKNESSES.substitute(
            job_description=job_description.strip(),
            resume_content=resume_content.strip()
        )
    
    @staticmethod
    def build_red_flags_prompt(resume_content: str) -> str:
        """Build red flags detection prompt."""
        return PromptTemplates.DETECT_RED_FLAGS.substitute(
            resume_content=resume_content.strip()
        )
    
    @staticmethod
    def build_interview_questions_prompt(
        job_description: str,
        candidate_info: str
    ) -> str:
        """Build interview questions generator prompt."""
        return PromptTemplates.GENERATE_INTERVIEW_QUESTIONS.substitute(
            job_description=job_description.strip(),
            candidate_info=candidate_info.strip()
        )
    
    @staticmethod
    def build_batch_ranking_prompt(
        job_description: str,
        candidates: List[Dict]
    ) -> str:
        """
        Build batch ranking prompt.
        
        Args:
            job_description: Job description
            candidates: List of candidate dicts with keys: name, experience, summary
        
        Returns:
            Formatted prompt
        """
        candidates_text = []
        for i, candidate in enumerate(candidates, 1):
            candidates_text.append(
                f"{i}. {candidate['name']}\n"
                f"   Experience: {candidate['experience']} years\n"
                f"   Summary: {candidate['summary']}"
            )
        
        return PromptTemplates.BATCH_RANKING.substitute(
            job_description=job_description.strip(),
            candidates_summary='\n\n'.join(candidates_text)
        )


# Pre-defined prompt variations for different use cases
class PromptVariations:
    """
    Different prompt variations for specific scenarios.
    """
    
    # For junior roles (adjust expectations)
    JUNIOR_ANALYSIS = Template("""Analyze this candidate for a JUNIOR role.

Job Description (Junior Level):
$job_description

Candidate Profile:
$candidate_info

For junior candidates, focus on:
- Learning potential over extensive experience
- Fundamental skills mastery
- Educational background
- Personal projects and initiative
- Growth mindset indicators

Provide analysis in JSON format:
{
    "match_score": <0-100>,
    "strengths": [...],
    "weaknesses": [...],
    "learning_potential": "low/medium/high",
    "key_qualifications": [...],
    "overall_assessment": "...",
    "recommendation": "..."
}

Be more forgiving of limited experience. Look for potential.

Respond with ONLY valid JSON.""")
    
    # For senior/leadership roles
    SENIOR_ANALYSIS = Template("""Analyze this candidate for a SENIOR/LEADERSHIP role.

Job Description (Senior Level):
$job_description

Candidate Profile:
$candidate_info

For senior candidates, focus on:
- Leadership experience and team management
- Strategic thinking and business impact
- Technical depth AND breadth
- Mentorship and influence
- Cross-functional collaboration
- Track record of delivering complex projects

Provide analysis in JSON format:
{
    "match_score": <0-100>,
    "strengths": [...],
    "weaknesses": [...],
    "leadership_score": <0-100>,
    "key_qualifications": [...],
    "overall_assessment": "...",
    "recommendation": "..."
}

Set high expectations. Leadership ability is critical.

Respond with ONLY valid JSON.""")
    
    # For remote roles
    REMOTE_ANALYSIS = Template("""Analyze this candidate for a REMOTE role.

Job Description (Remote Position):
$job_description

Candidate Profile:
$candidate_info

For remote candidates, additionally assess:
- Previous remote work experience
- Self-motivation and discipline
- Written communication skills (from resume quality)
- Independent problem-solving
- Time zone compatibility
- Virtual collaboration ability

Include in your JSON:
{
    "match_score": <0-100>,
    "strengths": [...],
    "weaknesses": [...],
    "remote_readiness": "low/medium/high",
    "remote_work_indicators": [...],
    "key_qualifications": [...],
    "overall_assessment": "...",
    "recommendation": "..."
}

Remote work ability is a key criterion.

Respond with ONLY valid JSON.""")


# Example usage and helper functions
def get_system_prompt(role: str = "recruiter") -> str:
    """
    Get appropriate system prompt.
    
    Args:
        role: 'recruiter', 'analyst', or 'comparator'
    
    Returns:
        System prompt string
    """
    prompts = {
        'recruiter': PromptTemplates.SYSTEM_RECRUITER,
        'analyst': PromptTemplates.SYSTEM_ANALYST,
        'comparator': PromptTemplates.SYSTEM_COMPARATOR
    }
    
    return prompts.get(role, PromptTemplates.SYSTEM_RECRUITER)


def build_custom_prompt(template: str, **kwargs) -> str:
    """
    Build custom prompt from template string.
    
    Args:
        template: Template string with $variables
        **kwargs: Variables to substitute
    
    Returns:
        Formatted prompt
    """
    return Template(template).substitute(**kwargs)