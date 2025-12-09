"""
GitHub Repository Analyzer Module
Analyzes candidate's GitHub repositories to assess project relevance to job descriptions.
"""

import logging
import re
from typing import List, Dict, Optional, Any
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class GitHubProject:
    """Represents a GitHub repository with relevant metadata."""
    name: str
    description: Optional[str]
    url: str
    language: Optional[str]
    topics: List[str]
    stars: int
    forks: int
    updated_at: str
    is_fork: bool


@dataclass
class GitHubAnalysis:
    """Analysis results for a candidate's GitHub profile."""
    username: str
    profile_url: str
    total_repos: int
    public_repos: int
    projects: List[GitHubProject]
    top_languages: List[str]
    relevance_score: float
    relevant_projects: List[Dict[str, Any]]
    summary: str


class GitHubAnalyzer:
    """Analyzes GitHub repositories to assess candidate's technical skills."""

    def __init__(self, llm_client=None, github_token: Optional[str] = None):
        self.llm_client = llm_client

        # Try to get token from parameter, then environment variable
        if github_token is None:
            import os
            github_token = os.getenv('GITHUB_TOKEN')

        self.github_token = github_token
        self.api_base = "https://api.github.com"
        self.headers = {"Accept": "application/vnd.github.v3+json"}

        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
            logger.info("GitHubAnalyzer initialized with authentication token")
        else:
            logger.warning("GitHubAnalyzer initialized without token - rate limit: 60 requests/hour")
            logger.warning("Set GITHUB_TOKEN environment variable for 5,000 requests/hour")

    def extract_github_username(self, github_url: str) -> Optional[str]:
        """Extract GitHub username from URL."""
        if not github_url:
            return None
        url = github_url.lower().strip()
        url = re.sub(r'^https?://', '', url)
        url = re.sub(r'^www\.', '', url)
        match = re.match(r'github\.com/([^/]+)', url)
        if match:
            username = match.group(1)
            if username not in ['login', 'join', 'explore', 'topics', 'trending', 'collections']:
                return username
        return None

    def fetch_user_repos(self, username: str, max_repos: int = 30) -> List[Dict[str, Any]]:
        """Fetch user's public repositories from GitHub API."""
        try:
            url = f"{self.api_base}/users/{username}/repos"
            params = {"sort": "updated", "per_page": min(max_repos, 100), "type": "owner"}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            repos = response.json()
            logger.info(f"Fetched {len(repos)} repositories for user: {username}")
            return repos
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"GitHub user not found: {username}")
            elif e.response.status_code == 403:
                logger.warning(f"GitHub API rate limit exceeded")
            else:
                logger.error(f"GitHub API error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error fetching repos for {username}: {str(e)}")
            return []

    def fetch_user_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetch user's profile information."""
        try:
            url = f"{self.api_base}/users/{username}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching profile for {username}: {str(e)}")
            return None

    def parse_repository(self, repo_data: Dict[str, Any]) -> GitHubProject:
        """Parse GitHub API repository data into GitHubProject."""
        return GitHubProject(
            name=repo_data.get('name', ''),
            description=repo_data.get('description'),
            url=repo_data.get('html_url', ''),
            language=repo_data.get('language'),
            topics=repo_data.get('topics', []),
            stars=repo_data.get('stargazers_count', 0),
            forks=repo_data.get('forks_count', 0),
            updated_at=repo_data.get('updated_at', ''),
            is_fork=repo_data.get('fork', False)
        )

    def fetch_readme(self, username: str, repo_name: str) -> Optional[str]:
        """
        Fetch README.md content for a repository.

        Returns the README content as plain text, or None if not found.
        Limits to first 3000 characters to avoid excessive API usage.
        """
        try:
            # Try common README filenames
            readme_variants = ['README.md', 'readme.md', 'README', 'Readme.md']

            for readme_file in readme_variants:
                try:
                    url = f"{self.api_base}/repos/{username}/{repo_name}/contents/{readme_file}"
                    response = requests.get(url, headers=self.headers, timeout=5)

                    if response.status_code == 200:
                        data = response.json()

                        # GitHub API returns content in base64
                        import base64
                        content = base64.b64decode(data.get('content', '')).decode('utf-8', errors='ignore')

                        # Limit to first 3000 chars to avoid too much data
                        # and focus on the important parts (usually at the top)
                        content = content[:3000]

                        # Remove markdown formatting for cleaner text analysis
                        content = self._clean_markdown(content)

                        logger.debug(f"Fetched README for {username}/{repo_name} ({len(content)} chars)")
                        return content

                except requests.RequestException:
                    continue

            # No README found
            logger.debug(f"No README found for {username}/{repo_name}")
            return None

        except Exception as e:
            logger.warning(f"Error fetching README for {username}/{repo_name}: {e}")
            return None

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting to get clean text."""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

        # Remove headers markdown but keep text
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

        # Remove bold/italic
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()

        return text

    def analyze_repository_relevance(self, project: GitHubProject, job_description: str, readme_content: Optional[str] = None) -> Dict[str, Any]:
        """Analyze how relevant a GitHub project is to the job description."""
        job_keywords = self._extract_technical_keywords(job_description.lower())

        # Build project text from multiple sources
        project_text = f"{project.name} {project.description or ''} {' '.join(project.topics)}"

        # Include README content if available (significantly improves matching)
        if readme_content:
            project_text += f" {readme_content}"

        project_text_lower = project_text.lower()

        matched_keywords = [kw for kw in job_keywords if kw in project_text_lower]
        keyword_score = (len(matched_keywords) / max(len(job_keywords), 1)) * 100

        language_match = False
        if project.language:
            language_match = project.language.lower() in job_description.lower()

        topic_matches = []
        for topic in project.topics:
            if topic.lower() in job_description.lower():
                topic_matches.append(topic)

        relevance_score = keyword_score
        if language_match:
            relevance_score += 20
        if topic_matches:
            relevance_score += min(len(topic_matches) * 10, 30)

        # Bonus for having a comprehensive README (shows documentation quality)
        has_readme = False
        if readme_content and len(readme_content) > 200:
            relevance_score += 10
            has_readme = True
            logger.debug(f"README bonus applied for {project.name} (length: {len(readme_content)})")

        relevance_score = min(relevance_score, 100)

        return {
            'project_name': project.name,
            'url': project.url,
            'language': project.language,
            'description': project.description,
            'topics': project.topics,
            'stars': project.stars,
            'relevance_score': round(relevance_score, 2),
            'matched_keywords': matched_keywords,
            'language_match': language_match,
            'topic_matches': topic_matches,
            'is_significant': project.stars >= 5 or not project.is_fork,
            'has_readme': has_readme
        }

    def analyze_github_profile(self, github_url: str, job_description: str, max_repos: int = 10) -> Optional[GitHubAnalysis]:
        """
        Comprehensive analysis of a candidate's GitHub profile.

        OPTIMIZED: Reduced default max_repos from 30 to 10, and only fetches READMEs for top 5 repos.
        """
        username = self.extract_github_username(github_url)
        if not username:
            logger.warning(f"Could not extract username from: {github_url}")
            return None

        profile = self.fetch_user_profile(username)
        if not profile:
            return None

        repos_data = self.fetch_user_repos(username, max_repos)
        if not repos_data:
            logger.warning(f"No repositories found for: {username}")
            return None

        projects = [self.parse_repository(repo) for repo in repos_data]
        original_projects = [p for p in projects if not p.is_fork]

        # OPTIMIZATION: Quick filter first without README (fast)
        logger.debug(f"Quick filtering {len(original_projects)} repositories...")
        quick_analyses = []
        for project in original_projects[:max_repos]:
            # Quick analysis without README
            analysis = self.analyze_repository_relevance(project, job_description, readme_content=None)
            quick_analyses.append((project, analysis))

        # Sort by relevance and get top candidates
        quick_analyses.sort(key=lambda x: x[1]['relevance_score'], reverse=True)
        top_candidates = quick_analyses[:min(5, len(quick_analyses))]  # Only analyze top 5 with README

        logger.debug(f"Fetching READMEs for top {len(top_candidates)} repositories...")

        # OPTIMIZATION: Parallel README fetching (3x faster!)
        import concurrent.futures

        def fetch_and_analyze(project_tuple):
            """Fetch README and analyze in parallel."""
            project, quick_analysis = project_tuple
            try:
                readme_content = self.fetch_readme(username, project.name)
                analysis = self.analyze_repository_relevance(project, job_description, readme_content)
                return analysis if analysis['relevance_score'] > 20 else None
            except Exception as e:
                logger.warning(f"Error analyzing {project.name}: {e}")
                return None

        # Fetch READMEs in parallel (up to 3 at a time to avoid rate limiting)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(fetch_and_analyze, top_candidates)

        relevant_projects = [r for r in results if r is not None]
        relevant_projects.sort(key=lambda x: x['relevance_score'], reverse=True)

        if relevant_projects:
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]
            weighted_scores = []
            for i, proj in enumerate(relevant_projects[:5]):
                weight = weights[i] if i < len(weights) else 0.2
                weighted_scores.append(proj['relevance_score'] * weight)
            overall_relevance = sum(weighted_scores) / sum(weights[:len(weighted_scores)])
        else:
            overall_relevance = 0.0

        languages = [p.language for p in projects if p.language]
        from collections import Counter
        language_counts = Counter(languages)
        top_languages = [lang for lang, _ in language_counts.most_common(5)]

        summary = self._generate_summary(username, len(original_projects), relevant_projects[:5], top_languages, overall_relevance)

        return GitHubAnalysis(
            username=username,
            profile_url=f"https://github.com/{username}",
            total_repos=profile.get('public_repos', 0),
            public_repos=len(original_projects),
            projects=original_projects[:10],
            top_languages=top_languages,
            relevance_score=round(overall_relevance, 2),
            relevant_projects=relevant_projects[:5],
            summary=summary
        )

    def _extract_technical_keywords(self, text: str) -> List[str]:
        """Extract technical keywords from job description."""
        keywords = []
        languages = ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring', 'express', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'node.js', 'nestjs', '.net', 'rails', 'laravel', 'next.js', 'svelte', 'flutter']
        technologies = ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'linux', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'graphql', 'rest', 'api', 'microservices', 'machine learning', 'deep learning', 'nlp', 'computer vision', 'data science', 'blockchain', 'cloud', 'devops', 'agile', 'scrum']
        all_keywords = languages + frameworks + technologies
        for keyword in all_keywords:
            if keyword in text:
                keywords.append(keyword)
        return keywords

    def _generate_summary(self, username: str, total_repos: int, relevant_projects: List[Dict], top_languages: List[str], overall_relevance: float) -> str:
        """Generate human-readable summary of GitHub analysis."""
        if not relevant_projects:
            return f"GitHub profile (@{username}) shows {total_repos} public repositories primarily using {', '.join(top_languages[:3]) if top_languages else 'various languages'}. However, no projects were found with strong relevance to the job description."

        if overall_relevance >= 70:
            relevance_level = "highly relevant"
        elif overall_relevance >= 50:
            relevance_level = "moderately relevant"
        else:
            relevance_level = "somewhat relevant"

        summary_parts = [f"GitHub profile (@{username}) has {total_repos} public repositories with {relevance_level} projects for this role."]

        if top_languages:
            summary_parts.append(f"Primary languages: {', '.join(top_languages[:3])}.")

        top_projects = relevant_projects[:3]
        if top_projects:
            project_names = [f"{p['project_name']} ({p['relevance_score']:.0f}% match)" for p in top_projects]
            summary_parts.append(f"Notable relevant projects: {', '.join(project_names)}.")

        total_stars = sum(p['stars'] for p in relevant_projects[:3])
        if total_stars > 0:
            summary_parts.append(f"These projects have earned {total_stars} stars, indicating community recognition.")

        return " ".join(summary_parts)


def analyze_candidate_github(github_url: str, job_description: str, llm_client=None) -> Optional[GitHubAnalysis]:
    """Convenience function to analyze a candidate's GitHub profile."""
    analyzer = GitHubAnalyzer(llm_client=llm_client)
    return analyzer.analyze_github_profile(github_url, job_description)
