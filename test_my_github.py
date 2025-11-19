#!/usr/bin/env python3
"""Test GitHub analyzer with Yuucas profile"""
import sys
sys.path.insert(0, 'src')

from src.agents.github_analyzer import GitHubAnalyzer

print('=' * 70)
print('Testing GitHub Analyzer with Your Profile (github.com/Yuucas)')
print('=' * 70)
print()

analyzer = GitHubAnalyzer()

# Your GitHub URL from resume
github_url = 'github.com/Yuucas'
job_description = '''
Python developer with machine learning, data science, and AI experience.
Looking for expertise in RAG systems, LangChain, ChromaDB, Docker, AWS, and MLOps.
Experience with PyTorch, Hugging Face, and generative AI tools.
'''

print(f'Analyzing: {github_url}')
print('Job Description: Python ML/AI developer with RAG, LangChain, ChromaDB...')
print()
print('Please wait (fetching from GitHub API)...')
print()

result = analyzer.analyze_github_profile(
    github_url=github_url,
    job_description=job_description,
    max_repos=20
)

if result:
    print('[SUCCESS] GitHub Analysis Completed!')
    print('=' * 70)
    print(f'Username: @{result.username}')
    print(f'Profile: {result.profile_url}')
    print(f'Total Repositories: {result.total_repos}')
    print(f'Public Repos (non-fork): {result.public_repos}')
    print(f'Relevance Score: {result.relevance_score:.1f}%')
    print(f'Top Languages: {", ".join(result.top_languages[:5])}')
    print()
    print('SUMMARY (This is what appears in analysis results):')
    print('-' * 70)
    print(result.summary)
    print()

    if result.relevant_projects:
        print(f'REPOSITORY NAMES (Top {min(5, len(result.relevant_projects))} Relevant Projects):')
        print('=' * 70)
        for i, proj in enumerate(result.relevant_projects[:5], 1):
            print(f'{i}. {proj["project_name"]} ({proj["relevance_score"]:.1f}% match)')
            print(f'   URL: {proj["url"]}')
            if proj.get('description'):
                desc = proj['description'][:80] if len(proj['description']) > 80 else proj['description']
                print(f'   Description: {desc}')
            print(f'   Language: {proj["language"]}, Stars: {proj["stars"]}')
            if proj.get('matched_keywords'):
                keywords = ', '.join(proj['matched_keywords'][:5])
                print(f'   Matched Keywords: {keywords}')
            print()

        print('=' * 70)
        print('[OK] This data will appear in your candidate analysis results')
        print('=' * 70)
    else:
        print('[INFO] No relevant projects found for the job description')
        print('       Try a different job description or check if repos are public')
else:
    print('[FAILED] GitHub analysis returned no results')
    print()
    print('Possible reasons:')
    print('  - Username not found (404)')
    print('  - No public repositories')
    print('  - GitHub API rate limit exceeded')
    print()
    print('Check GitHub API:')
    print(f'  curl https://api.github.com/users/Yuucas')
