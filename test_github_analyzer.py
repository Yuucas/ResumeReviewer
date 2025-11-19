#!/usr/bin/env python3
"""
Test script for GitHub Analyzer
Tests the GitHub profile analysis functionality independently.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.github_analyzer import GitHubAnalyzer

def test_github_analyzer():
    """Test GitHub analyzer with a real profile."""

    print("=" * 80)
    print("GitHub Analyzer Test")
    print("=" * 80)
    print()

    # Initialize analyzer
    analyzer = GitHubAnalyzer()
    print("✓ GitHubAnalyzer initialized")
    print()

    # Test with a well-known GitHub profile
    test_cases = [
        {
            "url": "github.com/torvalds",
            "job_description": "Linux kernel developer with C programming experience, systems programming, and low-level optimization"
        },
        {
            "url": "https://github.com/gvanrossum",
            "job_description": "Python developer with experience in language design, interpreter development, and open source contributions"
        },
        {
            "url": "github.com/nonexistentuser12345xyz",
            "job_description": "Test for non-existent user"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test['url']}")
        print("-" * 80)

        result = analyzer.analyze_github_profile(
            github_url=test['url'],
            job_description=test['job_description'],
            max_repos=10
        )

        if result:
            print(f"✓ Analysis successful!")
            print(f"  Username: @{result.username}")
            print(f"  Profile URL: {result.profile_url}")
            print(f"  Total Repositories: {result.total_repos}")
            print(f"  Public Repositories (non-fork): {result.public_repos}")
            print(f"  Top Languages: {', '.join(result.top_languages[:3])}")
            print(f"  Relevance Score: {result.relevance_score:.1f}%")
            print()
            print(f"  Summary:")
            print(f"  {result.summary}")
            print()

            if result.relevant_projects:
                print(f"  Top {min(3, len(result.relevant_projects))} Relevant Projects:")
                for j, proj in enumerate(result.relevant_projects[:3], 1):
                    print(f"    {j}. {proj['project_name']} ({proj['relevance_score']:.0f}% match)")
                    print(f"       URL: {proj['url']}")
                    if proj['description']:
                        print(f"       Description: {proj['description'][:80]}...")
                    print(f"       Language: {proj['language']}, Stars: {proj['stars']}")
            else:
                print("  No relevant projects found")
        else:
            print(f"✗ Analysis failed (user not found or no repositories)")

        print()
        print()

    print("=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_github_analyzer()
