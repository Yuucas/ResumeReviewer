#!/usr/bin/env python3
"""
Show GitHub information gathered for a specific resume.

Usage:
    python show_github_info.py "Yukselcan_Sevil_Resume.pdf"
"""

import sys
import requests
import json
from typing import Optional

def show_github_info(filename: str, job_description: Optional[str] = None):
    """Show GitHub information for a specific resume."""

    print('=' * 80)
    print(f'GitHub Information for: {filename}')
    print('=' * 80)
    print()

    # Step 1: Check ChromaDB metadata
    print('[STEP 1] Checking ChromaDB metadata...')
    print('-' * 80)

    try:
        response = requests.get('http://localhost:8000/api/debug/github-metadata')
        response.raise_for_status()
        debug_data = response.json()

        if debug_data.get('github_found'):
            github_url = debug_data.get('github_url')
            print(f'[OK] GitHub URL found in database: {github_url}')
            print(f'  Total chunks: {debug_data.get("total_chunks")}')
            print(f'  Chunks with GitHub: {sum(1 for m in debug_data.get("sample_metadata", []) if m.get("github_in_metadata"))}')
        else:
            print('[ERROR] No GitHub URL found in database!')
            print('  The resume may not have been parsed correctly.')
            return
    except Exception as e:
        print(f'[ERROR] Error checking database: {e}')
        return

    print()

    # Step 2: Perform a search to trigger GitHub analysis
    print('[STEP 2] Performing search to analyze GitHub profile...')
    print('-' * 80)

    if not job_description:
        job_description = "data scientist with Python and machine learning experience"

    print(f'Job description: "{job_description}"')
    print('This may take 1-2 minutes as we fetch and analyze GitHub repositories...')
    print()

    try:
        search_payload = {
            "job_description": job_description,
            "top_k": 5
        }

        response = requests.post(
            'http://localhost:8000/api/search',
            json=search_payload,
            timeout=300  # 5 minutes timeout
        )
        response.raise_for_status()
        search_results = response.json()

        if not search_results.get('success'):
            print(f'[ERROR] Search failed: {search_results.get("error", "Unknown error")}')
            return

        # Find the candidate
        candidates = search_results.get('results', [])
        candidate = None
        for c in candidates:
            if c.get('filename') == filename:
                candidate = c
                break

        if not candidate:
            print(f'[ERROR] Resume "{filename}" not found in search results.')
            print(f'  Total candidates found: {len(candidates)}')
            if candidates:
                print('  Available candidates:')
                for c in candidates:
                    print(f'    - {c.get("filename")}')
            return

        print(f'[OK] Found candidate: {filename}')
        print()

    except requests.Timeout:
        print('[ERROR] Search timed out (took longer than 5 minutes)')
        return
    except Exception as e:
        print(f'[ERROR] Search error: {e}')
        return

    # Step 3: Display GitHub information
    print('[STEP 3] GitHub Analysis Results')
    print('=' * 80)
    print()

    github_data = candidate.get('github')

    if not github_data:
        print('[ERROR] No GitHub analysis data in results!')
        print()
        print('Possible reasons:')
        print('  1. GitHub URL is invalid or profile does not exist')
        print('  2. GitHub API rate limit reached')
        print('  3. Error during GitHub analysis')
        print()
        print('Candidate data keys:', list(candidate.keys()))
        return

    # Display profile information
    print(f'GitHub Username: @{github_data.get("username")}')
    print(f'Profile URL: {github_data.get("profile_url")}')
    print(f'Relevance Score: {github_data.get("relevance_score", 0):.1f}%')
    print()

    # Display top languages
    top_languages = github_data.get('top_languages', [])
    if top_languages:
        print(f'Top Programming Languages ({len(top_languages)}):')
        for i, lang in enumerate(top_languages, 1):
            print(f'  {i}. {lang}')
        print()

    # Display relevant projects
    relevant_projects = github_data.get('relevant_projects', [])
    if relevant_projects:
        print(f'Relevant Projects ({len(relevant_projects)}):')
        print()
        for i, project in enumerate(relevant_projects, 1):
            print(f'  [{i}] {project.get("project_name")}')
            print(f'      URL: {project.get("url")}')
            print(f'      Relevance: {project.get("relevance_score", 0):.1f}%')
            print(f'      Language: {project.get("language", "N/A")}')
            print(f'      Stars: {project.get("stars", 0)}')

            # Display description if available
            description = project.get("description")
            if description:
                # Truncate long descriptions
                if len(description) > 100:
                    description = description[:97] + "..."
                print(f'      Description: {description}')
            print()
    else:
        print('No relevant projects found.')
        print()

    # Display summary
    summary = github_data.get('summary', '')
    if summary:
        print('Summary:')
        print('-' * 80)
        print(summary)
        print()

    # Display overall candidate match
    print('=' * 80)
    print(f'Overall Match Score: {candidate.get("match_score", 0):.1f}%')
    print(f'Recommendation: {candidate.get("recommendation", "N/A")}')
    print('=' * 80)
    print()

    # Export to JSON option
    print('[EXPORT] Full JSON data saved to: github_analysis_result.json')
    with open('github_analysis_result.json', 'w', encoding='utf-8') as f:
        json.dump(candidate, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point."""

    # Check if filename provided
    if len(sys.argv) < 2:
        print('Usage: python show_github_info.py "Resume_Filename.pdf" ["Optional job description"]')
        print()
        print('Examples:')
        print('  python show_github_info.py "Yukselcan_Sevil_Resume.pdf"')
        print('  python show_github_info.py "John_Doe.pdf" "software engineer with Python"')
        sys.exit(1)

    filename = sys.argv[1]
    job_description = sys.argv[2] if len(sys.argv) > 2 else None

    # Run the analysis
    show_github_info(filename, job_description)


if __name__ == '__main__':
    main()
