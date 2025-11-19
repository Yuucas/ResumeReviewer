#!/usr/bin/env python3
"""Test GitHub analyzer with Yukselcan_Sevil_Resume.pdf"""
import sys
import os
import re

sys.path.insert(0, 'src')

print("Testing GitHub Feature with Your Resume")
print("=" * 70)

# Step 1: Check if resume file exists
resume_path = "dataset/data_scientist/Yukselcan_Sevil_Resume.pdf"
if os.path.exists(resume_path):
    print(f"[OK] Resume found: {resume_path}")
else:
    print(f"[FAIL] Resume not found: {resume_path}")
    sys.exit(1)

# Step 2: Check if we can import GitHub analyzer
print("\nStep 1: Importing GitHub Analyzer...")
try:
    from src.agents.github_analyzer import GitHubAnalyzer
    print("[OK] GitHubAnalyzer imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import GitHubAnalyzer: {e}")
    sys.exit(1)

# Step 3: Initialize analyzer
print("\nStep 2: Initializing GitHub Analyzer...")
try:
    analyzer = GitHubAnalyzer()
    print("[OK] GitHubAnalyzer initialized")
except Exception as e:
    print(f"[FAIL] Failed to initialize: {e}")
    sys.exit(1)

# Step 4: Check if ChromaDB has your resume
print("\nStep 3: Checking ChromaDB for your resume...")
try:
    import chromadb
    client = chromadb.PersistentClient(path='chroma_db')
    collection = client.get_collection('resumes')
    results = collection.get(
        where={'filename': 'Yukselcan_Sevil_Resume.pdf'},
        include=['metadatas', 'documents']
    )

    if results['ids']:
        print(f"[OK] Resume found in database ({len(results['ids'])} chunks)")

        # Check for GitHub URL in metadata
        github_found = False
        github_url = None

        for meta in results['metadatas']:
            if meta and 'extracted_info' in meta:
                extracted = meta['extracted_info']
                if isinstance(extracted, dict) and 'github' in extracted:
                    github_url = extracted['github']
                    github_found = True
                    break
                elif isinstance(extracted, str):
                    # Try to parse if it's a JSON string
                    import json
                    try:
                        extracted_dict = json.loads(extracted)
                        if 'github' in extracted_dict:
                            github_url = extracted_dict['github']
                            github_found = True
                            break
                    except:
                        pass

        if github_found:
            print(f"[OK] GitHub URL found in resume: {github_url}")

            # Step 5: Test GitHub analysis
            print(f"\nStep 4: Analyzing GitHub profile...")
            print(f"GitHub URL: {github_url}")

            # Ensure URL has proper format
            if not github_url.startswith('http'):
                github_url = f"https://{github_url}"

            job_description = "Python developer with machine learning, data science, and AI experience. Looking for expertise in RAG systems, LangChain, ChromaDB, and MLOps."

            result = analyzer.analyze_github_profile(
                github_url=github_url,
                job_description=job_description,
                max_repos=20
            )

            if result:
                print("\n" + "=" * 70)
                print("GITHUB ANALYSIS RESULTS")
                print("=" * 70)
                print(f"Username: @{result.username}")
                print(f"Profile URL: {result.profile_url}")
                print(f"Total Repositories: {result.total_repos}")
                print(f"Public Repos (non-fork): {result.public_repos}")
                print(f"Relevance Score: {result.relevance_score:.1f}%")
                print(f"Top Languages: {', '.join(result.top_languages[:5])}")
                print(f"\nSummary:")
                print(f"{result.summary}")

                if result.relevant_projects:
                    print(f"\nTop {min(5, len(result.relevant_projects))} Relevant Projects:")
                    for i, proj in enumerate(result.relevant_projects[:5], 1):
                        print(f"\n  {i}. {proj['project_name']} ({proj['relevance_score']:.1f}% match)")
                        print(f"     URL: {proj['url']}")
                        if proj['description']:
                            desc = proj['description'][:80]
                            print(f"     Description: {desc}...")
                        print(f"     Language: {proj['language']}, Stars: {proj['stars']}")
                        if proj.get('matched_keywords'):
                            print(f"     Matched Keywords: {', '.join(proj['matched_keywords'][:5])}")
                else:
                    print("\n[INFO] No relevant projects found for the job description")

                print("\n" + "=" * 70)
                print("[SUCCESS] GitHub analysis completed!")
                print("\nThis data will appear in the API response under the 'github' field.")
            else:
                print("[FAIL] GitHub analysis returned no results")
                print("Possible reasons:")
                print("  - Invalid GitHub username")
                print("  - User not found (404)")
                print("  - No public repositories")
                print("  - GitHub API rate limit exceeded")

        else:
            print("[WARN] No GitHub URL found in resume metadata")
            print("\nTo add GitHub URL to your resume:")
            print("1. Add text 'github.com/yourusername' to your resume PDF")
            print("2. Re-upload the resume or reinitialize the system")
            print("3. Test again")

    else:
        print("[WARN] Resume not found in ChromaDB")
        print("You need to initialize/upload the resume first")
        print("Run: curl -X POST http://localhost:8000/api/initialize")

except Exception as e:
    print(f"[FAIL] Error accessing ChromaDB: {e}")
    print("\nMake sure ChromaDB is initialized.")

print("\n" + "=" * 70)
print("Test complete!")
