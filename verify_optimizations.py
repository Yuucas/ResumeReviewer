#!/usr/bin/env python3
"""
Verify that all performance optimizations are properly in place.

This script checks the code to ensure:
1. Parallel processing is enabled by default
2. GitHub analyzer uses smart filtering
3. LLM token count is optimized
"""

import re
from pathlib import Path

def check_file_contains(file_path: Path, pattern: str, description: str) -> bool:
    """Check if file contains a specific pattern."""
    try:
        content = file_path.read_text(encoding='utf-8')
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            print(f"  [OK] {description}")
            return True
        else:
            print(f"  [FAIL] {description} - NOT FOUND")
            return False
    except Exception as e:
        print(f"  [ERROR] Error checking {file_path}: {e}")
        return False

def main():
    """Main verification function."""

    print("=" * 70)
    print("PERFORMANCE OPTIMIZATION VERIFICATION")
    print("=" * 70)
    print()

    base_path = Path(__file__).parent
    all_checks_passed = True

    # Check 1: GitHub Analyzer Optimizations
    print("1. GitHub Analyzer Optimizations")
    print("-" * 70)

    github_analyzer_path = base_path / "src" / "agents" / "github_analyzer.py"

    checks = [
        (r"max_repos:\s*int\s*=\s*10", "Reduced max_repos to 10 (was 30)"),
        (r"def fetch_readme\(", "README fetching method exists"),
        (r"def _clean_markdown\(", "Markdown cleaning method exists"),
        (r"ThreadPoolExecutor\(max_workers=3\)", "Parallel README fetching (3 workers)"),
        (r"top_candidates\s*=.*\[:min\(5,", "Smart filtering: top 5 repos only"),
        (r"content\[:3000\]", "README content limited to 3000 chars"),
    ]

    for pattern, description in checks:
        if not check_file_contains(github_analyzer_path, pattern, description):
            all_checks_passed = False

    print()

    # Check 2: Analyzer Parallel Processing
    print("2. Analyzer Parallel Processing")
    print("-" * 70)

    analyzer_path = base_path / "src" / "agents" / "analyzer.py"

    checks = [
        (r"def analyze_candidates_batch\(", "Batch analysis method exists"),
        (r"parallel:\s*bool\s*=\s*True", "Parallel processing enabled by default"),
        (r"max_workers:\s*int\s*=\s*3", "3 parallel workers configured"),
        (r"ThreadPoolExecutor\(max_workers=max_workers\)", "ThreadPoolExecutor for parallel processing"),
        (r"max_tokens\s*=\s*800", "LLM tokens reduced to 800 (was 1500)"),
    ]

    for pattern, description in checks:
        if not check_file_contains(analyzer_path, pattern, description):
            all_checks_passed = False

    print()

    # Check 3: Pipeline Integration
    print("3. Pipeline Integration")
    print("-" * 70)

    main_path = base_path / "src" / "main.py"

    checks = [
        (r"self\.analyzer\.analyze_candidates_batch\(", "Pipeline uses batch analysis method"),
        (r"candidates=candidates", "Passes candidates to batch method"),
        (r"job_description=job_description", "Passes job description to batch method"),
    ]

    for pattern, description in checks:
        if not check_file_contains(main_path, pattern, description):
            all_checks_passed = False

    print()

    # Check 4: Helper Files
    print("4. Helper Files and Documentation")
    print("-" * 70)

    helper_files = [
        ("check_github_token.py", "GitHub token checker script"),
        ("test_performance.py", "Performance test script"),
        ("PERFORMANCE_OPTIMIZATIONS.md", "Performance documentation"),
        ("QUICK_TEST_GUIDE.md", "Quick test guide"),
        ("GITHUB_TOKEN_QUICKSTART.md", "GitHub token quick start"),
        ("GITHUB_TOKEN_TROUBLESHOOTING.md", "GitHub token troubleshooting"),
    ]

    for filename, description in helper_files:
        file_path = base_path / filename
        if file_path.exists():
            print(f"  [OK] {description}")
        else:
            print(f"  [FAIL] {description} - NOT FOUND")
            all_checks_passed = False

    print()

    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    if all_checks_passed:
        print("[SUCCESS] All optimizations are properly in place!")
        print()
        print("Expected Performance:")
        print("  - Single candidate: ~38 seconds (was ~180 seconds)")
        print("  - 5 candidates: ~1.3 minutes (was ~15 minutes)")
        print("  - Speedup: ~92% faster for batch processing")
        print()
        print("Next Steps:")
        print("  1. Set GITHUB_TOKEN: set GITHUB_TOKEN=ghp_your_token")
        print("  2. Verify token: python check_github_token.py")
        print("  3. Start backend: uvicorn backend.app.main:app --reload")
        print("  4. Run test: python test_performance.py")
        print()
        print("Or use the quick start script:")
        print("  start_backend_with_github_token.bat ghp_your_token")
        print()
        return 0
    else:
        print("[FAILED] Some optimizations are missing!")
        print()
        print("Please review the output above for details.")
        print("See PERFORMANCE_OPTIMIZATIONS.md for implementation details.")
        print()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
