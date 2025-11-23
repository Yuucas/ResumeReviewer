#!/usr/bin/env python3
"""
Performance testing script for optimized resume analysis.

Tests both sequential and parallel processing to demonstrate speedup.
"""

import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_parallel_performance():
    """Test the performance of parallel vs sequential processing."""

    print("=" * 70)
    print("PERFORMANCE TEST: Parallel vs Sequential Processing")
    print("=" * 70)
    print()

    # Add current directory to path for imports
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import after logging is setup
    from src.main import ResumeRAGPipeline

    # Sample job description
    job_description = """
    Senior Python Developer

    Requirements:
    - 5+ years of Python development experience
    - Strong experience with Django, FastAPI, or Flask
    - Experience with PostgreSQL and NoSQL databases
    - Knowledge of Docker and Kubernetes
    - Experience with REST APIs and microservices
    - Strong problem-solving skills
    - GitHub profile with open-source contributions preferred
    """

    # Initialize pipeline
    print("Initializing ResumeRAGPipeline...")
    try:
        pipeline = ResumeRAGPipeline()
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        print()
        print("This test requires:")
        print("  1. Database to be indexed with resumes")
        print("  2. Ollama to be running")
        print("  3. GITHUB_TOKEN environment variable set")
        print()
        print("Run 'python test_optimization_simple.py' instead for a simpler test.")
        return

    analyzer = pipeline.analyzer
    print()

    # Search for candidates
    print("Searching for top 5 candidates...")
    start_search = time.time()
    candidates = pipeline.search_candidates(
        job_description=job_description,
        role_category=None,
        min_experience=0.0,
        use_reranking=False,
        override_top_k=5
    )
    search_time = time.time() - start_search
    print(f"Found {len(candidates)} candidates in {search_time:.2f} seconds")
    print()

    if not candidates:
        print("[ERROR] No candidates found. Please ensure dataset is indexed.")
        print()
        print("To index resumes, run the ingestion script first.")
        return

    # Test 1: Sequential processing
    print("-" * 70)
    print("TEST 1: Sequential Processing (parallel=False)")
    print("-" * 70)
    start_seq = time.time()
    try:
        sequential_results = analyzer.analyze_candidates_batch(
            candidates=candidates,
            job_description=job_description,
            parallel=False  # Disable parallel processing
        )
        seq_time = time.time() - start_seq
        print(f"✓ Analyzed {len(sequential_results)} candidates sequentially")
        print(f"✓ Time: {seq_time:.2f} seconds")
        print(f"✓ Average: {seq_time/len(sequential_results):.2f} seconds per candidate")
    except Exception as e:
        print(f"✗ Sequential test failed: {e}")
        seq_time = None
    print()

    # Test 2: Parallel processing (3 workers)
    print("-" * 70)
    print("TEST 2: Parallel Processing (parallel=True, max_workers=3)")
    print("-" * 70)
    start_par = time.time()
    try:
        parallel_results = analyzer.analyze_candidates_batch(
            candidates=candidates,
            job_description=job_description,
            parallel=True,  # Enable parallel processing
            max_workers=3
        )
        par_time = time.time() - start_par
        print(f"✓ Analyzed {len(parallel_results)} candidates in parallel")
        print(f"✓ Time: {par_time:.2f} seconds")
        print(f"✓ Average: {par_time/len(parallel_results):.2f} seconds per candidate")
    except Exception as e:
        print(f"✗ Parallel test failed: {e}")
        par_time = None
    print()

    # Compare results
    if seq_time and par_time:
        print("=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)
        speedup = seq_time / par_time
        time_saved = seq_time - par_time
        percent_faster = ((seq_time - par_time) / seq_time) * 100

        print(f"Sequential time:    {seq_time:.2f} seconds")
        print(f"Parallel time:      {par_time:.2f} seconds")
        print(f"Time saved:         {time_saved:.2f} seconds")
        print(f"Speedup factor:     {speedup:.2f}x")
        print(f"Percent faster:     {percent_faster:.1f}%")
        print()

        if speedup >= 2.0:
            print("✓ EXCELLENT: Parallel processing is 2x+ faster!")
        elif speedup >= 1.5:
            print("✓ GOOD: Parallel processing provides significant speedup")
        elif speedup >= 1.2:
            print("✓ OK: Parallel processing provides moderate speedup")
        else:
            print("⚠ WARNING: Limited speedup from parallel processing")
        print()

    # GitHub analysis info
    print("=" * 70)
    print("GITHUB ANALYSIS OPTIMIZATIONS")
    print("=" * 70)
    print("✓ Smart filtering: Only top 5 repos analyzed with README")
    print("✓ Parallel README fetching: 3 concurrent requests")
    print("✓ README length limit: 3000 characters per README")
    print("✓ Quick pre-filter: Fast initial scoring without README")
    print()

    # LLM optimization info
    print("=" * 70)
    print("LLM OPTIMIZATIONS")
    print("=" * 70)
    print("✓ Reduced max_tokens: 1500 → 800 (faster generation)")
    print("✓ Temperature: 0.3 (consistent, focused results)")
    print()

    print("=" * 70)
    print("EXPECTED PERFORMANCE")
    print("=" * 70)
    print("Target: ~1 minute per candidate")
    print(f"Actual: ~{par_time/len(parallel_results):.2f} seconds per candidate")

    target_per_candidate = 60  # 1 minute
    actual_per_candidate = par_time / len(parallel_results)

    if actual_per_candidate <= target_per_candidate:
        print(f"✓ SUCCESS: Meeting performance target!")
    else:
        print(f"⚠ ATTENTION: {actual_per_candidate - target_per_candidate:.2f}s slower than target")
    print()


def main():
    """Main entry point."""
    try:
        test_parallel_performance()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
