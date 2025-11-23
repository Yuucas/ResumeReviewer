#!/usr/bin/env python3
"""
Debug LLM response to understand why JSON parsing fails.

This script captures and displays the actual LLM response to help diagnose the issue.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("LLM RESPONSE DEBUGGER")
print("=" * 70)
print()

try:
    from src.agents import create_llm_client
    from src.agents.prompt_templates import PromptBuilder, get_system_prompt

    # Create LLM client
    print("Initializing Ollama client...")
    llm_client = create_llm_client()
    print(f"✓ Connected to Ollama (model: {llm_client.model})")
    print()

    # Create a simple test prompt
    job_description = """
    Senior Python Developer

    Requirements:
    - 5+ years of Python experience
    - Django or FastAPI framework
    - PostgreSQL database
    - REST API development
    """

    candidate_info = """
    Candidate: John Doe
    Role Applied For: Software Engineer
    Total Experience: 6 years
    Email: john@example.com

    Resume Content:

    [EXPERIENCE]
    Senior Python Developer at Tech Corp (3 years)
    - Built REST APIs with FastAPI
    - Managed PostgreSQL databases
    - Led team of 5 developers

    [SKILLS]
    Python, Django, FastAPI, PostgreSQL, Docker, Git
    """

    # Build prompt
    prompt = PromptBuilder.build_analysis_prompt(
        job_description=job_description,
        candidate_info=candidate_info
    )

    system_prompt = get_system_prompt('recruiter')

    print("Sending request to LLM...")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"System prompt length: {len(system_prompt)} characters")
    print()

    # Generate response
    import time
    start_time = time.time()

    response = llm_client.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=1200,
        system_prompt=system_prompt
    )

    elapsed = time.time() - start_time

    print(f"✓ Response received in {elapsed:.2f} seconds")
    print(f"✓ Response length: {len(response)} characters")
    print()

    print("=" * 70)
    print("RAW LLM RESPONSE")
    print("=" * 70)
    print(response)
    print()
    print("=" * 70)

    # Try to extract JSON
    print()
    print("Attempting JSON extraction...")
    print()

    extracted = llm_client.extract_json(response)

    if extracted:
        print("✓ JSON extraction SUCCESSFUL!")
        print()
        print("Extracted JSON:")
        import json
        print(json.dumps(extracted, indent=2))
        print()

        # Check required fields
        required_fields = ['match_score', 'strengths', 'weaknesses', 'key_qualifications',
                          'overall_assessment', 'recommendation']
        missing_fields = [f for f in required_fields if f not in extracted]

        if missing_fields:
            print(f"⚠ WARNING: Missing fields: {', '.join(missing_fields)}")
        else:
            print("✓ All required fields present")
    else:
        print("✗ JSON extraction FAILED")
        print()
        print("Trying to identify the issue...")
        print()

        # Check for common issues
        import re

        # Check if there's JSON anywhere
        if '{' in response and '}' in response:
            print("  ✓ Response contains braces { }")

            # Try to find JSON block
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                print("  ✓ Found JSON-like structure")
                print()
                print("  Extracted structure:")
                print("  " + json_match.group(0)[:200] + "...")
            else:
                print("  ✗ Could not find valid JSON structure")
        else:
            print("  ✗ Response does not contain JSON braces")

        # Check for markdown code blocks
        if '```' in response:
            print("  ✓ Response contains markdown code blocks")

            if '```json' in response:
                print("  ✓ Found ```json marker")
            elif '```' in response:
                print("  ⚠ Found ``` but not ```json")
        else:
            print("  ✗ No markdown code blocks found")

        print()
        print("First 500 characters of response:")
        print("-" * 70)
        print(response[:500])
        print("-" * 70)
        print()
        print("Last 500 characters of response:")
        print("-" * 70)
        print(response[-500:])
        print("-" * 70)

except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print()
    print("Make sure you're running from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("Debug complete!")
print("=" * 70)
