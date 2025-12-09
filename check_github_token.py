#!/usr/bin/env python3
"""
Check if GitHub token is properly set and working.

Usage:
    python check_github_token.py
"""

import os
import sys
import requests

def check_token():
    """Check if GITHUB_TOKEN is set and valid."""

    print("="*70)
    print("GitHub Token Checker")
    print("="*70)
    print()

    # Check if token is set
    token = os.environ.get('GITHUB_TOKEN')

    if not token:
        print("[ERROR] GITHUB_TOKEN environment variable is NOT set!")
        print()
        print("How to fix:")
        print()
        print("Windows (Command Prompt):")
        print("  set GITHUB_TOKEN=ghp_your_token_here")
        print("  python check_github_token.py")
        print()
        print("Windows (PowerShell):")
        print("  $env:GITHUB_TOKEN=\"ghp_your_token_here\"")
        print("  python check_github_token.py")
        print()
        print("Linux/Mac:")
        print("  export GITHUB_TOKEN=ghp_your_token_here")
        print("  python check_github_token.py")
        print()
        print("Or use the helper script:")
        print("  Windows: start_backend_with_github_token.bat ghp_your_token")
        print("  Linux/Mac: ./start_backend_with_github_token.sh ghp_your_token")
        print()
        return False

    print(f"[OK] GITHUB_TOKEN is set")
    print(f"Token preview: {token[:10]}...{token[-4:]}")
    print()

    # Test the token by calling GitHub API
    print("Testing token with GitHub API...")

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }

    try:
        response = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            print("[OK] Token is VALID!")
            print()
            print("Rate Limits:")
            print(f"  Core API:")
            print(f"    Limit: {data['resources']['core']['limit']} requests/hour")
            print(f"    Remaining: {data['resources']['core']['remaining']} requests")
            print(f"    Used: {data['resources']['core']['limit'] - data['resources']['core']['remaining']} requests")
            print()

            if data['resources']['core']['limit'] == 5000:
                print("[SUCCESS] You have authenticated rate limit: 5,000 requests/hour!")
            else:
                print(f"[WARNING] Rate limit is {data['resources']['core']['limit']}/hour (expected 5,000)")

            print()
            print("Next steps:")
            print("1. Keep this terminal open")
            print("2. Start backend in this same terminal:")
            print("   uvicorn backend.app.main:app --reload")
            print()
            print("Or use the helper script:")
            print("   Windows: start_backend_with_github_token.bat")
            print("   Linux/Mac: ./start_backend_with_github_token.sh")
            print()
            return True

        elif response.status_code == 401:
            print("[ERROR] Token is INVALID!")
            print(f"Response: {response.json().get('message', 'Unknown error')}")
            print()
            print("Please check your token:")
            print("1. Go to: https://github.com/settings/tokens")
            print("2. Verify the token exists and hasn't expired")
            print("3. Generate a new token if needed")
            print("4. Make sure you copied the full token")
            print()
            return False

        else:
            print(f"[ERROR] Unexpected response: {response.status_code}")
            print(f"Message: {response.text}")
            return False

    except requests.RequestException as e:
        print(f"[ERROR] Failed to connect to GitHub API: {e}")
        print()
        print("Please check your internet connection")
        return False

def main():
    """Main entry point."""
    success = check_token()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
