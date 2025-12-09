#!/usr/bin/env python3
"""
Simple script to toggle GitHub analysis on/off in the RAG service.

Usage:
    python toggle_github_analysis.py on
    python toggle_github_analysis.py off
"""

import sys
import os

def toggle_github_analysis(enable: bool):
    """Toggle GitHub analysis by modifying the environment."""

    file_path = "backend/app/core/rag_service.py"

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the enable_github_analysis parameter
    if enable:
        # Enable GitHub analysis
        old_text = "enable_github_analysis=False"
        new_text = "enable_github_analysis=True"
        status = "ENABLED"
    else:
        # Disable GitHub analysis
        old_text = "enable_github_analysis=True"
        new_text = "enable_github_analysis=False"
        status = "DISABLED"

    if old_text in content:
        new_content = content.replace(old_text, new_text)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✓ GitHub analysis {status}")
        print(f"✓ Updated: {file_path}")
        print()
        print("Next steps:")
        print("1. Restart your backend server:")
        print("   uvicorn backend.app.main:app --reload")
        print()
        if enable:
            print("GitHub analysis is now ON")
            print("- Search will be slower (~10-15 sec per candidate)")
            print("- Results will include GitHub repository analysis")
        else:
            print("GitHub analysis is now OFF")
            print("- Search will be faster (~5-8 sec per candidate)")
            print("- Results will NOT include GitHub data")

    else:
        # Try to find the line to add the parameter
        search_pattern = "self.analyzer = create_analyzer("
        if search_pattern in content:
            print(f"✓ Found analyzer initialization")
            print()
            print(f"Please manually edit {file_path}")
            print(f"Find: {search_pattern}")
            print(f"Change the enable_github_analysis parameter to: {enable}")
        else:
            print(f"✗ Could not find GitHub analysis configuration in {file_path}")
            print()
            print("Manual configuration:")
            print(f"1. Open {file_path}")
            print("2. Find the create_analyzer() call")
            print(f"3. Set enable_github_analysis={enable}")


def main():
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ['on', 'off']:
        print("Usage: python toggle_github_analysis.py [on|off]")
        print()
        print("Examples:")
        print("  python toggle_github_analysis.py on   # Enable GitHub analysis")
        print("  python toggle_github_analysis.py off  # Disable for faster results")
        sys.exit(1)

    enable = sys.argv[1].lower() == 'on'
    toggle_github_analysis(enable)


if __name__ == "__main__":
    main()
