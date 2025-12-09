"""
Text utility functions.
"""

import re


def extract_job_title(job_description: str) -> str:
    """
    Extract job title from job description.

    Tries to identify the job title using common patterns:
    1. First line if it's short (likely the title)
    2. Lines starting with "Position:", "Role:", "Title:", etc.
    3. Capitalised text before "Job Description" or similar headers
    4. First 2-3 words if they look like a title

    Args:
        job_description: The job description text

    Returns:
        Extracted job title or "Job Search" as fallback
    """
    if not job_description or not job_description.strip():
        return "Job Search"

    lines = [line.strip() for line in job_description.strip().split('\n') if line.strip()]

    if not lines:
        return "Job Search"

    # Pattern 1: Explicit title markers
    title_patterns = [
        r'(?:position|role|title|job\s*title|job\s*role):\s*(.+)',
        r'(?:hiring|looking\s*for|seeking):\s*(.+)',
    ]

    for line in lines[:10]:  # Check first 10 lines
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if title and len(title) < 100:  # Reasonable title length
                    return clean_job_title(title)

    # Pattern 2: First line if it's short and looks like a title
    first_line = lines[0]

    # Skip common header words
    skip_words = ['job', 'description', 'about', 'overview', 'summary', 'details']
    first_word_lower = first_line.split()[0].lower() if first_line.split() else ''

    if (len(first_line) < 100 and
        len(first_line.split()) <= 8 and  # Titles are usually short
        first_word_lower not in skip_words and
        not first_line.endswith(':')):  # Not a section header
        return clean_job_title(first_line)

    # Pattern 3: Look for capitalised words before section headers
    for line in lines[:5]:
        # If line before a header section
        if re.match(r'^(job\s*description|about|overview|responsibilities|requirements|qualifications):',
                   line, re.IGNORECASE):
            # Previous line might be title
            idx = lines.index(line)
            if idx > 0:
                prev_line = lines[idx - 1]
                if len(prev_line) < 100 and len(prev_line.split()) <= 8:
                    return clean_job_title(prev_line)

    # Pattern 4: Extract first few words if they look title-like
    words = first_line.split()
    if len(words) >= 2:
        # Common job title patterns: "Senior Python Developer", "Data Scientist", etc.
        potential_title = ' '.join(words[:min(5, len(words))])
        if len(potential_title) < 80:
            return clean_job_title(potential_title)

    # Fallback: Use first line but limit length
    if len(first_line) > 50:
        return first_line[:47] + "..."

    return clean_job_title(first_line) if first_line else "Job Search"


def clean_job_title(title: str) -> str:
    """
    Clean and format job title.

    Args:
        title: Raw job title string

    Returns:
        Cleaned job title
    """
    # Remove common prefixes/suffixes
    title = re.sub(r'^(position|role|title|job):\s*', '', title, flags=re.IGNORECASE)

    # Remove markdown formatting
    title = re.sub(r'[#*_`]', '', title)

    # Remove extra whitespace
    title = ' '.join(title.split())

    # Remove trailing punctuation except essential ones
    title = re.sub(r'[,;.!?]+$', '', title)

    # Capitalize properly if all caps or all lowercase
    if title.isupper() or title.islower():
        # Title case, but keep acronyms
        words = title.split()
        formatted_words = []
        for word in words:
            if len(word) <= 3 and word.isupper():
                # Likely an acronym (API, CEO, etc.)
                formatted_words.append(word)
            else:
                formatted_words.append(word.capitalize())
        title = ' '.join(formatted_words)

    # Limit length
    if len(title) > 80:
        title = title[:77] + "..."

    return title.strip()
