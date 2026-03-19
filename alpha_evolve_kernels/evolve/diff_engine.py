"""Parse and apply AlphaEvolve-style SEARCH/REPLACE diffs.

Format:
    <<<<<<< SEARCH
    # Original code block to be found and replaced
    =======
    # New code block to replace the original
    >>>>>>> REPLACE
"""

import re
from dataclasses import dataclass


@dataclass
class DiffHunk:
    """A single SEARCH/REPLACE block."""
    search: str
    replace: str
    index: int = 0  # Position in the diff sequence


def parse_diff(llm_output: str) -> list[DiffHunk]:
    """Parse SEARCH/REPLACE blocks from LLM output.

    Handles variations in marker formatting:
    - <<<<<<< SEARCH / ======= / >>>>>>> REPLACE
    - <<<<<< SEARCH / ====== / >>>>>> REPLACE
    - <<<<< SEARCH / ===== / >>>>> REPLACE

    Args:
        llm_output: Raw LLM output text.

    Returns:
        List of DiffHunk objects.
    """
    pattern = re.compile(
        r'<{5,7}\s*SEARCH\s*\n'
        r'(.*?)\n'
        r'={5,7}\s*\n'
        r'(.*?)\n'
        r'>{5,7}\s*REPLACE',
        re.DOTALL
    )

    hunks = []
    for i, match in enumerate(pattern.finditer(llm_output)):
        search_text = match.group(1)
        replace_text = match.group(2)
        hunks.append(DiffHunk(search=search_text, replace=replace_text, index=i))

    return hunks


def apply_diff(source: str, hunks: list[DiffHunk]) -> tuple[str, list[str]]:
    """Apply diff hunks to source code sequentially.

    Args:
        source: Original source code.
        hunks: List of DiffHunk to apply in order.

    Returns:
        (modified_source, list_of_warnings)
        Warnings are generated for hunks where SEARCH text was not found.
    """
    result = source
    warnings = []

    for hunk in hunks:
        search = hunk.search.strip()
        if not search:
            warnings.append(f"Hunk {hunk.index}: empty SEARCH block, skipping")
            continue

        if search in result:
            result = result.replace(search, hunk.replace.strip(), 1)
        else:
            # Try with normalized whitespace
            normalized_search = _normalize_whitespace(search)
            normalized_result = _normalize_whitespace(result)

            if normalized_search in normalized_result:
                # Find the original text that matches when normalized
                result = _fuzzy_replace(result, search, hunk.replace.strip())
            else:
                warnings.append(
                    f"Hunk {hunk.index}: SEARCH text not found in source "
                    f"(first 60 chars: '{search[:60]}...')"
                )

    return result, warnings


def apply_or_fallback(source: str, llm_output: str) -> tuple[str, str]:
    """Try to apply diffs; if no diffs found, treat output as full replacement.

    Args:
        source: Original source code.
        llm_output: Raw LLM output.

    Returns:
        (new_code, method) where method is "diff" or "full_replace"
    """
    hunks = parse_diff(llm_output)

    if hunks:
        result, warnings = apply_diff(source, hunks)
        if not warnings:
            return result, "diff"
        # If some hunks failed, still return partial result
        return result, f"diff_partial({len(warnings)} warnings)"

    # No diff blocks found — check if it looks like complete code
    if "class ModelNew" in llm_output or "def forward" in llm_output:
        # Extract just the code (strip markdown fences etc.)
        code = _extract_code_block(llm_output)
        return code, "full_replace"

    return source, "no_change"


def _normalize_whitespace(s: str) -> str:
    """Normalize whitespace for fuzzy matching."""
    return re.sub(r'\s+', ' ', s).strip()


def _fuzzy_replace(source: str, search: str, replace: str) -> str:
    """Replace with normalized whitespace matching."""
    lines_source = source.split('\n')
    lines_search = search.strip().split('\n')

    if not lines_search:
        return source

    first_search = lines_search[0].strip()
    n = len(lines_search)

    for i in range(len(lines_source)):
        if lines_source[i].strip() == first_search:
            # Check if subsequent lines match
            match = True
            for j in range(1, n):
                if i + j >= len(lines_source):
                    match = False
                    break
                if lines_source[i + j].strip() != lines_search[j].strip():
                    match = False
                    break
            if match:
                lines_source[i:i + n] = replace.split('\n')
                return '\n'.join(lines_source)

    return source


def _extract_code_block(text: str) -> str:
    """Extract Python code from markdown-fenced output."""
    # Try ```python ... ``` first
    pattern = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # Try to find code starting from import/class/def
    for marker in ("import ", "from ", "class ", "def "):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].strip()

    return text.strip()
