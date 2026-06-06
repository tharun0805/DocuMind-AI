"""
utils/validator.py

Two responsibilities:
  1. validate_file()     — checks file exists, has supported extension, non-empty
  2. validate_question() — checks question is non-empty and long enough

HARDCODING PREVENTION (embedded, no new file needed)
─────────────────────────────────────────────────────
_DOMAIN_TERMS is a module-level registry of strings that must never appear
hardcoded in project source files OR in any LLM-generated code that gets
eval()'d at runtime.

Two public helpers expose this:
  • assert_no_domain_leakage(text, context="")
      Raises ValueError if any forbidden term is found.
      Called from dataframe_agent before eval()'ing LLM code, and from
      _smart_transform before building column mappings.

  • scan_source_files(root_dir)
      Walk every .py in root_dir, return list of (file, line, term) violations.
      Called once at app startup inside load_core() so the developer sees
      violations immediately on the first run — no separate script needed.
"""

import os
import re
from loguru import logger

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".ppt",
    ".xlsx", ".xls", ".csv", ".txt", ".md"
}

# ── Forbidden domain-specific terms ──────────────────────────────────────────
# These are strings that must NEVER be hardcoded in project source.
# They come from uploaded documents and belong to the user's data, not the app.
# Add a new entry whenever a new document type surfaces a leakage risk.
# Format: (compiled_regex, human_readable_label)
_DOMAIN_TERMS: list[tuple[re.Pattern, str]] = [
    # From assessment spreadsheet uploads
    (re.compile(r'\bBDI\b',             re.I), "assessment scale abbreviation"),
    (re.compile(r'\bBAI\b',             re.I), "assessment scale abbreviation"),
    (re.compile(r'feel\s+guilty',       re.I), "questionnaire item text"),
    (re.compile(r'feel\s+sad',          re.I), "questionnaire item text"),
    (re.compile(r'Total\s+BDI\s+Score', re.I), "hardcoded score column name"),
    (re.compile(r'Total\s+BAI\s+Score', re.I), "hardcoded score column name"),
    (re.compile(r'family\s+type',       re.I), "spreadsheet column name"),
    # Severity band triplets — a sign the LLM was primed with domain data
    (re.compile(r'minimal.*mild.*moderate', re.I), "hardcoded severity classification"),
    (re.compile(r'mild.*moderate.*severe',  re.I), "hardcoded severity classification"),
]

# Files that are exempt from source scanning (e.g. migration scripts)
_SCAN_SKIP_DIRS  = {".git", "__pycache__", ".venv", "venv", "node_modules", ".streamlit"}
_SCAN_SKIP_FILES: set[str] = {"validator.py"}  # skip self — pattern defs live here


def _is_skip_file(fname: str) -> bool:
    """Skip the validator itself and any legacy guard scripts."""
    return fname in _SCAN_SKIP_FILES or fname.startswith("check_hardcod")


def assert_no_domain_leakage(text: str, context: str = "") -> None:
    """
    Raise ValueError if `text` contains any forbidden domain-specific term.
    Call this before eval()'ing LLM-generated code or building column maps.

    Args:
        text:    The string to check (LLM output, column name list, etc.)
        context: Short label for the error message (e.g. "LLM code", "column map")
    """
    for pattern, label in _DOMAIN_TERMS:
        m = pattern.search(text)
        if m:
            raise ValueError(
                f"Domain leakage detected in {context or 'generated text'}: "
                f"found '{m.group()}' ({label}). "
                f"Values must be derived from the uploaded document at runtime, "
                f"not hardcoded in source."
            )


def scan_source_files(root_dir: str) -> list[tuple[str, int, str, str]]:
    """
    Walk every .py file under root_dir and return a list of violations:
      [(filepath, line_number, line_text, label), ...]

    Pure comments (#-only lines) are skipped.
    Designed to be called once at startup from load_core() so violations
    surface immediately without any separate script or CI step.
    """
    violations = []
    root = os.path.abspath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk doesn't descend
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIRS]

        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            if _is_skip_file(fname):
                continue

            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except Exception:
                continue

            for lineno, line in enumerate(lines, 1):
                if line.strip().startswith("#"):
                    continue
                for pattern, label in _DOMAIN_TERMS:
                    if pattern.search(line):
                        violations.append((fpath, lineno, line.rstrip(), label))
                        break   # one report per line is enough

    return violations


# ── Original validators (unchanged) ──────────────────────────────────────────

def validate_file(file_path: str) -> tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, "File not found. Please upload again."

    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported file type: {extension}. "
            f"Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT"
        )

    if os.path.getsize(file_path) == 0:
        return False, "File is empty."

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"File validated: {file_path} ({size_mb:.1f}MB)")
    return True, "Valid"


def validate_question(question: str) -> tuple[bool, str]:
    if not question or not question.strip():
        return False, "Question cannot be empty."
    if len(question.strip()) < 2:
        return False, "Question too short."
    return True, "Valid"