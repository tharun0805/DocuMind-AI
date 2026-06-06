import os
import re
from pathlib import Path
from loguru import logger
 
# Patterns that should never appear in source code
_SECRET_PATTERNS = [
    (r'AIza[0-9A-Za-z_-]{35}', "Google API key"),
    (r'gsk_[0-9A-Za-z]{50,}', "Groq API key"),
    (r'sk-[0-9A-Za-z]{48}', "OpenAI API key"),
    (r'(?i)(api[_-]?key|secret|password)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
]
 
# Files/folders to skip during scanning
_SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules", "models", "vector_db", "logs"}
_SKIP_EXTS = {".pyc", ".pyo", ".pkl", ".faiss", ".bin", ".zip"}
_SKIP_FILES = {".env", ".env.example"}
 
 
def scan_for_secrets(root_dir: str = ".") -> list[dict]:
    """
    Scan all Python source files for hardcoded secrets.
    Returns list of findings dicts.
    """
    findings = []
    root = Path(root_dir)
 
    for py_file in root.rglob("*.py"):
        # Skip unwanted directories
        if any(skip in py_file.parts for skip in _SKIP_DIRS):
            continue
        if py_file.name in _SKIP_FILES:
            continue
        if py_file.suffix in _SKIP_EXTS:
            continue
 
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
 
        for pattern, label in _SECRET_PATTERNS:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count("\n") + 1
                findings.append({
                    "file": str(py_file),
                    "line": line_num,
                    "type": label,
                    "snippet": match.group()[:40] + "...",
                })
                logger.critical(
                    f"[SECURITY] Potential {label} found in "
                    f"{py_file}:{line_num}"
                )
 
    return findings
 
 
def check_gitignore(root_dir: str = ".") -> dict:
    """
    Verify .gitignore properly covers sensitive files.
    Returns dict with status and missing entries.
    """
    required = {".env", "*.pyc", "__pycache__", "venv", "models", "vector_db", "logs"}
    gitignore = Path(root_dir) / ".gitignore"
 
    if not gitignore.exists():
        logger.error("[SECURITY] .gitignore not found")
        return {"ok": False, "missing": list(required), "message": ".gitignore not found"}
 
    content = gitignore.read_text(encoding="utf-8", errors="ignore")
    covered = {entry for entry in required if entry in content}
    missing = required - covered
 
    if missing:
        logger.warning(f"[SECURITY] .gitignore missing entries: {missing}")
    else:
        logger.info("[SECURITY] .gitignore covers all required entries")
 
    return {
        "ok": len(missing) == 0,
        "missing": list(missing),
        "covered": list(covered),
    }
 
 
def check_env_example(root_dir: str = ".") -> dict:
    """
    Verify .env.example exists and contains only placeholder values.
    """
    env_example = Path(root_dir) / ".env.example"
    if not env_example.exists():
        logger.warning("[SECURITY] .env.example not found")
        return {"ok": False, "message": ".env.example missing"}
 
    content = env_example.read_text(encoding="utf-8", errors="ignore")
 
    # Check for real keys (too long = likely real)
    suspicious = []
    for line in content.splitlines():
        if "=" in line and not line.startswith("#"):
            val = line.split("=", 1)[1].strip()
            if len(val) > 20 and not any(
                placeholder in val.lower()
                for placeholder in ["your_", "here", "example", "xxx", "placeholder"]
            ):
                suspicious.append(line[:60])
 
    if suspicious:
        logger.warning(f"[SECURITY] .env.example may contain real keys: {suspicious}")
        return {"ok": False, "suspicious_lines": suspicious}
 
    logger.info("[SECURITY] .env.example looks safe")
    return {"ok": True, "message": ".env.example uses only placeholders"}
 
 
def run_full_audit(root_dir: str = ".") -> dict:
    """
    Run complete security audit.
    Returns summary dict with all findings.
    """
    logger.info("[SECURITY] Starting full security audit...")
 
    secret_findings = scan_for_secrets(root_dir)
    gitignore_result = check_gitignore(root_dir)
    env_example_result = check_env_example(root_dir)
 
    passed = (
        len(secret_findings) == 0
        and gitignore_result["ok"]
        and env_example_result["ok"]
    )
 
    result = {
        "passed": passed,
        "secret_scan": {
            "clean": len(secret_findings) == 0,
            "findings": secret_findings,
        },
        "gitignore": gitignore_result,
        "env_example": env_example_result,
    }
 
    if passed:
        logger.info("[SECURITY] Audit PASSED - no issues found")
    else:
        logger.error("[SECURITY] Audit FAILED - review findings above")
 
    return result