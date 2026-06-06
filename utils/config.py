import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
 
load_dotenv()
 
 
# ── API Key Loaders ────────────────────────────────────────────────
 
def get_google_api_key() -> str:
    """
    Return the Google Gemini API key from .env.
    Raises ValueError if missing or still set to placeholder.
    """
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key or key == "your_google_api_key_here":
        logger.error("GOOGLE_API_KEY is missing or not set in .env")
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please add it to your .env file."
        )
    logger.debug("GOOGLE_API_KEY loaded successfully")
    return key
 
 
def get_groq_api_key() -> str:
    """
    Return the Groq API key from .env.
    Returns empty string if not set (Groq is optional — Gemini is fallback).
    """
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key == "your_groq_key_here":
        logger.debug("GROQ_API_KEY not set — will use Gemini fallback")
        return ""
    logger.debug("GROQ_API_KEY loaded successfully")
    return key
 
 
# ── Startup Validation ─────────────────────────────────────────────
 
def validate_startup() -> dict:
    """
    Run all startup security and environment checks.
    Returns a dict of results for display in the UI.
 
    Called once when app starts.
    """
    results = {
        "env_file": False,
        "google_key": False,
        "groq_key": False,
        "gitignore_safe": False,
        "model_cached": False,
        "logs_dir": False,
        "warnings": [],
        "errors": [],
    }
 
    # 1. Check .env file exists
    env_path = Path(".env")
    if env_path.exists():
        results["env_file"] = True
        logger.info("[STARTUP] .env file found")
    else:
        results["errors"].append(
            ".env file not found. Create it from .env.example"
        )
        logger.error("[STARTUP] .env file missing")
 
    # 2. Check Google API key
    try:
        get_google_api_key()
        results["google_key"] = True
        logger.info("[STARTUP] GOOGLE_API_KEY validated")
    except ValueError as e:
        results["errors"].append(str(e))
 
    # 3. Check Groq API key (optional)
    groq = get_groq_api_key()
    if groq:
        results["groq_key"] = True
        logger.info("[STARTUP] GROQ_API_KEY validated")
    else:
        results["warnings"].append(
            "GROQ_API_KEY not set. App will use Gemini (slower). "
            "Add Groq key for faster responses."
        )
 
    # 4. Check .gitignore protects .env
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text(encoding="utf-8")
        if ".env" in gitignore_content:
            results["gitignore_safe"] = True
            logger.info("[STARTUP] .env is in .gitignore")
        else:
            results["errors"].append(
                "SECURITY: .env is NOT in .gitignore. "
                "Your API keys may be exposed. Add .env to .gitignore immediately."
            )
            logger.critical("[STARTUP] .env NOT in .gitignore — security risk")
    else:
        results["warnings"].append(".gitignore file not found.")
 
    # 5. Check embedding model cached locally
    model_path = Path("models/embedding_model")
    if model_path.exists() and any(model_path.iterdir()):
        results["model_cached"] = True
        logger.info("[STARTUP] Local embedding model cache found")
    else:
        results["warnings"].append(
            "Embedding model not cached locally. "
            "First document upload will download the model (~90MB). "
            "Run the cache script for faster startup."
        )
 
    # 6. Check logs directory
    logs_path = Path("logs")
    try:
        logs_path.mkdir(exist_ok=True)
        results["logs_dir"] = True
        logger.info("[STARTUP] Logs directory ready")
    except Exception as e:
        results["warnings"].append(f"Could not create logs directory: {e}")
 
    # Summary
    error_count = len(results["errors"])
    warn_count = len(results["warnings"])
    logger.info(
        f"[STARTUP] Validation complete — "
        f"Errors: {error_count} | Warnings: {warn_count}"
    )
 
    return results