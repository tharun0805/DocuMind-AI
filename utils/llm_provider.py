import os
import time
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
 
load_dotenv()
 
# ── Retry configuration ───────────────────────────────────────────
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 5   # seconds
_BACKOFF_MULTIPLIER = 2
 
 
# ── Cached LLM instances ──────────────────────────────────────────
 
@st.cache_resource(show_spinner=False)
def _groq_llm(temperature: float = 0.3):
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key in ["your_groq_key_here", ""]:
        return None
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=key,
            temperature=temperature,
            max_tokens=2048,
        )
        logger.info("[LLM] Groq llama-3.3-70b-versatile instance cached")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] Groq init failed: {e}")
        return None
 
 
@st.cache_resource(show_spinner=False)
def _gemini_llm(temperature: float = 0.3):
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key or key == "your_google_api_key_here":
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=key,
            temperature=temperature,
        )
        logger.info("[LLM] Gemini 2.5 Flash instance cached")
        return llm
    except Exception as e:
        logger.warning(f"[LLM] Gemini init failed: {e}")
        return None
 
 
def get_shared_llm(temperature: float = 0.3):
    """
    Return the best available cached LLM.
    Priority: Groq → Gemini → raise error.
    """
    groq = _groq_llm(temperature)
    if groq is not None:
        logger.debug("[LLM] Using Groq")
        return groq
 
    gemini = _gemini_llm(temperature)
    if gemini is not None:
        logger.debug("[LLM] Using Gemini fallback")
        return gemini
 
    raise RuntimeError(
        "No LLM available. Please set GROQ_API_KEY or GOOGLE_API_KEY in your .env file."
    )
 
 
# ── Retry wrapper ─────────────────────────────────────────────────
 
def llm_call(prompt_template: str, inputs: dict, temp: float = 0.3) -> str:
    """
    Run a prompt with automatic rate-limit retry and timing.
 
    Args:
        prompt_template: A string template with {placeholders}.
        inputs:          Dict matching the placeholders.
        temp:            LLM temperature (0.0–1.0).
 
    Returns:
        The LLM response as a plain string.
 
    Raises:
        RuntimeError if all retries are exhausted.
    """
    from langchain_core.prompts import PromptTemplate
 
    prompt = PromptTemplate(
        input_variables=list(inputs.keys()),
        template=prompt_template,
    )
    backoff = _INITIAL_BACKOFF
 
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            t_start = time.perf_counter()
            llm = get_shared_llm(temperature=temp)
            chain = prompt | llm
            result = chain.invoke(inputs)
            duration = time.perf_counter() - t_start
 
            content = result.content if hasattr(result, "content") else str(result)
            logger.info(
                f"[LLM] Call completed in {duration:.3f}s "
                f"(attempt {attempt}/{_MAX_RETRIES})"
            )
            return content.strip()
 
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in [
                "429", "rate_limit", "resource_exhausted",
                "quota", "too many requests"
            ])
 
            if is_rate_limit and attempt < _MAX_RETRIES:
                logger.warning(
                    f"[LLM] Rate limit hit (attempt {attempt}/{_MAX_RETRIES}). "
                    f"Retrying in {backoff}s..."
                )
                time.sleep(backoff)
                backoff *= _BACKOFF_MULTIPLIER
                continue
 
            logger.error(f"[LLM] Call failed after {attempt} attempt(s): {e}")
            raise
 
    raise RuntimeError(
        f"LLM call failed after {_MAX_RETRIES} retries. "
        "The AI service may be temporarily unavailable."
    )