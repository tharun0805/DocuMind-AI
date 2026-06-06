"""
utils/llm_provider.py
FIX: Cache LLM instances with @st.cache_resource.
Previously created a brand-new LLM object on every single agent call,
adding significant overhead and logging noise.
"""
import os
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@st.cache_resource(show_spinner=False)
def _groq_llm():
    key = os.getenv("GROQ_API_KEY", "")
    if key and key not in ["your_groq_key_here", ""]:
        try:
            from langchain_groq import ChatGroq
            logger.debug("Groq LLM instance created")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=key,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")
    return None


@st.cache_resource(show_spinner=False)
def _gemini_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from utils.config import get_google_api_key
    logger.debug("Gemini LLM instance created")
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.3,
    )


def get_shared_llm(temperature: float = 0.3):
    """
    Return cached LLM. Created once, reused forever.
    Temperature variation applied at call time if needed.
    """
    llm = _groq_llm() or _gemini_llm()
    if temperature != 0.3:
        try:
            return llm.with_config(configurable={"temperature": temperature})
        except Exception:
            pass
    return llm