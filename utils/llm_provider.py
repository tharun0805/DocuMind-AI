import os
import streamlit as st
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_shared_llm(temperature: float = 0.3):
    groq_key = os.getenv("GROQ_API_KEY", "")

    if groq_key and groq_key not in ["your_groq_key_here", ""]:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
                temperature=temperature,
                max_tokens=2048
            )
            logger.info("Using Groq llama-3.3-70b — ultra fast and smart")
            return llm
        except Exception as e:
            logger.warning(f"Groq failed: {e}")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from utils.config import get_google_api_key
    logger.info("Using Gemini fallback")
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=temperature
    )