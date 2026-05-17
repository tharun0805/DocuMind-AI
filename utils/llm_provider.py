import streamlit as st
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_shared_llm(temperature: float = 0.3):
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key and groq_key != "your_groq_key_here":
        try:
            from langchain_groq import ChatGroq
            logger.info("Using Groq LLM — ultra fast")
            return ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_key,
                temperature=temperature
            )
        except Exception as e:
            logger.warning(f"Groq failed: {e}. Falling back to Gemini.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from utils.config import get_google_api_key
    logger.info("Using Gemini LLM")
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=temperature
    )