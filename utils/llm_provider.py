import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import get_google_api_key
from loguru import logger


@st.cache_resource(show_spinner=False)
def get_shared_llm(temperature: float = 0.3):
    logger.info("Initializing shared Gemini LLM...")
    api_key = get_google_api_key()
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature
    )
    logger.info("Shared LLM ready")
    return llm