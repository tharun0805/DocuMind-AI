import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        logger.error("GOOGLE_API_KEY not set in .env file")
        raise ValueError("Please add your GOOGLE_API_KEY to the .env file")
    logger.info("Google API key loaded successfully")
    return api_key


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        return ""
    return api_key