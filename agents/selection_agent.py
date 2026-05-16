import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.3
    )


def ask_about_selection(selected_text: str, action: str) -> str:
    logger.info(f"Processing selection with action: {action}")

    prompt = PromptTemplate(
        input_variables=["selected_text", "action"],
        template="""
        You are DocuMind AI analyzing a specific selected portion of a document.

        Selected Text:
        {selected_text}

        User Action: {action}

        Perform the requested action on this selected text.
        Be thorough, clear, and helpful.
        Explain in your own words — do not just repeat the text.
        Provide real value and deep analysis.

        Response:
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({
                "selected_text": selected_text,
                "action": action
            })

            logger.info("Selection analysis complete")
            return result.content.strip()

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return "Could not process selection. Please try again."