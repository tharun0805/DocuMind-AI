import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def ask_about_selection(selected_text: str, action: str) -> str:
    logger.info(f"Processing selection: {action}")

    prompt = PromptTemplate(
        input_variables=["selected_text", "action"],
        template="""
        You are DocuMind AI analyzing a specific selected portion of a document.

        Selected Text:
        {selected_text}

        User Action: {action}

        Perform the requested action thoroughly.
        Explain in your own words — do not just repeat the text.

        Response:
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.3)
            chain = prompt | llm
            result = chain.invoke({
                "selected_text": selected_text,
                "action": action
            })
            return result.content.strip()

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return "Could not process selection. Please try again."

    return "Could not process selection. Please try again."