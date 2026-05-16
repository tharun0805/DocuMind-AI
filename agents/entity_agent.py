import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0
    )


def extract_entities(text: str) -> dict:
    logger.info("Extracting entities from document...")

    sample = text[:3000]

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract key entities from this document text.
        For each type list up to 5 most important items.
        If none found write NONE.

        Text: {text}

        PEOPLE: [comma separated names]
        ORGANIZATIONS: [comma separated org names]
        DATES: [comma separated dates]
        LOCATIONS: [comma separated places]
        KEY_TERMS: [comma separated important terms]
        ACTION_ITEMS: [comma separated tasks or actions]
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({"text": sample})
            content = result.content.strip()

            entities = {}
            for line in content.split("\n"):
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if value and value.upper() != "NONE":
                        items = [
                            v.strip()
                            for v in value.split(",")
                            if v.strip()
                        ]
                        if items:
                            entities[key] = items

            logger.info(f"Extracted {len(entities)} entity types")
            return entities

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return {}