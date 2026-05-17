import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def extract_entities(text: str) -> dict:
    logger.info("Extracting entities from document...")

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
            llm = get_shared_llm(temperature=0)
            chain = prompt | llm
            result = chain.invoke({"text": text[:3000]})
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
                time.sleep(60 * (attempt + 1))
            else:
                return {}

    return {}