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


def extract_document_map(text: str) -> list[dict]:
    logger.info("Extracting document map...")

    sample = text[:5000]

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract the structure of this document.
        List all major sections headings and key topics found.

        Text: {text}

        Format each section exactly as:
        SECTION: [section title]
        DESCRIPTION: [one sentence description of what this section covers]

        List up to 10 sections.
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({"text": sample})
            content = result.content.strip()

            sections = []
            current_section = {}

            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("SECTION:"):
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        "title": line[8:].strip(),
                        "description": ""
                    }
                elif line.startswith("DESCRIPTION:") and current_section:
                    current_section["description"] = line[12:].strip()

            if current_section:
                sections.append(current_section)

            logger.info(f"Document map: {len(sections)} sections found")
            return sections

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return []