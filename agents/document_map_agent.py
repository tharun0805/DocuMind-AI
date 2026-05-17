import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def extract_document_map(text: str) -> list[dict]:
    logger.info("Extracting document map...")

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract the structure of this document.
        List all major sections headings and key topics.

        Text: {text}

        Format each section as:
        SECTION: [section title]
        DESCRIPTION: [one sentence description]

        List up to 10 sections.
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0)
            chain = prompt | llm
            result = chain.invoke({"text": text[:5000]})
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

            logger.info(f"Document map: {len(sections)} sections")
            return sections

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return []

    return []