import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def generate_insights(text: str) -> dict:
    logger.info("Generating living insights...")

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze this document and extract structured intelligence.

        Document: {text}

        SUMMARY: [2-3 sentence overview]

        KEY_FINDINGS: [3 most important findings one per line]

        ACTION_ITEMS: [up to 3 action items one per line]

        RISKS: [up to 3 risks identified one per line]

        DECISIONS: [up to 3 key decisions one per line]

        NEXT_STEPS: [2 recommended next steps one per line]
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.2)
            chain = prompt | llm
            result = chain.invoke({"text": text[:4000]})
            content = result.content.strip()

            insights = {}
            current_key = None
            current_lines = []

            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue

                found = False
                for key in [
                    "SUMMARY", "KEY_FINDINGS", "ACTION_ITEMS",
                    "RISKS", "DECISIONS", "NEXT_STEPS"
                ]:
                    if line.startswith(key + ":"):
                        if current_key:
                            insights[current_key] = "\n".join(current_lines)
                        current_key = key
                        rest = line[len(key)+1:].strip()
                        current_lines = [rest] if rest else []
                        found = True
                        break

                if not found and current_key:
                    current_lines.append(line)

            if current_key:
                insights[current_key] = "\n".join(current_lines)

            logger.info(f"Generated {len(insights)} insight sections")
            return insights

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return {}

    return {}