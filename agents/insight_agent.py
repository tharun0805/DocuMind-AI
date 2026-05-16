import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.2
    )


def generate_insights(text: str) -> dict:
    logger.info("Generating living insights from document...")

    sample = text[:4000]

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze this document and extract structured intelligence.

        Document: {text}

        Provide analysis with these exact sections:

        SUMMARY: [2-3 sentence overview of entire document]

        KEY_FINDINGS: [3 most important findings one per line]

        ACTION_ITEMS: [up to 3 action items mentioned one per line]

        RISKS: [up to 3 risks or concerns identified one per line]

        DECISIONS: [up to 3 key decisions or recommendations one per line]

        NEXT_STEPS: [2 recommended next steps for the reader one per line]
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({"text": sample})
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
                            insights[current_key] = "\n".join(
                                current_lines
                            )
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
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return {}