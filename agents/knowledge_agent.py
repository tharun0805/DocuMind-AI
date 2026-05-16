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


def expand_knowledge(question: str, answer: str) -> dict:
    logger.info("Expanding knowledge based on answer...")

    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="""
        Based on this document question and answer suggest learning resources.

        Question: {question}
        Answer: {answer}

        Provide exactly:

        YOUTUBE_SEARCHES: [3 specific YouTube search queries to learn more, one per line]

        RELATED_TOPICS: [3 related topics worth exploring, one per line]

        SIMILAR_RESOURCES: [3 types of resources to find like documentation papers tutorials, one per line]

        LEARN_MORE: [2 specific things to study next, one per line]
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({
                "question": question,
                "answer": answer
            })

            content = result.content.strip()
            knowledge = {}
            current_key = None
            current_items = []

            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue

                found = False
                for key in [
                    "YOUTUBE_SEARCHES",
                    "RELATED_TOPICS",
                    "SIMILAR_RESOURCES",
                    "LEARN_MORE"
                ]:
                    if line.startswith(key + ":"):
                        if current_key:
                            knowledge[current_key] = current_items
                        current_key = key
                        rest = line[len(key)+1:].strip()
                        current_items = [rest] if rest else []
                        found = True
                        break

                if not found and current_key and line:
                    current_items.append(line)

            if current_key:
                knowledge[current_key] = current_items

            logger.info("Knowledge expansion complete")
            return knowledge

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return {}