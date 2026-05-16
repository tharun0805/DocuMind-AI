import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.4
    )


def generate_suggestions(question: str, answer: str) -> list[str]:
    logger.info("Generating follow-up suggestions...")

    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="""
        Based on this question and answer generate exactly 3 smart
        follow-up questions to help the user explore further.

        Question: {question}
        Answer: {answer}

        Generate 3 short follow-up questions maximum 8 words each.
        One question per line. No numbering. No bullets.

        Questions:
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({
                "question": question,
                "answer": answer
            })

            lines = [
                line.strip()
                for line in result.content.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]

            suggestions = lines[:3]
            logger.info(f"Generated {len(suggestions)} suggestions")
            return suggestions

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

    return []