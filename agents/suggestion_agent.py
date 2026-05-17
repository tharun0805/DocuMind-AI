import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


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
            llm = get_shared_llm(temperature=0.4)
            chain = prompt | llm
            result = chain.invoke({
                "question": question,
                "answer": answer
            })

            lines = [
                line.strip()
                for line in result.content.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]

            return lines[:3]

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                return []

    return []